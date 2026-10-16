"""Regression tests for the lap wraparound in the MPC reference.

The car stalls on the real track if the reference window runs off the end of
the path array. The array holds exactly one lap, and the referee spawns the car
at waypoint 377 of 392 -- 2.8 m from the end -- so the failure fires within
seconds of every run, not at the end of a lap.

These tests pin the two places that must wrap:
  * `_reference_window`, which resamples the lookahead horizon, and
  * `_nearest_path_index`, whose search window used to stop at the array ends.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

from mpc_controller import MPCController  # noqa: E402
from speed_profile import speed_profile  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))
CENTERLINE = os.path.join(REPO, 'maps', 'icra26_centerline.csv')

# maps/tracks.yaml: on the bottom straight, facing along +x.
START_POSE = (-1.68, -0.01, 0.0)
SPAWN_WAYPOINT = 377
HORIZON = 8


def _centerline():
    path = np.loadtxt(CENTERLINE, delimiter=',', comments='#')
    return np.atleast_2d(path)[:, :2]


def _controller(path):
    mpc = MPCController(
        vehicle_model=VehicleModel(),
        prediction_horizon=HORIZON, control_horizon=HORIZON, dt=0.025)
    profile = speed_profile(path, v_max=3.0, a_lat_max=0.8, lock_margin=0.95,
                            a_accel_max=9.51, a_brake_max=0.8)
    mpc.set_reference_path(path, profile)
    return mpc


def _circular_distance(a, b, n):
    return abs((a - b + n // 2) % n - n // 2)


class TestReferenceWindowWraps(unittest.TestCase):
    """The lookahead horizon must advance at every point on the lap."""

    def setUp(self):
        self.path = _centerline()
        self.mpc = _controller(self.path)

    def test_no_anchor_collapses_to_a_single_point(self):
        """The original bug: every horizon state clamped onto waypoint 391.

        A window with one distinct point tells the optimiser the goal is to sit
        there, which is what froze the car.
        """
        for s in np.linspace(0.0, self.mpc._path_len, 400, endpoint=False):
            window = self.mpc._reference_window(s, HORIZON + 1)
            distinct = len(np.unique(np.round(window[:, :2], 6), axis=0))
            self.assertEqual(
                distinct, HORIZON + 1,
                f'anchor s={s:.2f} collapsed to {distinct} point(s)')

    def test_anchor_just_past_the_end_of_the_array(self):
        """The exact failure point: array span is 0.2 m short of a full lap."""
        span = float(self.mpc._path_arc[-1])
        self.assertLess(span, self.mpc._path_len)
        for s in (span, span + 0.05, self.mpc._path_len - 1e-6):
            window = self.mpc._reference_window(s, HORIZON + 1)
            distinct = len(np.unique(np.round(window[:, :2], 6), axis=0))
            self.assertEqual(distinct, HORIZON + 1, f'anchor s={s:.4f}')

    def test_spawn_window_advances_and_does_not_stop_at_the_last_waypoint(self):
        window = self.mpc._reference_window(
            self.mpc._path_arc[SPAWN_WAYPOINT], HORIZON + 1)
        steps = np.linalg.norm(np.diff(window[:, :2], axis=0), axis=1)
        self.assertTrue(np.all(steps > 0.0),
                        f'window repeated a coordinate: {steps}')


class TestNearestIndexWraps(unittest.TestCase):
    """The search window used to be a plain slice, so it could not reach
    across the finish line: a car past the end of the array kept matching
    waypoint 391 however far it travelled."""

    def setUp(self):
        self.path = _centerline()
        self.mpc = _controller(self.path)

    def test_hint_at_the_last_waypoint_still_finds_the_start(self):
        # A point just past waypoint 5 is only reachable by wrapping.
        target = self.path[5] + np.array([0.01, 0.0])
        found = self.mpc._nearest_path_index(target, len(self.path) - 1)
        self.assertLessEqual(
            _circular_distance(found, 5, len(self.path)), 2,
            f'expected a waypoint near 5, got {found}')

    def test_hint_at_the_first_waypoint_still_finds_the_end(self):
        found = self.mpc._nearest_path_index(self.path[-4], 0)
        self.assertLessEqual(
            _circular_distance(found, len(self.path) - 4, len(self.path)), 2)

    def test_closest_waypoint_is_reachable_from_the_spawn_hint(self):
        """Reproduces the stall: hint 391, car sitting near waypoint 11."""
        target = self.path[11] + np.array([0.0, 0.05])
        found = self.mpc._nearest_path_index(target, len(self.path) - 1)
        self.assertLessEqual(_circular_distance(found, 11, len(self.path)), 2)


class TestSolveAnchorsContinuously(unittest.TestCase):
    """solve() used to snap the reference to a waypoint index, so the anchor
    moved in 0.2 m jumps while the car moved v*dt (about 0.04 m)."""

    SPEED = 1.6

    def setUp(self):
        self.path = _centerline()
        self.mpc = _controller(self.path)

    def _state_at(self, s):
        """Car pose on the path at arc position s, heading along the path."""
        ref = self.mpc._reference_window(s % self.mpc._path_len, 2)
        return np.array([ref[0, 0], ref[0, 1], ref[0, 2], self.SPEED, 0.0])

    def test_anchor_advances_by_the_distance_actually_travelled(self):
        """With the car tracking the line, the anchor must advance by v*dt.

        Driving the car along a fixed heading instead of along the path would
        be the wrong test: the resync in solve() is *supposed* to correct a
        car that has genuinely left the line, and that correction is a whole
        waypoint by design.
        """
        s = float(self.mpc._path_arc[SPAWN_WAYPOINT])
        state = self._state_at(s)
        previous = None
        steps = []
        for _ in range(400):
            self.mpc.get_control(state)
            anchor = self.mpc._s_ref
            if previous is not None:
                steps.append(
                    abs((anchor - previous + self.mpc._path_len / 2)
                        % self.mpc._path_len - self.mpc._path_len / 2))
            previous = anchor
            s = (s + self.SPEED * self.mpc.dt) % self.mpc._path_len
            state = self._state_at(s)

        self.assertTrue(steps)
        expected = self.SPEED * self.mpc.dt
        self.assertLess(max(steps), self.mpc._mean_waypoint_spacing,
                        'reference anchor still jumps by a whole waypoint')
        self.assertLess(max(abs(np.array(steps) - expected)), 0.01,
                        f'anchor step drifted from v*dt = {expected:.4f} m')


if __name__ == '__main__':
    unittest.main()
