#!/usr/bin/env python3
"""The reference window advanced at one lap-wide mean speed (RC3).

`_reference_window` used to march its lookahead ahead by
`dt * mean(reference_speeds)` -- a single scalar for the whole lap, 1.86 m/s,
regardless of where the car actually was. Two separate faults live in that one
expression:

  * the speed is wrong on both halves of the track, and
  * the window advanced at the *reference's* pace rather than the *car's*, so
    the horizon sat at an arbitrary distance ahead instead of the distance the
    car covers in k*dt.

These tests pin the fix: the window must advance by the car's speed, and must
read position-dependent speed off the profile rather than a global mean.
"""

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, os.pardir, 'team_driver')
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from mpc_controller import MPCController  # noqa: E402
from speed_profile import closed_curvature, speed_profile  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

SPAWN_WAYPOINT = 377   # where the referee spawns the car on icra26
HORIZON = 8


def _centerline():
    """The real track, so the speed profile is the real one."""
    csv = os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir,
                       'maps', 'icra26_centerline.csv')
    path = np.loadtxt(csv, delimiter=',', comments='#')
    return np.atleast_2d(path)[:, :2]


def _controller(path):
    mpc = MPCController(vehicle_model=VehicleModel(),
                       prediction_horizon=HORIZON, control_horizon=HORIZON,
                       dt=0.025)
    profile = speed_profile(path, v_max=3.0, a_lat_max=0.8, lock_margin=0.95,
                            a_accel_max=9.51, a_brake_max=0.8)
    mpc.set_reference_path(path, profile)
    return mpc


class TestWindowAdvancesAtCarSpeed(unittest.TestCase):
    """The lookahead must march at the car's pace, not the reference mean's."""

    def setUp(self):
        self.path = _centerline()
        self.mpc = _controller(self.path)

    def test_window_spacing_matches_car_speed_not_profile_mean(self):
        """Consecutive reference states must be v_car*dt apart."""
        v_car = 1.6
        window = self.mpc._reference_window(12.0, HORIZON + 1, car_speed=v_car)
        pts = window[:, :2]

        gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.assertAlmostEqual(v_car * self.mpc.dt, float(np.median(gaps)),
                               delta=0.02,
                               msg=f'window spacing {np.median(gaps):.4f} m '
                                   f'is not v_car*dt = {v_car * self.mpc.dt:.4f} m')

    def test_spacing_tracks_car_speed(self):
        """Spacing must be max(car_speed, floor) * dt at every speed.

        Deliberately checked across the floor rather than by a ratio: the floor
        sits below the profile minimum on purpose, so a "double the speed,
        double the spacing" ratio would only hold for speeds above the floor.
        """
        for v in (0.0, 0.1, 0.3, 1.6, 3.0):
            with self.subTest(car_speed=v):
                window = self.mpc._reference_window(12.0, HORIZON + 1, car_speed=v)
                gaps = np.linalg.norm(np.diff(window[:, :2], axis=0), axis=1)
                expected = max(v, self.mpc._ref_speed_floor) * self.mpc.dt
                self.assertAlmostEqual(expected, float(np.median(gaps)), delta=0.002,
                                       msg=f'window spacing wrong at v={v}')

    def test_zero_speed_window_does_not_collapse_or_run_away(self):
        """A stopped car must still get a horizon it can act on.

        Spacing purely by car speed would freeze the window on one point, which
        is the stall the arc-length anchor was introduced to remove -- so the
        implementation carries a floor. At the floor the whole horizon spans
        only a few centimetres, which is the point: a stopped car should see
        only the track immediately in front of it.
        """
        window = self.mpc._reference_window(12.0, HORIZON + 1, car_speed=0.0)
        pts = window[:, :2]
        span = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        expected = self.mpc._ref_speed_floor * self.mpc.dt * HORIZON
        self.assertGreater(span, 0.5 * expected,
                           'zero-speed window collapsed to a single point')
        self.assertLess(span, 2.0 * expected,
                        'zero-speed window ran away down the track')


class TestWindowSpeedVariesAlongTrack(unittest.TestCase):
    """Reference speed must vary with position, not be one lap-wide mean."""

    def setUp(self):
        self.path = _centerline()
        self.mpc = _controller(self.path)

    def _window_at_waypoint(self, idx):
        return self.mpc._reference_window(float(self.mpc._path_arc[idx]),
                                          HORIZON + 1, car_speed=2.0)

    def test_speed_is_not_constant_around_the_lap(self):
        """The whole point of RC3: two corners must not both see the mean."""
        speeds = [float(np.mean(self._window_at_waypoint(i)[:, 3]))
                  for i in (0, 50, 120, 200, 280, 350)]

        self.assertGreater(max(speeds) - min(speeds), 0.4,
                           f'reference speed is flat around the lap: {speeds}')
        self.assertLess(max(speeds) - min(speeds), 3.0,
                        'reference speed looks implausibly jumpy, not profiled')

    def test_slow_corner_reads_slower_than_straight(self):
        """The tightest corner must ask for less speed than the fastest straight."""
        kappa = closed_curvature(self.mpc.reference_path)
        speeds = self.mpc.reference_speeds
        tight = int(np.argmax(kappa))
        fast = int(np.argmin(kappa))

        self.assertLess(speeds[tight], speeds[fast],
                        'tight corner is not slower than the straight')

        at_tight = float(np.mean(self._window_at_waypoint(tight)[:, 3]))
        at_fast = float(np.mean(self._window_at_waypoint(fast)[:, 3]))
        self.assertLess(at_tight, at_fast,
                        'window does not report a slower speed in the tight corner')

    def test_wrap_preserves_profile_speed(self):
        """Speed must still be interpolated across the finish line."""
        near_end = float(self.mpc._path_arc[-1]) - 0.1
        window = self.mpc._reference_window(near_end, HORIZON + 1, car_speed=2.0)
        self.assertTrue(np.all(window[:, 3] > 0.0),
                        'speed went non-positive across the wrap')
        self.assertTrue(np.all(np.isfinite(window[:, 3])))


class TestRefSpeedProperty(unittest.TestCase):
    """`ref_speed` was the flat lap mean. It is now a floored estimate of where
    the car is going, so the horizon is never degenerate when the car is slow."""

    def setUp(self):
        self.path = _centerline()
        self.mpc = _controller(self.path)

    def test_ref_speed_floor_is_positive(self):
        """The window must never advance by zero."""
        self.assertGreater(self.mpc.ref_speed, 0.0)

    def test_ref_speed_floor_is_below_the_profile(self):
        """The floor is a stall guard, not a speed estimate.

        It is deliberately allowed to sit *below* the slowest point on the
        profile: raising it to bound the profile would let the floor override
        the car's real speed through a slow corner, which is the bug this
        change exists to remove. It must still be a small fraction of the mean,
        so it only ever binds for a car that is barely moving.
        """
        mean = float(np.mean(self.mpc.reference_speeds))
        self.assertLess(self.mpc._ref_speed_floor, mean / 4.0,
                        'floor is high enough to distort a slow corner')
        self.assertGreater(self.mpc._ref_speed_floor, 0.0,
                           'floor must stay positive or the window collapses')

    def test_ref_speed_default_stays_the_profile_mean(self):
        """The no-car-speed fallback keeps its original meaning."""
        mean = float(np.mean(self.mpc.reference_speeds))
        self.assertAlmostEqual(mean, self.mpc.ref_speed, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
