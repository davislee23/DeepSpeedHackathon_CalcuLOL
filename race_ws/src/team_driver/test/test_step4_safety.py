"""Step 4: the output path re-based on the plant's speed loop, safety net kept."""

import dataclasses
import math
import os
import sys
import unittest
from collections import deque

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import driver_core  # noqa: E402
import lap_sim  # noqa: E402
import longitudinal  # noqa: E402
import plant_sim  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from vehicle_model import VehicleModel, VehicleParameters  # noqa: E402

DT = 0.025
VP = VehicleParameters()
LR = float(VP.lr)
CFG = driver_core.DriverConfig()
ANGLE_MIN = -3.0 * math.pi / 4.0
ANGLE_INC = math.radians(0.25)
ANGLES = ANGLE_MIN + np.arange(1081) * ANGLE_INC
MAX_RANGE = CFG.max_range
OVERHANG = longitudinal.front_overhang(VP)
FRONT_FROM_BASE_LINK = longitudinal.LIDAR_FORWARD + OVERHANG


def _cast(origin, yaw, segments):
    """Ranges from `origin` along ANGLES + yaw to the nearest segment, clipped."""
    o = np.asarray(origin, dtype=float)
    d = np.stack([np.cos(ANGLES + yaw), np.sin(ANGLES + yaw)], axis=1)
    ranges = np.full(len(ANGLES), MAX_RANGE)
    for a, b in segments:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        e = b - a
        denom = d[:, 0] * e[1] - d[:, 1] * e[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((a[0] - o[0]) * e[1] - (a[1] - o[1]) * e[0]) / denom
            u = ((a[0] - o[0]) * d[:, 1] - (a[1] - o[1]) * d[:, 0]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        ranges = np.where(hit, np.minimum(ranges, t), ranges)
    return ranges


def _lidar_scan(x, y, yaw, segments):
    """The scan a car at base_link (x, y, yaw) would see."""
    f = longitudinal.LIDAR_FORWARD
    return _cast((x + f * math.cos(yaw), y + f * math.sin(yaw)), yaw, segments)


def _wall(distance):
    """A flat wall across the lidar's path, `distance` ahead of it."""
    return _cast((0.0, 0.0), 0.0, [((distance, -20.0), (distance, 20.0))])


class _PathMPC:
    """Just enough of MPCController: scripted controls, an arc position, a plan."""

    dt = DT

    def __init__(self, arc_of, steer=0.0, accel=0.0, plan_of=None):
        self.arc_of = arc_of
        self.steer = steer
        self.accel = accel
        self.plan_of = plan_of
        self.arc_position = None
        self._plan = None

    def planned_path(self):
        return self._plan

    def speed_profile(self, **_kw):
        return np.full(4, 3.0)

    def set_reference_path(self, *_a):
        pass

    def get_control(self, state):
        self.arc_position = float(self.arc_of(state))
        self._plan = None if self.plan_of is None else self.plan_of(state)
        return np.array([self.steer, self.accel]), {'solve_time_ms': 0.0}


def _straight_plan(state, seconds=0.5):
    """What a plan holding the heading would drive through, from `state`."""
    reach = max(float(state[3]) * seconds, 0.5)
    steps = np.linspace(0.0, reach, 9)[:, None]
    return np.asarray(state[:2], dtype=float) + steps * [math.cos(state[2]), math.sin(state[2])]


def _line(x0=-1.0, length=80.0, spacing=0.1):
    xs = x0 + np.arange(0.0, length, spacing)
    return np.column_stack([xs, np.zeros_like(xs)])


def _core_on(path, speeds, arc_of, cfg=None, **mpc_kw):
    mpc = _PathMPC(arc_of, **mpc_kw)
    core = driver_core.DriverCore(cfg=cfg or driver_core.DriverConfig(), mpc_controller=mpc)
    core.set_reference(path, speeds)
    return core, mpc


def _closed_loop(core, plant, segments, seconds, stale, on_step):
    """Drive GymPlant from stale snapshots, as lap_sim does, until on_step says stop.

    Each 25 ms tick (3 then 2 physics steps) the core plans from the snapshot
    `stale` steps old; its command latches one physics step later.
    """
    ring = deque([(plant.state.copy(), plant.time)], maxlen=stale + 1)
    tick = 0
    while plant.time < seconds:
        state, t_snap = ring[0]
        x, y, yaw = plant_sim.base_link_pose_of(state, LR)
        vx, vy, wz = plant_sim.odom_twist_of(state)
        core.set_odom(x, y, yaw, math.hypot(vx, vy), beta=math.atan2(vy, vx),
                      yaw_rate=wz, t_sim=t_snap)
        core.set_scan(_lidar_scan(x, y, yaw, segments), ANGLE_MIN, ANGLE_INC, t_sim=t_snap)
        cmd = core.step(t_sim=t_snap)
        period = 3 if tick % 2 == 0 else 2
        tick += 1
        for i in range(period):
            if i == 1:
                plant.latch(cmd.steering_angle, cmd.speed)
                core.note_published(cmd, t_sim=plant.time)
            plant.step()
            ring.append((plant.state.copy(), plant.time))
            if on_step(plant):
                return True
    return False


def _plant_at(x_base_link, v):
    plant = plant_sim.GymPlant()
    plant.reset_state([x_base_link + LR, 0.0, 0.0, v, 0.0, 0.0, 0.0])
    return plant


def _x(plant):
    return plant_sim.base_link_pose_of(plant.state, LR)[0]


# ----------------------------------------------------------------------
# Clearance along the planned path
# ----------------------------------------------------------------------
class PathClearanceTest(unittest.TestCase):

    def _clear(self, ranges, bearing=0.0):
        return longitudinal.path_clearance(ranges, ANGLES, bearing, VP.width / 2.0, MAX_RANGE)

    def test_a_wall_across_the_bearing_is_its_distance(self):
        self.assertAlmostEqual(self._clear(_wall(2.0)), 2.0, places=6)

    def test_a_wall_seen_down_a_turned_bearing_is_further_along_it(self):
        # The corridor's near edge reaches a slanted wall first.
        clearance = self._clear(_wall(2.0), bearing=0.3)
        self.assertGreater(clearance, 2.0)
        self.assertLess(clearance, 2.0 / math.cos(0.3))

    def test_the_clearance_ignores_a_wall_straight_ahead_when_the_reference_turns_away(self):
        # A hairpin: a wall 2 m dead ahead, ending 0.4 m to the left, and the
        # line turning left past its end. The old straight-ahead cone saw 2 m.
        ranges = _cast((0.0, 0.0), 0.0, [((2.0, -3.0), (2.0, 0.4))])
        cone = ranges[np.abs(ANGLES) <= 0.2]
        self.assertAlmostEqual(float(np.percentile(cone, 10)), 2.0, delta=0.02)
        self.assertEqual(self._clear(ranges, bearing=math.radians(45.0)), MAX_RANGE)

    def test_walls_beside_the_corridor_do_not_count(self):
        ranges = _cast((0.0, 0.0), 0.0, [((-1.0, 0.3), (20.0, 0.3)),
                                          ((-1.0, -0.3), (20.0, -0.3))])
        self.assertEqual(self._clear(ranges), MAX_RANGE)

    def test_the_low_end_of_the_corridor_counts_not_its_mean(self):
        # A post 1.5 m ahead blocks a fifth of the corridor's beams.
        ranges = _wall(6.0)
        inside = np.where((np.abs(np.sin(ANGLES)) * ranges <= VP.width / 2.0)
                          & (np.cos(ANGLES) > 0.0))[0]
        post = inside[: max(1, len(inside) // 5)]
        ranges[post] = 1.5
        self.assertLess(self._clear(ranges), 1.6)

    def test_nothing_in_the_corridor_is_the_full_range(self):
        self.assertEqual(self._clear(np.full(len(ANGLES), MAX_RANGE)), MAX_RANGE)

    def test_the_bearing_looks_further_ahead_at_speed(self):
        self.assertEqual(longitudinal.bearing_lookahead(0.0), longitudinal.BEARING_LOOKAHEAD_MIN)
        self.assertEqual(longitudinal.bearing_lookahead(20.0), longitudinal.BEARING_LOOKAHEAD_MAX)
        self.assertGreater(longitudinal.bearing_lookahead(5.0), longitudinal.bearing_lookahead(3.0))
        self.assertEqual((longitudinal.BEARING_LOOKAHEAD_MIN, longitudinal.BEARING_LOOKAHEAD_MAX),
                         (1.0, 2.0))

    def test_the_overhang_is_the_body_ahead_of_the_lidar(self):
        # simulator._compute_collision_body_offset: the body box is centred
        # collision_body_center_x ahead of base_link; the lidar 0.275 m.
        self.assertAlmostEqual(OVERHANG, VP.collision_body_center_x + VP.length / 2.0 - 0.275,
                               places=12)
        self.assertGreater(OVERHANG, 0.15)


class AlongPathClearanceTest(unittest.TestCase):
    HALF = VP.width / 2.0

    @staticmethod
    def _wall_x(x, y0=-1.0, y1=1.0):
        ys = np.arange(y0, y1, 0.01)
        return np.column_stack([np.full_like(ys, x), ys])

    def _clear(self, points, path, length=6.0):
        return longitudinal.clearance_along_path(points, np.asarray(path, dtype=float),
                                                 self.HALF, length)

    def test_a_wall_across_the_path_is_its_arc_distance(self):
        self.assertAlmostEqual(self._clear(self._wall_x(3.0), [[0, 0], [4, 0]]), 3.0, places=9)

    def test_the_path_is_continued_straight_past_its_end(self):
        self.assertAlmostEqual(self._clear(self._wall_x(3.0), [[0, 0], [1, 0]]), 3.0, places=9)

    def test_the_outside_wall_of_a_bend_the_path_takes_is_not_in_the_corridor(self):
        phi = np.linspace(0.0, math.pi / 2.0, 20)
        bend = np.column_stack([np.sin(phi), 1.0 - np.cos(phi)])     # left, radius 1
        self.assertEqual(self._clear(self._wall_x(1.5, -1.0, 0.5), bend), 6.0)
        # The straight-ahead corridor would have seen it.
        self.assertLess(self._clear(self._wall_x(1.5, -1.0, 0.5), [[0, 0], [1, 0]]), 6.0)

    def test_walls_beside_the_path_are_not_in_it(self):
        ys = np.full(400, 0.3)
        xs = np.linspace(0.0, 4.0, 400)
        sides = np.vstack([np.column_stack([xs, ys]), np.column_stack([xs, -ys])])
        self.assertEqual(self._clear(sides, [[0, 0], [4, 0]]), 6.0)

    def test_nothing_seen_and_a_degenerate_path_are_the_full_length(self):
        self.assertEqual(self._clear(np.zeros((0, 2)), [[0, 0], [4, 0]]), 6.0)
        self.assertEqual(self._clear(self._wall_x(3.0), [[1, 1], [1, 1]]), 6.0)

    def test_scan_points_are_the_beams_that_returned(self):
        ranges = np.array([0.0, 2.0, MAX_RANGE])
        angles = np.array([0.0, math.pi / 2.0, 0.0])
        points = longitudinal.scan_points(ranges, angles, (1.0, 1.0), 0.0, MAX_RANGE)
        np.testing.assert_allclose(points, [[1.0, 3.0]], atol=1e-12)


class PlannedPathTest(unittest.TestCase):

    def test_the_mpc_reports_the_path_its_plan_drives(self):
        phi = np.arange(0.0, 2.0 * math.pi, 0.04)
        circle = np.column_stack([5.0 * np.cos(phi), 5.0 * np.sin(phi)])
        mpc = MPCController(vehicle_model=VehicleModel())
        self.assertIsNone(mpc.planned_path())
        mpc.set_reference_path(circle, np.full(len(circle), 3.0))
        mpc.get_control(np.array([5.0, 0.0, math.pi / 2.0, 3.0, 0.0, 0.0]))
        path = mpc.planned_path()
        self.assertEqual(path.shape, (mpc.prediction_horizon + 1, 2))
        np.testing.assert_allclose(path[0], [5.0, 0.0])
        driven = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        self.assertAlmostEqual(driven, 3.0 * sum(mpc.dt_schedule), delta=0.3)


class ClosedPathTest(unittest.TestCase):

    def test_points_and_values_interpolate_along_the_arc_and_wrap(self):
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        path = longitudinal.ClosedPath(square, np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(path.length, 4.0)
        np.testing.assert_allclose(path.point_at(0.5), [0.5, 0.0])
        np.testing.assert_allclose(path.point_at(3.5), [0.0, 0.5])      # closing segment
        np.testing.assert_allclose(path.point_at(4.5), [0.5, 0.0])      # wrapped
        self.assertAlmostEqual(path.value_at(1.5), 2.5)
        self.assertAlmostEqual(path.value_at(3.5), 2.5)                 # 4 back to 1
        self.assertAlmostEqual(path.value_at(-0.5), 2.5)

    def test_a_path_it_cannot_describe_is_refused(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        bad = [
            (line[:1], None),                              # one point
            (np.zeros((3, 2)), None),                      # zero length
            (line, np.array([1.0, 2.0])),                  # too few values
            (line, np.array([1.0, np.nan, 3.0])),          # a NaN speed
            (np.array([[0.0, 0.0], [np.inf, 0.0]]), None),
        ]
        for xy, values in bad:
            with self.subTest(xy=xy.tolist()), self.assertRaises(ValueError):
                longitudinal.ClosedPath(xy, values)

    def test_lookups_it_cannot_answer_are_refused(self):
        path = longitudinal.ClosedPath(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
        with self.assertRaises(ValueError):
            path.value_at(0.5)
        with self.assertRaises(ValueError):
            path.point_at(float('nan'))

    def test_the_arc_is_the_mpcs_arc(self):
        # The caps look positions up by the MPC's arc_position, so the two
        # parameterisations must be one: same lap length, same points.
        phi = np.arange(0.0, 2.0 * math.pi, 0.04)
        circle = np.column_stack([5.0 * np.cos(phi), 5.0 * np.sin(phi)])
        speeds = np.full(len(circle), 3.0)
        mpc = MPCController(vehicle_model=VehicleModel())
        mpc.set_reference_path(circle, speeds)
        path = longitudinal.ClosedPath(circle, speeds)
        self.assertAlmostEqual(path.length, mpc._path_len, places=9)
        for angle in (0.3, 3.0, 6.2):                   # 6.2: on the closing segment
            pose = (5.0 * math.cos(angle), 5.0 * math.sin(angle))
            mpc.get_control(np.array([pose[0], pose[1], angle + math.pi / 2.0, 2.0, 0.0, 0.0]))
            np.testing.assert_allclose(path.point_at(mpc.arc_position), pose, atol=0.01)
            mpc.set_reference_path(circle, speeds)      # forget the tracked arc


# ----------------------------------------------------------------------
# The emergency cap in the core
# ----------------------------------------------------------------------
def _hairpin(straight=0.3, radius=1.0):
    """Straight ahead from the origin, then a left half-turn."""
    xs = np.arange(0.0, straight, 0.05)
    first = np.column_stack([xs, np.zeros_like(xs)])
    phi = np.arange(0.0, math.pi, 0.05 / radius)
    turn = np.column_stack([straight + radius * np.sin(phi), radius - radius * np.cos(phi)])
    return np.vstack([first, turn])


class EmergencyCapTest(unittest.TestCase):

    def _cmd(self, path, ranges, v=6.0):
        core, _ = _core_on(path, np.full(len(path), 10.0), lambda _s: 0.0)
        core.set_odom(0.0, 0.0, 0.0, v, t_sim=0.0)
        core.set_scan(ranges, ANGLE_MIN, ANGLE_INC, t_sim=0.0)
        return core.step(t_sim=0.0), core

    def _hairpin_wall(self):
        return _cast((longitudinal.LIDAR_FORWARD, 0.0), 0.0, [((2.0, -3.0), (2.0, 0.4))])

    def test_a_hairpin_wall_straight_ahead_does_not_cap_the_car(self):
        cmd, core = self._cmd(_hairpin(), self._hairpin_wall())
        self.assertAlmostEqual(cmd.speed, 6.0, places=9)
        self.assertEqual(core.conditioning_stats().emergency_binds, 0)

    def test_the_same_wall_on_a_straight_reference_caps_it(self):
        cmd, core = self._cmd(_line(x0=0.0), self._hairpin_wall())
        self.assertLess(cmd.speed, 6.0)
        self.assertEqual(core.conditioning_stats().emergency_binds, 1)

    def test_the_cap_is_the_braking_distance_of_the_body_not_the_lidar(self):
        core, _ = _core_on(_line(), np.full(800, 10.0), lambda s: s[0] + 1.0)
        cap = core._speed_cap(_wall(3.0), ANGLES)
        self.assertAlmostEqual(
            cap, math.sqrt(2.0 * CFG.a_emergency * (3.0 - OVERHANG - CFG.stop_margin)), places=6)

    def test_along_the_plan_the_cap_is_measured_from_the_bumper(self):
        core, _ = _core_on(_line(), np.full(800, 10.0), lambda s: s[0] + 1.0)
        # At 5 m/s the stop needs ~1.8 m plus the hold's 0.125 m, so a wall 3 m
        # ahead of base_link is inside the corridor's reach.
        core.set_odom(0.0, 0.0, 0.0, 5.0, t_sim=0.0)
        ranges = _lidar_scan(0.0, 0.0, 0.0, [((3.0, -5.0), (3.0, 5.0))])
        plan = np.array([[0.0, 0.0], [4.0, 0.0]])
        cap = core._path_cap(ranges, ANGLES, plan)
        reach = 3.0 - FRONT_FROM_BASE_LINK - 5.0 * DT
        self.assertAlmostEqual(cap, math.sqrt(2.0 * CFG.a_emergency * reach), places=6)
        # With the wall out of reach of any stop, there is no bound at all.
        far = _lidar_scan(0.0, 0.0, 0.0, [((7.5, -5.0), (7.5, 5.0))])
        self.assertEqual(core._path_cap(far, ANGLES, plan), CFG.v_max)

    def test_a_hairpin_plan_does_not_cap_the_car_for_the_wall_ahead(self):
        core, _ = _core_on(_hairpin(), np.full(len(_hairpin()), 10.0), lambda _s: 0.0,
                           plan_of=lambda _s: _hairpin()[:40])
        core.set_odom(0.0, 0.0, 0.0, 6.0, t_sim=0.0)
        core.set_scan(self._hairpin_wall(), ANGLE_MIN, ANGLE_INC, t_sim=0.0)
        self.assertAlmostEqual(core.step(t_sim=0.0).speed, 6.0, places=9)
        self.assertEqual(core.conditioning_stats().emergency_binds, 0)

    def test_a_plan_into_the_wall_caps_the_car(self):
        core, _ = _core_on(_hairpin(), np.full(len(_hairpin()), 10.0), lambda _s: 0.0,
                           plan_of=_straight_plan)
        core.set_odom(0.0, 0.0, 0.0, 6.0, t_sim=0.0)
        core.set_scan(self._hairpin_wall(), ANGLE_MIN, ANGLE_INC, t_sim=0.0)
        self.assertLess(core.step(t_sim=0.0).speed, 6.0)
        self.assertEqual(core.conditioning_stats().emergency_binds, 1)

    def test_distance_already_covered_comes_off_the_clearance(self):
        core, _ = _core_on(_line(), np.full(800, 10.0), lambda s: s[0] + 1.0)
        cap = core._speed_cap(_wall(3.0), ANGLES, travelled=0.5)
        self.assertAlmostEqual(
            cap, math.sqrt(2.0 * CFG.a_emergency * (2.5 - OVERHANG - CFG.stop_margin)), places=6)


class FailClosedTest(unittest.TestCase):

    def test_a_nan_cap_coasts_rather_than_being_skipped(self):
        for caps in (dict(speed_cap=float('nan')), dict(profile_speed=float('nan'))):
            with self.subTest(**caps):
                core, _ = _core_on(_line(), np.full(800, 10.0), lambda s: s[0] + 1.0)
                core.speed = 5.0
                cmd = core._condition(0.0, 3.0, DT, **caps)
                self.assertEqual(core.conditioning_stats().coast_count, 1)
                self.assertLess(cmd.speed, 5.0)

    def test_the_reactive_branch_reads_no_stale_profile(self):
        # A solved cycle on a crawl-speed profile, then the reference is lost:
        # the gap-follower must not be braked against the old arc position.
        core, _ = _core_on(_line(), np.full(800, 1.0), lambda s: s[0] + 1.0)
        core.set_odom(0.0, 0.0, 0.0, 3.0, t_sim=0.0)
        core.set_scan(np.full(len(ANGLES), MAX_RANGE), ANGLE_MIN, ANGLE_INC, t_sim=0.0)
        self.assertLess(core.step(t_sim=0.0).speed, 3.0)
        core.reference_path = None
        core.set_scan(np.full(len(ANGLES), MAX_RANGE), ANGLE_MIN, ANGLE_INC, t_sim=DT)
        self.assertAlmostEqual(core.step(t_sim=DT).speed, driver_core.REACTIVE_SPEED, places=9)


class ReviewFixesTest(unittest.TestCase):
    """Fail closed, follow an overrunning timer, forget a reset episode."""

    PLAN = np.array([[0.0, 0.0], [3.0, 0.0]])

    def _core(self):
        core, mpc = _core_on(_line(), np.full(800, 10.0), lambda s: s[0] + 1.0)
        core.set_odom(0.0, 0.0, 0.0, 5.0, t_sim=0.0)
        return core, mpc

    def test_a_blind_scan_brakes_instead_of_lifting_the_cap(self):
        core, _ = self._core()
        self.assertEqual(core._path_cap(np.zeros(len(ANGLES)), ANGLES, self.PLAN),
                         CFG.crawl_speed)

    def test_an_open_scan_lifts_it(self):
        core, _ = self._core()
        self.assertEqual(core._path_cap(np.full(len(ANGLES), MAX_RANGE), ANGLES, self.PLAN),
                         CFG.v_max)

    def test_a_plan_with_no_extent_or_no_finite_points_falls_back(self):
        core, _ = self._core()
        wall = _lidar_scan(0.0, 0.0, 0.0, [((2.0, -5.0), (2.0, 5.0))])
        straight = core._speed_cap(wall, ANGLES, 0.0, 5.0 * DT)
        for plan in (np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 0.0]]),
                     np.array([[0.0, 0.0], [np.nan, 0.0]])):
            with self.subTest(plan=plan.tolist()):
                self.assertEqual(core._path_cap(wall, ANGLES, plan), straight)
        self.assertLess(straight, 5.0)

    def test_an_overrunning_timer_moves_the_predicted_latch_out(self):
        core, _ = self._core()
        for k in range(10):                         # solves taking 60 ms each
            core.note_published(driver_core.DriveCommand(0.0, 5.0), t_sim=k * 0.06)
        self.assertGreater(core._publish_interval, 0.055)
        core.set_odom(0.0, 0.0, 0.0, 5.0, t_sim=0.50)
        travel = core._predict_latch(0.50).travelled
        self.assertAlmostEqual(travel, 5.0 * (0.54 + core._publish_interval - 0.50), places=3)

    def test_the_alternating_harness_period_averages_to_the_control_period(self):
        core, _ = self._core()
        t = 0.0
        for k in range(40):
            t += 0.03 if k % 2 else 0.02
            core.note_published(driver_core.DriveCommand(0.0, 5.0), t_sim=t)
        self.assertAlmostEqual(core._publish_interval, 0.025, delta=0.003)

    def test_a_sim_reset_resets_the_mpcs_tracking(self):
        phi = np.arange(0.0, 2.0 * math.pi, 0.04)
        circle = np.column_stack([5.0 * np.cos(phi), 5.0 * np.sin(phi)])
        mpc = MPCController(vehicle_model=VehicleModel())
        mpc.set_reference_path(circle, np.full(len(circle), 3.0))
        mpc.get_control(np.array([5.0, 0.0, math.pi / 2.0, 3.0, 0.0, 0.0]))
        self.assertIsNotNone(mpc.arc_position)
        mpc.reset_tracking()
        self.assertIsNone(mpc.arc_position)
        self.assertIsNone(mpc.planned_path())
        self.assertIsNone(mpc._last_path_index)

    def test_the_core_resets_the_mpc_when_the_sim_resets(self):
        core, mpc = self._core()
        calls = []
        mpc.reset_tracking = lambda: calls.append(1)
        scan = (np.full(len(ANGLES), MAX_RANGE), ANGLE_MIN, ANGLE_INC)
        core.set_scan(*scan, t_sim=5.0)
        core.set_odom(0.0, 0.0, 0.0, 5.0, t_sim=5.0)
        core.step(t_sim=5.0)
        core.set_odom(0.0, 0.0, 0.0, 0.0, t_sim=0.0)    # /initialpose: back to zero
        core.set_scan(*scan, t_sim=0.0)
        core.step(t_sim=0.0)
        self.assertEqual(calls, [1])

    def test_the_safety_parameters_must_be_positive(self):
        import driver_params
        for name in driver_params.POSITIVE_PARAMS:
            for bad in (0.0, -1.0, float('nan')):
                with self.subTest(name=name, value=bad), self.assertRaises(ValueError):
                    driver_params.validate({**driver_params.DEFAULTS, name: bad})
        with self.assertRaises(ValueError):
            driver_params.validate({**driver_params.DEFAULTS, 'crawl_speed': 11.0})
        driver_params.validate(dict(driver_params.DEFAULTS))


class EmergencyStopTest(unittest.TestCase):
    """From 7 m/s, a wall ahead: down to crawl within v^2 / (2 a_emergency)."""

    def _stop(self, stale, plan_of=None):
        wall_x = 7.0
        core, _ = _core_on(_line(), np.full(800, 10.0), lambda s: s[0] + 1.0, accel=2.0,
                           plan_of=plan_of)
        plant = _plant_at(0.0, 7.0)
        seen = {}

        def on_step(p):
            x = _x(p)
            v = float(p.state[plant_sim.V])
            if 'bind' not in seen and core.conditioning_stats().emergency_binds:
                seen['bind'] = (x, v)
            seen['front_max'] = max(seen.get('front_max', -math.inf), x + FRONT_FROM_BASE_LINK)
            if 'bind' in seen and v <= CFG.crawl_speed + 0.05:
                seen['crawl'] = x
                return True
            return False

        reached = _closed_loop(core, plant, [((wall_x, -5.0), (wall_x, 5.0))], 3.0, stale,
                               on_step)
        self.assertTrue(reached, 'never shed speed to crawl')
        return wall_x, seen

    def _check(self, wall_x, seen):
        x_bind, v_bind = seen['bind']
        self.assertGreater(v_bind, 6.5)
        # The bind is seen at the plant's position when the command is planned;
        # the cap aims at where its hold ends, a latch and a hold further on.
        lead = v_bind * 2.0 * DT
        self.assertLessEqual(seen['crawl'] - x_bind,
                             v_bind ** 2 / (2.0 * CFG.a_emergency) + lead)
        self.assertLess(seen['front_max'], wall_x)

    def test_at_stale_9(self):
        self._check(*self._stop(9))

    def test_at_stale_14(self):
        self._check(*self._stop(14))

    def test_along_the_plan_at_stale_9(self):
        self._check(*self._stop(9, plan_of=_straight_plan))

    def test_along_the_plan_at_stale_14(self):
        self._check(*self._stop(14, plan_of=_straight_plan))


# ----------------------------------------------------------------------
# The profile cap: the plant never runs ahead of the profile
# ----------------------------------------------------------------------
def _braking_profile(path, v_top=8.0, v_corner=2.0, s_corner=30.0, a_brake=6.0):
    s = path[:, 0] - path[0, 0]
    return np.minimum(v_top, np.sqrt(v_corner ** 2 + 2.0 * a_brake * np.maximum(s_corner - s, 0.0)))


class ProfileCapTest(unittest.TestCase):

    def _run(self, stale):
        path = _line()
        speeds = _braking_profile(path)
        core, _ = _core_on(path, speeds, lambda s: s[0] - path[0, 0], accel=9.5)
        plant = _plant_at(0.0, 2.0)
        excess = []
        top = []

        def on_step(p):
            x = _x(p)
            v = float(p.state[plant_sim.V])
            excess.append(v - float(np.interp(x, path[:, 0], speeds)))
            top.append(v)
            return x > 35.0

        _closed_loop(core, plant, [], 12.0, stale, on_step)
        return max(excess), max(top), _x(plant)

    def test_the_plant_never_exceeds_the_profile_by_more_than_0_1_at_stale_9(self):
        worst, top, x = self._run(9)
        self.assertGreater(x, 35.0, 'the car never reached the corner')
        self.assertGreater(top, 7.5, 'the car never got near the profile')
        self.assertLessEqual(worst, 0.1)

    def test_the_plant_never_exceeds_the_profile_by_more_than_0_1_at_stale_14(self):
        worst, _, _ = self._run(14)
        self.assertLessEqual(worst, 0.1)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
class DefaultsTest(unittest.TestCase):

    def test_v_max_is_a_cap_that_should_not_bind(self):
        self.assertEqual(CFG.v_max, 10.0)

    def test_the_steering_limit_is_the_real_lock(self):
        self.assertEqual(CFG.steer_limit, VP.s_max)

    def test_the_emergency_budget_is_seven(self):
        self.assertEqual(CFG.a_emergency, 7.0)

    def test_the_double_counting_and_comfort_limits_are_gone(self):
        names = {f.name for f in dataclasses.fields(driver_core.DriverConfig)}
        self.assertNotIn('k_steer_derate', names)
        self.assertNotIn('a_brake', names)


class HarnessEmergencyGateTest(unittest.TestCase):

    def _result(self, **kw):
        base = {'laps': 10, 'laps_requested': 10, 'collisions': 0, 'sat_duty': 0.0,
                'cmd_slew_mean': 0.005, 'slew_bind_rate': 0.0, 'coast_count': 0}
        base.update(kw)
        return base

    def test_an_emergency_cap_that_binds_often_fails_the_verdict(self):
        reasons = lap_sim.verdict(self._result(emergency_bind_rate=0.02))
        self.assertEqual(len(reasons), 1)
        self.assertIn('emergency', reasons[0])

    def test_a_rarely_binding_cap_passes(self):
        self.assertEqual(lap_sim.verdict(self._result(emergency_bind_rate=0.005)), [])

    def test_the_gate_is_one_percent(self):
        self.assertEqual(lap_sim.MAX_EMERGENCY_BIND_RATE, 0.01)

    def test_profile_overshoot_is_measured_against_the_reference_speed(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        meter = lap_sim.LateralError(path, values=np.array([1.0, 2.0, 3.0, 4.0]))
        distance, value = meter.nearest((1.5, 0.2))
        self.assertAlmostEqual(distance, 0.2, places=2)
        self.assertAlmostEqual(value, 2.5, places=2)


if __name__ == '__main__':
    unittest.main()
