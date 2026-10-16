#!/usr/bin/env python3
"""The combined-accel (g-g) brake limit of the speed profile (sub-10 Step 2).

The judged plant's yaw dynamics go unstable under braking: load transfer unloads
the rear axle, and above ~6 m/s at -6 m/s^2 the sideslip runs away. A profile
that brakes at full rate into a corner therefore crashes at a high slip budget.
`gg_brake` makes the brake pass spend less deceleration where the corner already
uses the slip budget:

    a = a_brake * max(gg_floor, 1 - r^2)^(1/2),   r = min(v^2 kappa / budget, 1)

It is off by default, and off must reproduce today's profile bit for bit.
"""

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, 'team_driver'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import driver_core  # noqa: E402
import driver_params  # noqa: E402
import lap_floor  # noqa: E402
import speed_profile as sp  # noqa: E402

RACELINE_CSV = os.path.join(_HERE, os.pardir, 'config', 'icra26_raceline.csv')
CENTERLINE_CSV = lap_floor.default_centerline_path()
# The judged configuration the Step 1 scratch floors were priced at.
JUDGED = dict(lock_margin=0.95, a_accel_max=9.51, a_brake_max=6.0, v_max=10.0)
KINK = ((43.0, 47.0, 0.85),)
FLOOR_TOLERANCE = 0.005   # [s]


def _stadium(straight=6.0, radius=1.5, spacing=0.05):
    n_s = int(straight / spacing)
    n_c = int(math.pi * radius / spacing)
    bottom = np.column_stack([np.linspace(0, straight, n_s, endpoint=False),
                              np.full(n_s, -radius)])
    a = np.linspace(-np.pi / 2, np.pi / 2, n_c, endpoint=False)
    right = np.column_stack([straight + radius * np.cos(a), radius * np.sin(a)])
    top = np.column_stack([np.linspace(straight, 0, n_s, endpoint=False),
                           np.full(n_s, radius)])
    b = np.linspace(np.pi / 2, 3 * np.pi / 2, n_c, endpoint=False)
    left = np.column_stack([radius * np.cos(b), radius * np.sin(b)])
    return np.vstack([bottom, right, top, left])


STADIUM = dict(a_lat_max=12.0, lock_margin=0.95, a_accel_max=9.51, a_brake_max=6.0,
               v_max=10.0)


def _floor(path, a_lat, sectors=(), **kw):
    v = sp.speed_profile(path, a_lat_max=a_lat, sector_scale=sectors, **{**JUDGED, **kw})
    return sp.ideal_lap_time(path, v)


class TestGgDecel(unittest.TestCase):
    """The allowed deceleration at one point, from the lateral share it uses."""

    def decel(self, v, kappa, budget=20.0, a_brake=6.0, floor=0.1):
        return float(sp.gg_decel(v, kappa, budget, a_brake, floor))

    def test_a_straight_keeps_the_full_braking(self):
        self.assertAlmostEqual(self.decel(8.0, 0.0), 6.0)

    def test_the_budget_fully_used_leaves_the_floor_share(self):
        # r = 1: v^2 kappa == budget, so 1 - r^2 = 0 and the floor binds.
        self.assertAlmostEqual(self.decel(5.0, 0.8), 6.0 * math.sqrt(0.1))

    def test_a_share_of_0_6_leaves_0_8_of_the_braking(self):
        # 1 - 0.36 = 0.64, sqrt = 0.8
        self.assertAlmostEqual(self.decel(6.0, 0.6 * 20.0 / 36.0), 6.0 * 0.8)

    def test_more_than_the_budget_is_clipped_to_the_floor(self):
        self.assertAlmostEqual(self.decel(9.0, 0.8), 6.0 * math.sqrt(0.1))

    def test_a_floor_of_one_never_limits(self):
        self.assertEqual(self.decel(5.0, 0.8, floor=1.0), 6.0)

    def test_it_works_on_arrays(self):
        out = sp.gg_decel(np.array([8.0, 5.0]), np.array([0.0, 0.8]), 20.0, 6.0, 0.1)
        self.assertEqual(out.shape, (2,))
        self.assertAlmostEqual(float(out[0]), 6.0)


class TestOffIsToday(unittest.TestCase):
    """gg_brake=False (the default) must not change a single bit."""

    @staticmethod
    def _old_profile(path, *, a_lat_max, lock_margin, a_accel_max, a_brake_max, v_max,
                     sector_scale=()):
        # The profile as it was before the g-g limit, with the brake loop inlined so
        # a regression inside `_brake_pass` cannot hide on both sides.
        k = np.maximum(np.abs(sp.closed_curvature(path[:, :2])), sp.KAPPA_FLOOR)
        budget = sp.slip_budget(path[:, :2], a_lat_max, sector_scale, True)
        v = np.minimum(np.sqrt(budget / k), sp.lock_speed(k, lock_margin))
        v = np.minimum(v, v_max)
        ds = sp.segment_lengths(path[:, :2], True)
        n = len(v)
        for step in range(sp.PASS_LAPS * n):   # _brake_pass as it was, inlined
            i = (n - 1 - step) % n
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2.0 * a_brake_max * ds[i]))
        sp._accel_pass(v, ds, a_accel_max, True)
        return v

    def _paths(self):
        return {'stadium': _stadium(),
                'raceline': np.loadtxt(RACELINE_CSV, delimiter=',', comments='#')[:, :2],
                'centerline': np.loadtxt(CENTERLINE_CSV, delimiter=',', comments='#')[:, :2]}

    def test_default_equals_the_old_algorithm_bit_for_bit(self):
        sectors = ((2.0, 8.5, 0.6), (36.0, 43.0, 0.85))
        for name, path in self._paths().items():
            with self.subTest(path=name):
                spec = {**JUDGED, 'a_lat_max': 18.0, 'sector_scale': sectors}
                np.testing.assert_array_equal(sp.speed_profile(path, **spec),
                                              self._old_profile(path, **spec))

    def test_explicit_off_equals_the_default(self):
        for name, path in self._paths().items():
            with self.subTest(path=name):
                np.testing.assert_array_equal(
                    sp.speed_profile(path, **JUDGED, a_lat_max=20.0, gg_brake=False,
                                     gg_floor=0.5),
                    sp.speed_profile(path, **JUDGED, a_lat_max=20.0))

    def test_the_shipped_floor_is_the_pinned_config(self):
        # The driver's defaults (sub-10 Step 4) on the CSV they were tuned to.
        path = lap_floor.load_path(RACELINE_CSV)
        cfg = driver_core.DriverConfig()
        floor = _floor(path, cfg.a_lat_max, sp.sector_triples(cfg.sector_scale),
                       gg_brake=cfg.gg_brake, gg_floor=cfg.gg_floor)
        self.assertAlmostEqual(floor, 9.848, delta=FLOOR_TOLERANCE)


class TestGgBrakePass(unittest.TestCase):

    def setUp(self):
        self.path = _stadium()
        self.off = sp.speed_profile(self.path, **STADIUM)
        self.on = sp.speed_profile(self.path, **STADIUM, gg_brake=True)

    def test_a_floor_of_one_degenerates_to_the_plain_profile(self):
        v = sp.speed_profile(self.path, **STADIUM, gg_brake=True, gg_floor=1.0)
        np.testing.assert_array_equal(v, self.off)

    def test_it_is_never_faster_than_the_plain_profile(self):
        self.assertTrue(np.all(self.on <= self.off + 1e-12))

    def test_it_is_slower_somewhere_and_the_lap_takes_longer(self):
        self.assertTrue(np.any(self.on < self.off - 1e-3))
        self.assertGreater(sp.ideal_lap_time(self.path, self.on),
                           sp.ideal_lap_time(self.path, self.off))

    def test_the_corner_speed_itself_is_not_lowered(self):
        # The limit shapes the braking into a corner, not the corner's own speed.
        self.assertAlmostEqual(self.on.min(), self.off.min(), places=6)

    def test_braking_never_exceeds_the_plain_limit(self):
        ds = sp.segment_lengths(self.path, True)
        decel = (self.on ** 2 - np.roll(self.on, -1) ** 2) / (2.0 * ds)
        self.assertLessEqual(decel.max(), STADIUM['a_brake_max'] + 1e-6)

    def test_braking_stays_inside_the_ellipse_at_every_braking_point(self):
        # At every point that brakes, the deceleration taken is bounded by what the
        # ellipse leaves at that point's own speed (the iteration lands on the safe side).
        k = np.maximum(sp.closed_curvature(self.path), sp.KAPPA_FLOOR)
        ds = sp.segment_lengths(self.path, True)
        decel = (self.on ** 2 - np.roll(self.on, -1) ** 2) / (2.0 * ds)
        braking = decel > 1e-3
        self.assertGreater(int(np.count_nonzero(braking)), 20)
        allowed = sp.gg_decel(self.on, k, STADIUM['a_lat_max'], STADIUM['a_brake_max'], 0.1)
        self.assertTrue(np.all(decel[braking] <= allowed[braking] + 1e-6))

    def test_the_ellipse_is_measured_against_each_points_own_budget(self):
        # Same corner-speed caps, same curvature: only the budget the ellipse is
        # measured against differs, and only on the braking approach. Less budget left
        # means less braking, so the car must already be slower there.
        k = np.maximum(sp.closed_curvature(self.path), sp.KAPPA_FLOOR)
        ds = sp.segment_lengths(self.path, True)
        caps = np.minimum(np.sqrt(STADIUM['a_lat_max'] / k), STADIUM['v_max'])

        def brake(budget):
            v = caps.copy()
            sp._brake_pass(v, ds, STADIUM['a_brake_max'], True, (k, budget, 0.1))
            return v

        base = brake(np.full(len(k), STADIUM['a_lat_max']))
        approach = np.flatnonzero(base < caps - 1e-3)
        self.assertGreater(len(approach), 10)
        low = np.full(len(k), STADIUM['a_lat_max'])
        low[approach] *= 0.5
        lowered = brake(low)
        self.assertTrue(np.all(lowered <= base + 1e-12))
        self.assertTrue(np.any(lowered < base - 1e-3))

    def test_the_closed_lap_wraps_through_the_start_line(self):
        ds = sp.segment_lengths(self.path, True)
        decel = (self.on ** 2 - np.roll(self.on, -1) ** 2) / (2.0 * ds)
        seam = np.flatnonzero(decel > 0.5)[0] + 5     # inside the braking zone
        shift = len(self.path) - int(seam)
        rolled = sp.speed_profile(np.roll(self.path, shift, axis=0), **STADIUM, gg_brake=True)
        np.testing.assert_allclose(rolled, np.roll(self.on, shift), atol=1e-6)

    def test_an_open_path_is_supported_and_never_faster(self):
        path = self.path[:len(self.path) // 2]
        kw = dict(**STADIUM, closed=False)
        off = sp.speed_profile(path, **kw)
        on = sp.speed_profile(path, **kw, gg_brake=True)
        self.assertTrue(np.all(on <= off + 1e-12))

    def test_a_floor_outside_zero_to_one_is_refused(self):
        for bad in (0.0, -0.1, 1.5, float('nan')):
            with self.subTest(gg_floor=bad), self.assertRaisesRegex(ValueError, 'gg_floor'):
                sp.speed_profile(self.path, **STADIUM, gg_brake=True, gg_floor=bad)


class TestStep1Floors(unittest.TestCase):
    """The floors on config/icra26_raceline.csv, brake 6, g-g on.

    Step 1 priced these on the a18-refined line (a22 9.771, a22 + kink 9.813, a21
    9.908); Step 4 re-refined the line at the g-g a22 profile, which is these numbers.
    """

    @classmethod
    def setUpClass(cls):
        cls.path = lap_floor.load_path(RACELINE_CSV)

    def test_a22_uniform(self):
        self.assertAlmostEqual(_floor(self.path, 22.0, gg_brake=True), 9.613,
                               delta=FLOOR_TOLERANCE)

    def test_a22_with_the_chicane_exit_sector(self):
        self.assertAlmostEqual(_floor(self.path, 22.0, KINK, gg_brake=True), 9.654,
                               delta=FLOOR_TOLERANCE)

    def test_a21_uniform(self):
        self.assertAlmostEqual(_floor(self.path, 21.0, gg_brake=True), 9.780,
                               delta=FLOOR_TOLERANCE)

    def test_a_sub_10_judged_floor_is_within_reach(self):
        # Judged adds ~0.06-0.10 s to the floor; sub-10 needs a floor <= ~9.85.
        self.assertLessEqual(_floor(self.path, 22.0, KINK, gg_brake=True), 9.85)


class TestLapFloorOption(unittest.TestCase):

    def test_floor_table_takes_gg_brake(self):
        path = lap_floor.load_path(RACELINE_CSV)
        kw = dict(lock_margin=0.95, v_max=10.0, a_accel_max=9.51, a_brake_max=6.0)
        plain = lap_floor.floor_table(path, [22.0], **kw)[0].lap_time
        gg = lap_floor.floor_table(path, [22.0], gg_brake=True, **kw)[0].lap_time
        self.assertAlmostEqual(plain, 9.366, delta=FLOOR_TOLERANCE)
        self.assertAlmostEqual(gg, 9.613, delta=FLOOR_TOLERANCE)

    def test_the_cli_has_a_gg_flag(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lap_floor.main(['--path', RACELINE_CSV, '--a-lat', '22', '--v-max', '10',
                            '--a-brake', '6', '--gg'])
        self.assertIn('9.61', out.getvalue())


class TestHarnessOverride(unittest.TestCase):
    """`lap_sim --driver gg_brake=true` reaches the driver as a real bool."""

    def test_a_bool_field_takes_true_false_and_digits(self):
        import lap_sim
        for raw, want in (('true', True), ('True', True), ('1', True),
                          ('false', False), ('0', False)):
            with self.subTest(raw=raw):
                self.assertEqual(lap_sim.parse_driver_override(f'gg_brake={raw}'),
                                 ('gg_brake', want))

    def test_a_bool_field_refuses_anything_else(self):
        import argparse

        import lap_sim
        with self.assertRaisesRegex(argparse.ArgumentTypeError, 'true or false'):
            lap_sim.parse_driver_override('gg_brake=maybe')

    def test_the_override_reaches_the_driver_config(self):
        import lap_sim
        cfg = lap_sim.HarnessConfig(driver_overrides=(
            lap_sim.parse_driver_override('gg_brake=true'),
            lap_sim.parse_driver_override('gg_floor=0.2')))
        dc = lap_sim.driver_config(cfg)
        self.assertEqual((dc.gg_brake, dc.gg_floor), (True, 0.2))


class TestPlumbing(unittest.TestCase):

    def test_driver_config_defaults_are_the_pinned_ones(self):
        cfg = driver_core.DriverConfig()
        self.assertIs(cfg.gg_brake, True)       # on since sub-10 Step 4
        self.assertEqual(cfg.gg_floor, 0.1)

    def test_the_parameter_table_declares_both(self):
        self.assertIs(driver_params.DEFAULTS['gg_brake'], True)
        self.assertEqual(driver_params.DEFAULTS['gg_floor'], 0.1)

    def test_the_config_itself_refuses_a_bad_floor(self):
        for bad in (0.0, -1.0, 1.5, float('nan'), None, 'x'):
            with self.subTest(gg_floor=bad), self.assertRaisesRegex(ValueError, 'gg_floor'):
                driver_core.DriverConfig(gg_floor=bad)

    def test_validate_refuses_a_non_number_floor(self):
        with self.assertRaisesRegex(ValueError, 'gg_floor'):
            driver_params.validate({**driver_params.DEFAULTS, 'gg_floor': None})

    def test_validate_refuses_a_floor_outside_zero_to_one(self):
        base = dict(driver_params.DEFAULTS)
        driver_params.validate(base)
        for bad in (0.0, -1.0, 1.5, float('nan')):
            with self.subTest(gg_floor=bad), self.assertRaisesRegex(ValueError, 'gg_floor'):
                driver_params.validate({**base, 'gg_floor': bad})

    def test_the_core_passes_its_config_to_the_profile(self):
        class Spy:
            dt = 0.025
            state_dim = 7

            def set_reference_path(self, *_a):
                pass

        for cfg, want in ((driver_core.DriverConfig(), (True, 0.1)),
                          (driver_core.DriverConfig(gg_brake=False), (False, 0.1)),
                          (driver_core.DriverConfig(gg_brake=True, gg_floor=0.3),
                           (True, 0.3))):
            with self.subTest(cfg=want):
                with mock.patch.object(driver_core, 'speed_profile',
                                       side_effect=lambda path, **_: np.full(len(path), 3.0)
                                       ) as profile:
                    driver_core.DriverCore(cfg=cfg, mpc_controller=Spy())
                kw = profile.call_args.kwargs
                self.assertEqual((kw['gg_brake'], kw['gg_floor']), want)


if __name__ == '__main__':
    unittest.main()
