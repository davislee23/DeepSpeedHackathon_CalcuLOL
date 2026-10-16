"""Phase 2 Step 3 wiring: the new MPC options reach the node, the core and the harness.

`mpc.dt_schedule`, `mpc.feedforward` and `vehicle.model` are declared once in
driver_params.py, refused by `validate()` when the model cannot honour them, and
mirrored by lap_sim so a harness run is the node's car. The core hands an ST
controller the sideslip and yaw rate odometry already carries.
"""

import os
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'team_driver'))

import driver_core  # noqa: E402
import driver_params  # noqa: E402
import lap_sim  # noqa: E402

LONG = [0.025, 0.025, 0.05, 0.05, 0.075, 0.075, 0.1, 0.1]


def _values(**changes):
    values = dict(driver_params.DEFAULTS)
    values.update(changes)
    return values


class ParamTableTest(unittest.TestCase):

    def test_the_schedule_defaults_to_the_long_one(self):
        # Step 4: the 0.5 s lookahead the gate was run at.
        self.assertEqual(driver_params.DEFAULTS['mpc.dt_schedule'], [0.025, 0.025, 0.05, 0.05, 0.075, 0.075, 0.1, 0.1])

    def test_feedforward_defaults_off(self):
        self.assertIs(driver_params.DEFAULTS['mpc.feedforward'], False)

    def test_the_model_defaults_to_the_judged_single_track(self):
        self.assertEqual(driver_params.DEFAULTS['vehicle.model'], 'ST')

    def test_the_mpc_builder_passes_the_schedule_and_feedforward(self):
        kwargs = driver_params.mpc_kwargs(_values(**{'mpc.dt_schedule': LONG,
                                                     'mpc.feedforward': True}))
        self.assertEqual(kwargs['dt_schedule'], LONG)
        self.assertIs(kwargs['feedforward'], True)

    def test_the_vehicle_builder_takes_the_model_and_the_servo(self):
        model = driver_params.vehicle_model(_values(**{'vehicle.model': 'ST',
                                                       'servo.alpha': 0.5}))
        self.assertEqual(model.model_type, 'ST')
        self.assertEqual(model.servo.alpha, 0.5)


class ValidateTest(unittest.TestCase):

    def test_a_long_schedule_passes(self):
        driver_params.validate(_values(**{'mpc.dt_schedule': LONG}))

    def test_a_schedule_of_the_wrong_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'dt_schedule'):
            driver_params.validate(_values(**{'mpc.dt_schedule': LONG[:-1]}))

    def test_a_schedule_that_does_not_start_at_the_control_period_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'control_period'):
            driver_params.validate(_values(**{'mpc.dt_schedule': [0.05] + LONG[1:]}))

    def test_a_non_positive_step_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            driver_params.validate(_values(**{'mpc.dt_schedule': LONG[:7] + [0.0]}))

    def test_an_unknown_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'vehicle.model'):
            driver_params.validate(_values(**{'vehicle.model': 'MB'}))


class _RecordingMPC:
    """Just enough of MPCController for DriverCore.plan; records the state."""

    dt = 0.025

    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.states = []

    def speed_profile(self, **_kw):
        return np.full(4, 3.0)

    def set_reference_path(self, *_a):
        pass

    def get_control(self, state):
        self.states.append(np.asarray(state, dtype=float))
        return np.array([0.0, 0.0]), {'solve_time_ms': 0.0}


def _plan_with(state_dim, beta=0.05):
    mpc = _RecordingMPC(state_dim)
    core = driver_core.DriverCore(cfg=driver_core.DriverConfig(), mpc_controller=mpc)
    core.reference_path = np.zeros((4, 2))
    core.set_odom(1.0, 2.0, 0.3, 4.0, beta=beta, yaw_rate=1.2, t_sim=0.0)
    n = 181
    core.set_scan(np.full(n, 6.0), -np.pi / 2, np.pi / (n - 1), t_sim=0.0)
    core.step(t_sim=0.0)
    return mpc.states[-1]


class CoreStateTest(unittest.TestCase):

    def test_a_single_track_controller_gets_the_measured_slip_and_yaw_rate(self):
        state = _plan_with(7)
        self.assertEqual(len(state), 7)
        self.assertAlmostEqual(state[4], 0.05)
        self.assertAlmostEqual(state[6], 1.2)

    def test_reversing_is_not_handed_over_as_slip(self):
        # Odometry's speed is a magnitude, so reversing reads as beta ~ pi.
        self.assertEqual(_plan_with(7, beta=3.1)[4], 0.0)

    def test_a_kinematic_controller_still_assumes_no_slip(self):
        # The KS model holds beta constant over the horizon; handing it the
        # measured one would bend every prediction. Today's car is unchanged.
        state = _plan_with(6)
        self.assertEqual(len(state), 6)
        self.assertEqual(state[4], 0.0)


class HarnessFlagsTest(unittest.TestCase):

    def assertRejected(self, argv):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            lap_sim.parse_config(argv)

    def test_defaults_are_the_declared_car(self):
        cfg = lap_sim.parse_config([])
        self.assertEqual(cfg.model, 'ST')
        self.assertIsNone(cfg.dt_schedule)
        self.assertFalse(cfg.feedforward)
        self.assertEqual(cfg.driver_overrides, ())
        self.assertEqual(lap_sim.mpc_kwargs(cfg), lap_sim.JUDGED_MPC)

    def test_model_flag(self):
        self.assertEqual(lap_sim.parse_config(['--model', 'ST']).model, 'ST')
        self.assertRejected(['--model', 'MB'])

    def test_schedule_flag_reaches_the_controller(self):
        cfg = lap_sim.parse_config(['--dt-schedule', ','.join(map(str, LONG))])
        self.assertEqual(cfg.dt_schedule, tuple(LONG))
        self.assertEqual(lap_sim.mpc_kwargs(cfg)['dt_schedule'], LONG)

    def test_a_malformed_schedule_is_rejected(self):
        self.assertRejected(['--dt-schedule', '0.025,fast'])

    def test_feedforward_flag_reaches_the_controller(self):
        cfg = lap_sim.parse_config(['--feedforward'])
        self.assertIs(lap_sim.mpc_kwargs(cfg)['feedforward'], True)

    def test_driver_overrides_are_typed_by_the_field(self):
        cfg = lap_sim.parse_config(['--driver', 'v_max=8', '--driver', 'reference=raceline'])
        self.assertEqual(cfg.driver_overrides, (('v_max', 8.0), ('reference', 'raceline')))
        dc = lap_sim.driver_config(cfg)
        self.assertEqual((dc.v_max, dc.reference), (8.0, 'raceline'))

    def test_sector_scale_override_is_a_flat_list_of_sectors(self):
        cfg = lap_sim.parse_config(['--driver', 'sector_scale=29,37,0.8,38,43.5,0.5'])
        self.assertEqual(cfg.driver_overrides,
                         (('sector_scale', (29.0, 37.0, 0.8, 38.0, 43.5, 0.5)),))
        self.assertEqual(lap_sim.driver_config(cfg).sector_scale,
                         (29.0, 37.0, 0.8, 38.0, 43.5, 0.5))

    def test_a_malformed_sector_scale_is_rejected(self):
        self.assertRejected(['--driver', 'sector_scale=29,37'])
        self.assertRejected(['--driver', 'sector_scale=29,37,1.5'])
        self.assertRejected(['--driver', 'sector_scale=29,37,fast'])

    def test_an_unknown_driver_field_is_rejected(self):
        self.assertRejected(['--driver', 'warp_drive=1'])

    def test_a_non_numeric_value_for_a_numeric_field_is_rejected(self):
        self.assertRejected(['--driver', 'v_max=fast'])

    def test_an_invalid_reference_is_rejected(self):
        self.assertRejected(['--driver', 'reference=shortcut'])


class _Built(Exception):
    pass


class HarnessBuildsTheConfiguredCarTest(unittest.TestCase):

    def test_run_builds_the_model_schedule_and_driver_config_asked_for(self):
        seen = {}

        def capture(**kwargs):
            seen.update(kwargs)
            raise _Built

        cfg = lap_sim.HarnessConfig(laps=1, quiet=True, model='ST',
                                    dt_schedule=tuple(LONG), feedforward=True,
                                    driver_overrides=(('v_max', 6.0),))
        with mock.patch.object(lap_sim, 'MPCController', side_effect=capture), \
                self.assertRaises(_Built):
            lap_sim.run(cfg)
        self.assertEqual(seen['vehicle_model'].model_type, 'ST')
        self.assertEqual(seen['dt_schedule'], LONG)
        self.assertIs(seen['feedforward'], True)
        self.assertEqual(lap_sim.driver_config(cfg).v_max, 6.0)


class LateralErrorTest(unittest.TestCase):

    def test_distance_to_the_reference_polyline(self):
        square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        meter = lap_sim.LateralError(square)
        self.assertAlmostEqual(meter.distance((5.0, 0.3)), 0.3, places=2)
        self.assertAlmostEqual(meter.distance((-0.2, 5.0)), 0.2, places=2)  # closing side

    def test_a_point_on_the_line_is_zero(self):
        meter = lap_sim.LateralError(np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]]))
        self.assertLess(meter.distance((2.0, 0.0)), 0.01)


if __name__ == '__main__':
    unittest.main()
