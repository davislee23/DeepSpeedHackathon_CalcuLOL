"""Step 8: the node's parameters actually arrive, and the dead import is gone.

`driver_params.yaml` used to carry a top-level `mpc:` key. ROS reads a top-level
key as a NODE name, so every value under it was silently ignored and the node
ran its code defaults. The plumbing is now one table (`driver_params.py`) that
both `driver.py` and these tests read, so a name that is declared but not
settable, or settable but not declared, fails here rather than at the track.
"""

import ast
import dataclasses
import math
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(PACKAGE_DIR, 'team_driver'))

import driver_core  # noqa: E402
import driver_params  # noqa: E402
import lap_sim  # noqa: E402
import vehicle_model  # noqa: E402
from mpc_controller import MPCController  # noqa: E402

YAML_PATH = os.path.join(PACKAGE_DIR, 'config', 'driver_params.yaml')
VEHICLE_MODEL_PATH = os.path.join(PACKAGE_DIR, 'team_driver', 'vehicle_model.py')
NODE_NAME = 'driver'
# Declared on the node but deliberately not in the params file: the topic names
# are wiring, and use_sim_time is ROS's own, not ours to declare.
WIRING_ONLY = {'scan_topic', 'odom_topic', 'drive_topic'}
ROS_ONLY = {'use_sim_time'}
JUDGED_BLOCK_MAP = [0, 1, 1, 2, 2, 2, 2, 2]


def _flatten(tree, prefix=''):
    """ROS maps nested YAML to dotted parameter names."""
    flat = {}
    for key, value in tree.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(_flatten(value, f'{name}.'))
        else:
            flat[name] = value
    return flat


def _load_yaml():
    with open(YAML_PATH) as handle:
        return yaml.safe_load(handle)


def _node_params():
    return _flatten(_load_yaml()[NODE_NAME]['ros__parameters'])


def _defaults():
    return dict(driver_params.DEFAULTS)


class DeclarationTableTests(unittest.TestCase):

    def test_every_driver_config_field_is_declared_with_the_same_default(self):
        # stop_margin, coast_steer_rate and accel_limit were
        # core-only: the node could not override them.
        cfg = driver_core.DriverConfig()
        declared = driver_params.driver_config(driver_params.DEFAULTS)
        for field in dataclasses.fields(driver_core.DriverConfig):
            with self.subTest(field=field.name):
                self.assertIn(field.name, driver_params.DEFAULTS)
                # Through driver_config(): a tuple field is declared as a ROS list.
                self.assertEqual(getattr(declared, field.name), getattr(cfg, field.name))

    def test_servo_defaults_are_the_calibrated_model(self):
        servo = vehicle_model.ServoModel()
        self.assertEqual(driver_params.DEFAULTS['servo.alpha'], servo.alpha)
        self.assertEqual(driver_params.DEFAULTS['servo.d_max'], servo.d_max_cal)

    def test_params_source_defaults_to_the_code_defaults_sentinel(self):
        self.assertEqual(driver_params.DEFAULTS['params_source'], 'CODE_DEFAULTS')

    def test_control_period_default_equals_the_mpc_step(self):
        self.assertEqual(driver_params.DEFAULTS['control_period'],
                         driver_params.DEFAULTS['mpc.dt'])

    def test_names_are_unique(self):
        names = [spec.name for spec in driver_params.PARAM_SPECS]
        self.assertEqual(len(names), len(set(names)))

    def test_declare_all_declares_each_name_once_with_its_default(self):
        calls = []
        driver_params.declare_all(lambda name, default: calls.append((name, default)))
        self.assertEqual(dict(calls), driver_params.DEFAULTS)
        self.assertEqual(len(calls), len(driver_params.PARAM_SPECS))


class ParamsFileTests(unittest.TestCase):

    def test_the_only_top_level_key_is_the_node_name(self):
        # Any other top-level key (`mpc:` was one) names a node, not a namespace.
        self.assertEqual(set(_load_yaml()), {NODE_NAME})
        self.assertEqual(set(_load_yaml()[NODE_NAME]), {'ros__parameters'})

    def test_every_value_in_the_file_names_a_declared_parameter(self):
        # A name that is not declared is silently dropped by ROS: inert.
        unknown = set(_node_params()) - set(driver_params.DEFAULTS) - ROS_ONLY
        self.assertEqual(unknown, set())

    def test_every_tunable_is_settable_from_the_file(self):
        missing = set(driver_params.DEFAULTS) - set(_node_params()) - WIRING_ONLY
        self.assertEqual(missing, set())

    def test_the_file_is_a_no_op_against_the_code_defaults(self):
        # The values already matched the code; fixing the plumbing must change
        # nothing, or the regression gets blamed on whatever else shipped with it.
        defaults = _defaults()
        for name, value in _node_params().items():
            if name in ROS_ONLY or name == 'params_source':
                continue
            with self.subTest(name=name):
                self.assertEqual(value, defaults[name])

    def test_the_file_types_match_the_declared_types(self):
        # ROS refuses an int for a double parameter, so `3` for `3.0` would stop
        # the node at startup.
        defaults = _defaults()
        for name, value in _node_params().items():
            if name in ROS_ONLY:
                continue
            with self.subTest(name=name):
                self.assertIs(type(value), type(defaults[name]))
                if isinstance(value, list):
                    self.assertEqual([type(v) for v in value],
                                     [type(v) for v in defaults[name]])

    def test_the_file_sets_the_provenance_sentinel(self):
        self.assertEqual(_node_params()['params_source'], 'driver_params.yaml')

    def test_the_file_turns_sim_time_on(self):
        self.assertIs(_node_params()['use_sim_time'], True)

    def test_the_file_passes_the_startup_checks(self):
        values = {**_defaults(), **{k: v for k, v in _node_params().items()
                                    if k not in ROS_ONLY}}
        driver_params.validate(values)


STEP9_TUNED = {'mpc.r_delta_steer': 3.0, 'mpc.q_pos': 2.0, 'mpc.q_yaw': 2.0}


class Step9TunedWeightsTests(unittest.TestCase):
    """The Step 9 sweep on the 25 ms-cadence harness (10 laps, stale 9).

    Lowering r_delta_steer, as the plan proposed, made every metric worse: at
    0.1 the cmd slew is 0.022 and laps 0.7 s slower. Raising it to 3.0 is the
    knee: the first setting to pass the 0.012 slew gate (0.0111), 0 % rate
    saturation, and 0.1 s/lap faster; 5 and 8 only lose lap time. It also beats
    0.8 at stale 14. q_pos was flat over 1-4; q_yaw 4 is faster but fails the
    gate, q_yaw 1 is slower.
    """

    def test_the_code_defaults_carry_the_tuned_weights(self):
        for name, value in STEP9_TUNED.items():
            with self.subTest(name=name):
                self.assertEqual(driver_params.DEFAULTS[name], value)

    def test_the_params_file_carries_the_tuned_weights(self):
        params = _node_params()
        for name, value in STEP9_TUNED.items():
            with self.subTest(name=name):
                self.assertEqual(params[name], value)


SUB10_TUNED = {'a_lat_max': 21.0, 'a_brake_max': 6.0, 'accel_limit': 9.51,
               'lock_margin': 0.95, 'gg_brake': True, 'gg_floor': 0.1,
               'cap_follows_reference': True,
               'sector_scale': [43.0, 47.0, 0.85, 3.0, 7.0, 0.95]}


class Sub10TunedProfileTests(unittest.TestCase):
    """The sub-10 profile (Steps 1-4), on the min-time line re-refined at the g-g a22 profile.

    Step 1: the crashes above a_lat 18 are braking-induced yaw instability, not
    steering lock. Step 2 limits braking by the share of the slip budget a corner
    already uses (gg_brake), which replaces the hand-tuned sectors. Step 3 stops the
    emergency cap braking at corner entries for walls the car is turning away from
    (cap_follows_reference). Step 4 re-refines the line at that profile and finds
    two spots left: s~45.3, the chicane-exit tracking transient (43-47 at 0.85), and
    s~4.8-5.2, the first corner on the new line (3-7 at 0.95; 0.9 let a22 hit s 54).
    The sectors are arc positions on the min-time CSV.

    Grid at a_lat 21, a_brake_max 6: stale 5/9/14 x fixed 5/7/12 ms (5 laps), a
    10-lap run, 5 measured-timing runs (<= 3 jobs) and the centreline fallback, all
    16 clean, best 9.82 s; 0 collisions in 24 runs under 12-job host load. The fastest
    clean budget, a22 (9.71 s with 3-7 at 0.85), hit s 54 in 2 of 18 loaded runs, and
    a23 collides there on 15 of 16 grid runs, so the pin backs off to 21.
    """

    def test_the_code_defaults_carry_the_tuned_profile(self):
        for name, value in SUB10_TUNED.items():
            with self.subTest(name=name):
                self.assertEqual(driver_params.DEFAULTS[name], value)

    def test_the_params_file_carries_the_tuned_profile(self):
        params = _node_params()
        for name, value in SUB10_TUNED.items():
            with self.subTest(name=name):
                self.assertEqual(params[name], value)


class ValidateTests(unittest.TestCase):

    def test_the_defaults_pass(self):
        driver_params.validate(_defaults())

    def test_a_control_period_that_differs_from_the_mpc_step_is_refused(self):
        values = {**_defaults(), 'control_period': 0.05}
        with self.assertRaisesRegex(ValueError, r'control_period.*mpc\.dt'):
            driver_params.validate(values)

    def test_a_rounding_sized_difference_is_accepted(self):
        values = {**_defaults(), 'control_period': 0.025 + 1e-12}
        driver_params.validate(values)

    def test_a_block_map_of_the_wrong_length_is_refused(self):
        values = {**_defaults(), 'mpc.block_map': [0, 1, 2]}
        with self.assertRaisesRegex(ValueError, r'block_map.*prediction_horizon'):
            driver_params.validate(values)

    def test_a_sector_scale_that_is_not_whole_sectors_is_refused(self):
        values = {**_defaults(), 'sector_scale': [29.0, 37.0]}
        with self.assertRaisesRegex(ValueError, r'sector'):
            driver_params.validate(values)

    def test_a_sector_scale_above_one_is_refused(self):
        values = {**_defaults(), 'sector_scale': [29.0, 37.0, 1.5]}
        with self.assertRaisesRegex(ValueError, r'sector'):
            driver_params.validate(values)

    def test_list_defaults_are_non_empty_so_ros_can_type_them(self):
        # rclpy types an empty list as a byte array, and the node would then
        # refuse the file's list of doubles at startup.
        for name, value in _defaults().items():
            if isinstance(value, list):
                with self.subTest(name=name):
                    self.assertTrue(value)

    def test_a_bad_sector_scale_is_refused_by_the_config_itself(self):
        # Not only by validate(): a DriverCore built without it must not quietly
        # fall back to the reactive controller.
        with self.assertRaisesRegex(ValueError, r'sector'):
            driver_core.DriverConfig(sector_scale=(29.0, 37.0))

    def test_sector_scale_reaches_the_driver_config_as_a_tuple(self):
        cfg = driver_params.driver_config({**_defaults(), 'sector_scale': [29.0, 37.0, 0.8]})
        self.assertEqual(cfg.sector_scale, (29.0, 37.0, 0.8))

    def test_a_longer_horizon_without_a_longer_block_map_is_refused(self):
        values = {**_defaults(), 'mpc.prediction_horizon': 10}
        with self.assertRaisesRegex(ValueError, r'block_map.*prediction_horizon'):
            driver_params.validate(values)


class BuilderTests(unittest.TestCase):

    def test_driver_config_carries_every_field_from_the_values(self):
        values = {**_defaults(), 'stop_margin': 0.2, 'accel_limit': 3.5,
                  'v_max': 2.5}
        cfg = driver_params.driver_config(values)
        self.assertEqual(cfg.stop_margin, 0.2)
        self.assertEqual(cfg.accel_limit, 3.5)
        self.assertEqual(cfg.v_max, 2.5)
        self.assertEqual(cfg.a_emergency, driver_core.DriverConfig().a_emergency)

    def test_default_driver_config_equals_the_cores_own_default(self):
        self.assertEqual(driver_params.driver_config(_defaults()),
                         driver_core.DriverConfig())

    def test_servo_model_takes_alpha_and_d_max(self):
        values = {**_defaults(), 'servo.alpha': 0.5, 'servo.d_max': 0.07}
        servo = driver_params.servo_model(values)
        self.assertEqual(servo.alpha, 0.5)
        self.assertEqual(servo.d_max_cal, 0.07)

    def test_default_servo_model_equals_the_calibrated_one(self):
        self.assertEqual(driver_params.servo_model(_defaults()),
                         vehicle_model.ServoModel())

    def test_mpc_kwargs_build_the_judged_controller(self):
        kwargs = driver_params.mpc_kwargs(_defaults())
        self.assertEqual(kwargs, lap_sim.JUDGED_MPC)
        controller = MPCController(vehicle_model=vehicle_model.VehicleModel(), **kwargs)
        self.assertEqual(controller.prediction_horizon, 8)
        self.assertEqual(controller.block_map.tolist(), JUDGED_BLOCK_MAP)

    def test_mpc_kwargs_copy_the_block_map(self):
        values = _defaults()
        kwargs = driver_params.mpc_kwargs(values)
        kwargs['block_map'].append(99)
        self.assertEqual(values['mpc.block_map'], JUDGED_BLOCK_MAP)

    def test_read_all_asks_for_every_declared_name(self):
        asked = []

        def get(name):
            asked.append(name)
            return type('P', (), {'value': driver_params.DEFAULTS[name]})()

        values = driver_params.read_all(get)
        self.assertEqual(values, driver_params.DEFAULTS)
        self.assertEqual(sorted(asked), sorted(driver_params.DEFAULTS))


class EchoTests(unittest.TestCase):

    def test_the_echo_names_the_source_first(self):
        lines = driver_params.echo_lines({**_defaults(), 'params_source': 'driver_params.yaml'})
        self.assertEqual(lines[0], 'params_source = driver_params.yaml')

    def test_the_echo_lists_every_parameter_exactly_once(self):
        lines = driver_params.echo_lines(_defaults())
        names = [line.split(' = ')[0] for line in lines]
        self.assertCountEqual(names, driver_params.DEFAULTS)

    def test_the_echo_shows_the_resolved_values(self):
        lines = driver_params.echo_lines({**_defaults(), 'mpc.r_delta_steer': 0.123})
        self.assertIn('mpc.r_delta_steer = 0.123', lines)
        self.assertIn('mpc.block_map = [0, 1, 1, 2, 2, 2, 2, 2]', lines)


def _eval_number(node):
    """Evaluate a literal or constant arithmetic expression, nothing else."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_number(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = _eval_number(node.left), _eval_number(node.right)
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}
        return ops[type(node.op)](left, right)
    raise ValueError(f'not a constant expression: {ast.dump(node)}')


def _upstream_vehicle_parameters():
    """The judged car, read as TEXT from the gym's dynamic_models/__init__.py."""
    here = PACKAGE_DIR
    for _ in range(8):
        candidate = os.path.join(here, 'external', 'f1tenth_gym', 'f1tenth_gym',
                                 'envs', 'dynamic_models', '__init__.py')
        if os.path.isfile(candidate):
            break
        here = os.path.dirname(here)
    else:
        return None
    with open(candidate) as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'F1TENTH_VEHICLE_PARAMETERS'
                        for t in node.targets)):
            return {kw.arg: _eval_number(kw.value) for kw in node.value.keywords}
    return None


class VehicleParametersPinTests(unittest.TestCase):

    def test_vehicle_model_imports_nothing_from_the_gym(self):
        # The gym's vehicle_dynamics_ks_cog has state [x, y, delta, v, psi] and
        # control [steer_vel, accl] over a flat numpy params array. If that
        # import ever succeeded it would swap state indices 2 and 4 and describe
        # a different vehicle without raising. Deleted, not repaired.
        with open(VEHICLE_MODEL_PATH) as handle:
            tree = ast.parse(handle.read())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or '')
        offenders = [m for m in modules if 'f1tenth_gym' in m or m.startswith('external')]
        self.assertEqual(offenders, [])

    def test_vehicle_model_has_no_import_fallback_left(self):
        with open(VEHICLE_MODEL_PATH) as handle:
            tree = ast.parse(handle.read())
        handlers = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        self.assertEqual(handlers, [])

    def test_the_local_parameters_match_the_judged_cars_numbers(self):
        upstream = _upstream_vehicle_parameters()
        if upstream is None:
            self.skipTest('external/f1tenth_gym not found above the package')
        local = vehicle_model.F1TENTH_VEHICLE_PARAMETERS
        self.assertGreaterEqual(len(upstream), 19)
        for name, expected in upstream.items():
            with self.subTest(name=name):
                self.assertTrue(hasattr(local, name), f'local VehicleParameters lacks {name}')
                self.assertAlmostEqual(getattr(local, name), expected, places=12)

    def test_the_local_parameters_are_the_default_instance(self):
        self.assertEqual(vehicle_model.F1TENTH_VEHICLE_PARAMETERS,
                         vehicle_model.VehicleParameters())

    def test_the_kinematic_model_uses_the_local_state_order(self):
        # [x, y, psi, v, beta] with control [delta, a]: the order the MPC assumes.
        model = vehicle_model.VehicleModel()
        derivative = model.dynamics_continuous(
            np.asarray([0.0, 0.0, 0.3, 2.0, 0.0]), (0.1, 0.5))
        self.assertAlmostEqual(derivative[0], 2.0 * math.cos(0.3))
        self.assertAlmostEqual(derivative[3], 0.5)


if __name__ == '__main__':
    unittest.main()
