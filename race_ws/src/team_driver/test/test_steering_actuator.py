"""Step 6: the steering actuator is an MPC state, estimated by DriverCore."""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import driver_core  # noqa: E402
import lap_sim  # noqa: E402
import plant_sim  # noqa: E402
import vehicle_model  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

DT = 0.025
PHYSICS_DT = 0.01
JUDGED_BLOCK_MAP = [0, 1, 1, 2, 2, 2, 2, 2]
PARAMS = VehicleModel().params
WHEELBASE = PARAMS.lf + PARAMS.lr
SLSQP_FD_STEP = 1.4901161193847656e-08   # sqrt(float64 eps), scipy's default


def _plant_move(error, n_steps, delta0=0.0):
    """How far the judged servo moves the wheel in n physics steps."""
    plant = plant_sim.GymPlant()
    state = np.zeros(7)
    state[plant_sim.DELTA] = delta0
    plant.reset_state(state)
    plant.latch(delta0 + error, 0.0)
    for _ in range(n_steps):
        plant.step()
    return float(plant.state[plant_sim.DELTA]) - delta0


def _plant_move_over_one_mpc_step(error):
    # A 25 ms timer on a 10 ms physics grid holds each command for 2 or 3
    # steps, alternately; 2.5 on average.
    return 0.5 * (_plant_move(error, 2) + _plant_move(error, 3))


# ----------------------------------------------------------------------
# The servo model the MPC predicts with
# ----------------------------------------------------------------------
class ServoModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.servo = vehicle_model.ServoModel()
        cls.errors = np.linspace(-0.5, 0.5, 101)
        cls.plant = np.array([_plant_move_over_one_mpc_step(e) for e in cls.errors])
        cls.model = np.array([cls.servo.move(e, DT) for e in cls.errors])

    def test_calibrated_model_is_within_5pct_of_d_max_of_the_judged_servo(self):
        worst = np.max(np.abs(self.model - self.plant))
        self.assertLess(worst, 0.05 * self.servo.d_max(DT))

    def test_small_errors_move_the_wheel_at_the_plants_linear_rate(self):
        e = 0.005
        self.assertAlmostEqual(self.servo.move(e, DT) / e,
                               _plant_move_over_one_mpc_step(e) / e, delta=0.01)

    def test_a_saturated_move_is_sv_max_times_dt(self):
        self.assertAlmostEqual(self.servo.d_max(DT), PARAMS.sv_max * DT, delta=0.002)

    def test_move_strictly_increases_with_the_error_even_deep_in_saturation(self):
        errors = np.linspace(-0.84, 0.84, 2001)
        moves = np.array([self.servo.move(e, DT) for e in errors])
        self.assertTrue(np.all(np.diff(moves) > 0.0))

    def test_move_is_odd_symmetric(self):
        for e in (0.01, 0.08, 0.3):
            self.assertAlmostEqual(self.servo.move(-e, DT), -self.servo.move(e, DT))

    def test_the_wheel_never_overshoots_the_command(self):
        for e in np.linspace(-0.84, 0.84, 169):
            self.assertLessEqual(abs(self.servo.move(e, DT)), abs(e) + 1e-12)

    def test_a_longer_step_compounds_the_linear_rate_and_doubles_the_rate_limit(self):
        alpha = self.servo.move(1e-4, DT) / 1e-4
        alpha2 = self.servo.move(1e-4, 2 * DT) / 1e-4
        self.assertAlmostEqual(alpha2, 1.0 - (1.0 - alpha) ** 2, places=4)
        self.assertAlmostEqual(self.servo.d_max(2 * DT), 2 * self.servo.d_max(DT))


# ----------------------------------------------------------------------
# The vehicle model: the wheel angle is a state, rows 0-4 are unchanged
# ----------------------------------------------------------------------
class ActuatedVehicleModelTest(unittest.TestCase):

    def setUp(self):
        self.model = VehicleModel()

    def test_the_car_turns_with_the_wheel_not_with_the_command(self):
        state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        nxt = self.model.actuated_step(state, np.array([0.4, 0.0]), DT)
        self.assertEqual(nxt[2], 0.0)
        self.assertGreater(nxt[5], 0.0)

    def test_the_wheel_moves_by_the_servo_model(self):
        state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.1])
        nxt = self.model.actuated_step(state, np.array([0.3, 0.0]), DT)
        self.assertAlmostEqual(nxt[5], 0.1 + self.model.servo.move(0.2, DT))

    def test_rows_0_to_4_are_the_kinematic_model_driven_by_the_wheel(self):
        state = np.array([1.0, 2.0, 0.3, 2.0, 0.0, 0.15])
        nxt = self.model.actuated_step(state, np.array([-0.2, 1.5]), DT)
        expected = self.model.dynamics_discrete(state[:5], np.array([0.15, 1.5]), DT)
        np.testing.assert_allclose(nxt[:5], expected)


# ----------------------------------------------------------------------
# The MPC predicts with the actuator
# ----------------------------------------------------------------------
def _controller(**overrides):
    kwargs = dict(vehicle_model=VehicleModel(), prediction_horizon=8,
                  block_map=JUDGED_BLOCK_MAP, dt=DT)
    kwargs.update(overrides)
    mpc = MPCController(**kwargs)
    mpc.set_reference_path(np.array([[x, 0.0] for x in np.arange(0.0, 20.0, 0.2)]))
    return mpc


class MpcActuatorStateTest(unittest.TestCase):

    def test_state_is_six_wide(self):
        self.assertEqual(_controller().state_dim, 6)

    def test_the_wheel_angle_is_not_tracked_by_the_state_cost(self):
        mpc = _controller()
        self.assertEqual(len(mpc.Q_diag_masked), 6)
        self.assertEqual(mpc.Q_diag_masked[5], 0.0)

    def test_the_predicted_trajectory_carries_the_wheel_angle(self):
        mpc = _controller()
        x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        states = mpc._predict_trajectory(x0, np.array([0.4, 0.0] * 3))
        self.assertEqual(states.shape, (9, 6))
        self.assertAlmostEqual(states[1, 5], mpc.vehicle_model.servo.move(0.4, DT))

    def test_no_predicted_wheel_move_exceeds_the_rate_limit(self):
        mpc = _controller()
        rng = np.random.default_rng(0)
        d_max = mpc.vehicle_model.servo.d_max(DT)
        for _ in range(20):
            x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, rng.uniform(-0.4, 0.4)])
            u = np.ravel([[rng.uniform(-0.42, 0.42), 0.0] for _ in range(3)])
            states = mpc._predict_trajectory(x0, u)
            self.assertLessEqual(np.max(np.abs(np.diff(states[:, 5]))), d_max + 1e-12)

    def test_the_first_command_accounts_for_where_the_wheel_already_is(self):
        # Same pose; the wheel is turned left in one case and right in the
        # other. Unwinding a left-turned wheel needs a more rightward command.
        left, _ = _controller().solve(np.array([2.0, 0.0, 0.0, 2.0, 0.0, 0.3]))
        right, _ = _controller().solve(np.array([2.0, 0.0, 0.0, 2.0, 0.0, -0.3]))
        self.assertLess(left[0], right[0])

    # The initial state reaches the optimiser through _prepare_objective (the
    # fused objective, Phase 2 Step 3), so that is where these spy.
    def _initial_state_seen(self, mpc, state):
        seen = []
        real = mpc._prepare_objective

        def spy(x0, ref):
            seen.append(np.array(x0))
            return real(x0, ref)

        mpc._prepare_objective = spy
        mpc.solve(state)
        return seen[0]

    def test_a_five_state_input_assumes_the_wheel_is_at_the_last_command(self):
        mpc = _controller()
        mpc.prev_control = np.array([0.2, 0.0])
        x0 = self._initial_state_seen(mpc, np.array([2.0, 0.0, 0.0, 2.0, 0.0]))
        self.assertEqual(x0[5], 0.2)

    def test_a_four_state_input_assumes_no_sideslip_and_the_last_command(self):
        mpc = _controller()
        mpc.prev_control = np.array([-0.1, 0.0])
        x0 = self._initial_state_seen(mpc, np.array([2.0, 0.0, 0.0, 2.0]))
        self.assertEqual((x0[4], x0[5]), (0.0, -0.1))

    def test_other_state_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            _controller().solve(np.zeros(7))

    def test_slsqp_sees_a_gradient_in_the_first_command_when_rate_limited(self):
        # With effort and rate costs off, only physics links u0 to the cost.
        # 0.3 rad of error is ~2.6x the rate-limit knee. Step by SLSQP's own
        # finite-difference size: a hard clip makes these bit-identical.
        mpc = _controller(r_steer=0.0, r_delta_steer=0.0)
        x0 = np.array([2.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        ref = mpc._reference_window(2.0, 9, car_speed=2.0).copy()
        u = np.array([0.3, 0.0, 0.1, 0.0, 0.0, 0.0])
        c1 = mpc._cost_function(u, x0, ref, ref[:, 3])
        u[0] += SLSQP_FD_STEP
        c2 = mpc._cost_function(u, x0, ref, ref[:, 3])
        self.assertNotEqual(c1, c2)

    def test_deep_in_saturation_the_command_does_not_move_the_wheel_further(self):
        # Honest physics, not a defect: past ~0.5 rad of error every command
        # moves the wheel by d_max, so the effort and rate costs carry u0's
        # gradient there and pull it back toward the knee.
        servo = VehicleModel().servo
        gain = (servo.move(0.8 + SLSQP_FD_STEP, DT) - servo.move(0.8, DT)) / SLSQP_FD_STEP
        self.assertLess(abs(gain), 1e-6)

    def test_a_zero_length_step_does_not_move_the_wheel(self):
        self.assertEqual(VehicleModel().servo.move(0.3, 0.0), 0.0)


# ----------------------------------------------------------------------
# The estimator DriverCore hands the MPC
# ----------------------------------------------------------------------
class _ServoReplay:
    """Drives GymPlant's servo with timed latches, for ground truth."""

    def __init__(self):
        self.plant = plant_sim.GymPlant()
        self.plant.reset((0.0, 0.0, 0.0))
        self.time = 0.0

    def run_to(self, t):
        while self.time < t - 1e-9:
            self.plant.step()
            self.time += PHYSICS_DT

    @property
    def delta(self):
        return float(self.plant.state[plant_sim.DELTA])


class SteeringEstimatorPredictTest(unittest.TestCase):

    def setUp(self):
        self.est = driver_core.SteeringEstimator()

    def test_starts_centred(self):
        self.assertEqual(self.est.delta_hat, 0.0)

    def test_the_first_prediction_only_starts_the_clock(self):
        self.est.on_publish(0.3, 0.0)
        self.est.predict_to(1.0)
        self.assertEqual(self.est.delta_hat, 0.0)

    def test_with_no_command_the_wheel_stays_centred(self):
        self.est.predict_to(0.0)
        self.est.predict_to(0.5)
        self.assertEqual(self.est.delta_hat, 0.0)

    def test_a_command_acts_only_from_its_latch_time(self):
        self.est.predict_to(0.0)
        self.est.on_publish(0.3, 0.05)
        self.est.predict_to(0.05)
        self.assertEqual(self.est.delta_hat, 0.0)
        self.est.predict_to(0.06)
        self.assertGreater(self.est.delta_hat, 0.0)

    def test_matches_the_judged_servo_step_for_step(self):
        truth = _ServoReplay()
        self.est.predict_to(0.0)
        latches = [(0.00, 0.30), (0.07, -0.35), (0.19, 0.05), (0.22, 0.40),
                   (0.40, -0.10), (0.43, 0.0)]
        worst = 0.0
        pending = list(latches)
        for t_latch, cmd in latches:
            self.est.on_publish(cmd, t_latch)
        for t in np.arange(0.0, 0.8, 0.01):
            # Latches are queued up front; the estimator must apply each one
            # only once its time comes, exactly as the plant does.
            while pending and pending[0][0] <= t + 1e-9:
                truth.plant.latch(pending.pop(0)[1], 0.0)
            truth.run_to(t + PHYSICS_DT)
            self.est.predict_to(t + PHYSICS_DT)
            worst = max(worst, abs(self.est.delta_hat - truth.delta))
        self.assertLess(worst, 1e-5)

    def test_a_published_command_is_clipped_to_the_lock(self):
        self.est.predict_to(0.0)
        self.est.on_publish(1.0, 0.0)
        self.est.predict_to(2.0)
        self.assertAlmostEqual(self.est.delta_hat, PARAMS.s_max, places=6)

    def test_a_wheel_at_the_lock_is_held_there(self):
        self.est.delta_hat = PARAMS.s_max
        self.est.predict_to(0.0)
        self.est.on_publish(PARAMS.s_max, 0.0)
        self.est.predict_to(0.1)
        self.assertEqual(self.est.delta_hat, PARAMS.s_max)

    def test_prediction_never_runs_backwards(self):
        self.est.predict_to(0.0)
        self.est.on_publish(0.3, 0.0)
        self.est.predict_to(0.1)
        before = self.est.delta_hat
        self.est.predict_to(0.05)
        self.assertEqual(self.est.delta_hat, before)

    def test_a_sim_reset_reseeds_the_wheel_to_centre(self):
        # /initialpose makes the bridge call env.reset(): sim_time -> 0 and the
        # wheel re-centres. The estimate must follow, not freeze.
        self.est.predict_to(0.0)
        self.est.on_publish(0.4, 0.0)
        self.est.predict_to(5.0)
        self.est.on_publish(-0.3, 5.2)          # pending when the reset lands
        self.est.predict_to(0.02)
        self.assertEqual(self.est.delta_hat, 0.0)
        self.assertAlmostEqual(self.est.t, 0.02)
        self.est.predict_to(0.5)
        self.assertEqual(self.est.delta_hat, 0.0)

    def test_a_newer_publish_supersedes_pending_ones(self):
        self.est.predict_to(0.0)
        self.est.on_publish(0.3, 0.10)
        self.est.on_publish(-0.3, 0.05)         # published later, latches earlier
        self.est.predict_to(0.5)
        self.assertLess(self.est.delta_hat, 0.0)

    def test_an_immediate_publish_supersedes_pending_ones(self):
        self.est.predict_to(0.0)
        self.est.on_publish(0.3, 0.10)
        self.est.on_publish(-0.3)               # unstamped: acts from now
        self.est.predict_to(0.5)
        self.assertLess(self.est.delta_hat, 0.0)

    def test_an_interval_off_the_physics_grid_is_advanced_in_full(self):
        self.est.predict_to(0.0)
        self.est.on_publish(0.3, 0.0)
        self.est.predict_to(0.015)
        # One full 10 ms step at sv_max, then half a step more.
        after_one = PARAMS.sv_max * PHYSICS_DT
        self.assertGreater(self.est.delta_hat, after_one)
        self.assertLess(self.est.delta_hat, 2 * after_one)


class SteeringEstimatorCorrectTest(unittest.TestCase):

    def _yaw_rate_for(self, delta, v, beta=0.0):
        return v * math.cos(beta) * math.tan(delta) / WHEELBASE

    def test_no_correction_below_the_speed_threshold(self):
        est = driver_core.SteeringEstimator()
        est.correct(self._yaw_rate_for(0.2, 0.5), 0.5, 0.0, DT)
        self.assertEqual(est.delta_hat, 0.0)

    def test_pulls_toward_the_yaw_rate_measurement(self):
        est = driver_core.SteeringEstimator()
        est.correct(self._yaw_rate_for(0.2, 2.0), 2.0, 0.0, DT)
        gain = 1.0 - math.exp(-DT / driver_core.STEER_CORRECT_TAU)
        self.assertAlmostEqual(est.delta_hat, gain * 0.2, places=9)

    def test_the_gain_is_about_five_percent_per_25_ms(self):
        gain = 1.0 - math.exp(-DT / driver_core.STEER_CORRECT_TAU)
        self.assertAlmostEqual(gain, 0.05, delta=0.002)

    def test_sideslip_is_used_in_the_inversion(self):
        est = driver_core.SteeringEstimator()
        est.delta_hat = 0.2
        est.correct(self._yaw_rate_for(0.2, 2.0, beta=0.1), 2.0, 0.1, DT)
        self.assertAlmostEqual(est.delta_hat, 0.2, places=9)

    def test_residual_summary_reports_p95_and_resets(self):
        est = driver_core.SteeringEstimator()
        for _ in range(20):
            est.correct(self._yaw_rate_for(0.1, 2.0), 2.0, 0.0, DT)
        self.assertGreater(est.residual_p95(), 0.0)
        est.reset_stats()
        self.assertEqual(est.residual_p95(), 0.0)


# ----------------------------------------------------------------------
# DriverCore wiring
# ----------------------------------------------------------------------
class _RecordingMPC:
    dt = DT

    def __init__(self):
        self.states = []

    def speed_profile(self, **_kw):
        return np.full(4, 3.0)

    def set_reference_path(self, *_a):
        pass

    def get_control(self, state):
        self.states.append(np.array(state))
        return np.array([0.0, 0.0]), {'solve_time_ms': 0.0}


def _core():
    mpc = _RecordingMPC()
    core = driver_core.DriverCore(cfg=driver_core.DriverConfig(), mpc_controller=mpc)
    core.reference_path = np.zeros((4, 2))
    return core, mpc


def _feed(core, t, v=0.5, yaw_rate=0.0):
    core.set_odom(0.0, 0.0, 0.0, v, beta=0.0, yaw_rate=yaw_rate, t_sim=t)
    core.set_scan(np.full(181, 6.0), -np.pi / 2, np.pi / 180, t_sim=t)


class DriverCoreSteeringTest(unittest.TestCase):

    def test_the_mpc_state_is_six_wide_with_zero_sideslip(self):
        core, mpc = _core()
        _feed(core, 0.0)
        core.step(t_sim=0.0)
        self.assertEqual(len(mpc.states[0]), 6)
        self.assertEqual(mpc.states[0][4], 0.0)

    def test_the_mpc_receives_the_estimated_wheel_angle(self):
        core, mpc = _core()
        _feed(core, 0.0)
        core.step(t_sim=0.0)
        core.note_published(driver_core.DriveCommand(0.3, 1.0), t_sim=0.0)
        _feed(core, 0.1)
        core.step(t_sim=0.1)
        self.assertGreater(mpc.states[-1][5], 0.2)
        self.assertEqual(mpc.states[-1][5], core.delta_hat)

    def test_the_live_residual_summary_reports_then_clears(self):
        core, _ = _core()
        yaw_rate = 2.0 * math.tan(0.1) / WHEELBASE   # wheel really at 0.1 rad
        for i in range(10):
            _feed(core, i * DT, v=2.0, yaw_rate=yaw_rate)
            core.step(t_sim=i * DT)
        self.assertGreater(core.steering_residual_summary(), 0.05)
        self.assertEqual(core.steering_residual_summary(), 0.0)

    def test_the_estimate_is_brought_to_the_odom_stamp_the_pose_came_from(self):
        core, mpc = _core()
        _feed(core, 0.0)
        core.step(t_sim=0.0)
        core.note_published(driver_core.DriveCommand(0.3, 1.0), t_sim=0.0)
        core.set_odom(0.0, 0.0, 0.0, 0.5, t_sim=0.05)       # pose at 0.05 ...
        core.set_scan(np.full(181, 6.0), -np.pi / 2, np.pi / 180, t_sim=0.06)  # scan a tick on
        core.step(t_sim=0.06)
        self.assertAlmostEqual(core.steering.t, 0.05)

    def test_an_unstamped_publish_latches_at_the_current_estimate_time(self):
        core, mpc = _core()
        _feed(core, 0.0)
        core.step(t_sim=0.0)
        core.note_published(driver_core.DriveCommand(0.3, 1.0))
        _feed(core, 0.1)
        core.step(t_sim=0.1)
        self.assertGreater(mpc.states[-1][5], 0.2)


# ----------------------------------------------------------------------
# Live latch time: the node's clock is frozen while it solves
# ----------------------------------------------------------------------
class LatchClockTest(unittest.TestCase):

    def test_before_any_rate_is_seen_sim_runs_at_wall_speed(self):
        clock = driver_core.LatchClock()
        self.assertAlmostEqual(clock.latch_time(10.0, 0.005), 10.005)

    def test_the_real_time_factor_comes_from_sim_versus_wall_progress(self):
        clock = driver_core.LatchClock()
        for i in range(200):
            clock.update(i * 0.025, i * 0.025 / 2.2)   # sim runs 2.2x wall
        self.assertAlmostEqual(clock.rtf, 2.2, places=6)

    def test_a_command_latches_a_solve_later_in_sim_time(self):
        clock = driver_core.LatchClock()
        for i in range(200):
            clock.update(i * 0.025, i * 0.025 / 2.0)
        # 6 ms of wall solve at RTF 2 is 12 ms of plant time.
        self.assertAlmostEqual(clock.latch_time(5.0, 0.006), 5.012, places=6)

    def test_a_sim_reset_or_a_stalled_wall_clock_is_ignored(self):
        clock = driver_core.LatchClock()
        for i in range(200):
            clock.update(i * 0.025, i * 0.025 / 2.0)
        clock.update(0.0, 100.0)          # sim time went backwards
        clock.update(0.025, 100.0)        # no wall time passed
        self.assertAlmostEqual(clock.rtf, 2.0, places=6)


# ----------------------------------------------------------------------
# The harness measures the estimator against the plant's own wheel
# ----------------------------------------------------------------------
class HarnessEstimatorGateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = lap_sim.HarnessConfig(laps=1, max_time=3.0, stale_steps=9,
                                    fixed_solve_ms=18.0, quiet=True)
        cls.result = lap_sim.run(cfg)

    def test_the_gate_is_two_hundredths_of_a_radian(self):
        self.assertEqual(lap_sim.MAX_DELTA_EST_ERR, 0.02)

    def test_the_estimate_error_is_reported(self):
        for key in ('delta_est_err_mean', 'delta_est_err_p95', 'delta_est_err_max'):
            self.assertIn(key, self.result)

    def test_the_estimate_tracks_the_stale_snapshots_wheel(self):
        self.assertGreater(self.result['distance_m'], 0.5)
        self.assertLess(self.result['delta_est_err_max'], lap_sim.MAX_DELTA_EST_ERR)

    def test_a_drifting_estimate_fails_the_verdict(self):
        reasons = lap_sim.verdict(dict(laps=10, laps_requested=10, collisions=0,
                                       sat_duty=0.0, cmd_slew_mean=0.005,
                                       delta_est_err_max=0.03))
        self.assertEqual(len(reasons), 1)
        self.assertIn('wheel estimate', reasons[0])


if __name__ == '__main__':
    unittest.main()
