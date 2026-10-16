"""Phase 2 Step 3: the MPC's dynamic single-track model against the judged one.

The judged plant is the gym's `vehicle_dynamics_st` (linear tyres, RK4 at
10 ms). The MPC's KS model pins sideslip at zero, which at the target lateral
acceleration (~10 m/s^2, ~0.18 rad of rear slip) is the largest modelling
error left. `VehicleModel(model_type='ST')` ports the gym's model to the MPC's
state order and frame:

    [x, y, yaw, v, beta, delta, yaw_rate]   at the rear axle (base_link)

The reference here is the gym's own function, imported through plant_sim, so
the test cannot drift from the judge.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, 'team_driver')))

import plant_sim  # noqa: E402
from vehicle_model import VehicleModel  # noqa: E402

# The reference integrates the gym's function with its own RK4 on a grid that
# divides every schedule entry (0.025 is not a whole number of 10 ms steps).
REFERENCE_DT = 0.005
# The gym's parameters are a float32 array, and under NumPy 2 a float32 scalar
# times a Python float stays float32, so its derivatives carry float32 rounding
# (~1e-5 after the yaw equation's cancellation). A wrong term is O(1).
RHS_RTOL = 2e-5
PARAMS = VehicleModel().params
LR = PARAMS.lr
WHEELBASE = PARAMS.lf + PARAMS.lr
SCHEDULE = [0.025, 0.025, 0.05, 0.05, 0.075, 0.075, 0.075, 0.075]
X, Y, YAW, V, BETA, DELTA, YAW_RATE = range(7)


def _gym_rhs(state7_ours, a):
    """The gym's ST derivative for our state, wheel held (zero steer velocity)."""
    x, y, yaw, v, beta, delta, r = state7_ours
    cog = np.array([x + LR * math.cos(yaw), y + LR * math.sin(yaw),
                    delta, v, yaw, r, beta])
    return plant_sim.vehicle_dynamics_st(
        cog, np.array([0.0, a]), plant_sim.F1TENTH_VEHICLE_PARAMETERS.to_array())


def _gym_run(cog, seconds, a=0.0):
    """Integrate the gym's model with the wheel held, as the judge would."""
    params = plant_sim.F1TENTH_VEHICLE_PARAMETERS.to_array()
    state = np.array(cog, dtype=np.float64)
    for _ in range(int(round(seconds / REFERENCE_DT))):
        state = plant_sim.rk4_integration(plant_sim.vehicle_dynamics_st, state,
                                          np.array([0.0, a]), REFERENCE_DT, params)
    return state


def _gym_settle(v, delta, seconds=2.0):
    """A gym CoG state cornering steadily at speed v on wheel angle delta."""
    state = np.zeros(7)
    state[plant_sim.V] = v
    state[plant_sim.DELTA] = delta
    return _gym_run(state, seconds)


def _ours_from_gym(cog):
    """The MPC's 7-state, at base_link, for a gym CoG state."""
    x, y, yaw = plant_sim.base_link_pose_of(cog, LR)
    return np.array([x, y, yaw, cog[plant_sim.V], cog[plant_sim.BETA],
                     cog[plant_sim.DELTA], cog[plant_sim.PSI_DOT]])


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class StructureTest(unittest.TestCase):

    def test_kinematic_model_keeps_its_six_actuated_states(self):
        self.assertEqual(VehicleModel().actuated_dim, 6)

    def test_single_track_model_adds_the_yaw_rate(self):
        self.assertEqual(VehicleModel(model_type='ST').actuated_dim, 7)

    def test_an_unknown_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'model'):
            VehicleModel(model_type='MB')


class ContinuousDynamicsTest(unittest.TestCase):
    """Our right-hand side is the gym's, moved from the CoG to the rear axle."""

    CASES = [  # (v, beta, delta, yaw_rate, a)
        (1.0, 0.02, 0.10, 0.20, 0.0),
        (3.0, -0.05, 0.20, 1.50, 1.0),
        (5.0, 0.08, -0.15, -2.0, -2.0),
        (7.0, 0.10, 0.05, 1.0, 0.5),
    ]

    def _both(self, v, beta, delta, r, a):
        model = VehicleModel(model_type='ST')
        state = np.array([0.3, -0.2, 0.7, v, beta, delta, r])
        ours = model.dynamics_continuous(state[[0, 1, 2, 3, 4, 6]], (delta, a))
        return state, ours, _gym_rhs(state, a)

    def test_yaw_speed_slip_and_yaw_rate_rates_match_the_gym(self):
        for case in self.CASES:
            with self.subTest(v=case[0]):
                _, ours, gym = self._both(*case)
                np.testing.assert_allclose(
                    ours[2:6],
                    [gym[plant_sim.PSI], gym[plant_sim.V], gym[plant_sim.BETA],
                     gym[plant_sim.PSI_DOT]],
                    rtol=RHS_RTOL, atol=RHS_RTOL)

    def test_the_rear_axle_moves_as_the_rigid_body_says(self):
        # d/dt (X - lr cos psi, Y - lr sin psi) for the gym's CoG motion.
        for case in self.CASES:
            with self.subTest(v=case[0]):
                state, ours, gym = self._both(*case)
                yaw, r = state[YAW], state[YAW_RATE]
                np.testing.assert_allclose(
                    ours[:2],
                    [gym[plant_sim.X] + LR * math.sin(yaw) * r,
                     gym[plant_sim.Y] - LR * math.cos(yaw) * r],
                    rtol=RHS_RTOL, atol=RHS_RTOL)

    def test_below_the_switch_speed_it_is_the_gyms_kinematic_branch(self):
        model = VehicleModel(model_type='ST')
        state = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0])
        ours = model.dynamics_continuous(state, (0.2, 0.0))
        beta_hat = math.atan(math.tan(0.2) * LR / WHEELBASE)
        self.assertAlmostEqual(
            ours[2], 0.3 * math.cos(beta_hat) * math.tan(0.2) / WHEELBASE, places=12)
        self.assertAlmostEqual(ours[1], 0.0, places=12)   # rear axle: no side slip


class DiscreteStepTest(unittest.TestCase):
    """The MPC's step, with the wheel held, against the gym's RK4 at 10 ms."""

    def _compare(self, v, delta, dts, a=0.0):
        cog = _gym_settle(v, delta)
        state = _ours_from_gym(cog)
        model = VehicleModel(model_type='ST')
        for dt in dts:
            state = model.actuated_step(state, (delta, a), dt)
        truth = _ours_from_gym(_gym_run(cog, sum(dts), a))
        return state, truth

    def test_one_control_step_matches_the_gym_at_racing_speeds(self):
        for v in (3.0, 5.0, 7.0):
            with self.subTest(v=v):
                state, truth = self._compare(v, 0.08, [0.025])
                self.assertLess(math.hypot(*(state[:2] - truth[:2])), 1e-4)
                self.assertLess(abs(_wrap(state[YAW] - truth[YAW])), 1e-4)
                self.assertLess(abs(state[BETA] - truth[BETA]), 1e-4)
                self.assertLess(abs(state[YAW_RATE] - truth[YAW_RATE]), 1e-3)

    def test_the_whole_long_schedule_stays_within_a_centimetre(self):
        for v in (3.0, 5.0, 7.0):
            with self.subTest(v=v):
                state, truth = self._compare(v, 0.08, SCHEDULE)
                self.assertLess(math.hypot(*(state[:2] - truth[:2])), 0.01)
                self.assertLess(abs(_wrap(state[YAW] - truth[YAW])), 0.01)

    def test_a_steering_step_is_followed_through_the_slip_transient(self):
        # Settled straight, then the wheel is at 0.1 rad: beta and r must
        # build up the way the gym's do, not instantly.
        model = VehicleModel(model_type='ST')
        cog = _gym_settle(5.0, 0.0)
        cog[plant_sim.DELTA] = 0.1
        state = _ours_from_gym(cog)
        for dt in SCHEDULE:
            state = model.actuated_step(state, (0.1, 0.0), dt)
        truth = _ours_from_gym(_gym_run(cog, sum(SCHEDULE)))
        self.assertLess(abs(state[YAW_RATE] - truth[YAW_RATE]),
                        0.02 * abs(truth[YAW_RATE]))
        self.assertLess(abs(state[BETA] - truth[BETA]), 0.02 * abs(truth[BETA]) + 1e-4)

    def test_acceleration_moves_the_speed(self):
        model = VehicleModel(model_type='ST')
        state = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
        nxt = model.actuated_step(state, (0.0, 2.0), 0.05)
        self.assertAlmostEqual(nxt[V], 4.1, places=12)

    def test_it_is_stable_at_low_speed_on_the_longest_step(self):
        # Explicit Euler diverges here (|1 + h lambda| = 7.5 at 1 m/s, 75 ms).
        model = VehicleModel(model_type='ST')
        for v in (0.6, 1.0, 2.0):
            with self.subTest(v=v):
                state = np.array([0.0, 0.0, 0.0, v, 0.0, 0.2, 0.0])
                for _ in range(40):
                    state = model.actuated_step(state, (0.2, 0.0), 0.075)
                self.assertTrue(np.all(np.isfinite(state)))
                self.assertLess(abs(state[BETA]), 0.2)
                # Settled on the kinematic yaw rate to within the understeer.
                r_kin = v * math.tan(0.2) / WHEELBASE
                self.assertAlmostEqual(state[YAW_RATE], r_kin, delta=0.05 * r_kin)

    def test_below_the_switch_speed_slip_follows_the_wheel(self):
        model = VehicleModel(model_type='ST')
        state = np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.0])
        nxt = model.actuated_step(state, (0.2, 0.0), 0.025)
        beta_hat = math.atan(math.tan(0.2) * LR / WHEELBASE)
        self.assertAlmostEqual(nxt[BETA], beta_hat, places=9)
        self.assertAlmostEqual(
            nxt[YAW_RATE], 0.3 * math.cos(beta_hat) * math.tan(0.2) / WHEELBASE,
            places=9)

    def test_the_wheel_moves_through_the_same_servo_as_the_kinematic_model(self):
        ks = VehicleModel().actuated_step(
            np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.05]), (0.3, 0.0), 0.05)
        st = VehicleModel(model_type='ST').actuated_step(
            np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.05, 0.0]), (0.3, 0.0), 0.05)
        self.assertAlmostEqual(st[DELTA], ks[DELTA], places=15)


def _gym_with_inputs(cog, seconds, steer_velocity=0.0, a=0.0, dt=0.0005):
    """The gym's model under a constant steering velocity and acceleration."""
    params = plant_sim.F1TENTH_VEHICLE_PARAMETERS.to_array()
    state = np.array(cog, dtype=np.float64)
    for _ in range(int(round(seconds / dt))):
        state = plant_sim.rk4_integration(plant_sim.vehicle_dynamics_st, state,
                                          np.array([steer_velocity, a]), dt, params)
    return state


class LongStepTest(unittest.TestCase):
    """What a 100 ms dt_schedule step has to get right that a 25 ms one hides."""

    def test_the_tyres_see_the_wheel_as_it_stood_at_the_start_of_the_step(self):
        # Deliberate, and measured: the mid-step wheel predicts a sweeping
        # wheel better open-loop but drove worse closed-loop (see scalar_step).
        # This pins the choice so it is not "fixed" back without a harness run.
        model = VehicleModel(model_type='ST')
        state = (0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0)
        swept = model.scalar_step(state, 0.4, 0.0, 0.1)
        held = model.scalar_step(state, 0.0, 0.0, 0.1)
        self.assertEqual(swept[YAW_RATE], held[YAW_RATE])
        self.assertGreater(swept[DELTA], 0.3)

    def test_acceleration_across_a_long_step_uses_the_mid_step_speed(self):
        cog = _gym_settle(3.0, 0.1)
        truth = _ours_from_gym(_gym_with_inputs(cog, 0.1, a=5.0))
        ours = VehicleModel(model_type='ST').actuated_step(_ours_from_gym(cog), (0.1, 5.0), 0.1)
        self.assertAlmostEqual(ours[YAW_RATE], truth[YAW_RATE], delta=0.06 * truth[YAW_RATE])

    def test_the_step_is_finite_where_the_lateral_matrix_is_singular(self):
        # Braking above ~12 m/s, load transfer makes the car oversteer and one
        # eigenvalue of the slip/yaw-rate block crosses zero: det(A) = 0.
        constants = VehicleModel(model_type='ST')._st
        a = -2.853

        def det(v):
            a11, a12, a21, a22, _, _ = constants.lateral(v, a)
            return a11 * a22 - a12 * a21

        lo, hi = 10.0, 14.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if det(mid) > 0.0 else (lo, mid)
        cog = np.zeros(7)
        cog[plant_sim.V] = lo + 0.5 * a * 0.05     # singular at mid-step
        cog[plant_sim.DELTA] = 0.02
        cog[plant_sim.BETA] = 0.01
        cog[plant_sim.PSI_DOT] = 0.5
        ours = VehicleModel(model_type='ST').actuated_step(_ours_from_gym(cog), (0.02, a), 0.05)
        truth = _ours_from_gym(_gym_with_inputs(cog, 0.05, a=a))
        self.assertTrue(np.all(np.isfinite(ours)))
        self.assertAlmostEqual(ours[YAW_RATE], truth[YAW_RATE], delta=0.01)
        self.assertAlmostEqual(ours[BETA], truth[BETA], delta=0.002)


class RolloutTest(unittest.TestCase):
    """The fused horizon rollout the MPC calls is the step, repeated."""

    def test_rollout_equals_repeated_actuated_steps(self):
        model = VehicleModel(model_type='ST')
        x0 = np.array([1.0, 2.0, 0.4, 5.0, 0.03, 0.05, 0.8])
        controls = np.array([[0.1, 1.0], [0.12, 0.5], [0.12, 0.5], [0.0, -2.0],
                             [0.0, -2.0], [0.0, -2.0], [0.0, -2.0], [0.0, -2.0]])
        out = np.zeros((len(SCHEDULE) + 1, 7))
        model.rollout(x0, controls, np.asarray(SCHEDULE), out)
        state = x0
        np.testing.assert_allclose(out[0], x0)
        for k, dt in enumerate(SCHEDULE):
            state = model.actuated_step(state, controls[k], dt)
            np.testing.assert_allclose(out[k + 1], state, rtol=0, atol=1e-12)

    def test_kinematic_rollout_equals_repeated_actuated_steps(self):
        model = VehicleModel()
        x0 = np.array([1.0, 2.0, 0.4, 3.0, 0.0, 0.05])
        controls = np.tile([0.1, 1.0], (4, 1))
        out = np.zeros((5, 6))
        model.rollout(x0, controls, np.full(4, 0.025), out)
        state = x0
        for k in range(4):
            state = model.actuated_step(state, controls[k], 0.025)
            np.testing.assert_array_equal(out[k + 1], state)


class SteadyStateTest(unittest.TestCase):

    def test_the_wheel_angle_a_corner_needs_includes_the_understeer(self):
        # delta = atan(L kappa) + K v^2 kappa, the steady state the profile uses.
        model = VehicleModel(model_type='ST')
        kappa, v = 0.5, 5.0
        delta = model.steady_steer(kappa, v)
        state = np.array([0.0, 0.0, 0.0, v, 0.0, delta, 0.0])
        for _ in range(200):
            state = model.actuated_step(state, (delta, 0.0), 0.025)
        self.assertAlmostEqual(state[YAW_RATE] / v, kappa, delta=0.01 * kappa)

    def test_the_kinematic_model_needs_only_the_geometry(self):
        self.assertAlmostEqual(VehicleModel().steady_steer(0.5, 5.0),
                               math.atan(WHEELBASE * 0.5), places=12)

    def test_steady_steer_is_odd_in_curvature(self):
        model = VehicleModel(model_type='ST')
        self.assertAlmostEqual(model.steady_steer(-0.5, 5.0),
                               -model.steady_steer(0.5, 5.0), places=12)


if __name__ == '__main__':
    unittest.main()
