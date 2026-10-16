#!/usr/bin/env python3
"""Vehicle model wrapper for MPC controller.

Provides interface to F1TENTH vehicle dynamics for use in MPC prediction.
"""

import cmath
import math
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

# The judged steering servo over one MPC step, calibrated against GymPlant by
# least squares (test/test_steering_actuator.py pins it). A 25 ms command on the
# 10 ms physics grid is held for 2 then 3 steps, so the target is the mean move
# of those two holds. The continuous-time alpha, 1 - exp(-kp*dt) = 0.615, is
# wrong here: the plant's servo is a discrete Euler step, which closes more of
# the error per step than its continuous limit would.
SERVO_CAL_DT = 0.025      # [s] the step the constants below are fitted for
SERVO_ALPHA = 0.6903      # fraction of a small error closed per SERVO_CAL_DT
SERVO_D_MAX = 0.0801      # [rad] wheel travel per SERVO_CAL_DT at the rate limit
# Knee sharpness of the soft clip. tanh (the plan's first choice) fits no
# better than 6.7 % of d_max and overstates the small-signal rate by 26 %; this
# fits within 1.5 % and is exact in the linear regime.
SERVO_SHARPNESS = 8


@dataclass(frozen=True)
class ServoModel:
    """The wheel's move over one step toward a held command, smoothly saturated.

    move = d_max * x / (1 + |x|^p)^(1/p),  x = alpha * error / d_max

    Linear with slope alpha for small errors, tending to d_max in saturation,
    and strictly monotone: the derivative (1 + |x|^p)^-(1 + 1/p) is never zero.
    That matters because SLSQP differentiates by finite differences over the
    command; a hard clip reads a gradient of exactly zero the moment the rate
    limit binds. Here the gradient stays visible to SLSQP's 1.5e-8 step well
    past the knee (tested at 2.6x it). Past ~0.5 rad of error it does fade to
    numerically nothing -- honestly: every such command moves the wheel by
    d_max, and there the effort and rate costs carry the gradient, pulling the
    command back toward the knee. It never overshoots the command.
    """

    alpha: float = SERVO_ALPHA
    d_max_cal: float = SERVO_D_MAX
    sharpness: int = SERVO_SHARPNESS
    cal_dt: float = SERVO_CAL_DT

    def alpha_at(self, dt: float) -> float:
        """Linear fraction closed over dt, compounding the calibrated step."""
        return 1.0 - (1.0 - self.alpha) ** (dt / self.cal_dt)

    def d_max(self, dt: float) -> float:
        """[rad] Most the wheel can travel in dt."""
        return self.d_max_cal * dt / self.cal_dt

    def move(self, error: float, dt: float) -> float:
        """[rad] How far the wheel moves in dt toward a command `error` away."""
        if dt <= 0.0:
            return 0.0
        d = self.d_max(dt)
        x = error * self.alpha_at(dt) / d
        p = self.sharpness
        return d * x / (1.0 + abs(x) ** p) ** (1.0 / p)


# The judged car, transcribed from external/f1tenth_gym's dynamic_models/__init__.py
# (F1TENTH_VEHICLE_PARAMETERS). It is deliberately NOT imported from there: the
# gym's `vehicle_dynamics_ks_cog` has state [x, y, delta, v, psi] and control
# [steer_vel, accl] over a flat numpy params array, so a successful import would
# swap state indices 2 and 4 and describe a different vehicle without raising.
# test_driver_params.py reads the upstream file as text and pins these numbers,
# so an upstream bump is caught without importing anything.
@dataclass(frozen=True)
class VehicleParameters:
    """Typed view over the vehicle parameters used by the dynamics models."""
    # Core parameters required by the simplified single-track models
    mu: float = 1.0489
    C_Sf: float = 4.718
    C_Sr: float = 5.4562
    lf: float = 0.15875
    lr: float = 0.17145
    h: float = 0.074
    m: float = 3.74
    I: float = 0.04712
    s_min: float = -0.4189
    s_max: float = 0.4189
    sv_min: float = -3.2
    sv_max: float = 3.2
    v_switch: float = 7.319
    a_max: float = 9.51
    v_min: float = -5.0
    v_max: float = 20.0
    width: float = 0.31
    length: float = 0.58
    collision_body_center_x: float = 0.1651


F1TENTH_VEHICLE_PARAMETERS = VehicleParameters()

PLANT_STEP_DT = 0.01      # [s] gym_bridge.py sim_timestep: one speed-loop evaluation each
PID_ACCL_FORWARD_SCALE = 10.0   # utils.py pid_accl: kp = 10 a_max / v_limit moving forward
PID_ACCL_REVERSE_SCALE = 2.0    # ... and 2 a_max / v_limit at rest or in reverse


@dataclass(frozen=True)
class SpeedLoop:
    """The judged speed loop: utils.py `pid_accl`, then `accl_constraints`.

    /drive carries a speed, and the plant turns it into an acceleration with a
    proportional loop evaluated once per physics step from the pre-step speed:
    k_accel = 10 a_max / v_max = 4.755 /s when speeding up and
    k_brake = 10 a_max / -v_min = 19.02 /s when slowing down (pid_accl uses
    weaker gains at rest and in reverse), then clamped to +-a_max and to
    a_max v_switch / v above v_switch. test_speed_loop.py pins it to the gym.
    """

    a_max: float = 9.51
    v_max: float = 20.0
    v_min: float = -5.0
    v_switch: float = 7.319
    step: float = PLANT_STEP_DT

    def __post_init__(self):
        if not (self.step > 0.0 and self.a_max > 0.0 and self.v_max > 0.0
                and self.v_min < 0.0 and self.v_switch > 0.0):
            raise ValueError(f'not a speed loop the plant could have: {self}')

    @classmethod
    def of(cls, params: Optional[VehicleParameters] = None) -> 'SpeedLoop':
        p = params if params is not None else F1TENTH_VEHICLE_PARAMETERS
        return cls(a_max=float(p.a_max), v_max=float(p.v_max),
                   v_min=float(p.v_min), v_switch=float(p.v_switch))

    @property
    def k_accel(self) -> float:
        """[1/s] Gain while moving forward and speeding up."""
        return PID_ACCL_FORWARD_SCALE * self.a_max / self.v_max

    @property
    def k_brake(self) -> float:
        """[1/s] Gain while moving forward and slowing down."""
        return PID_ACCL_FORWARD_SCALE * self.a_max / -self.v_min

    def accel_ceiling(self, v: float) -> float:
        """[m/s^2] Most the plant accelerates at speed v (utils.py upper_accel_limit)."""
        return self.a_max * self.v_switch / v if v > self.v_switch else self.a_max

    def accel(self, v_cmd: float, v: float) -> float:
        """[m/s^2] What the plant applies for setpoint `v_cmd` at speed `v`."""
        error = v_cmd - v
        if v > 0.0:
            gain = self.k_accel if error > 0.0 else self.k_brake
        else:
            gain = PID_ACCL_REVERSE_SCALE * self.a_max / (self.v_max if error > 0.0
                                                          else -self.v_min)
        a = gain * error
        if (v <= self.v_min and a <= 0.0) or (v >= self.v_max and a >= 0.0):
            return 0.0
        return min(max(a, -self.a_max), self.accel_ceiling(v))

    def command(self, v: float, a_des: float, hold: float) -> float:
        """[m/s] The setpoint that gains the plant `a_des * hold` of speed over `hold`.

        Held for n steps of h, the setpoint error shrinks by (1 - k h) each
        step, so the setpoint must lead by a_des * hold / (1 - (1 - k h)^n).
        As the hold shrinks to one step this is v + a_des / k, the plan's
        inversion; over a real 25 ms hold that would brake ~15 % short. A
        demand past what the plant can give is cut to it first, so the lead is
        never bigger than the next command has to take back. The forward gains
        are used throughout: at rest the plant's weaker gain lasts one step.
        """
        a = min(max(a_des, -self.a_max), self.accel_ceiling(max(v, 0.0)))
        if a == 0.0 or hold <= 0.0:
            return v
        kh = (self.k_accel if a > 0.0 else self.k_brake) * self.step
        n = hold / self.step
        whole = math.floor(n)
        left = (1.0 - kh) ** whole
        closed = 1.0 - left + left * kh * (n - whole)   # a part step closes linearly
        return v + a * hold / closed


def vehicle_dynamics_ks_cog(state, u, params):
    """Simplified kinematic bicycle model, state [x, y, psi, v, beta]."""
    x, y, psi, v, beta = state
    delta, a = u  # steering angle, acceleration

    # Kinematic bicycle model
    x_dot = v * math.cos(psi + beta)
    y_dot = v * math.sin(psi + beta)
    psi_dot = v * math.cos(beta) * math.tan(delta) / (params.lf + params.lr)
    v_dot = a
    # Simplified sideslip approximation
    beta_dot = 0.0  # Placeholder

    return np.array([x_dot, y_dot, psi_dot, v_dot, beta_dot])


MODEL_TYPES = ('KS', 'ST')
GRAVITY = 9.81            # [m/s^2] as single_track.py has it
# single_track.py:103 -- below this the gym switches to its kinematic branch,
# where the tyre model's 1/v terms would blow up.
ST_SWITCH_SPEED = 0.5     # [m/s]
# |s^2 - det| below which the lateral block's eigenvalues count as repeated.
_REPEATED_ROOT_TOL = 1e-9


@dataclass(frozen=True)
class _SingleTrackConstants:
    """The terms of single_track.py's tyre model that do not depend on the state."""

    mu: float
    lf: float
    lr: float
    wheelbase: float
    h: float
    c_sf: float
    c_sr: float
    k_yaw: float           # mu m / (I (lf + lr)), the yaw equation's prefactor
    understeer: float      # K = (1 / (mu g)) (1 / C_Sf - 1 / C_Sr), rad per m/s^2

    @classmethod
    def of(cls, p: 'VehicleParameters') -> '_SingleTrackConstants':
        wheelbase = p.lf + p.lr
        return cls(mu=p.mu, lf=p.lf, lr=p.lr, wheelbase=wheelbase, h=p.h,
                   c_sf=p.C_Sf, c_sr=p.C_Sr, k_yaw=p.mu * p.m / (p.I * wheelbase),
                   understeer=(1.0 / (p.mu * GRAVITY)) * (1.0 / p.C_Sf - 1.0 / p.C_Sr))

    def lateral(self, v: float, a: float):
        """single_track.py's beta and yaw-rate equations as x' = A x + b delta.

        x = [beta, r]. The load transfer (glr, glf) makes A depend on the
        acceleration as well as the speed; both are held over a step.
        Returns (a11, a12, a21, a22, b1, b2).
        """
        cf = self.c_sf * (GRAVITY * self.lr - a * self.h)
        cr = self.c_sr * (GRAVITY * self.lf + a * self.h)
        mu_vl = self.mu / (v * self.wheelbase)
        return (-mu_vl * (cr + cf),
                mu_vl * (cr * self.lr - cf * self.lf) / v - 1.0,
                self.k_yaw * (self.lr * cr - self.lf * cf),
                -self.k_yaw * (self.lf * self.lf * cf + self.lr * self.lr * cr) / v,
                mu_vl * cf,
                self.k_yaw * self.lf * cf)

    def kinematic_slip(self, delta: float) -> float:
        """The gym's beta below the switch speed: the no-slip geometry."""
        return math.atan(math.tan(delta) * self.lr / self.wheelbase)


def vehicle_dynamics_st_rear(state, u, c: _SingleTrackConstants):
    """single_track.py's model, at the rear axle, state [x, y, psi, v, beta, r].

    The gym integrates the CoG. Odometry reports the rear axle (base_link),
    and so does the MPC's reference, so the position rows here are the rear
    axle's velocity: the CoG's, v (cos(psi + beta), sin(psi + beta)), plus
    omega x (-lr) along the heading. In the body frame that is
    (v cos beta, v sin beta - lr r): the rear axle slides sideways by exactly
    the rear tyre's slip.
    """
    x, y, psi, v, beta, r = state
    delta, a = u
    if v < ST_SWITCH_SPEED:
        # The gym's kinematic branch: no tyre slip, beta from the geometry.
        beta = c.kinematic_slip(delta)
        r = v * math.cos(beta) * math.tan(delta) / c.wheelbase
        return np.array([v * math.cos(beta) * math.cos(psi),
                         v * math.cos(beta) * math.sin(psi), r, a, 0.0, 0.0])
    a11, a12, a21, a22, b1, b2 = c.lateral(v, a)
    return np.array([v * math.cos(psi + beta) + c.lr * r * math.sin(psi),
                     v * math.sin(psi + beta) - c.lr * r * math.cos(psi),
                     r,
                     a,
                     a11 * beta + a12 * r + b1 * delta,
                     a21 * beta + a22 * r + b2 * delta])


def _phi1(lam: float, h: float) -> float:
    """(e^{lam h} - 1) / lam, i.e. the integral of e^{lam t} over [0, h]."""
    return h if lam == 0.0 else math.expm1(lam * h) / lam


def _lateral_zoh(c: _SingleTrackConstants, v, a, beta, r, delta, h):
    """Exact zero-order-hold step of [beta, r] with v, a and delta held over h.

    The block is stiff at low speed (eigenvalues ~ -113/v and -52/v), so
    explicit Euler diverges below ~2.5 m/s on a 25 ms step. It is linear for a
    held speed, so step it exactly instead: x(h) = E x + G b delta, with
    E = e^{A h} and G the integral of e^{A t} over [0, h]. For a 2x2 matrix
    both are c0 I + c1 (A - s I), s = tr(A)/2, with scalar c0, c1 from the
    eigenvalues s +/- q. G is built from (e^{lam h} - 1)/lam per eigenvalue,
    not from A^{-1}: det(A) does reach 0 -- braking hard above ~12 m/s, where
    load transfer makes the car oversteer and one eigenvalue crosses zero.
    """
    a11, a12, a21, a22, b1, b2 = c.lateral(v, a)
    s = 0.5 * (a11 + a22)
    disc = s * s - (a11 * a22 - a12 * a21)
    if disc > _REPEATED_ROOT_TOL:
        q = math.sqrt(disc)
        e_hi, e_lo = math.exp((s + q) * h), math.exp((s - q) * h)
        p_hi, p_lo = _phi1(s + q, h), _phi1(s - q, h)
        e0, e1 = 0.5 * (e_hi + e_lo), 0.5 * (e_hi - e_lo) / q
        g0, g1 = 0.5 * (p_hi + p_lo), 0.5 * (p_hi - p_lo) / q
    elif disc < -_REPEATED_ROOT_TOL:
        w = math.sqrt(-disc)
        lam = complex(s, w)
        e = cmath.exp(lam * h)
        g = (e - 1.0) / lam            # |lam|^2 = det > s^2 > 0 here: no zero
        e0, e1 = e.real, e.imag / w
        g0, g1 = g.real, g.imag / w
    else:
        e0, e1 = math.exp(s * h), h * math.exp(s * h)
        g0 = _phi1(s, h)
        g1 = h * h / 2.0 if s == 0.0 else (h * math.exp(s * h) - g0) / s
    f1, f2 = b1 * delta, b2 * delta
    return ((e0 + e1 * (a11 - s)) * beta + e1 * a12 * r
            + (g0 + g1 * (a11 - s)) * f1 + g1 * a12 * f2,
            e1 * a21 * beta + (e0 + e1 * (a22 - s)) * r
            + g1 * a21 * f1 + (g0 + g1 * (a22 - s)) * f2)


class VehicleModel:
    """Vehicle model wrapper for MPC prediction."""

    def __init__(self, params: VehicleParameters = None, model_type: str = 'KS',
                 servo: ServoModel = None):
        """Initialize vehicle model.

        Args:
            params: Vehicle parameters (defaults to F1TENTH)
            model_type: 'KS' for kinematic single track, beta pinned at 0;
                'ST' for the judged dynamic single track (linear tyres), which
                adds the yaw rate as a seventh actuated state
            servo: Steering servo model for `actuated_step` (defaults to the
                one calibrated against the judged plant)
        """
        self.params = params if params is not None else F1TENTH_VEHICLE_PARAMETERS
        self.model_type = model_type
        self.servo = servo if servo is not None else ServoModel()
        self._st = _SingleTrackConstants.of(self.params)

        # State dimensions
        if model_type == 'KS':
            self.state_dim = 5  # [x, y, yaw, velocity, sideslip]
            self.actuated_dim = 6  # + wheel angle
        elif model_type == 'ST':
            self.state_dim = 6  # [x, y, yaw, velocity, sideslip, yaw rate]
            self.actuated_dim = 7  # [x, y, yaw, v, beta, delta, yaw rate]
        else:
            raise ValueError(f"Unsupported model type: {model_type}; "
                             f"choose from {MODEL_TYPES}")
        self.control_dim = 2  # [steering_angle, acceleration]

    def dynamics_continuous(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Continuous-time dynamics function.

        Args:
            state: KS [x, y, yaw, v, beta]; ST [x, y, yaw, v, beta, yaw_rate]
            control: [steering_angle, acceleration]

        Returns:
            The state's derivative, in the same order
        """
        if self.model_type == 'KS':
            return vehicle_dynamics_ks_cog(state, control, self.params)
        return vehicle_dynamics_st_rear(state, control, self._st)

    def dynamics_discrete(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Discrete-time dynamics: Euler for KS, an exact lateral step for ST.

        Args:
            state: Current state, as `dynamics_continuous` takes it
            control: Control input [steering_angle, acceleration]
            dt: Time step

        Returns:
            Next state, in the same order
        """
        if self.model_type == 'KS':
            state_dot = self.dynamics_continuous(state, control)
            return state + state_dot * dt
        x, y, psi, v, beta, r = (float(s) for s in state)
        return np.array(self.st_step(x, y, psi, v, beta, r,
                                      float(control[0]), float(control[1]), dt))

    def st_step(self, x, y, psi, v, beta, r, delta, a, h):
        """One single-track step of length h, wheel and acceleration held.

        Slip and yaw rate advance exactly (`_lateral_zoh`) for the speed at
        mid-step -- the tyre terms go as 1/v, and on a 100 ms step under
        acceleration the start-of-step speed is visibly wrong. Heading and
        position follow by the trapezoid on them, which is second order.
        """
        c = self._st
        v1 = v + a * h
        if v < ST_SWITCH_SPEED:
            beta1 = c.kinematic_slip(delta)
            cos_b = math.cos(beta1)
            r1 = v1 * cos_b * math.tan(delta) / c.wheelbase
            r_mid = 0.5 * (v + v1) * cos_b * math.tan(delta) / c.wheelbase
            psi_m = psi + 0.5 * h * r_mid
            ground = 0.5 * (v + v1) * cos_b
            return (x + h * ground * math.cos(psi_m), y + h * ground * math.sin(psi_m),
                    psi + h * r_mid, v1, beta1, r1)
        beta1, r1 = _lateral_zoh(c, v + 0.5 * a * h, a, beta, r, delta, h)
        psi1 = psi + 0.5 * h * (r + r1)
        psi_m = 0.5 * (psi + psi1)
        course = psi_m + 0.5 * (beta + beta1)
        v_m = 0.5 * (v + v1)
        lr_r = c.lr * 0.5 * (r + r1)
        return (x + h * (v_m * math.cos(course) + lr_r * math.sin(psi_m)),
                y + h * (v_m * math.sin(course) - lr_r * math.cos(psi_m)),
                psi1, v1, beta1, r1)

    def actuated_step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """One step with the steering servo in the loop.

        The wheel angle is the sixth state. The car turns with the wheel as it
        stands, not with the command: the dynamics are stepped on state[5],
        and the command only moves the wheel, through the servo model. So the
        rate limit is a property of every predicted trajectory, at no cost in
        decision variables or constraints.

        Args:
            state: KS [x, y, yaw, v, beta, delta];
                ST [x, y, yaw, v, beta, delta, yaw_rate]
            control: [delta_cmd, acceleration]
            dt: Time step

        Returns:
            Next state, in the same order
        """
        delta = state[5]
        if self.model_type == 'ST':
            return np.array(self.scalar_step(tuple(float(s) for s in state),
                                             float(control[0]), float(control[1]), dt))
        nxt = np.empty(6)
        nxt[:5] = self.dynamics_discrete(state[:5], (delta, control[1]), dt)
        nxt[5] = delta + self.servo.move(control[0] - delta, dt)
        return nxt

    def scalar_step(self, state: tuple, cmd: float, a: float, h: float) -> tuple:
        """`actuated_step` on a tuple of Python floats, with no arrays.

        The MPC's fused objective calls this a few thousand times a solve, where
        a per-step numpy array costs more than the physics. KS is the same
        arithmetic in the same order as `actuated_step`, so it agrees to the bit.
        """
        if self.model_type == 'ST':
            # The tyres are driven by the wheel as it stood at the start of the
            # step, as in KS. The mid-step wheel predicts a sweeping wheel
            # better open-loop (yaw within 3 % over a 100 ms sweep, against
            # ~2x low here), but closed-loop it was worse: 10 harness laps at
            # a_lat_max 5 went from 0 collisions / 17.18 s to 2 / 17.80 s. The
            # ~100 ms of command dead time the MPC does not model is the likely
            # reason; the start-of-step wheel's lag offsets part of it. Revisit
            # together with latency compensation.
            x, y, psi, v, beta, delta, r = state
            x, y, psi, v, beta, r = self.st_step(x, y, psi, v, beta, r, delta, a, h)
            return (x, y, psi, v, beta, delta + self.servo.move(cmd - delta, h), r)
        x, y, psi, v, beta, delta = state
        wheelbase = self.params.lf + self.params.lr
        return (x + v * math.cos(psi + beta) * h,
                y + v * math.sin(psi + beta) * h,
                psi + v * math.cos(beta) * math.tan(delta) / wheelbase * h,
                v + a * h,
                beta,
                delta + self.servo.move(cmd - delta, h))

    def rollout(self, state0: np.ndarray, controls: np.ndarray, dts: np.ndarray,
                out: np.ndarray) -> np.ndarray:
        """`actuated_step` over a horizon, written into `out` (N+1 rows).

        For ST the loop runs on Python floats with no per-step array; KS keeps
        its original step, so its predictions are unchanged to the bit.
        """
        out[0] = state0
        if self.model_type != 'ST':
            for k in range(len(dts)):
                out[k + 1] = self.actuated_step(out[k], controls[k], dts[k])
            return out
        state = tuple(float(s) for s in state0)
        for k in range(len(dts)):
            cmd, a = controls[k]
            state = self.scalar_step(state, float(cmd), float(a), float(dts[k]))
            out[k + 1] = state
        return out

    def steady_steer(self, kappa: float, v: float) -> float:
        """[rad] The wheel angle that holds curvature `kappa` at speed `v`.

        KS: the geometry, atan(L kappa). ST is linear in the wheel angle, so
        its steady state is (L + K v^2) kappa with the understeer
        K = 0.00279 rad per m/s^2: the steady state the speed profile's lock
        limit is built on (speed_profile.py).
        """
        if self.model_type == 'KS':
            return math.atan(self._st.wheelbase * kappa)
        return (self._st.wheelbase + self._st.understeer * v * v) * kappa

    def get_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get control input bounds.

        Returns:
            (min_control, max_control) arrays
        """
        min_control = np.array([
            self.params.s_min,  # steering angle
            self.params.a_max * -0.3  # acceleration (conservative reverse)
        ])

        max_control = np.array([
            self.params.s_max,  # steering angle
            self.params.a_max   # acceleration
        ])

        return min_control, max_control

