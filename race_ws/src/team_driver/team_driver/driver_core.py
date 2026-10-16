#!/usr/bin/env python3
"""The driver's control path, with no ROS in it.

`driver.py` used to hold both the ROS wiring and the algorithm. That meant the
offline harness had to keep its own copy of the control path, and the two
drifted: `lap_sim.py` fed the MPC its previous *MPC* steer while `driver.py`
fed it the *gap-follower's* steer, so the harness was not testing the thing the
judge runs. Anything measured against that copy was measuring a different car.

So the algorithm lives here and the node is wiring. There is one control path
and both the node and the harness drive it.

Nothing in this module imports rclpy. Logging goes through a duck-typed object
with `info`/`warn`/`error`; the node passes `self.get_logger()` and the harness
passes whatever it likes (or nothing, for silence).
"""

import functools
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple, Optional

import numpy as np

from mpc_controller import MPCController
from vehicle_model import F1TENTH_VEHICLE_PARAMETERS, PLANT_STEP_DT, SpeedLoop, VehicleModel
import lap_floor
import raceline
from speed_profile import check_gg_floor, sector_triples, slip_budget, speed_profile
from latch_predictor import (LATCH_TIME_TOL, LatchPredictor, Prediction,
                             plant_servo_rate, servo_kp)
from longitudinal import (LIDAR_FORWARD, ClosedPath, bearing_lookahead,
                          clearance_along_path, front_overhang, path_clearance,
                          scan_points)


# Swallows log calls, so the core runs with no logger attached.
_NULL_LOG = SimpleNamespace(info=lambda _msg: None, warn=lambda _msg: None,
                            error=lambda _msg: None)


@dataclass(frozen=True)
class DriverConfig:
    """Everything the control path is allowed to be tuned by.

    `driver_params.py` declares one node parameter per field here, with these
    defaults, so a forgotten `--driver-params` flag is a provable no-op rather
    than a silently different car.
    """

    crawl_speed: float = 1.0        # [m/s] the emergency cap never asks for less
    max_range: float = 8.0          # [m] clip the scan here
    v_max: float = 10.0             # [m/s] caps the profile and emergency bound; should not bind
    # Speed profile (speed_profile.py). `a_lat_max` is a slip budget, not a grip
    # limit: the plant's tyres are linear, so rear slip is a_y / 56.1 rad.
    # [m/s^2] Cornering budget: ~21 deg of rear slip, held lower in `sector_scale`.
    # Sub-10 Step 4, on the min-time line refined at the g-g a22 profile with the
    # g-g brake limit and the capped emergency corridor on: 21, one step under the
    # fastest clean budget. 16/16 clean (fixed 5/7/12 ms x stale 5/9/14, 10 laps,
    # 5 measured runs, centreline), best 9.82 s; 0 collisions in 24 runs under
    # 12-job host load. a22 is clean on the same grid (9.71 s) but hit s 54 in 2 of
    # 18 loaded runs, a23 on 15 of 16 grid runs.
    a_lat_max: float = 21.0
    lock_margin: float = 0.95       # fraction of the steering lock the profile may use
    a_accel_max: float = 9.51       # [m/s^2] acceleration the profile plans (plant: 9.51)
    # [m/s^2] Braking the profile plans, and the most the conditioner passes on
    # from the MPC (plant: 9.51). The profile and emergency caps may brake harder.
    a_brake_max: float = 6.0
    # Where the slip budget is lowered: flat (s_start [m], s_end [m], scale) triples,
    # s the arc length from the reference's first waypoint, scale in (0, 1].
    # Flat because ROS parameters are flat lists (speed_profile.sector_triples);
    # never empty, or rclpy types it as a byte array: (0, 1, 1) turns it off.
    # Sub-10 Step 4, on the re-refined min-time line: 43-47 @0.85 for the chicane-exit
    # tracking transient (s~45.3 collides without it), 3-7 @0.95 for the first corner
    # (s~4.8-5.2 collided on the new line). The older sectors are gone: the g-g brake
    # limit covers the braking-yaw sites they patched. Off the raceline, see
    # _slip_budget_for.
    sector_scale: tuple[float, ...] = (43.0, 47.0, 0.85, 3.0, 7.0, 0.95)
    # Combined-accel limit on the profile's braking (speed_profile.gg_decel): the
    # deceleration taken into a corner shrinks with the share of the slip budget the
    # corner already uses, because braking unloads the rear axle and the plant's yaw
    # goes unstable (sub-10 Step 1). Off reproduces the plain profile bit for bit.
    gg_brake: bool = True
    gg_floor: float = 0.1           # braking never drops below a_brake_max*sqrt(gg_floor), in (0, 1]
    # [m/s^2] Braking the emergency cap assumes, and the rate a coast sheds speed.
    a_emergency: float = 7.0
    # Continue the emergency cap's corridor past the MPC's 0.5 s plan along the
    # reference (offset kept), not straight: straight, it ran into the outside
    # wall of each corner the plan had not reached, and braked the car there at
    # up to -10.6 m/s^2 where the profile asked -2 (sub-10 Step 3).
    cap_follows_reference: bool = True
    steer_limit: float = 0.4189     # [rad] the real lock, VehicleParameters.s_max
    # Reference line the MPC tracks: 'centerline' (the shipped CSV) or 'raceline'
    # (the shipped offline min-time line when it checks out against the map, else
    # minimum curvature built from the map at startup; see raceline.py).
    reference: str = 'raceline'
    # [m] Clearance beyond the car's half-width kept from walls. 0.10 was too thin
    # once the car ran at profile speed (DQ at a_lat_max 5); 0.25 gave 0 collisions
    # and was faster than 0.20, since nothing brakes the car mid-corner.
    raceline_margin: float = 0.25
    # [m] Clearance the emergency cap keeps in hand, beyond the body's own
    # overhang ahead of the lidar (longitudinal.front_overhang).
    stop_margin: float = 0.0
    coast_steer_rate: float = 1.0   # [rad/s] steering decay toward straight while coasting
    # [m/s^2] Most acceleration the conditioner passes on. The MPC's reference window
    # assumes the plant ceiling (a_max 9.51), so keep this at it: lower, and the
    # reference outruns what the conditioner lets through.
    accel_limit: float = 9.51
    # Plan from where the car will be when the command latches, replaying the
    # published commands over the dead time (latch_predictor.py); False plans
    # from the measured, dead-time-old state.
    predict_latch: bool = True

    def __post_init__(self):
        # Here, not only in driver_params.validate(): a bad list raised later, inside
        # the reference loader, would quietly drop the car to the reactive controller.
        sector_triples(self.sector_scale)
        check_gg_floor(self.gg_floor)


class DriveCommand(NamedTuple):
    """One command destined for /drive.

    `steering_angle_velocity` is the rate the conditioner implies,
    (steer - previous steer) / dt. The judged config runs the simulator in
    STEERING_ANGLE mode and ignores it, so it documents the command rather than
    drives the plant.
    """

    steering_angle: float
    speed: float
    steering_angle_velocity: float = 0.0


class ConditioningStats(NamedTuple):
    """How hard the output conditioner has had to work, since construction."""

    steps: int          # commands conditioned (coasts are counted separately)
    slew_binds: int     # of those, commands the slew limiter shortened
    coast_count: int    # commands replaced by `_coast()`
    emergency_binds: int = 0    # conditioned commands the emergency cap slowed

    @property
    def slew_bind_rate(self) -> float:
        return self.slew_binds / self.steps if self.steps else 0.0

    @property
    def emergency_bind_rate(self) -> float:
        return self.emergency_binds / self.steps if self.steps else 0.0


class Plan(NamedTuple):
    """What `plan` wants, before the conditioner holds it to what the car can do."""

    steering: float                 # [rad]
    accel: float                    # [m/s^2] demanded over the command's hold
    profile_speed: float = math.inf  # [m/s] the profile where the hold ends
    speed_cap: float = math.inf     # [m/s] emergency bound, at the latch


REFERENCES = ('centerline', 'raceline')
CACHED_RACELINE = 'icra26_raceline.csv'   # config/: the fallback if the runtime build fails

DT_CLAMP_LOW = 0.5     # x nominal: a burst of scans must not zero the interval
DT_CLAMP_HIGH = 3.0    # x nominal: a dropped scan must not hand over a 1 s dt

STEER_CORRECT_MIN_SPEED = 0.8   # [m/s] below this the yaw-rate inversion is ill-conditioned
STEER_CORRECT_TAU = 0.5         # [s] time constant of the yaw-rate leash (~5 % per 25 ms)
# [s] Sim time jumping back further than this is a reset (/initialpose makes
# the bridge call env.reset(), which zeroes sim_time and re-centres the wheel).
# Smaller backward steps are an out-of-order stamp and are ignored.
SIM_RESET_BACKSTEP = 0.1
RTF_SMOOTHING = 0.05            # EMA weight per control step for the live RTF
# [m/s] Where the gap-follower heads when there is no reference to track: the
# old v_max it was tuned at, not today's cap, which should never bind.
REACTIVE_SPEED = 3.0
# [m] Corridor kept beyond the stopping distance along the plan, so a wall just
# past it still shows as the clearance it is rather than as "nothing seen".
PATH_CAP_SLACK = 1.0
# [m] Point spacing of the corridor continued along the reference past the plan
# (cap_follows_reference): a 0.005 m sagitta on the tightest corner (kappa 1.05).
CAP_CONTINUATION_SPACING = 0.2
# The continuation is where the car goes only while it is following the
# reference. Judged on the car, not the plan's end: move blocking holds the
# plan's last move for 0.4 s, so at a corner entry the plan runs on straight,
# up to 1.4 m and 0.8 rad off the line, while the car turns in on it (lateral
# error <= 0.22 m). Past these the car is sliding or off line (a24 spins): the
# corridor stays straight, failing closed. Clean laps at a22 never reach them.
CAP_CONTINUATION_MAX_OFFSET = 0.5     # [m] car to the reference at its arc position
CAP_CONTINUATION_MAX_HEADING = 0.3    # [rad] car's velocity heading to the reference's
# The plan's period, learnt from the publish stamps: a solve that overruns the
# timer stretches it, and the next command then lands later than one period on.
PUBLISH_INTERVAL_SMOOTHING = 0.2    # EMA weight per published command
PUBLISH_INTERVAL_MAX = 4.0          # x mpc.dt: a stall must not predict seconds ahead
# [rad] Largest odometry sideslip handed to the ST model; beyond it the car is
# reversing or spinning.
MPC_BETA_LIMIT = 0.5


class DtTracker:
    """The interval the plant advanced between sensor stamps, clamped.

    Stamps come from the bridge's own sim clock, so consecutive deltas are the
    real hold period regardless of wall-clock load or real-time factor.
    """

    def __init__(self, nominal):
        self.nominal = nominal
        self.dt = nominal
        self._last_stamp = None
        self._history = []

    def update(self, stamp):
        if stamp is None:
            return
        if self._last_stamp is not None:
            self.dt = float(np.clip(stamp - self._last_stamp,
                                    DT_CLAMP_LOW * self.nominal,
                                    DT_CLAMP_HIGH * self.nominal))
            self._history.append(self.dt)
        self._last_stamp = stamp

    def summary(self):
        """(mean, p95, max) of the dt seen since the last reset."""
        if not self._history:
            return (self.nominal,) * 3
        h = np.asarray(self._history)
        return float(h.mean()), float(np.percentile(h, 95)), float(h.max())

    def reset_stats(self):
        self._history = []


class SteeringEstimator:
    """Where the judged servo has put the wheel, for the MPC's sixth state.

    Nothing measures the wheel angle, so it is predicted: the plant's own
    servo -- `pid_steer` then `steering_constraint`, evaluated once per 10 ms
    physics step with a hard clip, since this simulates the plant rather than
    being differentiated by anyone -- is replayed over the commands the driver
    published, each from the plant time it latched. A command does not act at
    the scan it was planned from: it reaches the plant a solve later. Replaying
    by latch time is what keeps the estimate on the same instant as the pose it
    is paired with, however stale that pose is.

    Odometry's yaw rate leashes the prediction against drift. Inverting the
    kinematic yaw rate gives delta_meas = atan(L * psi_dot / (v cos beta)),
    well conditioned only at speed, and the estimate is pulled toward it with a
    ~0.5 s time constant. The predictor is primary; a stronger pull would feed
    yaw-rate noise and the tyres' ~2 % understeer into the MPC's initial state.
    """

    def __init__(self, params=None):
        p = params if params is not None else F1TENTH_VEHICLE_PARAMETERS
        self._params = p
        self.s_min = float(p.s_min)
        self.s_max = float(p.s_max)
        self.sv_max = float(p.sv_max)
        # action.py:47-48 -- the bridge passes no steer_kp, so the plant derives it.
        self.kp = servo_kp(p)
        self.wheelbase = float(p.lf + p.lr)

        self.delta_hat = 0.0      # [rad] the car starts on the grid, wheel centred
        self.t = None             # [s] plant time delta_hat refers to
        self._cmd = 0.0           # command latched at self.t
        self._pending = []        # (t_latch, cmd) not yet reached, time-ordered
        self._residuals = []      # |delta_hat - delta_meas| before each correction
        self.reseeds = 0          # sim resets seen; others key their own memory off this

    def on_publish(self, delta_cmd, t_latch=None):
        """A command went on the wire and latches at plant time `t_latch`.

        Clipped to the lock exactly as gym_bridge.py:595 does. A command with
        no time, or one whose time the estimate has already passed, acts from
        the estimate's own instant: the past cannot be replayed differently.
        """
        cmd = float(np.clip(delta_cmd, self.s_min, self.s_max))
        if self.t is None or t_latch is None or t_latch <= self.t + LATCH_TIME_TOL:
            # The newest command wins: anything published earlier is moot.
            self._pending = []
            self._cmd = cmd
            return
        t_latch = float(t_latch)
        # A newer command supersedes older ones due at or after it.
        self._pending = [entry for entry in self._pending if entry[0] < t_latch]
        self._pending.append((t_latch, cmd))

    def predict_to(self, t):
        """Advance the estimate to plant time `t`. Returns the interval advanced."""
        if t is None:
            return 0.0
        if self.t is not None and t < self.t - SIM_RESET_BACKSTEP:
            self._reseed()
        if self.t is None:
            self.t = float(t)
            return 0.0
        start = self.t
        while self.t < t - LATCH_TIME_TOL:
            self._latch_due()
            h = min(PLANT_STEP_DT, t - self.t)
            self.delta_hat = self._servo_step(self.delta_hat, self._cmd, h)
            self.t += h
        # Resynchronise on the caller's clock so float sums never drift.
        self.t = max(self.t, float(t))
        self._latch_due()
        return self.t - start

    def correct(self, yaw_rate, v, beta, dt):
        """Pull the estimate toward the wheel angle the yaw rate implies."""
        if v < STEER_CORRECT_MIN_SPEED or dt <= 0.0:
            return
        delta_meas = math.atan(self.wheelbase * yaw_rate / (v * math.cos(beta)))
        self._residuals.append(abs(self.delta_hat - delta_meas))
        gain = 1.0 - math.exp(-dt / STEER_CORRECT_TAU)
        self.delta_hat = float(np.clip(
            self.delta_hat + gain * (delta_meas - self.delta_hat),
            self.s_min, self.s_max))

    def residual_p95(self):
        """p95 of |delta_hat - delta_meas| since the last reset; 0 if none."""
        if not self._residuals:
            return 0.0
        return float(np.percentile(self._residuals, 95))

    def reset_stats(self):
        self._residuals = []

    def _reseed(self):
        """The sim was reset: the wheel is centred and old commands are void."""
        self.delta_hat = 0.0
        self.t = None
        self._cmd = 0.0
        self._pending = []
        self.reseeds += 1

    def _latch_due(self):
        while self._pending and self._pending[0][0] <= self.t + LATCH_TIME_TOL:
            self._cmd = self._pending.pop(0)[1]

    def _servo_step(self, delta, cmd, h):
        """utils.py pid_steer + steering_constraint, one step of length h."""
        return delta + plant_servo_rate(self._params, self.kp, delta, cmd) * h


class LatchClock:
    """Plant time at which a command published now will latch, live.

    The node's sim clock is updated by a /clock callback on the same
    single-threaded executor as the control timer, so it stands still while
    the MPC solves -- but the bridge free-runs and keeps stepping. A command
    therefore latches about `solve_wall * rtf` of plant time after the clock
    reading the node holds. The real-time factor is learnt from how far sim
    and wall time move between control steps. Transport lag is not modelled,
    so this is still a slight lower bound; the live residual log is the check.
    """

    def __init__(self):
        self.rtf = None
        self._last = None

    def update(self, t_sim, t_wall):
        """Feed one (sim, wall) clock pair per control step."""
        if self._last is not None:
            d_sim = t_sim - self._last[0]
            d_wall = t_wall - self._last[1]
            if d_sim > 0.0 and d_wall > 0.0:
                ratio = d_sim / d_wall
                self.rtf = (ratio if self.rtf is None
                            else self.rtf + RTF_SMOOTHING * (ratio - self.rtf))
        self._last = (t_sim, t_wall)

    def latch_time(self, t_sim_now, solve_wall_s):
        """[s] Plant time a command published after `solve_wall_s` will latch."""
        rtf = self.rtf if self.rtf is not None else 1.0
        return t_sim_now + solve_wall_s * rtf


def _share_dir():
    """The installed package's share directory, or None outside a ROS install."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('team_driver')
    except Exception:
        return None


def _path_length(path: np.ndarray) -> float:
    """[m] Length of an open polyline; 0 for one point."""
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


@functools.lru_cache(maxsize=4)
def _built_raceline(centerline_path, margin):
    """(xy, build seconds) of the raceline for a centreline file and margin.

    Built once per process: the node builds it once anyway, and the harness
    and the tests construct many cores from the same map.

    Raises if the weighted solve fell back to the uniform line: the sectors and
    `a_lat_max` are tuned to the reweighted geometry, and the caller's cached
    CSV is that line.
    """
    yaml_path = raceline.find_map_yaml(centerline_path)
    if yaml_path is None:
        raise FileNotFoundError(f'no map yaml beside {centerline_path}')
    line = raceline.build_raceline(lap_floor.load_path(centerline_path),
                                   raceline.load_occupancy(yaml_path), margin=margin)
    if not line.reweighted:
        raise RuntimeError('the weighted solve fell back to the uniform line, '
                           'which the tuned speed profile does not fit')
    line.xy.setflags(write=False)
    return line.xy, line.build_seconds


@functools.lru_cache(maxsize=4)
def _offline_min_time_line(csv_path, centerline_path, required):
    """xy of the min-time line cached in `csv_path`, checked against the map.

    Raises unless the CSV says it is the `raceline.py --refine` line and every
    point keeps `required` from the sealed walls of the map beside the centreline:
    shipped data is only trusted as far as the map it is driven on agrees.
    """
    method = raceline.cached_method(csv_path)
    if method != raceline.MIN_TIME_METHOD:
        raise ValueError(f'{csv_path} holds the {method} line')
    xy, _ = raceline.read_cached(csv_path)
    yaml_path = raceline.find_map_yaml(centerline_path)
    if yaml_path is None:
        raise FileNotFoundError(f'no map yaml beside {centerline_path}')
    sealed = raceline.seal_cones(raceline.load_occupancy(yaml_path))
    closest = float(raceline.wall_clearance(sealed, xy).min())
    if closest < required - raceline.CLEARANCE_EPS:
        raise ValueError(f'clearance {closest:.3f} m on the map, {required:.3f} m required')
    xy.setflags(write=False)
    return xy


class DriverCore:
    """The algorithm: reference path, MPC, and the reactive safety layer."""

    def __init__(self,
                 cfg: DriverConfig = None,
                 mpc_controller: MPCController = None,
                 logger=None):
        self.cfg = cfg if cfg is not None else DriverConfig()
        self.log = logger if logger is not None else _NULL_LOG

        self.mpc_controller = (mpc_controller if mpc_controller is not None
                               else MPCController(vehicle_model=VehicleModel()))
        self.mpc_dt = self.mpc_controller.dt
        self._dt = DtTracker(self.mpc_dt)
        self._planned_scan_t = None
        self._last_planned = DriveCommand(0.0, 0.0, 0.0)
        vehicle = getattr(self.mpc_controller, 'vehicle_model', None)
        params = vehicle.params if vehicle is not None else F1TENTH_VEHICLE_PARAMETERS
        self.steering = SteeringEstimator(params)
        # The plant's speed loop, inverted to send a speed for an acceleration,
        # and the whole plant, replayed to where the next command will land.
        self.speed_loop = SpeedLoop.of(params)
        self.latch_predictor = LatchPredictor(params)
        self._half_width = float(params.width) / 2.0
        self._overhang = front_overhang(params)
        self._plant_a_max = float(params.a_max)
        # The plant at the latch of the command being planned; None until the
        # first pose, when the measured speed stands in.
        self._latch = None
        # The MPC's arc position and planned path for this cycle's solve; None
        # when the cycle did not solve (reactive), so no cap reads a stale one.
        self._solved_arc = None
        self._planned_path = None

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]
        # Sideslip and yaw rate arrive free on /ego_racecar/odom. Both leash the
        # steering estimate, and the ST model starts its predictions from them.
        self.beta = 0.0           # [rad]
        self.yaw_rate = 0.0       # [rad/s]
        self.t_odom = None        # [s] stamp the pose was measured at

        # Latest scan, as the three fields preprocess() needs.
        self._scan_ranges = None
        self._scan_angle_min = 0.0
        self._scan_angle_inc = 0.0
        self.t_scan = None        # [s] stamp the scan was measured at

        self.reference_path = None
        self.reference_speeds = None
        self._reference = None    # ClosedPath over both, for the speed caps

        # What actually went on the wire last, and when.
        self._last_cmd = DriveCommand(0.0, 0.0, 0.0)
        self._last_publish_t = None
        self._publish_interval = None     # [s] smoothed latch-to-latch interval

        # The conditioner's own memory of its last steering output: the base the
        # slew limit steps from. None until the first command, which starts from
        # the wheel estimate instead. Speed needs no memory: each setpoint is
        # inverted afresh from the predicted speed.
        self._cond_steer = None
        self._seen_reseeds = 0
        self._cond_steps = 0
        self._slew_binds = 0
        self._emergency_binds = 0
        self._coast_count = 0
        self._coasting = False
        self._diag_counter = -1   # the first solve logs

        self._load_reference_path()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def set_odom(self, x, y, yaw, v, beta=0.0, yaw_rate=0.0, t_sim=None):
        """Latest pose and velocity, already unpacked from whatever carried it."""
        self.position = (x, y)
        self.yaw = yaw
        self.speed = v
        self.beta = beta
        self.yaw_rate = yaw_rate
        self.t_odom = t_sim

    def set_scan(self, ranges, angle_min, angle_inc, t_sim=None):
        """Latest LiDAR scan, as the raw range array plus its angular frame."""
        self._scan_ranges = ranges
        self._scan_angle_min = angle_min
        self._scan_angle_inc = angle_inc
        if t_sim != self.t_scan:
            self._dt.update(t_sim)
        self.t_scan = t_sim

    @property
    def dt_measured(self):
        """[s] Hold interval between the last two distinct scan stamps."""
        return self._dt.dt

    def dt_summary(self):
        """(mean, p95, max) of dt_measured; clears the window."""
        summary = self._dt.summary()
        self._dt.reset_stats()
        return summary

    @property
    def delta_hat(self):
        """[rad] Estimated wheel angle at the last planned pose's stamp."""
        return self.steering.delta_hat

    def steering_residual_summary(self):
        """p95 of |delta_hat - delta_meas| (yaw-rate inversion); clears the window."""
        p95 = self.steering.residual_p95()
        self.steering.reset_stats()
        return p95

    # ------------------------------------------------------------------
    # Reference path loading
    # ------------------------------------------------------------------
    def _centerline_path(self):
        """icra26_centerline.csv: the share directory's maps/ first, then the repo's.

        The package is installed under race_ws/install/... but the map ships at
        the repo root, so the obvious 'share directory' guess does not exist.
        """
        share = _share_dir()
        if share is not None:
            candidate = os.path.join(share, 'maps', lap_floor.CENTERLINE_NAME)
            if os.path.exists(candidate):
                return candidate
        try:
            return lap_floor.default_centerline_path()
        except FileNotFoundError:
            return None

    def _cached_raceline_candidates(self):
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [os.path.join(os.path.dirname(here), 'config', CACHED_RACELINE)]
        share = _share_dir()
        if share is not None:
            candidates.append(os.path.join(share, 'config', CACHED_RACELINE))
        return candidates

    def _raceline_reference(self, centerline_path, centerline_xy):
        """The offline min-time line, else the runtime raceline, else the cached
        one, else the centreline.

        The min-time line (`raceline.py --refine --write`) takes ~60 s to compute,
        past the startup budget, so it ships as the cached CSV and is driven first
        once it checks out against the map. Otherwise the minimum-curvature line is
        built from the map at startup, and the CSV is used only when that build
        fails, and only if it was built with at least the margin asked for.
        """
        cached = next((p for p in self._cached_raceline_candidates() if os.path.exists(p)), None)
        rejected = None
        if cached is not None:
            try:
                xy = _offline_min_time_line(
                    cached, centerline_path, raceline.HALF_WIDTH + float(self.cfg.raceline_margin))
                self.log.info(f'raceline: offline min-time line {cached}, {len(xy)} points')
                return xy.copy()
            except Exception as e:
                rejected = e
                self.log.warn(f'raceline: not driving the offline min-time line ({e}); '
                              'building from the map; the sector profile was tuned on the '
                              'min-time line, so expect a slower, untested lap')
        try:
            xy, seconds = _built_raceline(centerline_path, float(self.cfg.raceline_margin))
            self.log.info(f'raceline: {len(xy)} points, built in {seconds:.2f} s')
            return xy.copy()
        except Exception as e:
            reason = f'build failed: {e}'

        if cached is None:
            self.log.error(f'raceline: {reason}; no cached raceline found, '
                           'using the centreline')
            return centerline_xy
        try:
            xy, margin = raceline.read_cached(cached)
        except Exception as e:
            self.log.error(f'raceline: {reason}; cached raceline unusable ({e}), '
                           'using the centreline')
            return centerline_xy
        if margin < self.cfg.raceline_margin:
            self.log.error(f'raceline: {reason}; the cached raceline keeps {margin} m of '
                           f'margin, less than the {self.cfg.raceline_margin} m asked for, '
                           'using the centreline')
            return centerline_xy
        if rejected is not None and raceline.cached_method(cached) == raceline.MIN_TIME_METHOD:
            self.log.error(f'raceline: {reason}; the cached raceline failed its map check '
                           f'({rejected}), using the centreline')
            return centerline_xy
        self.log.warn(f'raceline: {reason}; using the cached raceline {cached} '
                      f'(margin {margin} m)')
        return xy

    def _load_reference_path(self):
        """Load centerline as reference path."""
        try:
            centerline_path = self._centerline_path()

            if centerline_path is None:
                self.log.error(
                    'icra26_centerline.csv not found in the share directory or '
                    'above the package. MPC will NOT be used; '
                    'falling back to the reactive gap-following controller.')
                self.reference_path = None
                return

            centerline_xy = lap_floor.load_path(centerline_path)
            self.log.info(
                f'Loaded {len(centerline_xy)} centerline points from {centerline_path}')

            path = centerline_xy
            if self.cfg.reference == 'raceline':
                path = self._raceline_reference(centerline_path, centerline_xy)
            on_raceline = path is not centerline_xy
            a_lat_max, sectors = self._slip_budget_for(path, on_raceline=on_raceline)

            # Curvature-limited speeds, not a flat 3 m/s. The track is ~0.5 m
            # wide and the centerline peaks well above a curvature that 3 m/s
            # can hold, so a constant reference drives straight into the
            # outside wall at the first real corner.
            speeds = speed_profile(
                path=path, v_max=self.cfg.v_max,
                a_lat_max=a_lat_max, lock_margin=self.cfg.lock_margin,
                a_accel_max=self.cfg.a_accel_max, a_brake_max=self.cfg.a_brake_max,
                sector_scale=sectors,
                gg_brake=self.cfg.gg_brake, gg_floor=self.cfg.gg_floor)
            self.log.info(f'Speed profile: {speeds.min():.2f}-{speeds.max():.2f} m/s')
            self.set_reference(path, speeds)

        except Exception as e:
            self.log.error(f'Error loading reference path: {e}')
            self.reference_path = None

    def _slip_budget_for(self, path, on_raceline):
        """(a_lat_max, sectors) for the profile of `path`.

        The sectors are arc positions on the raceline. On the centreline they would
        land on the wrong corners, so there the whole lap gets the lowest budget
        any sector allows instead.
        """
        sectors = sector_triples(self.cfg.sector_scale)
        if not sectors:
            return self.cfg.a_lat_max, ()
        if not on_raceline:
            a_lat = self.cfg.a_lat_max * min(scale for _, _, scale in sectors)
            self.log.warn(f'sector_scale is tuned to the raceline; on the centreline the '
                          f'slip budget is a uniform {a_lat:.2f} m/s^2')
            return a_lat, ()
        for sector in sectors:
            lowered = int(np.count_nonzero(slip_budget(path, 1.0, [sector]) < 1.0))
            if lowered == 0:
                self.log.warn(f'sector_scale {sector} lowers no waypoint of the '
                              f'{len(path)}-point raceline')
        return self.cfg.a_lat_max, sectors

    def set_reference(self, path: np.ndarray, speeds: np.ndarray) -> None:
        """Track `path` at `speeds`: the MPC's reference, and the speed caps' lookup.

        The lookup is built first, so a mismatched pair changes nothing.
        """
        lookup = ClosedPath(path, speeds)
        self.reference_path = path
        self.reference_speeds = speeds
        self._reference = lookup
        self.mpc_controller.set_reference_path(path, speeds, self.cfg.a_brake_max)
        if not hasattr(self.mpc_controller, 'arc_position'):
            self.log.warn(f'{type(self.mpc_controller).__name__} reports no arc_position: '
                          'the profile cap is off and the emergency cap looks straight ahead')

    # ------------------------------------------------------------------
    # One control cycle
    # ------------------------------------------------------------------
    def step(self, t_sim=None) -> DriveCommand:
        """Plan one command from the latest sensor data."""
        if self._scan_ranges is None:
            # No scan has arrived yet; the fixed-rate node can beat the first
            # one. Driving blind is worse than standing still.
            return self._coast(self.dt_measured, 'no scan yet')

        # A fixed-rate caller can beat the next scan; replaying the last command
        # is better than solving the same scan twice.
        if self.t_scan is not None and self.t_scan == self._planned_scan_t:
            return self._last_planned
        self._planned_scan_t = self.t_scan

        # Bring the wheel estimate to the pose's instant -- odom's stamp, which
        # can trail the scan by a tick -- so the two describe the same moment
        # when they reach the MPC together, and the yaw rate it is corrected
        # with is from that moment too.
        t_pose = self.t_odom if self.t_odom is not None else self.t_scan
        advanced = self.steering.predict_to(t_pose)
        self.steering.correct(self.yaw_rate, self.speed, self.beta, advanced)
        if self.steering.reseeds != self._seen_reseeds:
            # The sim was reset: the car is back on the grid with the wheel
            # centred, so the old setpoints are not where any limit steps from.
            self._seen_reseeds = self.steering.reseeds
            self._cond_steer = None
            self._coasting = False
            self.latch_predictor.reset()
            self._last_publish_t = None
            self._publish_interval = None
            # The MPC's arc tracking and plan belong to the old episode too.
            reset_tracking = getattr(self.mpc_controller, 'reset_tracking', None)
            if callable(reset_tracking):
                reset_tracking()
        self._latch = self._predict_latch(t_pose)

        ranges, angles = self.preprocess()
        planned = self.plan(ranges, angles)
        dt = self.dt_measured
        if planned is None:
            self._last_planned = self._coast(dt, 'no usable plan')
        else:
            self._last_planned = self._condition(planned.steering, planned.accel, dt,
                                                 speed_cap=planned.speed_cap,
                                                 profile_speed=planned.profile_speed)
        return self._last_planned

    def _measured_state(self):
        """(x, y, yaw, v, beta, wheel, yaw_rate) as measured, made safe to replay.

        Odometry's beta is atan2(vy, vx) beside a speed that is a magnitude,
        so reversing reads as beta ~ pi. Past the model's slip bound it is that,
        or a spin, not a slip the tyre model describes.
        """
        def finite(value):
            return value if math.isfinite(value) else 0.0
        beta = finite(self.beta)
        return (self.position[0], self.position[1], self.yaw, finite(self.speed),
                beta if abs(beta) <= MPC_BETA_LIMIT else 0.0,
                self.steering.delta_hat, finite(self.yaw_rate))

    def _predict_latch(self, t_pose) -> Optional[Prediction]:
        """The plant when the command planned now latches; None with no pose.

        The pose is a dead time old. Commands go out on a fixed timer, so this
        one latches a period after the last one did: the control period, or
        longer while solves overrun it.
        """
        if self.position is None:
            return None
        state = self._measured_state()
        if not self.cfg.predict_latch:
            return Prediction(state, 0.0)
        period = max(self.mpc_dt, self._publish_interval or 0.0)
        t_latch = (self._last_publish_t + period
                   if self._last_publish_t is not None else None)
        return self.latch_predictor.predict(state, t_pose, t_latch)

    def _latch_or_measured(self) -> Prediction:
        """The plant at the latch, or as measured when no latch is predicted."""
        return self._latch if self._latch is not None else Prediction(self._measured_state(), 0.0)

    def _lidar_xy(self):
        """(x, y) of the lidar in the map, from the measured pose."""
        return (self.position[0] + LIDAR_FORWARD * math.cos(self.yaw),
                self.position[1] + LIDAR_FORWARD * math.sin(self.yaw))

    def _latch_speed(self) -> float:
        """[m/s] The speed the next command acts from; the measured one if unknown."""
        v = self._latch.speed if self._latch is not None else self.speed
        return v if math.isfinite(v) else 0.0

    def _latch_travel(self) -> float:
        """[m] Distance covered between the pose's stamp and the next latch."""
        return self._latch.travelled if self._latch is not None else 0.0

    def conditioning_stats(self) -> ConditioningStats:
        """Slew-limiter, coast and emergency-cap counts since construction."""
        return ConditioningStats(self._cond_steps, self._slew_binds, self._coast_count,
                                 self._emergency_binds)

    # ------------------------------------------------------------------
    # Output conditioning: the single path to the wire
    # ------------------------------------------------------------------
    def _previous_steer(self):
        """[rad] The steering the next command's slew limit steps from."""
        steer = self._cond_steer if self._cond_steer is not None else self.steering.delta_hat
        # A NaN here would be stored as the new memory and never wash out.
        return steer if math.isfinite(steer) else 0.0

    def _issue(self, steer, speed, prev_steer, dt) -> DriveCommand:
        self._cond_steer = steer
        rate = (steer - prev_steer) / dt if dt > 0.0 else 0.0
        return DriveCommand(steer, speed, rate)

    def _setpoint(self, accel: float) -> float:
        """[m/s] The /drive speed that makes the plant accelerate at `accel`."""
        return max(self.speed_loop.command(self._latch_speed(), accel, self.mpc_dt), 0.0)

    def _condition(self, raw_steer: float, accel: float, dt: float, *,
                   speed_cap: float = math.inf,
                   profile_speed: float = math.inf) -> DriveCommand:
        """Hold a raw command to what the servo and the speed loop can follow.

        The steering step is limited to exactly `sv_max * dt`: at that rate the
        limiter only removes motion the wheel could not have followed, so it
        adds no lag. Tighter would remove motion the wheel could have followed.

        Speed is handled as the acceleration it is. /drive speed feeds the
        plant's proportional loop, so the setpoint is inverted from that loop
        (`SpeedLoop.command`) from the speed the car will have when it latches.
        The MPC's demand is clamped to [-a_brake_max, accel_limit]; then the
        hold may end no faster than `profile_speed`, then no faster than
        `speed_cap`. Those two may brake past the clamp, to the plant's limit:
        the emergency cap is a safety bound and must not be delayed by it.
        A NaN cap would be skipped by every comparison below -- failing open --
        so it coasts instead; +inf is "no cap".
        """
        if not (math.isfinite(raw_steer) and math.isfinite(accel)):
            return self._coast(dt, 'non-finite command')
        if math.isnan(speed_cap) or math.isnan(profile_speed):
            return self._coast(dt, 'non-finite speed cap')
        self._coasting = False

        prev_steer = self._previous_steer()
        reach = self.steering.sv_max * dt
        target = float(np.clip(raw_steer, -self.cfg.steer_limit, self.cfg.steer_limit))
        steer = float(np.clip(target, prev_steer - reach, prev_steer + reach))
        steer = float(np.clip(steer, -self.cfg.steer_limit, self.cfg.steer_limit))

        v = self._latch_speed()
        a = min(max(accel, -self.cfg.a_brake_max), self.cfg.accel_limit)
        a = min(a, (profile_speed - v) / self.mpc_dt)
        a_cap = (speed_cap - v) / self.mpc_dt
        if a_cap < a:
            a = a_cap
            self._emergency_binds += 1
        a = max(a, -self._plant_a_max)

        self._cond_steps += 1
        if steer != target:
            self._slew_binds += 1
        return self._issue(steer, self._setpoint(a), prev_steer, dt)

    def _coast(self, dt, why='') -> DriveCommand:
        """Degrade smoothly when there is no usable plan: never snap.

        Steering unwinds toward straight at `coast_steer_rate` and speed is shed
        at `a_emergency`, so a solver dropout at speed does not coast into a
        wall. The old fallbacks jumped the wheel to 0.0 from wherever it was, at
        the moment the controller had admitted it did not know what was
        happening.
        """
        prev_steer = self._previous_steer()
        decay = self.cfg.coast_steer_rate * dt
        steer = prev_steer - math.copysign(min(abs(prev_steer), decay), prev_steer)

        self._coast_count += 1
        if not self._coasting:
            self._coasting = True
            self.log.warn(f'Coasting ({why or "no usable command"}); '
                          f'{self._coast_count} coast steps so far')
        return self._issue(steer, self._setpoint(-self.cfg.a_emergency), prev_steer, dt)

    def _speed_cap(self, ranges: np.ndarray, angles: np.ndarray, bearing: float = 0.0,
                   travelled: float = 0.0) -> float:
        """[m/s] Emergency bound: the speed the body can still stop from at `a_emergency`.

        Clearance is looked for along `bearing` (the planned path's), measured
        from the lidar, less the body's overhang ahead of it, less the distance
        covered since the scan. A braking-distance bound, so it has no steps.
        """
        clearance = path_clearance(ranges, angles, bearing, self._half_width, self.cfg.max_range)
        return self._braking_bound(clearance - travelled - self._overhang)

    def _path_cap(self, ranges, angles, path) -> float:
        """[m/s] Emergency bound along the plan being executed.

        The corridor is the MPC's own predicted path, one car wide, from the
        latch pose -- so it bends with a corner the car is taking, and starts
        from where the car really points. A wall the plan runs into is in it;
        the outside wall of a bend the plan follows is not. The scan is from
        the measured pose, and is placed in the map from there.

        It fails closed: a plan with no extent or no finite points, or a scan
        that saw nothing because its beams read zero (NaN, or closer than the
        sensor sees), falls back to the straight corridor, which brakes for
        zero ranges.
        """
        v = self._latch_speed()
        fallback = self._latch_travel() + v * self.mpc_dt
        path = np.asarray(path, dtype=np.float64)
        extent = _path_length(path)
        if not (math.isfinite(self.yaw) and all(math.isfinite(c) for c in self.position)
                and np.all(np.isfinite(path)) and extent > 1e-6):
            return self._speed_cap(ranges, angles, 0.0, fallback)
        points = scan_points(ranges, angles, self._lidar_xy(), self.yaw, self.cfg.max_range)
        if len(points) == 0 and np.any(ranges <= 0.0):
            return self._speed_cap(ranges, angles, 0.0, fallback)
        front = LIDAR_FORWARD + self._overhang     # base_link to the bumper
        hold = v * self.mpc_dt
        needed = (v * v / (2.0 * self.cfg.a_emergency) + front + hold + self.cfg.stop_margin
                  + PATH_CAP_SLACK)
        path, hop = self._continued_plan(path, needed)
        clearance = clearance_along_path(points, path, self._half_width, needed + hop)
        if clearance > extent:      # past the plan: the hop back to the line is no travel
            clearance = max(clearance - hop, extent)
        if clearance >= needed:
            return self.cfg.v_max       # nothing within reach of a stop: no bound
        return self._braking_bound(clearance - front - hold)

    def _continued_plan(self, path: np.ndarray, length: float) -> tuple[np.ndarray, float]:
        """(`path` continued along the reference to `length` of travel, hop) if shorter.

        The plan starts at the MPC's arc position `s` for the latch pose, so the
        continuation picks the reference up at `s` plus the plan's extent,
        shifted by the car's own offset from the line. A plan that ran on
        straight (move blocking) ends off the line; `hop` [m] is how much longer
        than one spacing the segment back to it is -- arc along the corridor
        that is not travel. With `cap_follows_reference` off, no arc position,
        or a car off the line or heading away from it, the plan is returned as
        it is, hop 0, and `clearance_along_path` continues it straight.
        """
        if not self.cfg.cap_follows_reference:
            return path, 0.0
        s = self._arc_position()
        if s is None:
            return path, 0.0
        extent = _path_length(path)
        if extent >= length:
            return path, 0.0
        offset = self._offset_from_reference(s)
        if offset is None:
            return path, 0.0
        count = math.ceil((length - extent) / CAP_CONTINUATION_SPACING)
        tail = self._reference.points_at(
            s + extent + CAP_CONTINUATION_SPACING * np.arange(1, count + 1)) + offset
        hop = max(float(np.linalg.norm(tail[0] - path[-1])) - CAP_CONTINUATION_SPACING, 0.0)
        return np.vstack([path, tail]), hop

    def _offset_from_reference(self, s: float) -> Optional[np.ndarray]:
        """[m] The latch pose less the reference at `s`; None if the car is not following it.

        Following it: within CAP_CONTINUATION_MAX_OFFSET of the line, its
        velocity (yaw plus sideslip) within CAP_CONTINUATION_MAX_HEADING of
        the reference's direction there.
        """
        x, y, yaw, _v, beta, _wheel, _r = self._latch_or_measured().state
        anchor = self._reference.point_at(s)
        tangent = self._reference.point_at(s + CAP_CONTINUATION_SPACING) - anchor
        offset = np.array([x, y]) - anchor
        heading = yaw + beta - math.atan2(tangent[1], tangent[0])
        heading = (heading + math.pi) % (2.0 * math.pi) - math.pi
        if (np.linalg.norm(offset) > CAP_CONTINUATION_MAX_OFFSET
                or abs(heading) > CAP_CONTINUATION_MAX_HEADING):
            return None
        return offset

    def _braking_bound(self, reach) -> float:
        """[m/s] The speed the body stops from at `a_emergency` within `reach`."""
        v_brake = math.sqrt(2.0 * self.cfg.a_emergency * max(reach - self.cfg.stop_margin, 0.0))
        return float(np.clip(v_brake, self.cfg.crawl_speed, self.cfg.v_max))

    def _arc_position(self) -> Optional[float]:
        """[m] The MPC's arc position for this cycle's pose, or None if it did not solve."""
        if self._reference is None or self._solved_arc is None:
            return None
        return self._solved_arc if math.isfinite(self._solved_arc) else None

    def _reference_bearing(self) -> float:
        """[rad] Bearing, in the scan's frame, of the reference point ahead; 0 if unknown.

        The scan and the pose are the same instant, so the bearing is taken
        from that pose, at the lidar.
        """
        s = self._arc_position()
        if (s is None or self.position is None or not math.isfinite(self.yaw)
                or not all(math.isfinite(c) for c in self.position)):
            return 0.0
        target = self._reference.point_at(s + bearing_lookahead(self._latch_speed()))
        lx, ly = self._lidar_xy()
        bearing = math.atan2(target[1] - ly, target[0] - lx) - self.yaw
        return (bearing + math.pi) % (2.0 * math.pi) - math.pi

    def _profile_speed(self) -> float:
        """[m/s] The profile at the arc position where this command's hold ends.

        The MPC solved from the latch pose, so its arc position is already
        there; the hold is what is left to go.
        """
        s = self._arc_position()
        if s is None:
            return math.inf
        return self._reference.value_at(s + self._latch_speed() * self.mpc_dt)

    def note_published(self, cmd: DriveCommand, t_sim=None):
        """Record what actually went on the wire.

        `t_sim` is the plant time the command takes effect -- when it is
        published, not the stamp of the scan it was planned from. The steering
        estimate replays the servo from exactly that instant.
        """
        self._last_cmd = cmd
        last = self._last_publish_t
        if t_sim is not None and last is not None and t_sim > last:
            gap = min(t_sim - last, PUBLISH_INTERVAL_MAX * self.mpc_dt)
            smoothed = self._publish_interval
            self._publish_interval = (gap if smoothed is None else
                                      smoothed + PUBLISH_INTERVAL_SMOOTHING * (gap - smoothed))
        self._last_publish_t = t_sim
        self.steering.on_publish(cmd.steering_angle, t_sim)
        self.latch_predictor.on_publish(cmd.steering_angle, cmd.speed, t_sim)

    def preprocess(self):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(self._scan_ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.cfg.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.cfg.max_range)

        angles = self._scan_angle_min + np.arange(len(ranges)) * self._scan_angle_inc
        return ranges, angles

    def _mpc_state(self):
        """The MPC's initial state, at the length its model takes.

        [x, y, yaw, v, beta, wheel], plus the yaw rate for the ST model, at
        the instant the command will latch (or as measured, if prediction is
        off). Odometry carries the real sideslip and yaw rate at the CoG,
        which is what the ST model's beta and r are. The KS model gets
        beta = 0: it holds beta constant across the horizon, so the measured
        value would bend every prediction rather than inform it.
        """
        dynamic = getattr(self.mpc_controller, 'state_dim', 6) == 7
        x, y, yaw, v, beta, wheel, yaw_rate = self._latch_or_measured().state
        state = [x, y, yaw, v, beta if dynamic else 0.0, wheel]
        if dynamic:
            state.append(yaw_rate)
        return np.array(state)

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """MPC-based planning function.

        Uses Model Predictive Control to compute optimal steering and
        acceleration based on vehicle dynamics, reference path, and current
        state. Returns a `Plan` before conditioning, or None when there is
        nothing usable to plan from and the caller should coast.
        """
        self._solved_arc = None
        self._planned_path = None
        # 1. Restrict view to forward field of view [-100 deg, +100 deg] (~ -1.75 to +1.75 rad)
        fov_mask = (angles >= -1.75) & (angles <= 1.75)
        fov_ranges = ranges[fov_mask]
        fov_angles = angles[fov_mask]

        if len(fov_ranges) == 0:
            # No valid LiDAR data.
            return None

        # 2. Basic obstacle avoidance - keep existing safety logic for robustness
        # Locate nearest obstacle point inside FOV
        min_idx = np.argmin(fov_ranges)
        min_dist = fov_ranges[min_idx]

        # Create a safety bubble around nearest obstacle (radius ~0.35m)
        car_radius = 0.35
        bubble_rad = math.atan2(car_radius, max(min_dist, 0.1))

        proc_ranges = fov_ranges.copy()
        bubble_mask = np.abs(fov_angles - fov_angles[min_idx]) <= bubble_rad
        proc_ranges[bubble_mask] = 0.0

        # 3. Find the largest continuous gap beyond threshold distance
        gap_threshold = 1.5  # meters
        free_space = proc_ranges > gap_threshold

        if not np.any(free_space):
            # Fallback: steer towards furthest visible point if no clear gap
            best_idx = np.argmax(proc_ranges)
            target_angle = fov_angles[best_idx]
        else:
            # Detect contiguous gap ranges
            bounded = np.pad(free_space.astype(int), (1, 1), 'constant')
            diffs = np.diff(bounded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            # Choose the widest gap
            gap_lengths = ends - starts
            max_gap_idx = np.argmax(gap_lengths)

            start_i = starts[max_gap_idx]
            end_i = ends[max_gap_idx]

            # Target the center of the largest gap
            gap_angles = fov_angles[start_i:end_i]
            target_angle = (gap_angles[0] + gap_angles[-1]) / 2.0

        # 4. Reactive steering: steer at the middle of the widest gap. Used as
        #    the fallback if the MPC fails; the MPC keeps its own command memory.
        reactive_steering = float(np.clip(0.85 * target_angle,
                                          -self.cfg.steer_limit, self.cfg.steer_limit))

        # 5. Use MPC when we have both a path and a state. Without a reference
        #    path the MPC has nothing to track and would drive the car to the
        #    origin, so stay reactive in that case. Either way the speed is
        #    bounded by `_speed_cap`, a braking-distance bound with no steps.
        if self.position is not None and self.reference_path is not None:
            try:
                current_state = self._mpc_state()

                mpc_control, diagnostics = self.mpc_controller.get_control(
                    current_state)
                self._solved_arc = getattr(self.mpc_controller, 'arc_position', None)
                planner = getattr(self.mpc_controller, 'planned_path', None)
                self._planned_path = planner() if callable(planner) else None

                # /drive takes a SPEED, not an acceleration -- the simulator
                # closes that loop itself. _condition inverts it.
                steering = float(np.clip(mpc_control[0], -self.cfg.steer_limit, self.cfg.steer_limit))
                accel = float(mpc_control[1])

                self._diag_counter += 1
                if self._diag_counter % 50 == 0:  # Every ~2 s at 40 Hz
                    self.log.info(
                        f'MPC: steer={steering:.3f}, accel={accel:.2f} '
                        f'(v {self.speed:.2f}, at latch {self._latch_speed():.2f}), '
                        f'solve={diagnostics["solve_time_ms"]:.1f}ms')

            except Exception as e:
                self.log.warn(f'MPC failed: {e}')
                return None
        else:
            # No odometry or no reference path - use reactive
            steering = reactive_steering
            target = min(REACTIVE_SPEED, self.cfg.v_max)
            accel = (target - self._latch_speed()) / self.mpc_dt

        # The caps stop the MPC talking the car past the profile or into a
        # wall; _condition applies them as ceilings on where the hold ends, so
        # the clearance loses the hold's travel as well as the dead time's.
        if self._planned_path is not None:
            cap = self._path_cap(ranges, angles, self._planned_path)
        else:
            travelled = self._latch_travel() + self._latch_speed() * self.mpc_dt
            cap = self._speed_cap(ranges, angles, self._reference_bearing(), travelled)
        return Plan(steering, accel, self._profile_speed(), cap)
