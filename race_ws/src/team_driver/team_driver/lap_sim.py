#!/usr/bin/env python3
"""Closed-loop harness: the judged plant, sensors and timing, driving DriverCore.

    python3 lap_sim.py                          # 2 laps, judged timing
    python3 lap_sim.py --laps 10 --json r.json  # a scored-length run
    python3 lap_sim.py --stale-steps 0          # what Step 3's QoS fix is worth
    python3 lap_sim.py --plant instant          # ablation: no steering servo

WHY THE TIMING MODEL
    The judged run sets `use_sim_time_bridge`, which zeroes both bridge timers:
    the bridge steps physics and publishes scan+odom as fast as the CPU allows
    (RTF ~1.4). The driver solves one scan in ~30-40 ms of wall time, so its
    depth-10 subscription queues are always full and every scan it takes is ~10
    physics steps old. While it solves, the sim keeps running -- so each command
    reaches the plant `solve_wall * rtf` of sim time after its tick. Together
    that is ~150 ms of dead time, against an MPC that plans 200 ms ahead and a
    steering servo that needs 290 ms for a full reversal. A harness that applies
    commands instantly to the current state -- the old `lap_sim.py` -- cannot see
    any of it.

    Since Step 3 the node plans on a fixed timer, so ticks are `--control-period`
    (25 ms) apart: 3 and 2 physics steps alternately on the 10 ms grid. A solve
    that outlasts the period stretches it, as a real overrunning timer would.

WHAT IS EXACT, WHAT IS NOT
    Plant, servo, LiDAR, collision detection and the referee's 1 Hz collision
    debounce are the judge's own code or line-for-line ports (plant_sim,
    scan_sim). Collision RESPONSE is a restitution-0 approximation, and the
    queue model is a fixed staleness rather than a simulated DDS executor.
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import driver_params  # noqa: E402
import lap_floor  # noqa: E402
import plant_sim  # noqa: E402
import scan_sim  # noqa: E402
from driver_core import REFERENCES, DriverConfig, DriverCore  # noqa: E402
from mpc_controller import MPCController  # noqa: E402
from speed_profile import sector_triples  # noqa: E402
from vehicle_model import MODEL_TYPES, PLANT_STEP_DT, VehicleModel  # noqa: E402

PHYSICS_DT = PLANT_STEP_DT      # s, gym_bridge.py sim_timestep
# Scan and odom are both 9 physics steps old when the driver reads them: the
# depth-10 queue stays full, and its oldest entry is 9 behind the newest.
# Measured live with a /clock probe (run 20260930T091032Z, mode 9 of 1536).
DEFAULT_STALE_STEPS = 9
DEFAULT_RTF = 1.4               # sim seconds per wall second, measured live
# The node plans on a fixed timer (driver.py create_timer, mpc.dt). 0 restores
# the old free-running loop, where each command is held for its own solve time.
DEFAULT_CONTROL_PERIOD = 0.025  # s
# Live, consecutive scans the old scan-callback driver used were ~74 ms of sim
# apart, against ~32 ms for core.step here: 74 / (32 * 1.4) = 1.65. That fixes
# how long a solve takes in the container; since Step 3 the tick spacing is the
# control period, and this only sets the solve wait inside it.
DEFAULT_SOLVE_SCALE = 1.65
SCAN_CULL_REACH = 9.0           # m; exact below it, and the driver clips at 8 m

# maps/tracks.yaml: spawn behind the line, and the line itself.
SPAWN_POSE = (-1.68, -0.01, 0.0)
FINISH_LINE_X = 1.32
FINISH_LINE_Y = (-0.76, 0.89)

# The node's declared MPC parameters and model, read off its parameter table.
# DriverCore's own fallback builds horizon 10, so these are what the judge
# runs, with or without driver_params.yaml.
JUDGED_MPC = driver_params.mpc_kwargs(driver_params.DEFAULTS)
JUDGED_MODEL = driver_params.DEFAULTS['vehicle.model']
# [m] Spacing of the resampled reference the lateral-error meter measures to;
# the distance it reports is high by at most half of it.
LATERAL_ERROR_SPACING = 0.01
# What `--mpc KEY=VALUE` may override for one run (the Step 9 sweep). Horizon,
# block map, dt and maxiter are structure, held by the node's startup checks.
TUNABLE_MPC_KEYS = ('q_pos', 'q_yaw', 'q_vel', 'r_steer', 'r_accel',
                    'r_delta_steer', 'r_delta_accel', 'solve_budget_ms')

# Controller gates. The slew gate is the old harness's, kept so the two can be
# compared; the saturation gate is what Step 6 has to reach.
MAX_MEAN_CMD_SLEW = 0.012       # rad per control step
MAX_SAT_DUTY = 0.05             # fraction of physics steps rate-saturated
# Step 6: the wheel estimate handed to the MPC, against the plant's own wheel
# at the same instant. Past this the actuator state does net harm.
MAX_DELTA_EST_ERR = 0.02        # rad
# Step 7: the slew limiter sits at exactly what the servo can follow, so with
# an accurate actuator model it should almost never bind; frequent binding
# means the model is wrong or the estimate has drifted. Coasting is a fallback
# for a controller with nothing to say, so a clean run never needs it.
MAX_SLEW_BIND_RATE = 0.02       # fraction of commands the limiter shortened
# Step 4: the emergency cap is a safety net under the profile. If it binds
# often, the line or the profile is wrong, not the cap.
MAX_EMERGENCY_BIND_RATE = 0.01  # fraction of commands the emergency cap slowed
# Step 2's binding gate on the harness itself: today's controller must fail the
# way it fails when judged (7 collisions / 10 laps in the judged run).
GATE_MIN_COLLISIONS_PER_10 = 3
# Rate-saturation duty of the live command stream (run 20260930T091032Z replayed
# through the judge's servo). A faithful harness lands either side of it, so the
# gate asks for agreement with it rather than a one-sided threshold.
LIVE_SAT_DUTY = 0.207
GATE_SAT_TOLERANCE = 0.03
JUDGED_BASELINE = 'results/CalcuLOL__20260929T172052Z.json: 10 laps, 7 collisions'


@dataclass(frozen=True)
class HarnessConfig:
    laps: int = 2
    max_time: Optional[float] = None  # sim s; default scales with laps
    stale_steps: int = DEFAULT_STALE_STEPS
    rtf: float = DEFAULT_RTF
    control_period: float = DEFAULT_CONTROL_PERIOD
    fixed_solve_ms: Optional[float] = None
    solve_scale: float = DEFAULT_SOLVE_SCALE
    plant: str = 'st'
    seed: Optional[int] = None  # plant noise RNG; inert unless noise is on
    quiet: bool = False
    mpc_overrides: tuple[tuple[str, float], ...] = ()  # over JUDGED_MPC, in order
    model: str = JUDGED_MODEL
    dt_schedule: Optional[tuple[float, ...]] = None    # None: JUDGED_MPC's
    feedforward: bool = False
    driver_overrides: tuple[tuple[str, object], ...] = ()  # over DriverConfig()

    @property
    def time_limit(self):
        return self.max_time if self.max_time is not None else 30.0 + 90.0 * self.laps


def mpc_kwargs(cfg):
    """JUDGED_MPC with this run's overrides applied; JUDGED_MPC itself untouched."""
    kwargs = {**JUDGED_MPC, **dict(cfg.mpc_overrides)}
    kwargs['block_map'] = list(kwargs['block_map'])
    kwargs['dt_schedule'] = list(cfg.dt_schedule if cfg.dt_schedule is not None
                                 else kwargs['dt_schedule'])
    if cfg.feedforward:
        kwargs['feedforward'] = True
    return kwargs


def driver_config(cfg):
    """The driver's configuration for this run: DriverConfig() with the overrides."""
    return dataclasses.replace(DriverConfig(), **dict(cfg.driver_overrides))


_DRIVER_FIELDS = {f.name: f for f in dataclasses.fields(DriverConfig)}
BOOL_WORDS = {'true': True, '1': True, 'false': False, '0': False}


def parse_driver_override(text):
    """`KEY=VALUE` -> (key, value) for a DriverConfig field, typed by its default."""
    key, sep, raw = text.partition('=')
    if not sep:
        raise argparse.ArgumentTypeError(f'expected KEY=VALUE, got {text!r}')
    if key not in _DRIVER_FIELDS:
        raise argparse.ArgumentTypeError(
            f'{key!r} is not a DriverConfig field; choose from {", ".join(_DRIVER_FIELDS)}')
    default = _DRIVER_FIELDS[key].default
    if isinstance(default, bool):
        if raw.lower() not in BOOL_WORDS:
            raise argparse.ArgumentTypeError(f'{key}: {raw!r} is not true or false')
        return key, BOOL_WORDS[raw.lower()]
    if isinstance(default, str):
        if key == 'reference' and raw not in REFERENCES:
            raise argparse.ArgumentTypeError(f'reference must be one of {REFERENCES}')
        return key, raw
    if isinstance(default, tuple):
        # sector_scale: `s_start,s_end,scale,...`, the node's flat list.
        try:
            flat = tuple(float(part) for part in raw.split(','))
            sector_triples(flat)
        except ValueError as err:
            raise argparse.ArgumentTypeError(f'{key}: {err}') from None
        return key, flat
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{key}: {raw!r} is not a number') from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f'{key}: {value} must be finite')
    return key, value


def parse_schedule(text):
    """`0.025,0.025,0.05,...` -> a tuple of positive step lengths [s]."""
    try:
        steps = tuple(float(part) for part in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a list of numbers: {text!r}') from None
    if not all(math.isfinite(s) and s > 0.0 for s in steps):
        raise argparse.ArgumentTypeError(f'steps must be positive and finite: {text!r}')
    return steps


class LateralError:
    """Distance from a point to a closed reference polyline, and the value there.

    The line is resampled every LATERAL_ERROR_SPACING, closing segment
    included, and the nearest sample stands for the nearest point: high by at
    most half a spacing, and only for a point almost on the line. `values`
    (one per waypoint, e.g. the speed profile) are interpolated the same way.
    `arc` is the arc length there, from the first waypoint, along the line.
    """

    def __init__(self, path, spacing=LATERAL_ERROR_SPACING, values=None):
        p = np.asarray(path, dtype=np.float64)[:, :2]
        ends = np.roll(p, -1, axis=0)
        v = None if values is None else np.asarray(values, dtype=np.float64)
        v_ends = None if v is None else np.roll(v, -1)
        samples, sample_values, sample_arcs = [], [], []
        s0 = 0.0
        for i, (a, b) in enumerate(zip(p, ends, strict=True)):
            length = math.hypot(*(b - a))
            count = max(1, int(math.ceil(length / spacing)))
            f = np.arange(count) / count
            samples.append(a + np.outer(f, b - a))
            sample_arcs.append(s0 + f * length)
            s0 += length
            if v is not None:
                sample_values.append(v[i] + f * (v_ends[i] - v[i]))
        self._tree = cKDTree(np.vstack(samples))
        self._arcs = np.concatenate(sample_arcs)
        self._values = np.concatenate(sample_values) if v is not None else None

    def arc(self, xy):
        """[m] Arc length of the nearest point, from the first waypoint."""
        return float(self._arcs[self._tree.query(np.asarray(xy[:2], dtype=np.float64))[1]])

    def distance(self, xy):
        return float(self._tree.query(np.asarray(xy[:2], dtype=np.float64))[0])

    def nearest(self, xy):
        """(distance, value) at the nearest point; the value is None without values."""
        d, i = self._tree.query(np.asarray(xy[:2], dtype=np.float64))
        return float(d), (None if self._values is None else float(self._values[i]))


def parse_mpc_override(text):
    """`KEY=VALUE` -> (key, float), for a tunable key and a finite value >= 0."""
    key, sep, raw = text.partition('=')
    if not sep:
        raise argparse.ArgumentTypeError(f'expected KEY=VALUE, got {text!r}')
    if key not in TUNABLE_MPC_KEYS:
        raise argparse.ArgumentTypeError(
            f'{key!r} is not tunable here; choose from {", ".join(TUNABLE_MPC_KEYS)}')
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{key}: {raw!r} is not a number') from None
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(f'{key}: {value} must be finite and >= 0')
    if key == 'solve_budget_ms' and value == 0.0:
        raise argparse.ArgumentTypeError('solve_budget_ms: 0 aborts every solve; use 1e6 '
                                         'to lift it')
    return key, value


def json_record(cfg, result):
    """What `--json` writes: the config, the MPC weights actually used, the result."""
    return {'config': asdict(cfg), 'mpc': mpc_kwargs(cfg), 'result': result}


def hold_steps(solve_wall_s, rtf, dt=PHYSICS_DT):
    """Physics steps the sim advances while the driver solves one command.

    That is the wait before the command latches, whatever the tick spacing.
    """
    return max(1, int(round(solve_wall_s * rtf / dt)))


def hold_for(cfg, measured_solve_s):
    """Hold for one command: the fixed solve time, else the scaled measured one."""
    if cfg.fixed_solve_ms is not None:
        return hold_steps(cfg.fixed_solve_ms / 1e3, cfg.rtf)
    return hold_steps(measured_solve_s * cfg.solve_scale, cfg.rtf)


def cadence_steps(k, period, dt=PHYSICS_DT):
    """Physics steps in control period k, so no period's rounding accumulates.

    A 25 ms timer on the 10 ms grid is 3 and 2 steps alternately: tick k sits at
    round(k * period / dt) steps, and this is the gap to tick k + 1.
    """
    def tick(n):
        return math.floor(n * period / dt + 0.5 + 1e-9)
    return max(1, tick(k + 1) - tick(k))


def period_for(cfg, k, latency):
    """Physics steps from tick k to tick k + 1.

    The timer sets it. A solve that outlasts the period overruns it: the next
    tick fires the moment the callback returns, so the period stretches to the
    solve. With no control period the loop is free-running and the solve is it.
    """
    if cfg.control_period <= 0.0:
        return latency
    return max(cadence_steps(k, cfg.control_period), latency)


class SnapshotRing:
    """The last few published (state, stamp) pairs; the driver reads a stale one."""

    def __init__(self, stale_steps, initial):
        if stale_steps < 0:
            raise ValueError(f'stale_steps must be >= 0, got {stale_steps}')
        self._ring = deque([initial], maxlen=stale_steps + 1)

    def push(self, snapshot):
        self._ring.append(snapshot)

    def stale(self):
        return self._ring[0]


class FinishLine:
    """Counts laps the way f110_env does: the first crossing is the out-lap."""

    def __init__(self, x=FINISH_LINE_X, y_min=FINISH_LINE_Y[0], y_max=FINISH_LINE_Y[1]):
        self.x = float(x)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.crossings = 0
        self.lap_times = []
        self._last_cross_t = None
        # A backwards crossing must be undone by a forward one before it counts.
        self._debt = 0

    @property
    def laps(self):
        return len(self.lap_times)

    def update(self, prev_xy, xy, t):
        """Feed one step of base_link motion. Returns True if a lap was scored."""
        x0, x1 = float(prev_xy[0]), float(xy[0])
        if not self.y_min <= float(xy[1]) <= self.y_max:
            return False
        if x0 >= self.x > x1:
            self._debt += 1
            return False
        if not x0 < self.x <= x1:
            return False
        if self._debt:
            self._debt -= 1
            return False
        self.crossings += 1
        scored = self._last_cross_t is not None
        if scored:
            self.lap_times.append(t - self._last_cross_t)
        self._last_cross_t = t
        return scored


def count_reversals(steers):
    """Sign flips in the command's direction of change; ties are not flips."""
    signs = np.sign(np.diff(np.asarray(steers, dtype=float)))
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


class InstantSteerPlant(plant_sim.GymPlant):
    """Ablation: the judged plant with the steering servo removed.

    Steering snaps to the command every step, so `pid_steer` sees zero error.
    Comparing against the real plant isolates what the servo costs.
    """

    def step(self):
        self.state[plant_sim.DELTA] = np.float32(self.steer_cmd)
        return super().step()


PLANTS = {
    'st': plant_sim.GymPlant,               # the judged plant
    'kinematic': plant_sim.KinematicPlant,  # same servo, no tyre slip
    'instant': InstantSteerPlant,           # same tyres, no servo
}


def make_plant(name, seed=None):
    """Build the plant a `--plant` flag names."""
    if name not in PLANTS:
        raise ValueError(f'unknown plant {name!r}; choose from {sorted(PLANTS)}')
    return PLANTS[name](seed=seed)


def _repo_root():
    return os.path.dirname(os.path.dirname(lap_floor.default_centerline_path()))


def _feed_driver(core, scanner, state, t_sim, lr):
    """Hand the driver one snapshot, in the frames the bridge publishes."""
    x, y, yaw = plant_sim.base_link_pose_of(state, lr)
    vx, vy, wz = plant_sim.odom_twist_of(state)
    core.set_odom(x, y, yaw, math.hypot(vx, vy), beta=math.atan2(vy, vx),
                  yaw_rate=wz, t_sim=t_sim)
    ranges = scanner.scan((float(state[plant_sim.X]), float(state[plant_sim.Y]),
                           float(state[plant_sim.PSI])))
    core.set_scan(ranges, scanner.angle_min, scanner.angle_increment, t_sim=t_sim)


class _Log:
    """Per-step and per-command traces, reduced to metrics at the end."""

    def __init__(self):
        self.cmd_steer = []
        self.cmd_speed = []
        self.solve_ms = []
        self.holds = []        # physics steps from one tick to the next
        self.latencies = []    # of those, steps before the new command latches
        self.saturated = []
        self.delta = []
        self.track_err = []
        self.speed = []
        self.delta_est_err = []
        self.distance = 0.0
        self.lateral = None    # LateralError against the driver's reference
        self.lateral_err = []  # [m] per racing-lap physics step, base_link to the reference
        self.overshoot = []    # [m/s] per racing-lap physics step, plant speed - profile
        self.beta_peak = 0.0   # [rad] the largest |sideslip| the plant reached
        self.collisions = []   # collision_event per scored collision


def collision_event(*, t, xy, speed, beta, lap, lateral):
    """Where and how a scored collision happened; `s` is None without a reference."""
    return {'t': float(t), 'lap': int(lap),
            's': None if lateral is None else lateral.arc(xy),
            'v': float(speed), 'beta': abs(float(beta))}


def _stat(fn, values):
    """A summary statistic, or 0.0 for a run that logged nothing."""
    return float(fn(values)) if len(values) else 0.0


def _controller_stats(mpc):
    """Solves and fallbacks of the MPC, from its own counters."""
    solves = int(getattr(mpc, 'solve_count', 0))
    fallbacks = (int(getattr(mpc, 'budget_exceeded_count', 0))
                 + int(getattr(mpc, 'solver_failure_count', 0)))
    return {'solves': solves, 'fallbacks': fallbacks,
            'fallback_rate': fallbacks / solves if solves else 0.0}


def _summarise(cfg, log, plant, counter, line, wall_s, contact, conditioning,
               controller_stats=None):
    cmd = np.asarray(log.cmd_steer)
    cmd_speed = np.asarray(log.cmd_speed)
    delta = np.asarray(log.delta)
    err = np.asarray(log.track_err)
    holds = np.asarray(log.holds, dtype=float)
    latencies = np.asarray(log.latencies, dtype=float)
    reversals = count_reversals(cmd)
    return {
        'plant': cfg.plant,
        'seed': cfg.seed,
        'stale_steps': cfg.stale_steps,
        'rtf': cfg.rtf,
        'fixed_solve_ms': cfg.fixed_solve_ms,
        'solve_scale': cfg.solve_scale,
        'sim_time': plant.time,
        'wall_time': wall_s,
        'laps_requested': cfg.laps,
        'laps': line.laps,
        'lap_times': list(line.lap_times),
        'best_lap': min(line.lap_times) if line.lap_times else None,
        'distance_m': log.distance,
        'mean_speed': float(np.mean(log.speed)) if log.speed else 0.0,
        'collisions': counter.events,
        'collision_events': list(log.collisions),
        'contact_steps': counter.steps_in_contact,
        'penalty_s': counter.penalty,
        'disqualified': counter.disqualified,
        'contact_model_approximate': contact.approximate,
        'sat_duty': float(np.mean(log.saturated)) if log.saturated else 0.0,
        'cmd_slew_mean': float(np.mean(np.abs(np.diff(cmd)))) if cmd.size > 1 else 0.0,
        'cmd_slew_max': float(np.max(np.abs(np.diff(cmd)))) if cmd.size > 1 else 0.0,
        'actual_slew_mean': (float(np.mean(np.abs(np.diff(delta))))
                             if delta.size > 1 else 0.0),
        'track_err_mean': float(np.mean(err)) if err.size else 0.0,
        'track_err_p95': float(np.percentile(err, 95)) if err.size else 0.0,
        'reversals': reversals,
        'reversals_per_m': reversals / log.distance if log.distance > 0 else 0.0,
        'slew_bind_rate': conditioning.slew_bind_rate,
        'coast_count': conditioning.coast_count,
        'emergency_bind_rate': conditioning.emergency_bind_rate,
        # Racing laps only, like the lateral error: the plant against the profile
        # at its nearest point on the reference.
        'profile_overshoot_max': _stat(np.max, log.overshoot),
        'profile_overshoot_p99': _stat(lambda v: np.percentile(v, 99), log.overshoot),
        'cmd_speed_step_max': (float(np.max(np.abs(np.diff(cmd_speed))))
                               if cmd_speed.size > 1 else 0.0),
        'delta_est_err_mean': _stat(np.mean, log.delta_est_err),
        'delta_est_err_p95': _stat(lambda v: np.percentile(v, 95), log.delta_est_err),
        'delta_est_err_max': _stat(np.max, log.delta_est_err),
        'solve_ms_mean': _stat(np.mean, log.solve_ms),
        'solve_ms_p95': _stat(lambda v: np.percentile(v, 95), log.solve_ms),
        'solve_ms_max': _stat(np.max, log.solve_ms),
        'hold_steps_mean': _stat(np.mean, holds),
        'hold_steps_max': int(holds.max()) if holds.size else 0,
        'latency_steps_mean': _stat(np.mean, latencies),
        # Scan age plus the wait for the solve; the period is not dead time.
        'dead_time_ms_mean': (cfg.stale_steps + _stat(np.mean, latencies)) * PHYSICS_DT * 1e3,
        'model': cfg.model,
        # Chatter as a sign-reversal rate: fast legitimate steering through a
        # slalom moves the wheel a lot per step but reverses no more often.
        'reversals_per_s': reversals / plant.time if plant.time > 0 else 0.0,
        'lateral_err_mean': _stat(np.mean, log.lateral_err),
        'lateral_err_p95': _stat(lambda v: np.percentile(v, 95), log.lateral_err),
        'lateral_err_max': _stat(np.max, log.lateral_err),
        'beta_peak': log.beta_peak,
        **(controller_stats or {'solves': 0, 'fallbacks': 0, 'fallback_rate': 0.0}),
    }


def _advance(steps, plant, contact, counter, line, ring, log, prev_xy, cfg):
    """Run `steps` physics steps under the latched command.

    Returns (finished, base_link xy); finished means the laps are done or the
    car was disqualified, so the caller stops stepping.
    """
    for _ in range(steps):
        plant.step()
        scored = counter.update(plant.time, contact.resolve(plant.state))
        log.saturated.append(plant.is_rate_saturated)
        log.delta.append(float(plant.state[plant_sim.DELTA]))
        log.track_err.append(abs(plant.steer_cmd - log.delta[-1]))
        log.speed.append(float(plant.state[plant_sim.V]))
        beta = float(plant.state[plant_sim.BETA])
        log.beta_peak = max(log.beta_peak, abs(beta))
        xy = plant.base_link_pose[:2]
        if scored:
            log.collisions.append(collision_event(t=plant.time, xy=xy, speed=log.speed[-1],
                                                  beta=beta, lap=line.laps,
                                                  lateral=log.lateral))
        # Racing laps only: the spawn sits on the centreline, so the out-lap
        # measures joining the reference, not holding it.
        if log.lateral is not None and line.crossings:
            distance, profile = log.lateral.nearest(xy)
            log.lateral_err.append(distance)
            if profile is not None:
                log.overshoot.append(log.speed[-1] - profile)
        log.distance += math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])
        if line.update(prev_xy, xy, plant.time) and not cfg.quiet:
            print(f'  lap {line.laps}: {line.lap_times[-1]:.2f} s')
        prev_xy = xy
        ring.push((plant.state.copy(), plant.time))
        if counter.disqualified or line.laps >= cfg.laps:
            return True, prev_xy
    return False, prev_xy


def run(cfg: HarnessConfig):
    """Drive DriverCore against the judged plant until the laps, a DQ, or the limit."""
    track = scan_sim.load_map(os.path.join(_repo_root(), 'maps', 'icra26.yaml'))
    scanner = scan_sim.ScanSimulator(track, cull_reach=SCAN_CULL_REACH)
    contact = scan_sim.ContactResolver(track)
    plant = make_plant(cfg.plant, cfg.seed)
    plant.reset(SPAWN_POSE)
    lr = float(plant.params.lr)
    core = DriverCore(cfg=driver_config(cfg),
                      mpc_controller=MPCController(
                          vehicle_model=VehicleModel(model_type=cfg.model),
                          **mpc_kwargs(cfg)))

    counter = scan_sim.CollisionCounter()
    line = FinishLine()
    ring = SnapshotRing(cfg.stale_steps, (plant.state.copy(), plant.time))
    log = _Log()
    if core.reference_path is not None:
        speeds = core.reference_speeds
        log.lateral = LateralError(
            core.reference_path,
            values=speeds if speeds is not None and len(speeds) == len(core.reference_path)
            else None)
    prev_xy = plant.base_link_pose[:2]
    tick = 0
    next_report = 5.0
    wall0 = time.perf_counter()

    while (line.laps < cfg.laps and not counter.disqualified
           and plant.time < cfg.time_limit):
        state, t_snap = ring.stale()
        _feed_driver(core, scanner, state, t_snap, lr)
        # CPU time of this thread, not wall time: other processes on the host
        # must not lengthen the hold. The solver is single-threaded (small
        # SLSQP problem), so thread time is all of its work.
        t0 = time.thread_time()
        cmd = core.step(t_sim=t_snap)
        solve_s = time.thread_time() - t0
        log.solve_ms.append(solve_s * 1e3)
        # The estimate refers to the snapshot's instant, like the pose it rides with.
        log.delta_est_err.append(abs(core.delta_hat - float(state[plant_sim.DELTA])))

        latency = hold_for(cfg, solve_s)
        period = period_for(cfg, tick, latency)
        tick += 1
        log.holds.append(period)
        log.latencies.append(latency)
        log.cmd_steer.append(cmd.steering_angle)
        log.cmd_speed.append(cmd.speed)

        # The sim keeps running the previous command while this one solves; the
        # new one reaches the bridge only when the solve ends, and holds until
        # the next timer tick.
        finished, prev_xy = _advance(latency, plant, contact, counter, line, ring, log,
                                     prev_xy, cfg)
        plant.latch(cmd.steering_angle, cmd.speed)
        # Published now, a solve after the scan it was planned from.
        core.note_published(cmd, t_sim=plant.time)
        if not finished:
            # Whether this one finishes the run is read by the `while` above.
            _, prev_xy = _advance(period - latency, plant, contact, counter, line, ring,
                                  log, prev_xy, cfg)

        if not cfg.quiet and plant.time >= next_report:
            next_report += 5.0
            print(f'  t={plant.time:6.1f}s  laps={line.laps}  s={log.distance:6.1f}m  '
                  f'v={log.speed[-1]:.2f}  coll={counter.events}  '
                  f'sat={np.mean(log.saturated):.1%}')

    return _summarise(cfg, log, plant, counter, line, time.perf_counter() - wall0, contact,
                      core.conditioning_stats(), _controller_stats(core.mpc_controller))


def verdict(r):
    """Reasons the controller fails; empty means it passes."""
    reasons = []
    if r['laps'] < r['laps_requested']:
        reasons.append(f"completed {r['laps']}/{r['laps_requested']} laps")
    if r['collisions']:
        reasons.append(f"{r['collisions']} collisions")
    if r['sat_duty'] > MAX_SAT_DUTY:
        reasons.append(f"rate-saturated {r['sat_duty']:.1%} of steps > {MAX_SAT_DUTY:.0%}")
    if r['cmd_slew_mean'] > MAX_MEAN_CMD_SLEW:
        reasons.append(f"cmd slew {r['cmd_slew_mean']:.4f} > {MAX_MEAN_CMD_SLEW}")
    if r.get('delta_est_err_max', 0.0) >= MAX_DELTA_EST_ERR:
        reasons.append(f"wheel estimate off by {r['delta_est_err_max']:.4f} rad "
                       f">= {MAX_DELTA_EST_ERR}")
    if r.get('slew_bind_rate', 0.0) > MAX_SLEW_BIND_RATE:
        reasons.append(f"slew limiter bound on {r['slew_bind_rate']:.1%} of commands "
                       f"> {MAX_SLEW_BIND_RATE:.0%}")
    if r.get('coast_count', 0):
        reasons.append(f"{r['coast_count']} coast steps in a clean run")
    if r.get('emergency_bind_rate', 0.0) > MAX_EMERGENCY_BIND_RATE:
        reasons.append(f"emergency cap bound on {r['emergency_bind_rate']:.1%} of commands "
                       f"> {MAX_EMERGENCY_BIND_RATE:.0%}")
    return reasons


def collisions_per_10_laps(r):
    return r['collisions'] * 10.0 / max(r['laps'], 1)


def reproduces_judged_failure(r):
    """Whether a run on today's controller fails the way the judged run did."""
    return (collisions_per_10_laps(r) >= GATE_MIN_COLLISIONS_PER_10
            and abs(r['sat_duty'] - LIVE_SAT_DUTY) <= GATE_SAT_TOLERANCE)


def report(r):
    laps = ', '.join(f'{t:.2f}' for t in r['lap_times']) or '-'
    per10 = collisions_per_10_laps(r)
    print(f"\nplant={r['plant']}  stale={r['stale_steps']} steps  rtf={r['rtf']}  "
          f"solve={'measured x' + str(r['solve_scale']) if r['fixed_solve_ms'] is None else str(r['fixed_solve_ms']) + ' ms'}")
    print(f"  sim / wall        {r['sim_time']:.1f} s / {r['wall_time']:.1f} s")
    print(f"  laps              {r['laps']}/{r['laps_requested']}  [{laps}] s")
    print(f"  distance, speed   {r['distance_m']:.1f} m, mean {r['mean_speed']:.2f} m/s")
    print(f"  collisions        {r['collisions']} ({per10:.1f} per 10 laps; "
          f"{r['contact_steps']} contact steps){'  DQ' if r['disqualified'] else ''}")
    for e in r.get('collision_events', []):
        where = '-' if e['s'] is None else f"{e['s']:.1f}"
        print(f"    t={e['t']:.2f} s  lap {e['lap']}  s={where} m  v={e['v']:.2f} m/s  "
              f"|beta|={e['beta']:.3f} rad")
    print(f"  rate-sat duty     {r['sat_duty']:.1%}  (gate <= {MAX_SAT_DUTY:.0%})")
    print(f"  cmd slew          mean {r['cmd_slew_mean']:.4f}, max {r['cmd_slew_max']:.4f} "
          f"rad/cmd  (old gate < {MAX_MEAN_CMD_SLEW})")
    print(f"  actual slew       mean {r['actual_slew_mean']:.4f} rad/physics step")
    print(f"  |cmd - delta|     mean {r['track_err_mean']:.3f}, p95 {r['track_err_p95']:.3f} rad")
    print(f"  |delta_hat - delta| mean {r['delta_est_err_mean']:.4f}, "
          f"p95 {r['delta_est_err_p95']:.4f}, max {r['delta_est_err_max']:.4f} rad "
          f"(gate < {MAX_DELTA_EST_ERR})")
    print(f"  reversals         {r['reversals']} ({r['reversals_per_m']:.2f} per m)")
    print(f"  slew limiter      bound on {r['slew_bind_rate']:.2%} of commands "
          f"(gate <= {MAX_SLEW_BIND_RATE:.0%}); coast steps {r['coast_count']} (gate 0); "
          f"max speed step {r['cmd_speed_step_max']:.3f} m/s")
    print(f"  solve             mean {r['solve_ms_mean']:.1f}, p95 {r['solve_ms_p95']:.1f}, "
          f"max {r['solve_ms_max']:.1f} ms; fallbacks {r.get('fallbacks', 0)}/"
          f"{r.get('solves', 0)} ({r.get('fallback_rate', 0.0):.2%})")
    print(f"  model             {r.get('model', JUDGED_MODEL)}; peak |beta| "
          f"{r.get('beta_peak', 0.0):.3f} rad; reversals {r.get('reversals_per_s', 0.0):.2f}/s")
    print(f"  lateral error     mean {r.get('lateral_err_mean', 0.0):.3f}, p95 "
          f"{r.get('lateral_err_p95', 0.0):.3f}, max {r.get('lateral_err_max', 0.0):.3f} m "
          f"from the reference")
    print(f"  speed caps        emergency bound on "
          f"{r.get('emergency_bind_rate', 0.0):.2%} of commands "
          f"(gate <= {MAX_EMERGENCY_BIND_RATE:.0%}); plant over profile max "
          f"{r.get('profile_overshoot_max', 0.0):+.3f}, p99 "
          f"{r.get('profile_overshoot_p99', 0.0):+.3f} m/s")
    print(f"  dead time         {r['dead_time_ms_mean']:.0f} ms mean "
          f"(solve wait {r['latency_steps_mean']:.1f} steps; period "
          f"{r['hold_steps_mean']:.1f} steps, max {r['hold_steps_max']})")
    print(f"  judged baseline   {JUDGED_BASELINE}")
    fails = verdict(r)
    print(f"  VERDICT           {'FAIL -- ' + '; '.join(fails) if fails else 'PASS'}")
    print(f"  judged failure    {'REPRODUCED' if reproduces_judged_failure(r) else 'not reproduced'}"
          f"  (needs >= {GATE_MIN_COLLISIONS_PER_10} collisions/10 laps and "
          f"sat {LIVE_SAT_DUTY:.1%} +/- {GATE_SAT_TOLERANCE:.0%})")


def _parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--laps', type=int, default=2)
    ap.add_argument('--max-time', type=float, default=None, help='sim seconds')
    ap.add_argument('--stale-steps', type=int, default=DEFAULT_STALE_STEPS)
    ap.add_argument('--rtf', type=float, default=DEFAULT_RTF)
    ap.add_argument('--control-period', type=float, default=DEFAULT_CONTROL_PERIOD,
                    help='s between control ticks, as the node timer; 0 holds each '
                         'command for its own solve time (the pre-Step-3 loop)')
    ap.add_argument('--fixed-solve-ms', type=float, default=None,
                    help='use this solve time for every command (deterministic): the '
                         'wait before it latches, and the period when it overruns')
    ap.add_argument('--solve-scale', type=float, default=DEFAULT_SOLVE_SCALE,
                    help='multiply measured solve times by this (container vs host)')
    ap.add_argument('--plant', choices=sorted(PLANTS), default='st')
    ap.add_argument('--seed', type=int, default=None,
                    help='plant noise seed; measured-solve runs still vary with host timing')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--mpc', type=parse_mpc_override, action='append', default=[],
                    metavar='KEY=VALUE',
                    help='override one MPC weight for this run (repeatable): '
                         + ', '.join(TUNABLE_MPC_KEYS))
    ap.add_argument('--model', choices=MODEL_TYPES, default=JUDGED_MODEL,
                    help="the MPC's prediction model (the node's vehicle.model)")
    ap.add_argument('--dt-schedule', type=parse_schedule, default=None,
                    metavar='DT,DT,...',
                    help='prediction step lengths, one per step, the first the '
                         'control period (the node\'s mpc.dt_schedule)')
    ap.add_argument('--feedforward', action='store_true',
                    help='seed the first steering move from the reference curvature')
    ap.add_argument('--driver', type=parse_driver_override, action='append', default=[],
                    metavar='KEY=VALUE',
                    help='override one DriverConfig field for this run (repeatable), '
                         'e.g. reference=raceline, v_max=6, a_lat_max=5')
    ap.add_argument('--json', default=None, help='write the result dict here')
    return ap


def _config_from(args):
    return HarnessConfig(laps=args.laps, max_time=args.max_time,
                        stale_steps=args.stale_steps, rtf=args.rtf,
                        control_period=args.control_period,
                        fixed_solve_ms=args.fixed_solve_ms, solve_scale=args.solve_scale,
                        plant=args.plant, seed=args.seed, quiet=args.quiet,
                        mpc_overrides=tuple(args.mpc), model=args.model,
                        dt_schedule=args.dt_schedule, feedforward=args.feedforward,
                        driver_overrides=tuple(args.driver))


def parse_config(argv=None):
    """Turn command-line flags into a HarnessConfig."""
    return _config_from(_parser().parse_args(argv))


def main():
    args = _parser().parse_args()
    cfg = _config_from(args)
    r = run(cfg)
    report(r)
    if args.json:
        with open(args.json, 'w') as handle:
            json.dump(json_record(cfg, r), handle, indent=2)


if __name__ == '__main__':
    main()
