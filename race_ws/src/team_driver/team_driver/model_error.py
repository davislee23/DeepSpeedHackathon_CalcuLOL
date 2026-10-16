#!/usr/bin/env python3
"""How well each MPC model predicts the judged plant, measured.

    python3 model_error.py                     # 3, 5 and 7 m/s, 4 s each
    python3 model_error.py --speeds 2 4 6 8 --seconds 6

The judged plant (GymPlant: the gym's single-track model, servo and RK4) is
held at a speed and steered through a sine sweep on the 25 ms control cadence.
At every tick each model is started from what odometry reports there -- the
rear-axle pose, speed, the wheel (known exactly here), and for ST the sideslip
and yaw rate; KS gets beta = 0, as the driver hands it -- and asked to predict:

  * one control period ahead, under the command the plant then held;
  * over the controller's own horizon: the long dt_schedule (0.5 s, steps of
    25-100 ms) with one command held per block of the block map, exactly as
    the MPC plans. The plant is re-run from that tick under the same held
    commands, so a long step's error is the model's, not the plan's.

The acceleration each model is given over a step is the plant's own speed
change over it, so the comparison isolates the lateral model.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plant_sim  # noqa: E402
from driver_params import DEFAULTS  # noqa: E402
from lap_sim import cadence_steps  # noqa: E402
from vehicle_model import PLANT_STEP_DT, VehicleModel  # noqa: E402

SPEEDS = (3.0, 5.0, 7.0)          # [m/s]
SECONDS = 4.0                     # [s] of plant time per speed
CONTROL_PERIOD = DEFAULTS['control_period']   # [s] the node's timer
# The controller's horizon: the long schedule and the judged block map.
SCHEDULE = tuple(DEFAULTS['mpc.dt_schedule'])   # [s], 0.5 s
BLOCK_MAP = tuple(DEFAULTS['mpc.block_map'])
HORIZON_STRIDE = 4                # start a horizon every this many ticks
PHYSICS_DT = PLANT_STEP_DT       # [s] the judged plant's step
# A sweep that reaches ~8-10 m/s^2 of lateral acceleration at 7 m/s: the slip
# regime the ST model exists for.
STEER_AMPLITUDE = 0.10            # [rad]
STEER_FREQUENCY = 1.0             # [Hz]


class Tick(NamedTuple):
    """The plant at one control tick, and the command it then held."""

    t: float
    state: np.ndarray     # (7,) gym CoG state
    command: float        # [rad] steering latched at t


@dataclass(frozen=True)
class ErrorRow:
    speed: float
    model: str
    step_pos_mean: float      # [m]
    step_pos_max: float       # [m]
    step_yaw_mean: float      # [rad]
    horizon_pos_mean: float   # [m]
    horizon_pos_max: float    # [m]
    horizon_yaw_mean: float   # [rad]


def plant_trace(speed, seconds=SECONDS, amplitude=STEER_AMPLITUDE,
                frequency=STEER_FREQUENCY):
    """Run GymPlant at `speed` through a steering sweep; one Tick per control tick."""
    plant = plant_sim.GymPlant()
    start = np.zeros(7)
    start[plant_sim.V] = speed
    plant.reset_state(start)
    trace = []
    k = 0
    while plant.time < seconds:
        command = amplitude * math.sin(2.0 * math.pi * frequency * plant.time)
        plant.latch(command, speed)
        trace.append(Tick(plant.time, plant.state.astype(np.float64), command))
        for _ in range(cadence_steps(k, CONTROL_PERIOD)):
            plant.step()
        k += 1
    trace.append(Tick(plant.time, plant.state.astype(np.float64), 0.0))
    return trace


def odom_state(cog, model_type, lr):
    """The MPC's initial state for a gym CoG state, as the driver builds it."""
    x, y, yaw = plant_sim.base_link_pose_of(cog, lr)
    v, delta = cog[plant_sim.V], cog[plant_sim.DELTA]
    if model_type == 'KS':
        return np.array([x, y, yaw, v, 0.0, delta])
    return np.array([x, y, yaw, v, cog[plant_sim.BETA], delta, cog[plant_sim.PSI_DOT]])


def _pose_error(predicted, cog, lr):
    x, y, yaw = plant_sim.base_link_pose_of(cog, lr)
    d_yaw = math.atan2(math.sin(predicted[2] - yaw), math.cos(predicted[2] - yaw))
    return math.hypot(predicted[0] - x, predicted[1] - y), abs(d_yaw)


def _predict_step(model, trace, k):
    """One control period from tick k, under the command the plant held."""
    lr = float(model.params.lr)
    dt = trace[k + 1].t - trace[k].t
    accel = (trace[k + 1].state[plant_sim.V] - trace[k].state[plant_sim.V]) / dt
    state = model.actuated_step(odom_state(trace[k].state, model.model_type, lr),
                                (trace[k].command, accel), dt)
    return _pose_error(state, trace[k + 1].state, lr)


def plan_segments(start_state, start_t, speed, amplitude=STEER_AMPLITUDE,
                  frequency=STEER_FREQUENCY):
    """Re-run the plant over the horizon with one command held per block.

    Returns [(command, dt, accel)] per schedule step and the plant's final
    CoG state. Step ends fall on the 10 ms physics grid, so each step's dt is
    what the plant actually ran.
    """
    plant = plant_sim.GymPlant()
    plant.reset_state(start_state)
    segments, held_block, command = [], None, 0.0
    boundary, steps_done = 0.0, 0
    for dt_nominal, block in zip(SCHEDULE, BLOCK_MAP):
        if block != held_block:
            held_block = block
            t = start_t + steps_done * PHYSICS_DT
            command = amplitude * math.sin(2.0 * math.pi * frequency * t)
        boundary += dt_nominal
        steps = int(round(boundary / PHYSICS_DT)) - steps_done
        v0 = float(plant.state[plant_sim.V])
        plant.latch(command, speed)
        for _ in range(steps):
            plant.step()
        steps_done += steps
        dt = steps * PHYSICS_DT
        segments.append((command, dt, (float(plant.state[plant_sim.V]) - v0) / dt))
    return segments, plant.state.astype(np.float64)


def _predict_horizon(model, tick, speed):
    """The controller's horizon from `tick`, against the plant under the same plan."""
    lr = float(model.params.lr)
    segments, truth = plan_segments(tick.state, tick.t, speed)
    state = odom_state(tick.state, model.model_type, lr)
    for command, dt, accel in segments:
        state = model.actuated_step(state, (command, accel), dt)
    return _pose_error(state, truth, lr)


def measure(trace, model_type, speed):
    """(step errors, horizon errors), each a list of (position, yaw) pairs."""
    model = VehicleModel(model_type=model_type)
    steps = [_predict_step(model, trace, k) for k in range(len(trace) - 1)]
    horizons = [_predict_horizon(model, trace[k], speed)
                for k in range(0, len(trace) - 1, HORIZON_STRIDE)]
    return steps, horizons


def error_table(speeds=SPEEDS, seconds=SECONDS):
    """One ErrorRow per (speed, model)."""
    rows = []
    for speed in speeds:
        trace = plant_trace(speed, seconds)
        for model_type in ('KS', 'ST'):
            steps, horizons = measure(trace, model_type, speed)
            s, h = np.asarray(steps), np.asarray(horizons)
            rows.append(ErrorRow(speed, model_type,
                                 float(s[:, 0].mean()), float(s[:, 0].max()),
                                 float(s[:, 1].mean()),
                                 float(h[:, 0].mean()), float(h[:, 0].max()),
                                 float(h[:, 1].mean())))
    return rows


def format_table(rows):
    lines = [f"{'v m/s':>6} {'model':>5} | {'1-step pos mm':>17} {'yaw mrad':>9} | "
             f"{'horizon pos cm':>17} {'yaw mrad':>9}",
             f"{'':>6} {'':>5} | {'mean':>8} {'max':>8} {'mean':>9} | "
             f"{'mean':>8} {'max':>8} {'mean':>9}"]
    for r in sorted(rows, key=lambda r: (r.speed, r.model)):
        lines.append(
            f"{r.speed:6.1f} {r.model:>5} | {r.step_pos_mean * 1e3:8.3f} "
            f"{r.step_pos_max * 1e3:8.3f} {r.step_yaw_mean * 1e3:9.3f} | "
            f"{r.horizon_pos_mean * 1e2:8.2f} {r.horizon_pos_max * 1e2:8.2f} "
            f"{r.horizon_yaw_mean * 1e3:9.2f}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--speeds', type=float, nargs='+', default=list(SPEEDS))
    ap.add_argument('--seconds', type=float, default=SECONDS)
    args = ap.parse_args(argv)
    print(f'steering sweep {STEER_AMPLITUDE} rad at {STEER_FREQUENCY} Hz, '
          f'{args.seconds} s per speed, horizon {sum(SCHEDULE):.2f} s as {list(SCHEDULE)}')
    print(format_table(error_table(args.speeds, args.seconds)))


if __name__ == '__main__':
    main()
