#!/usr/bin/env python3
"""Where the car will be when the command being planned now latches.

The pose and scan the driver plans from are a dead time old: the bridge's
queue holds them ~9 physics steps, and the command lands a solve after the
tick. At 5 m/s that is half a metre the MPC never sees. But every command that
acts over that interval is known -- the driver published it, and knows the
plant time it latched -- so the interval is replayed: the plant's own servo
(`pid_steer`), speed loop (`pid_accl`) and single-track dynamics, one 10 ms
physics step at a time, from the measured state to the next latch. The MPC
then plans from where the car will be when its command takes effect, and the
speed caps act from the speed it will have.
"""

import math
from typing import List, NamedTuple, Optional, Tuple

from vehicle_model import (F1TENTH_VEHICLE_PARAMETERS, PLANT_STEP_DT, SpeedLoop,
                           VehicleModel, VehicleParameters)

LATCH_TIME_TOL = 1e-6     # [s] stamps are float sums of 0.01; compare with slack
# [s] Commands kept for the replay. Odometry is never this stale, and a gap
# longer than it is replayed only this far.
HISTORY_SPAN = 1.0
STEER_KP_SCALE = 10.0     # action.py:47-48: kp = 10 sv_max / (s_max - s_min)


def servo_kp(params: VehicleParameters) -> float:
    """[1/s] The plant's steering gain; the bridge passes none, so it is derived."""
    return STEER_KP_SCALE * params.sv_max / (params.s_max - params.s_min)


def plant_servo_rate(params: VehicleParameters, kp: float, delta: float, cmd: float) -> float:
    """[rad/s] utils.py pid_steer then steering_constraint, from the pre-step wheel."""
    sv = min(max(kp * (cmd - delta), -params.sv_max), params.sv_max)
    if (delta <= params.s_min and sv <= 0.0) or (delta >= params.s_max and sv >= 0.0):
        return 0.0
    return sv


class Prediction(NamedTuple):
    """The plant at a latch: the ST state at base_link, and the path driven to it."""

    state: Tuple[float, ...]   # (x, y, yaw, v, beta, delta, yaw_rate)
    travelled: float           # [m] path length since the measured state

    @property
    def speed(self) -> float:
        return self.state[3]


class LatchPredictor:
    """Replays published commands through the plant from a measured state."""

    def __init__(self, params: Optional[VehicleParameters] = None):
        self.params = params if params is not None else F1TENTH_VEHICLE_PARAMETERS
        self.model = VehicleModel(self.params, model_type='ST')
        self.loop = SpeedLoop.of(self.params)
        self.kp = servo_kp(self.params)
        self.step = PLANT_STEP_DT
        self._history: List[Tuple[float, float, float]] = []   # (t_latch, steer, speed)

    def __len__(self) -> int:
        """Commands held for the replay."""
        return len(self._history)

    def on_publish(self, steer: float, speed: float, t_latch: Optional[float]) -> None:
        """A command went on the wire and latches at plant time `t_latch`.

        The steering is clipped to the lock as gym_bridge.py:595 does; the
        speed is left to the plant's loop, as the bridge leaves it.
        """
        if t_latch is None:
            return      # it cannot be placed in time; the replay holds the last known one
        t = float(t_latch)
        steer = min(max(float(steer), self.params.s_min), self.params.s_max)
        # A newer command supersedes older ones due at or after it.
        self._history = [e for e in self._history if e[0] < t - LATCH_TIME_TOL]
        self._history.append((t, steer, float(speed)))
        # Keep the newest command at or before the span's start: it is the one
        # acting there.
        cutoff = t - HISTORY_SPAN
        while len(self._history) > 1 and self._history[1][0] <= cutoff:
            self._history.pop(0)

    def reset(self) -> None:
        """Forget every command: after a sim reset none of them is acting."""
        self._history = []

    def predict(self, state0: Tuple[float, ...], t0: Optional[float],
                t1: Optional[float]) -> Prediction:
        """The plant at `t1`, from ST state `state0` measured at `t0`.

        With no command known the plant is taken to hold its wheel and speed.
        """
        state0 = tuple(float(s) for s in state0)
        if t0 is None or t1 is None or t1 <= t0 + LATCH_TIME_TOL:
            return Prediction(state0, 0.0)
        t1 = min(t1, t0 + HISTORY_SPAN)
        history = self._history
        i = 0
        cmd = None
        while i < len(history) and history[i][0] <= t0 + LATCH_TIME_TOL:
            cmd = history[i]
            i += 1
        x, y, psi, v, beta, delta, r = state0
        t, travelled = float(t0), 0.0
        while t < t1 - LATCH_TIME_TOL:
            h = min(self.step, t1 - t)
            if i < len(history):
                h = min(h, max(history[i][0] - t, LATCH_TIME_TOL))
            sv = 0.0 if cmd is None else plant_servo_rate(self.params, self.kp, delta, cmd[1])
            a = 0.0 if cmd is None else self.loop.accel(cmd[2], v)
            # The plant holds sv across its RK4 step, so the wheel moves
            # linearly through it: the tyres see it at mid-step.
            x1, y1, psi, v, beta, r = self.model.st_step(
                x, y, psi, v, beta, r, delta + 0.5 * sv * h, a, h)
            travelled += math.hypot(x1 - x, y1 - y)
            x, y = x1, y1
            delta += sv * h
            t += h
            while i < len(history) and history[i][0] <= t + LATCH_TIME_TOL:
                cmd = history[i]
                i += 1
        return Prediction((x, y, psi, v, beta, delta, r), travelled)
