#!/usr/bin/env python3
"""Model Predictive Controller for F1TENTH autonomous racing.

Implements a basic MPC that optimizes steering and acceleration over a prediction horizon
using a kinematic bicycle model.
"""

import numpy as np
import math
from typing import Tuple, Optional, Sequence
from scipy.optimize import minimize
import time
from vehicle_model import ST_SWITCH_SPEED, SpeedLoop, VehicleModel
import speed_profile as _speed_profile

# scipy's SLSQP status for "iteration limit reached". The iterate is still a
# feasible improvement on the warm start, so a capped solve is applied, not
# discarded: with a warm start the cap makes the solve a continuation method.
SLSQP_ITERATION_LIMIT = 9
# [s] Slack when comparing step lengths that are sums of floats.
SCHEDULE_DT_TOLERANCE = 1e-9
# SLSQP's default finite-difference step (its `eps` option), absolute. The
# fused gradient uses the same step so it finds the same optimum.
SLSQP_FD_STEP = 1.4901161193847656e-08


class SolveBudgetExceeded(Exception):
    """Raised inside the objective once the wall-clock solve budget is spent."""


class MPCController:
    """Model Predictive Controller for vehicle path following."""

    def __init__(self,
                 vehicle_model: VehicleModel = None,
                 prediction_horizon: int = 10,
                 control_horizon: int = None,
                 dt: float = 0.025,
                 q_pos: float = 2.0,
                 q_yaw: float = 2.0,
                 q_vel: float = 0.5,
                 r_steer: float = 0.08,
                 r_accel: float = 0.01,
                 r_delta_steer: float = 0.8,
                 r_delta_accel: float = 0.005,
                 block_map: Optional[Sequence[int]] = None,
                 maxiter: int = 25,
                 solve_budget_ms: float = 15.0,
                 dt_schedule: Optional[Sequence[float]] = None,
                 feedforward: bool = False):
        """Initialize MPC controller.

        Args:
            vehicle_model: Vehicle model for predictions
            prediction_horizon: Number of steps to predict ahead
            control_horizon: Number of free control moves (None = one per
                prediction step, or the block count when block_map is given)
            dt: Time step for discrete dynamics (seconds)
            q_pos: Weight on position tracking error
            q_y: Weight on lateral error (inf = hard constraint)
            q_yaw: Weight on heading error
            q_vel: Weight on velocity tracking error
            r_steer: Weight on steering control effort
            r_accel: Weight on acceleration control effort
            r_delta_steer: Weight on steering rate of change. The tuned
                value (3.0, Step 9) is in driver_params.py, which the node
                and lap_sim always pass; these defaults are not the judged car.
            r_delta_accel: Weight on acceleration rate of change
            block_map: Move blocking. Entry k is the index of the free control
                applied at prediction step k, so len == prediction_horizon and
                it counts up from 0 one block at a time. [0,1,1,2,2,2,2,2]
                optimises 3 moves (6 variables) over 8 steps. None holds the
                last of `control_horizon` moves to the end of the horizon.
            maxiter: SLSQP iteration cap per solve
            solve_budget_ms: Wall-clock budget per solve. Once spent, the solve
                is abandoned and the previous plan, advanced one step, is used.
            dt_schedule: Length of each prediction step [s], one per step.
                None is `dt` throughout. The first entry must be `dt`, the
                control period: the servo latch and the warm-start shift both
                assume a command is held for exactly one step. Later steps may
                be longer, buying lookahead without decision variables.
            feedforward: Seed the warm start's first steering move with the
                wheel angle the reference curvature needs (steady_steer), so
                the solver corrects around it instead of finding it.
        """
        self.vehicle_model = vehicle_model if vehicle_model is not None else VehicleModel()
        self.prediction_horizon = prediction_horizon
        self.block_map = self._resolve_block_map(
            prediction_horizon, control_horizon, block_map)
        self.control_horizon = int(self.block_map[-1]) + 1
        self._block_starts = np.flatnonzero(np.diff(self.block_map, prepend=-1))
        self._block_counts = np.diff(np.append(self._block_starts, prediction_horizon))
        self.dt = dt
        self.dt_schedule = self._resolve_dt_schedule(prediction_horizon, dt, dt_schedule)
        self._uniform = bool(np.all(self.dt_schedule == dt))
        self._t_offsets = self._time_offsets(prediction_horizon + 1)
        self.lookahead = float(self._t_offsets[-1])
        self._shift_index = self._resolve_shift_index(self.dt_schedule, self._t_offsets)
        self.feedforward = bool(feedforward)
        self.maxiter = maxiter
        self.solve_budget_ms = solve_budget_ms
        self._deadline = math.inf        # only armed while solve() is running
        self.last_fallback = None        # None, 'budget', 'solver' or 'error'
        self.budget_exceeded_count = 0
        self.solver_failure_count = 0
        self.solve_count = 0

        # State and control dimensions. The wheel angle is a state because the
        # judged servo is rate-limited: the car turns with the wheel, and the
        # command only moves the wheel (VehicleModel.actuated_step). The ST
        # model adds the yaw rate as a seventh.
        self.state_dim = self.vehicle_model.actuated_dim  # [x, y, yaw, v, beta, delta(, r)]
        self.control_dim = 2  # [steering_command, acceleration]
        untracked = [0.0] * (self.state_dim - 4)

        # Cost function weights. Neither beta, the wheel angle nor the yaw rate
        # is tracked: the wheel is where it is, and the command is costed
        # through R.
        # Diagonal weights, for the vectorized cost. q_yaw is handled separately
        # (angle wrapping) so it is zeroed here and re-added wrapped.
        self.Q_diag_masked = np.array([q_pos, q_pos, 0.0, q_vel] + untracked)
        self.q_yaw_wrap = q_yaw
        self.R_diag = np.array([r_steer, r_accel])
        self.R_delta_diag = np.array([r_delta_steer, r_delta_accel])
        self._lr = float(self.vehicle_model.params.lr)

        # The fused objective (`_objective` / `_gradient`): the same cost and
        # the same forward differences as SLSQP's own, on Python floats. False
        # hands SLSQP `_cost_function` and lets scipy difference it.
        self.fused_objective = True
        self._weights = (float(q_pos), float(q_vel), float(q_yaw), float(r_steer),
                         float(r_accel), float(r_delta_steer), float(r_delta_accel))
        self._block_list = self.block_map.tolist()
        self._dts_list = [float(h) for h in self.dt_schedule]
        self._base = None   # (u, states, cumulative stage cost) of the last _objective

        self.control_min, self.control_max = self.vehicle_model.get_control_bounds()

        # Previous control for rate limiting
        self.prev_control = np.zeros(self.control_dim)
        # Last full optimised sequence, flattened; the next solve shifts it one
        # step to warm-start SLSQP.
        self._u_prev = np.zeros(self.control_horizon * self.control_dim)
        self._last_state = None    # the state the last solve planned from

        # Reference path (will be set externally)
        self.reference_path = None
        self.reference_speeds = None
        # [m/s^2] Deceleration the reference window assumes the car can hold
        # (set_reference_path); None paces the window at the car's own speed.
        self.brake_limit = None
        self._speed_loop = SpeedLoop.of(self.vehicle_model.params)
        self._mean_waypoint_spacing = 0.2  # overwritten by set_reference_path
        self._last_path_index = None

        # Continuous arc position of the reference, in [0, _path_len). The
        # path array holds exactly one lap, so a *continuously increasing* arc
        # coordinate would walk off the end partway round. Carrying the anchor
        # wrapped into [0, _path_len) is what lets a lap close on itself.
        self._s_ref = None
        self._path_len = 0.0

        # Reused across solves so the cost function stays allocation-light:
        # SLSQP calls the objective a few hundred times per solve.
        self._states = np.zeros((prediction_horizon + 1, self.state_dim))
        self._scratch = np.zeros((prediction_horizon + 1, self.state_dim))
        self._du = np.zeros((self.control_horizon, 2))
        self._ref_cache = np.zeros((prediction_horizon + 1, self.state_dim))

        print(f"MPC Controller initialized:")
        print(f"  Prediction horizon: {self.prediction_horizon}")
        print(f"  Control horizon: {self.control_horizon} (block map {self.block_map.tolist()})")
        print(f"  SLSQP maxiter: {self.maxiter}, solve budget: {self.solve_budget_ms} ms")
        print(f"  DT: {self.dt}s ({1/dt:.1f} Hz), schedule {self.dt_schedule.tolist()} "
              f"({self.lookahead:.3f} s lookahead)")
        print(f"  Model: {self.vehicle_model.model_type}, feed-forward: {self.feedforward}")
        print(f"  State dim: {self.state_dim}, Control dim: {self.control_dim}")

    @staticmethod
    def _resolve_block_map(prediction_horizon: int,
                           control_horizon: Optional[int],
                           block_map: Optional[Sequence[int]]) -> np.ndarray:
        """Validate `block_map`, or build the hold-last map from `control_horizon`."""
        if block_map is None:
            moves = control_horizon if control_horizon is not None else prediction_horizon
            if not 1 <= moves <= prediction_horizon:
                raise ValueError(
                    f"control_horizon must be in [1, {prediction_horizon}], got {moves}")
            return np.minimum(np.arange(prediction_horizon), moves - 1)

        blocks = np.asarray(block_map, dtype=int)
        if blocks.shape != (prediction_horizon,):
            raise ValueError(
                f"block_map needs one entry per prediction step "
                f"({prediction_horizon}), got shape {blocks.shape}")
        steps = np.diff(blocks)
        if blocks[0] != 0 or np.any((steps != 0) & (steps != 1)):
            raise ValueError(
                f"block_map must start at 0 and count up one block at a time, "
                f"got {blocks.tolist()}")
        if control_horizon is not None and control_horizon != blocks[-1] + 1:
            raise ValueError(
                f"control_horizon {control_horizon} contradicts block_map, "
                f"which has {blocks[-1] + 1} blocks")
        return blocks

    @staticmethod
    def _resolve_dt_schedule(prediction_horizon: int, dt: float,
                             dt_schedule: Optional[Sequence[float]]) -> np.ndarray:
        """Validate `dt_schedule`, or hold `dt` for every step."""
        if dt_schedule is None:
            return np.full(prediction_horizon, float(dt))
        steps = np.asarray(dt_schedule, dtype=np.float64)
        if steps.shape != (prediction_horizon,):
            raise ValueError(
                f"dt_schedule needs one entry per prediction step "
                f"({prediction_horizon}), got shape {steps.shape}")
        if not np.all(np.isfinite(steps) & (steps > 0.0)):
            raise ValueError(f"dt_schedule entries must be positive, got {steps.tolist()}")
        if abs(steps[0] - dt) > SCHEDULE_DT_TOLERANCE:
            raise ValueError(
                f"dt_schedule's first step {steps[0]} must be the control period "
                f"dt {dt}: a command is held for exactly one step")
        return steps

    def _time_offsets(self, count: int) -> np.ndarray:
        """[s] Time of each of `count` reference states after the first.

        A uniform schedule keeps the old `k * dt`, so it reproduces the
        controller as it was to the bit; a cumulative sum differs in the last
        place. Past the horizon the last step length repeats.
        """
        if self._uniform:
            return np.arange(count) * self.dt
        steps = np.resize(self.dt_schedule, max(count - 1, 0))
        steps[len(self.dt_schedule):] = self.dt_schedule[-1]
        return np.concatenate([[0.0], np.cumsum(steps)])

    @staticmethod
    def _resolve_shift_index(dt_schedule: np.ndarray, t_offsets: np.ndarray) -> np.ndarray:
        """Step of the last plan that covers each step's start, one control period on.

        Advancing a plan by one control period is a one-step shift only when
        every step is that long. With longer later steps, step k of the next
        plan starts dt_schedule[0] later on the old plan's clock, inside the
        step this returns. The tolerance absorbs float sums, so a uniform
        schedule gives exactly k + 1.
        """
        starts = t_offsets[:len(dt_schedule)]
        due = starts + dt_schedule[0] + SCHEDULE_DT_TOLERANCE
        index = np.searchsorted(starts, due, side='right') - 1
        return np.minimum(index, len(dt_schedule) - 1)

    def set_reference_path(self, path: np.ndarray, speeds: np.ndarray = None,
                           brake_limit: float = None):
        """Set reference path for tracking.

        Args:
            path: Array of shape (N, 2) with [x, y] waypoints
            speeds: Optional array of shape (N,) with target speeds
            brake_limit: [m/s^2] Deceleration the car is held to. Given, the
                reference window is paced by speeds the plant can reach
                (`_reachable_arc`); None keeps the car's-own-speed pace.
        """
        self.reference_path = path
        self.reference_speeds = speeds
        self.brake_limit = brake_limit

        # If no speeds provided, compute based on curvature or use constant
        if speeds is None and path is not None:
            self.reference_speeds = np.full(len(path), 3.0)  # Default 3 m/s

        # Cache spacing so the reference can advance by distance per step
        # instead of one index per step (the centerline is 0.2 m sampled but
        # the car only covers v*dt, so indexing by step overshoots).
        if path is not None and len(path) > 1:
            path = np.asarray(path, dtype=np.float64)
            seg = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
            self._mean_waypoint_spacing = float(np.mean(seg)) or 0.2
            self._path_arc = np.concatenate([[0.0], np.cumsum(seg)])
            # The CSV omits the closing segment back to waypoint 0, so the
            # drivable lap is arc[n-1] + that last gap, not arc[n-1].
            self._path_len = float(self._path_arc[-1]
                                   + np.linalg.norm(path[0, :2] - path[-1, :2]))
            # Signed (left positive) and wrapped at the seam, for feed-forward:
            # the same curvature raceline_table writes in its kappa column.
            kappa = _speed_profile.closed_curvature(path[:, :2], signed=True)
            self._ff_arc = np.append(self._path_arc, self._path_len)
            self._ff_kappa = np.append(kappa, kappa[0])
        else:
            self._mean_waypoint_spacing = 0.2
            self._path_arc = None
            self._path_len = 0.0
            self._ff_arc = None
            self._ff_kappa = None
        self._window_s = None

        # A new path invalidates the local search window.
        self._last_path_index = None
        self._s_ref = None

    @property
    def arc_position(self) -> Optional[float]:
        """[m] Arc position the last solve anchored its reference at, or None.

        It is the arc of the state that solve was given -- the pose the driver
        planned from -- in `set_reference_path`'s parameterisation.
        """
        return self._s_ref

    def reset_tracking(self) -> None:
        """Forget where the car was and what it planned: the sim was reset.

        The nearest-waypoint search only looks near the last index, so a car
        respawned elsewhere would otherwise stay localised to its old arc.
        """
        self._last_path_index = None
        self._s_ref = None
        self._window_s = None
        self._u_prev = np.zeros(self.control_horizon * self.control_dim)
        self.prev_control = np.zeros(self.control_dim)
        self._last_state = None

    def planned_path(self) -> Optional[np.ndarray]:
        """(N+1, 2) positions the plan being executed drives through, or None.

        The last solve's controls (or, after a fallback, the plan it fell back
        to) rolled out from the state it planned from.
        """
        if self._last_state is None:
            return None
        return self._predict_trajectory(self._last_state, self._u_prev)[:, :2].copy()

    @property
    def search_window(self) -> int:
        """How many waypoints either side of the last index to search."""
        return max(10, int(round(2.0 / max(self._mean_waypoint_spacing, 1e-3))))

    @property
    def ref_speed(self) -> float:
        """Default pace for the reference window when the car's speed is unknown.

        Callers that know the car's speed pass it to `_reference_window`
        directly; this is the fallback for the ones that do not. It is the mean
        of the profile, unchanged in spirit from the original scalar.
        """
        if self.reference_speeds is None or len(self.reference_speeds) == 0:
            return 3.0
        return float(np.mean(self.reference_speeds))

    @property
    def _ref_speed_floor(self) -> float:
        """Minimum speed the reference window may advance at, m/s.

        Guards one degenerate case only: a car that is stopped, or rolling
        backwards, would otherwise collapse the entire window onto one point --
        the stall the arc-length anchor exists to prevent. It is a small
        fraction of the profile mean, so it only ever binds when the car is
        barely moving and never overrides the car's real speed otherwise.
        """
        if self.reference_speeds is None or len(self.reference_speeds) == 0:
            return 0.25
        return float(np.mean(self.reference_speeds)) / 8.0

    def _nearest_path_index(self, position: np.ndarray, hint: int = None) -> int:
        """Index of the path waypoint nearest a point.

        When the caller passes the previous index as a hint we only search a
        window around it. Pure nearest-neighbour search over the whole path is
        wrong on a closed track: the loop doubles back on itself, so a point on
        the far side of the lap is closer in straight-line distance and the
        reference index jumps to the wrong part of the track.

        The window **wraps**, because the path is a closed lap: a plain slice
        `path[lo:hi]` cannot reach across the finish line, so a car that runs
        off the end of the array would keep matching the final waypoint no
        matter how far it travelled -- the same failure as an unwrapped
        reference window, one stage earlier in the pipeline.
        """
        path = self.reference_path
        n = len(path)
        if hint is not None and 0 <= hint < n:
            w = self.search_window
            idx = (np.arange(hint - w, hint + w + 1) % n)
            d = np.sum((path[idx] - position) ** 2, axis=1)
            return int(idx[int(np.argmin(d))])
        return int(np.argmin(np.sum((path - position) ** 2, axis=1)))

    def _project_arc(self, position: np.ndarray, idx: int) -> float:
        """Arc position of `position` projected onto the path near `idx`.

        The nearest *waypoint* is up to half a spacing away from where the car
        actually sits on the line, so using the waypoint's own arc position as
        the resync target would drag the anchor by up to half a spacing every
        step. Projecting onto the segment instead makes the residual zero for a
        car that is on the line, and a whole waypoint only when it is not.

        The nearest waypoint is not always the right segment to project onto:
        a car that has fallen half a spacing behind waypoint `idx` still finds
        `idx` as its nearest, and a projection clamped to the segment *after*
        `idx` would snap it forward onto that waypoint. So the preceding
        segment is included, and the closer of the two is used.
        """
        n = len(self.reference_path)
        candidates = []
        for j in (idx - 1, idx, idx + 1) if n > 2 else (idx,):
            j %= n
            a = self.reference_path[j, :2]
            b = self.reference_path[(j + 1) % n, :2]
            ab = b - a
            denom = float(ab @ ab)
            f = 0.0 if denom < 1e-12 else float((position - a) @ ab / denom)
            f = min(max(f, 0.0), 1.0)
            # Closing segment: last waypoint -> waypoint 0, which the array
            # has no entry for.
            span = (self._path_len - float(self._path_arc[-1]) if j == n - 1
                    else float(self._path_arc[j + 1] - self._path_arc[j]))
            px = a + f * ab
            candidates.append((float((position - px) @ (position - px)),
                               float(self._path_arc[j]) + f * max(span, 1e-9)))
        candidates.sort()
        return candidates[0][1]

    def _reachable_arc(self, s0: float, v0: float, offsets: np.ndarray) -> np.ndarray:
        """Arc positions of the window if the car chases the profile at its limits.

        The speed follows the profile at `s`, but moves up by at most the
        plant's acceleration ceiling (power-limited above ST_SWITCH_SPEED) and
        down by at most `brake_limit`; each step covers its mean speed.
        """
        arc, L, profile = self._path_arc, self._path_len, self.reference_speeds
        s = np.empty(len(offsets))
        s[0] = s0
        v = v0
        for k in range(1, len(offsets)):
            h = float(offsets[k] - offsets[k - 1])
            target = float(np.interp(s[k - 1] % L, arc, profile, period=L))
            v_next = min(max(target, v - self.brake_limit * h),
                         v + self._speed_loop.accel_ceiling(v) * h)
            s[k] = s[k - 1] + 0.5 * (v + v_next) * h
            v = v_next
        return s

    def _reference_window(self, s_start: float, count: int,
                          car_speed: float = None) -> np.ndarray:
        """Resample `count` reference states marching forward along the path.

        The k-th state must be roughly where the car will be in k*dt seconds.
        The centerline is sampled at 0.2 m intervals, so stepping one index
        per control step advances the reference 0.2 m while the car only moves
        v*dt (about 0.075 m at 3 m/s) -- the reference would run away from the
        car roughly 2.5x too fast. We advance by distance instead.

        With a `brake_limit` the distance is instead what the plant can reach
        (`_reachable_arc`). Pacing at the car's own speed puts the horizon
        behind where an accelerating car will be, and q_pos (4x q_vel) then
        penalises the acceleration the profile asks for.

        Otherwise that distance is the *car's*, floored at `ref_speed`. Spacing by the
        profile mean instead (the old behaviour) put the horizon at a distance
        set by the track rather than by the car: on the fast half it ran ahead
        of anything the car could reach, and at a slow corner it reached
        forward into the straight beyond, asking the car to straighten before
        the corner. The floor keeps a stopped car from collapsing the window.

        `s_start` is an arc position in [0, self._path_len) and the window is
        **wrapped** at the end of the lap. Wrapping is not optional: the path
        array holds exactly one lap, so a window that runs past its end would
        clip every horizon state onto the final waypoint -- a single repeated
        point behind the car, whose position cost is minimised by sitting still.
        That is what stalled the car on the real track, where the spawn sits at
        waypoint 377 of 392 and the reference hits the end within seconds.
        """
        path = self.reference_path
        n = len(path)
        out = self._ref_cache
        out = out[:count]

        # Cumulative arc length is cached in set_reference_path; recomputing it
        # here would cost more than the resample itself.
        arc = self._path_arc
        L = self._path_len
        if car_speed is None:
            pace = self.ref_speed
        else:
            pace = max(float(car_speed), self._ref_speed_floor)
        # State k is where the car will be after the first k steps, which
        # with a dt_schedule is a cumulative time, not k * dt.
        offsets = (self._t_offsets if count == len(self._t_offsets)
                   else self._time_offsets(count))
        if (self.brake_limit is not None and car_speed is not None
                and self.reference_speeds is not None
                and len(self.reference_speeds) == n):
            s = self._reachable_arc(s_start % L, pace, offsets)
        else:
            s = (s_start % L) + offsets * pace
        self._window_s = s

        # The path array stores a single lap and no closing waypoint, so
        # sampling is only an interpolation while s stays inside the array.
        # Past that -- the last spacing, and anything the horizon spills over
        # the finish line -- walk the closing segment explicitly, so the window
        # still advances instead of collapsing onto the last waypoint.
        span = float(arc[-1])
        ref_pos = np.empty((count, 2), dtype=np.float64)
        speed = np.empty(count, dtype=np.float64)
        heading = np.empty(count, dtype=np.float64)
        speeds = self.reference_speeds
        have_speeds = speeds is not None and len(speeds) == n

        def _lerp(a, b, f):
            return (path[a, 0] + f * (path[b, 0] - path[a, 0]),
                    path[a, 1] + f * (path[b, 1] - path[a, 1]))

        for k in range(count):
            sk = s[k]
            if sk >= span:
                # Closing segment: last waypoint -> waypoint 0.
                f = (sk - span) / max(L - span, 1e-9)
                x, y = _lerp(n - 1, 0, f)
                # Tangent of the closing segment, in the direction of travel.
                d = path[0, :2] - path[-1, :2]
                heading[k] = math.atan2(d[1], d[0])
                speed[k] = (speeds[0] if have_speeds else 3.0) * (1.0 - f) + \
                           (speeds[-1] if have_speeds else 3.0) * f
            else:
                j = int(np.searchsorted(arc, sk, side='right')) - 1
                j = min(max(j, 0), n - 2)
                d = arc[j + 1] - arc[j]
                f = (sk - arc[j]) / d if d > 1e-9 else 0.0
                x, y = _lerp(j, j + 1, f)
                seg = path[j + 1, :2] - path[j, :2]
                heading[k] = math.atan2(seg[1], seg[0])
                speed[k] = ((speeds[j] * (1.0 - f) + speeds[j + 1] * f)
                            if have_speeds else 3.0)
            ref_pos[k] = (x, y)

        out[:, 0] = ref_pos[:, 0]
        out[:, 1] = ref_pos[:, 1]
        out[:, 2] = heading
        out[:, 3] = speed
        out[:, 4:] = 0.0    # beta, wheel angle (and yaw rate); untracked (Q weight 0)
        return out

    def _predict_trajectory(self,
                           initial_state: np.ndarray,
                           control_sequence: np.ndarray) -> np.ndarray:
        """Predict trajectory over horizon given control sequence.

        Args:
            initial_state: Starting state [x, y, yaw, v, beta, delta]
            control_sequence: Flattened free controls, one per block of
                `block_map` (control_horizon of them)

        Returns:
            Predicted states of shape (N+1, state_dim)
        """
        # Expand the free controls to one per prediction step.
        per_step = control_sequence.reshape(
            (self.control_horizon, self.control_dim))[self.block_map]

        # Propagate dynamics, with the steering servo in the loop
        return self.vehicle_model.rollout(initial_state, per_step, self.dt_schedule,
                                          self._states)

    def _cost_function(self,
                      control_sequence: np.ndarray,
                      initial_state: np.ndarray,
                      reference_states: np.ndarray,
                      reference_speeds: np.ndarray) -> float:
        """Compute MPC cost function.

        Args:
            control_sequence: Flattened control sequence to optimize
            initial_state: Current vehicle state
            reference_states: Reference states over horizon (N+1, state_dim)
            reference_speeds: Reference speeds over horizon (N+1,)

        Returns:
            Total cost

        Raises:
            SolveBudgetExceeded: once the wall-clock budget armed by solve() is spent
        """
        self._check_deadline()

        # Predict trajectory
        states = self._predict_trajectory(initial_state, control_sequence)

        u = control_sequence.reshape((self.control_horizon, self.control_dim))

        # State tracking cost: err^T Q err summed over the horizon. Q is a
        # fixed diagonal, so this collapses to a weighted sum of squares
        # instead of a Python loop of N matmuls.
        err = self._scratch
        np.subtract(states, reference_states, out=err)
        cost = float(np.sum(err * (self.Q_diag_masked * err)))

        if self.state_dim == 7:
            # The reference heading is the path's tangent, which a sliding
            # car's rear axle follows while its body points elsewhere: its
            # velocity is (v cos beta, v sin beta - lr r) in the body frame.
            # Compare the direction of travel, or the cost charges the slip
            # the corner needs as a heading error. KS pins beta at 0, so its
            # rear axle travels along the yaw and needs no correction. Below
            # the switch speed the model's rear axle has no side slip anyway,
            # and the angle would flip by pi as a predicted v crosses zero.
            v, beta = states[:, 3], states[:, 4]
            drift = np.arctan2(v * np.sin(beta) - self._lr * states[:, 6], v * np.cos(beta))
            err[:, 2] += np.where(v > ST_SWITCH_SPEED, drift, 0.0)

        # yaw is an angle, not a coordinate: compare it wrapped to +/-pi so the
        # cost does not spike when the reference heading crosses the branch cut.
        yaw_err = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
        cost += float(self.q_yaw_wrap * np.sum(yaw_err * yaw_err))

        # Control effort cost
        cost += float(np.sum(u * (self.R_diag * u)))

        # Control rate cost (smoothness), including the jump from the last
        # command actually applied.
        d = self._du
        d[0] = u[0] - self.prev_control
        if self.control_horizon > 1:
            np.subtract(u[1:], u[:-1], out=d[1:])
        cost += float(np.sum(d * (self.R_delta_diag * d)))

        return cost

    # ------------------------------------------------------------------
    # The fused objective: `_cost_function` and SLSQP's forward differences,
    # on Python floats
    # ------------------------------------------------------------------
    def _prepare_objective(self, initial_state: np.ndarray,
                           reference_states: np.ndarray) -> None:
        """Cache what every evaluation of one solve shares, as Python floats."""
        self._x0 = tuple(float(s) for s in initial_state)
        self._ref_rows = [tuple(float(c) for c in row[:4]) for row in reference_states]
        self._prev_u = (float(self.prev_control[0]), float(self.prev_control[1]))
        self._upper = [float(b) for b in np.tile(self.control_max, self.control_horizon)]
        self._stage0 = self._stage_cost(self._x0, self._ref_rows[0])
        self._base = None

    def _stage_cost(self, s: tuple, ref: tuple) -> float:
        """One predicted state's tracking cost, as `_cost_function` weighs it."""
        q_pos, q_vel, q_yaw = self._weights[:3]
        ex, ey, ev = s[0] - ref[0], s[1] - ref[1], s[3] - ref[3]
        e_yaw = s[2] - ref[2]
        v = s[3]
        if self.state_dim == 7 and v > ST_SWITCH_SPEED:   # as in _cost_function
            beta = s[4]
            e_yaw += math.atan2(v * math.sin(beta) - self._lr * s[6], v * math.cos(beta))
        e_yaw = math.atan2(math.sin(e_yaw), math.cos(e_yaw))
        return q_pos * (ex * ex + ey * ey) + q_vel * ev * ev + q_yaw * e_yaw * e_yaw

    def _control_cost(self, u: list) -> float:
        """Effort and rate cost of the free moves, as `_cost_function` weighs them."""
        r_s, r_a, r_ds, r_da = self._weights[3:]
        prev_s, prev_a = self._prev_u
        cost = 0.0
        for b in range(self.control_horizon):
            s, a = u[2 * b], u[2 * b + 1]
            ds, da = s - prev_s, a - prev_a
            cost += r_s * s * s + r_a * a * a + r_ds * ds * ds + r_da * da * da
            prev_s, prev_a = s, a
        return cost

    def _track_from(self, u: list, start: int, state: tuple, acc: float,
                    states: list = None, cumulative: list = None) -> float:
        """Roll the moves `u` on from step `start` and add up the stage costs.

        `state` is the predicted state at `start` and `acc` the stage cost of
        states 0..start. When `states` / `cumulative` are given, each state and
        running total is recorded so a perturbed rollout can restart mid-way.
        """
        step = self.vehicle_model.scalar_step
        blocks, dts, refs = self._block_list, self._dts_list, self._ref_rows
        for k in range(start, self.prediction_horizon):
            b = blocks[k]
            state = step(state, u[2 * b], u[2 * b + 1], dts[k])
            acc += self._stage_cost(state, refs[k + 1])
            if states is not None:
                states.append(state)
                cumulative.append(acc)
        return acc

    def _check_deadline(self) -> None:
        if time.perf_counter() >= self._deadline:
            raise SolveBudgetExceeded

    def _objective(self, control_sequence: np.ndarray) -> float:
        """`_cost_function` for the prepared solve, on Python floats."""
        self._check_deadline()
        u = control_sequence.tolist()
        states, cumulative = [self._x0], [self._stage0]
        track = self._track_from(u, 0, self._x0, self._stage0, states, cumulative)
        total = track + self._control_cost(u)
        self._base = (u, states, cumulative, total)
        return total

    def _gradient(self, control_sequence: np.ndarray) -> np.ndarray:
        """SLSQP's own forward difference of `_objective`, restarted mid-horizon.

        Same step (SLSQP_FD_STEP) and same bound handling as scipy's
        approx_derivative: a step that would cross the upper bound is taken
        backwards. A move only acts from its block's first step on, so the
        perturbed rollout restarts there from the recorded base trajectory.
        """
        self._check_deadline()
        u = control_sequence.tolist()
        if self._base is None or self._base[0] != u:
            self._objective(control_sequence)
        _, states, cumulative, f0 = self._base
        starts = self._block_starts
        grad = np.empty(len(u))
        for i in range(len(u)):
            h = SLSQP_FD_STEP
            if u[i] + h > self._upper[i]:
                h = -h
            moved = list(u)
            moved[i] = u[i] + h
            start = int(starts[i // 2])
            f = (self._track_from(moved, start, states[start], cumulative[start])
                 + self._control_cost(moved))
            grad[i] = (f - f0) / (moved[i] - u[i])
        return grad

    def solve(self, current_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve MPC optimization problem.

        The smoothness anchor and the warm start both come from this
        controller's own previous output; callers cannot inject either.

        Args:
            current_state: [x, y, yaw, v, beta, delta], plus the yaw rate for
                the ST model. A caller with no wheel estimate may pass
                [x, y, yaw, v, beta] or [x, y, yaw, v]: beta then defaults to
                0 and the wheel to the last command, i.e. a wheel that has
                caught up with it. A missing yaw rate defaults to 0.

        Returns:
            (optimal_control, solve_time)
        """
        start_time = time.perf_counter()
        self.last_fallback = None
        self.solve_count += 1

        current_state = self._complete_state(current_state)
        self._last_state = current_state.copy()

        # Improved approach: look ahead along the path from current vehicle position
        if self.reference_path is not None and len(self.reference_path) > 1:
            current_pos = current_state[0:2]

            # Search near last time to stay on the correct part of a closed loop.
            closest_idx = self._nearest_path_index(current_pos, self._last_path_index)
            self._last_path_index = closest_idx

            # Anchor the reference in continuous arc length, not in waypoint
            # index. The car moves v*dt per step (about 0.04 m at 1.6 m/s) but a
            # waypoint is 0.2 m apart, so a snapped index can only either sit on
            # the car or jump a whole waypoint -- noise of up to half a spacing
            # against 0.04 m of real travel, which the optimiser then chases.
            # Advance by the distance actually covered and resynchronise on the
            # car's projection onto the path, so the window starts where the
            # car is.
            s_proj = self._project_arc(current_pos, closest_idx)
            v = max(float(current_state[3]), 0.0)
            step = v * self.dt
            L = self._path_len
            if self._s_ref is None:
                self._s_ref = s_proj
            else:
                # Resynchronise: |s_proj - s_ref| is how far off the tracked
                # arc position has drifted, and the modulo puts it in
                # [-L/2, L/2] -- the shortest way round a closed lap.
                s_ref = self._s_ref + step
                s_ref += (s_proj - s_ref + L / 2.0) % L - L / 2.0
                self._s_ref = s_ref

            reference_states = self._reference_window(
                self._s_ref, self.prediction_horizon + 1, car_speed=v)
            reference_speeds = (reference_states[:, 3] if self.reference_speeds is not None
                                else np.full(self.prediction_horizon + 1, 3.0))
        else:
            # Zero reference if no path set
            reference_states = np.zeros((self.prediction_horizon + 1, self.state_dim))
            reference_speeds = np.zeros(self.prediction_horizon + 1)

        # A fallback continues the plan itself, so the feed-forward seed goes
        # into a copy: a blown budget must not jump the wheel to the seed.
        warm_start = self._warm_start()
        initial_guess = warm_start.copy()
        if self.feedforward and self._ff_kappa is not None and self._window_s is not None:
            initial_guess[0] = self._feedforward_steer(float(current_state[3]))

        # Bounds for optimization
        bounds = list(zip(self.control_min, self.control_max)) * self.control_horizon

        if self.fused_objective:
            self._prepare_objective(current_state, reference_states)
            fun, problem = self._objective, dict(jac=self._gradient)
        else:
            fun, problem = self._cost_function, dict(
                args=(current_state, reference_states, reference_speeds))
        # Optimize. The deadline is only armed around the solver call so a
        # direct _cost_function call never trips it.
        self._deadline = start_time + self.solve_budget_ms / 1e3
        try:
            result = minimize(
                fun,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': self.maxiter, 'ftol': 1e-6},
                **problem
            )
        except SolveBudgetExceeded:
            self.budget_exceeded_count += 1
            return self._fall_back('budget', warm_start, start_time)
        except Exception as e:
            print(f"MPC optimization error: {e}")
            self.solver_failure_count += 1
            return self._fall_back('error', warm_start, start_time)
        finally:
            self._deadline = math.inf

        usable = np.all(np.isfinite(result.x)) and (
            result.success or result.status == SLSQP_ITERATION_LIMIT)
        if not usable:
            print(f"MPC optimization failed: {result.message}")
            self.solver_failure_count += 1
            return self._fall_back('solver', warm_start, start_time)

        self._u_prev = result.x.copy()
        optimal_control = self._u_prev.reshape(
            (self.control_horizon, self.control_dim))[0].copy()
        self.prev_control = optimal_control.copy()
        return optimal_control, time.perf_counter() - start_time

    def _complete_state(self, current_state) -> np.ndarray:
        """The state at the model's length, padding what the caller left out."""
        state = np.asarray(current_state, dtype=np.float64)
        wheel = self.prev_control[0]
        if len(state) == 4:
            state = np.append(state, [0.0, wheel])
        elif len(state) == 5:
            state = np.append(state, wheel)
        if len(state) == 6 and self.state_dim == 7:
            state = np.append(state, 0.0)
        if len(state) != self.state_dim:
            raise ValueError(
                f"Expected state of length {sorted({4, 5, 6, self.state_dim})}, "
                f"got {len(np.atleast_1d(current_state))}")
        return state

    def _feedforward_steer(self, v: float) -> float:
        """[rad] The wheel angle the reference needs over the first block.

        The curvature is averaged over the reference states the first block
        drives towards (its steps' end points), and turned into a wheel angle
        by the model's own steady state, so ST includes the understeer.
        """
        first_block = int(self._block_counts[0])
        s = self._window_s[1:first_block + 1] % self._path_len
        kappa = float(np.mean(np.interp(s, self._ff_arc, self._ff_kappa)))
        steer = self.vehicle_model.steady_steer(kappa, max(v, 0.0))
        return float(np.clip(steer, self.control_min[0], self.control_max[0]))

    def _warm_start(self) -> np.ndarray:
        """The last plan advanced one control period, refitted to the blocks.

        With blocking, one slot of `_u_prev` is a block of steps, so shifting by
        a slot would jump the plan ahead by a whole block. Expand to one control
        per step, advance by one control period, and average each block back
        down. With equal steps that is a least-squares fit of the advanced
        plan onto the blocked parameterisation and the advance is a one-step
        shift. With a dt_schedule each step reads the old step covering its
        start time one period later (`_shift_index`), and the block mean is
        over steps, not weighted by their length -- a warm start, not a fit.
        """
        per_step = self._u_prev.reshape(
            (self.control_horizon, self.control_dim))[self.block_map]
        shifted = per_step[self._shift_index]
        sums = np.add.reduceat(shifted, self._block_starts, axis=0)
        return (sums / self._block_counts[:, None]).ravel()

    def _fall_back(self, reason: str, warm_start: np.ndarray,
                   start_time: float) -> Tuple[np.ndarray, float]:
        """Continue the previous plan one step on instead of applying a bad solve.

        The command is continuous with the last one (it is the plan's own next
        move), and the memory advances with it so the next solve warm-starts
        from where the car now is rather than from a plan one step stale.
        """
        self._u_prev = warm_start.copy()
        control = warm_start.reshape(
            (self.control_horizon, self.control_dim))[0].copy()
        self.prev_control = control.copy()
        self.last_fallback = reason
        return control, time.perf_counter() - start_time

    def get_control(self, current_state: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Get MPC control command.

        Args:
            current_state: Current vehicle state [x, y, yaw, v]

        Returns:
            (control_command, diagnostics)
        """
        control, solve_time = self.solve(current_state)

        diagnostics = {
            'solve_time_ms': solve_time * 1000,
            'fallback': self.last_fallback
        }

        return control, diagnostics

