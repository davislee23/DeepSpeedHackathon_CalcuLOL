#!/usr/bin/env python3
"""Closed-loop speed profile for a reference path, with no ROS in it.

The profile is the fastest speed the plant can be asked to hold at every point:

    v(kappa) = min( sqrt(a_lat_max / kappa), v_lock(kappa), v_max )

then lowered by a backward braking pass and a forward acceleration pass, both
wrapped round the lap so the corner at the finish line is braked for.

`a_lat_max` is a *slip budget*, not a grip limit. The judged plant is the
single-track model with linear tyres (`single_track.py`): force grows with slip
angle and never peaks, so there is no friction circle. The practical limit is
how much rear slip the car may carry, alpha_r = a_y / (mu * C_Sr * g) = a_y / 56.1
rad. Braking does eat into it, though: load transfer unloads the rear axle and
the yaw dynamics go unstable (above ~6 m/s at -6 m/s^2), so `gg_brake` optionally
limits the deceleration into a corner by an ellipse (see `gg_decel`).

`v_lock` is the steering lock. The plant's steady turn needs
delta = L * kappa + K * v^2 * kappa, with the understeer gradient
K = (1 / (mu g)) (1 / C_Sf - 1 / C_Sr) = 0.00279 rad per m/s^2. Solving for v at
delta = lock_margin * s_max gives `lock_speed`. It only binds in hairpins.

Every plant constant comes from `VehicleParameters`; none is a tunable.
"""

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np

from vehicle_model import F1TENTH_VEHICLE_PARAMETERS as _VP

GRAVITY = 9.81                    # [m/s^2] single_track.py's g
MIN_CORNER_SPEED = 0.5            # [m/s] the plant's kinematic switch; a floor, not a target
KAPPA_FLOOR = 1e-6                # [1/m] a straight, for the divisions below
CURVATURE_SMOOTH = 9              # [points] centred moving average, as MPCController
POWER_SPEED_FLOOR = 0.1           # [m/s] keeps 9.51 * 7.319 / v finite at a standstill
PASS_LAPS = 2                     # a wrapped pass needs two laps to settle
GG_EXPONENT = 2.0                 # shape of the braking/cornering ellipse
GG_ITERATIONS = 3                 # fixed-point passes: the share depends on the entry speed
GG_FLOOR = 0.1                    # default share of the ellipse braking may always use

WHEELBASE = _VP.lf + _VP.lr
UNDERSTEER_GRADIENT = (1.0 / (_VP.mu * GRAVITY)) * (1.0 / _VP.C_Sf - 1.0 / _VP.C_Sr)
REAR_SLIP_STIFFNESS = _VP.mu * _VP.C_Sr * GRAVITY   # a_y per radian of rear slip


def segment_lengths(path: np.ndarray, closed: bool) -> np.ndarray:
    """Distance from each point to the next; the last wraps when closed."""
    nxt = np.roll(path, -1, axis=0) if closed else path[1:]
    start = path if closed else path[:-1]
    return np.linalg.norm(nxt - start, axis=1)


def closed_curvature(path: np.ndarray, smooth: int = CURVATURE_SMOOTH,
                     closed: bool = True, signed: bool = False) -> np.ndarray:
    """Smoothed curvature, k = 4 Area / (|ab| |bc| |ac|), wrapped at the seam.

    `MPCController._curvature` zeroes the two end points, which on a closed lap
    erases the curvature at the finish line. `signed` keeps the turn direction
    (left positive) for steering feed-forward; the profile only needs the size.
    """
    if smooth % 2 == 0:
        raise ValueError(f'smooth must be odd to stay centred, got {smooth}')
    p = np.asarray(path, dtype=np.float64)[:, :2]
    n = len(p)
    if n < 3:
        return np.zeros(n)
    smooth = min(smooth, n if n % 2 else n - 1)
    a, b, c = np.roll(p, 1, axis=0), p, np.roll(p, -1, axis=0)
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    cross = cross if signed else np.abs(cross)
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    kappa = np.where(denom > 1e-9, 2.0 * cross / np.maximum(denom, 1e-9), 0.0)
    if not closed:
        kappa[[0, -1]] = 0.0
    kern = np.ones(smooth) / smooth
    half = smooth // 2
    padded = np.concatenate([kappa[-half:], kappa, kappa[:half]]) if closed \
        else np.pad(kappa, half, mode='edge')
    return np.convolve(padded, kern, mode='valid')


def lock_speed(kappa: np.ndarray, lock_margin: float) -> np.ndarray:
    """Fastest steady speed the steering lock can hold at curvature `kappa`.

    A point tighter than the lock radius has no steady solution at any speed. It
    is a line error, so it gets `MIN_CORNER_SPEED` rather than a NaN.
    """
    k = np.maximum(np.abs(np.asarray(kappa, dtype=np.float64)), KAPPA_FLOOR)
    headroom = _VP.s_max * lock_margin - WHEELBASE * k
    v_sq = np.where(headroom > 0.0, headroom / (UNDERSTEER_GRADIENT * k), 0.0)
    return np.maximum(np.sqrt(v_sq), MIN_CORNER_SPEED)


def rear_slip(v: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Rear slip angle [rad] of a steady turn at speed `v`, curvature `kappa`."""
    return np.asarray(v) ** 2 * np.abs(kappa) / REAR_SLIP_STIFFNESS


def body_slip(v: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Magnitude of the body sideslip beta [rad]: beta = lr * kappa - alpha_r."""
    return np.abs(_VP.lr * np.abs(kappa) - rear_slip(v, kappa))


def _accel_limit(v: float, a_accel_max: float) -> float:
    """Acceleration the plant delivers at speed `v` (utils.py accl_constraints)."""
    return min(a_accel_max, _VP.a_max * _VP.v_switch / max(v, POWER_SPEED_FLOOR))


def check_gg_floor(floor: float) -> None:
    """Raise ValueError unless `floor` is in (0, 1]; also refuses NaN."""
    if not (isinstance(floor, (int, float)) and 0.0 < floor <= 1.0):
        raise ValueError(f'gg_floor {floor!r} must be in (0, 1]')


def gg_decel(v, kappa, budget, a_brake: float, floor: float):
    """[m/s^2] Braking left at speed `v` where the corner already uses its slip budget.

    The judged plant's yaw dynamics go unstable under braking (load transfer
    unloads the rear axle), so deceleration and cornering share one ellipse:

        a = a_brake * max(floor, 1 - r^2)^(1/2),   r = min(v^2 kappa / budget, 1)

    A straight keeps all of `a_brake`; a corner held at its budget keeps only
    `sqrt(floor)` of it. Works on scalars and arrays.
    """
    r = np.minimum(np.asarray(v) ** 2 * np.abs(kappa) / budget, 1.0)
    return a_brake * np.maximum(floor, 1.0 - r ** GG_EXPONENT) ** (1.0 / GG_EXPONENT)


def _gg_rate(v_next: float, ds: float, a_brake: float, kappa: float, budget: float,
             floor: float) -> float:
    """[m/s^2] Braking into a point, from the share of the ellipse its entry speed uses."""
    a = a_brake
    for _ in range(GG_ITERATIONS):
        v_here = math.sqrt(v_next ** 2 + 2.0 * a * ds)
        a = float(gg_decel(v_here, kappa, budget, a_brake, floor))
    return a


def _brake_pass(v: np.ndarray, ds: np.ndarray, a_brake: float, closed: bool,
                gg: Optional[tuple] = None) -> None:
    """Lower `v` in place so each point can be braked to from the one before it.

    `gg` is (kappa, budget, floor) per point to brake inside the ellipse of
    `gg_decel`; None brakes at `a_brake` everywhere.
    """
    n = len(v)
    first = n - 1 if closed else n - 2
    for step in range(PASS_LAPS * n if closed else n - 1):
        i = (first - step) % n
        j = (i + 1) % n
        a = a_brake if gg is None else _gg_rate(v[j], ds[i], a_brake, gg[0][i], gg[1][i], gg[2])
        v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2.0 * a * ds[i]))


def _accel_pass(v: np.ndarray, ds: np.ndarray, a_accel_max: float, closed: bool) -> None:
    """Lower `v` in place so each point is reachable from the one before it."""
    n = len(v)
    for step in range(PASS_LAPS * n if closed else n - 1):
        i = step % n
        j = (i + 1) % n
        reach = np.sqrt(v[i] ** 2 + 2.0 * _accel_limit(v[i], a_accel_max) * ds[i])
        v[j] = min(v[j], reach)


def _check_sector(s_start: float, s_end: float, scale: float) -> None:
    if not (np.isfinite(s_start) and np.isfinite(s_end) and 0.0 < scale <= 1.0):
        raise ValueError(f'sector ({s_start}, {s_end}, {scale}): bounds must be '
                         'finite and the scale in (0, 1]')


def sector_triples(flat: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """A flat parameter list [s_start, s_end, scale, ...] as validated sectors.

    ROS parameters are flat lists of doubles, so the node stores sectors that way.
    """
    if len(flat) % 3:
        raise ValueError(f'sector_scale has {len(flat)} numbers; it needs whole '
                         '(s_start, s_end, scale) sectors')
    sectors = tuple(tuple(float(x) for x in flat[i:i + 3]) for i in range(0, len(flat), 3))
    for sector in sectors:
        _check_sector(*sector)
    return sectors


def slip_budget(path: np.ndarray, a_lat_max: float,
                sectors: Sequence[Sequence[float]] = (), closed: bool = True) -> np.ndarray:
    """[m/s^2] The slip budget per waypoint: `a_lat_max`, lowered in sectors.

    Each sector is (s_start, s_end, scale): waypoints whose arc length from
    waypoint 0 lies in [s_start, s_end) get `scale * a_lat_max`. A sector with
    s_start > s_end wraps through the start line. Where sectors overlap, the
    lower scale wins. A scale outside (0, 1] is refused: sectors only back off.
    """
    pts = np.asarray(path, dtype=np.float64)[:, :2]
    scale = np.ones(len(pts))
    if len(sectors):
        # Open: n-1 segments, so the last point's arc is the whole length.
        s = np.concatenate([[0.0], np.cumsum(segment_lengths(pts, closed))])[:len(pts)]
        for s_start, s_end, c in sectors:
            _check_sector(s_start, s_end, c)
            inside = ((s >= s_start) & (s < s_end) if s_start <= s_end
                      else (s >= s_start) | (s < s_end))
            scale = np.where(inside, np.minimum(scale, c), scale)
    return float(a_lat_max) * scale


def speed_profile(path: np.ndarray, *, a_lat_max: float, lock_margin: float,
                  a_accel_max: float, a_brake_max: float, v_max: float,
                  kappa: Optional[np.ndarray] = None, closed: bool = True,
                  v_start: float = 0.0,
                  sector_scale: Sequence[Sequence[float]] = (),
                  gg_brake: bool = False, gg_floor: float = GG_FLOOR) -> np.ndarray:
    """Speed per waypoint, shape (N,).

    Args:
        path: (N, 2+) waypoints. When `closed`, point N-1 joins point 0.
        a_lat_max: slip budget [m/s^2], the cornering limit before the lock.
        lock_margin: fraction of the steering lock `v_lock` may use (< 1).
        a_accel_max, a_brake_max: longitudinal limits [m/s^2]; the plant's own
            ceiling is 9.51, and above 7.319 m/s acceleration falls as 1/v.
        v_max: speed ceiling [m/s].
        kappa: curvature per point; computed from `path` when omitted.
        closed: a lap (wrapped passes) rather than a one-way path.
        v_start: speed at point 0 of an open path (a standing start).
        sector_scale: (s_start, s_end, scale) sectors that lower `a_lat_max`
            (see `slip_budget`); empty keeps it uniform.
        gg_brake: brake inside the ellipse of `gg_decel`, so the deceleration
            into a corner shrinks with the slip budget the corner already uses.
            Off reproduces the plain profile bit for bit.
        gg_floor: braking never drops below `a_brake_max` * sqrt(gg_floor), in (0, 1].
    """
    check_gg_floor(gg_floor)
    pts = np.asarray(path, dtype=np.float64)[:, :2]
    n = len(pts)
    if n < 3:
        return np.zeros(n)
    k = closed_curvature(pts, closed=closed) if kappa is None else np.asarray(kappa, float)
    k = np.maximum(np.abs(k), KAPPA_FLOOR)

    budget = slip_budget(pts, a_lat_max, sector_scale, closed)
    v = np.minimum(np.sqrt(budget / k), lock_speed(k, lock_margin))
    v = np.minimum(v, v_max)
    if not closed:
        v[0] = min(v[0], v_start)

    ds = segment_lengths(pts, closed)
    _brake_pass(v, ds, a_brake_max, closed, (k, budget, gg_floor) if gg_brake else None)
    _accel_pass(v, ds, a_accel_max, closed)
    return v


def ideal_lap_time(path: np.ndarray, v: np.ndarray, closed: bool = True) -> float:
    """Time to drive `path` at profile `v`: the sum of ds over the mean segment speed."""
    pts = np.asarray(path, dtype=np.float64)[:, :2]
    ds = segment_lengths(pts, closed)
    v_next = np.roll(v, -1) if closed else v[1:]
    v_here = v if closed else v[:-1]
    return float(np.sum(2.0 * ds / np.maximum(v_here + v_next, 1e-6)))
