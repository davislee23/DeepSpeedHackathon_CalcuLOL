#!/usr/bin/env python3
"""What lap time does a path physically allow? A table over the slip budget.

For each `a_lat_max` it builds the closed-loop speed profile and prints the
ideal lap time (sum of ds over segment speed) and the rear slip and body slip
the profile implies at its worst point. It assumes a perfect car on the line, so
it is a floor: the driver will not beat it, and the gap to it is tracking loss.

    python lap_floor.py                       # icra26 centreline, 5-15 m/s^2
    python lap_floor.py --path raceline.csv --a-lat 8 10 12
"""

import argparse
import math
import os
import sys
from typing import NamedTuple, Sequence

import numpy as np

import speed_profile as sp

DEFAULT_A_LAT = (5.0, 7.5, 10.0, 12.5, 15.0)   # [m/s^2], about 5-15 deg of rear slip
DEFAULT_LOCK_MARGIN = 0.95
DEFAULT_V_MAX = 20.0                            # [m/s] the plant's own ceiling
DEFAULT_A_ACCEL = 9.51                          # [m/s^2] the plant's own ceiling
DEFAULT_A_BRAKE = 9.51                          # [m/s^2] the plant's own ceiling
CENTERLINE_NAME = 'icra26_centerline.csv'
MAPS_SEARCH_DEPTH = 8


class FloorRow(NamedTuple):
    a_lat_max: float
    lap_time: float         # [s]
    mean_speed: float       # [m/s]
    peak_speed: float       # [m/s]
    peak_rear_slip: float   # [rad]
    peak_body_slip: float   # [rad]
    lock_bound: float       # fraction of points the steering lock limits


def default_centerline_path() -> str:
    """The shipped centreline, found by walking up to the repo's maps/."""
    here = os.path.dirname(os.path.abspath(__file__))
    node = here
    for _ in range(MAPS_SEARCH_DEPTH):
        node = os.path.dirname(node)
        candidate = os.path.join(node, 'maps', CENTERLINE_NAME)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'{CENTERLINE_NAME} not found above {here}')


def load_path(csv_path: str) -> np.ndarray:
    """(N, 2) waypoints from a commented CSV whose first columns are x, y."""
    return np.atleast_2d(np.loadtxt(csv_path, delimiter=',', comments='#'))[:, :2]


def floor_table(path: np.ndarray, a_lat_values: Sequence[float] = DEFAULT_A_LAT, *,
                lock_margin: float = DEFAULT_LOCK_MARGIN, v_max: float = DEFAULT_V_MAX,
                a_accel_max: float = DEFAULT_A_ACCEL,
                a_brake_max: float = DEFAULT_A_BRAKE, gg_brake: bool = False,
                gg_floor: float = sp.GG_FLOOR) -> list[FloorRow]:
    """One `FloorRow` per slip budget, on a closed `path`."""
    kappa = sp.closed_curvature(path)
    lock = sp.lock_speed(kappa, lock_margin)
    length = float(np.sum(sp.segment_lengths(path, True)))
    rows = []
    for a_lat in a_lat_values:
        v = sp.speed_profile(path, a_lat_max=a_lat, lock_margin=lock_margin,
                             a_accel_max=a_accel_max, a_brake_max=a_brake_max,
                             v_max=v_max, kappa=kappa, gg_brake=gg_brake,
                             gg_floor=gg_floor)
        lap = sp.ideal_lap_time(path, v)
        slip_limit = np.sqrt(a_lat / np.maximum(kappa, sp.KAPPA_FLOOR))
        rows.append(FloorRow(
            a_lat_max=float(a_lat), lap_time=lap, mean_speed=length / lap,
            peak_speed=float(v.max()),
            peak_rear_slip=float(sp.rear_slip(v, kappa).max()),
            peak_body_slip=float(sp.body_slip(v, kappa).max()),
            lock_bound=float(np.mean(lock < np.minimum(slip_limit, v_max)))))
    return rows


def format_table(rows: Sequence[FloorRow], length: float) -> str:
    head = (f'{"a_lat_max":>10} {"lap [s]":>8} {"mean v":>7} {"peak v":>7} '
            f'{"rear slip":>10} {"body beta":>10} {"lock-bound":>11}')
    lines = [f'path length {length:.2f} m', head, '-' * len(head)]
    for r in rows:
        lines.append(
            f'{r.a_lat_max:10.1f} {r.lap_time:8.2f} {r.mean_speed:7.2f} {r.peak_speed:7.2f} '
            f'{math.degrees(r.peak_rear_slip):9.1f}d {math.degrees(r.peak_body_slip):9.1f}d '
            f'{100.0 * r.lock_bound:10.1f}%')
    return '\n'.join(lines)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--path', help='CSV of x,y waypoints (default: the icra26 centreline)')
    parser.add_argument('--a-lat', type=float, nargs='+', default=list(DEFAULT_A_LAT),
                        help='slip budgets a_lat_max [m/s^2]')
    parser.add_argument('--lock-margin', type=float, default=DEFAULT_LOCK_MARGIN)
    parser.add_argument('--v-max', type=float, default=DEFAULT_V_MAX)
    parser.add_argument('--a-accel', type=float, default=DEFAULT_A_ACCEL)
    parser.add_argument('--a-brake', type=float, default=DEFAULT_A_BRAKE)
    parser.add_argument('--gg', action='store_true',
                        help='brake inside the combined-accel ellipse (speed_profile.gg_decel)')
    parser.add_argument('--gg-floor', type=float, default=sp.GG_FLOOR,
                        help='braking never drops below --a-brake*sqrt(this), in (0, 1]')
    args = parser.parse_args(argv)

    path = load_path(args.path or default_centerline_path())
    rows = floor_table(path, args.a_lat, lock_margin=args.lock_margin, v_max=args.v_max,
                       a_accel_max=args.a_accel, a_brake_max=args.a_brake,
                       gg_brake=args.gg, gg_floor=args.gg_floor)
    length = float(np.sum(sp.segment_lengths(path, True)))
    print(format_table(rows, length))
    return 0


if __name__ == '__main__':
    sys.exit(main())
