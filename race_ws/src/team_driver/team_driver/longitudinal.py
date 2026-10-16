#!/usr/bin/env python3
"""The speed caps' geometry: the reference by arc length, and clearance along it.

The scan is a dead time old; the distance the car covers meanwhile comes from
`latch_predictor.py` and is taken off the clearance the scan saw.

The emergency speed bound looks along the planned path, not straight ahead: a
corridor one car wide toward the reference point 1-2 m ahead. In a hairpin a
straight-ahead cone sees the outside wall and caps the car although the line
turns away from it.
"""

import math
from typing import Optional

import numpy as np

from vehicle_model import VehicleParameters

# [m] base_link -> laser, gym_bridge's `lidar_base_link_to_lidar_tf` default.
LIDAR_FORWARD = 0.275
# [m] The reference point the corridor aims at: ~0.3 s ahead, within 1-2 m.
BEARING_LOOKAHEAD_MIN = 1.0
BEARING_LOOKAHEAD_MAX = 2.0
BEARING_LOOKAHEAD_TIME = 0.3      # [s]
# The near end of the corridor, not its mean: one long beam down a gap must
# not lift the bound. 'lower' takes a real return rather than interpolating
# toward the far ones, which over the ~5-15 beams in a corridor sits well
# above the short returns. Under ~11 beams that is the minimum, so a single
# short beam sets the bound: deliberately, it fails toward braking.
CLEARANCE_PERCENTILE = 10


def front_overhang(params: VehicleParameters) -> float:
    """[m] How far the body reaches ahead of the lidar.

    simulator._compute_collision_body_offset centres the body box
    `collision_body_center_x` ahead of base_link.
    """
    return float(params.collision_body_center_x + params.length / 2.0 - LIDAR_FORWARD)


def bearing_lookahead(v: float) -> float:
    """[m] Arc distance ahead of the car to aim the clearance corridor at."""
    return min(max(v * BEARING_LOOKAHEAD_TIME, BEARING_LOOKAHEAD_MIN), BEARING_LOOKAHEAD_MAX)


def path_clearance(ranges: np.ndarray, angles: np.ndarray, bearing: float,
                   half_width: float, max_range: float) -> float:
    """[m] Free distance along `bearing`, in a corridor `half_width` either side.

    Distances are measured along the bearing from the lidar. A beam at
    `max_range` has no return and blocks nothing; a zero range (closer than the
    sensor can see, and what preprocess() makes of NaN) is an obstacle at the
    sensor. A corridor no beam falls in -- including an empty scan -- is free
    to `max_range`; the profile and the MPC still bound the speed there.
    """
    rel = angles - bearing
    cos_rel = np.cos(rel)
    along = ranges * cos_rel
    inside = ((cos_rel > 0.0) & (ranges < max_range)
              & (np.abs(ranges * np.sin(rel)) <= half_width))
    if not np.any(inside):
        return float(max_range)
    return float(np.percentile(along[inside], CLEARANCE_PERCENTILE, method='lower'))


def scan_points(ranges: np.ndarray, angles: np.ndarray, lidar_xy, lidar_yaw: float,
                max_range: float) -> np.ndarray:
    """(M, 2) map-frame points of the beams that returned.

    A beam at `max_range` saw nothing; a zero range (closer than the sensor
    sees, or a NaN) carries no position, so neither is a point.
    """
    hit = (ranges > 0.0) & (ranges < max_range)
    r, a = ranges[hit], angles[hit] + lidar_yaw
    return np.column_stack([lidar_xy[0] + r * np.cos(a), lidar_xy[1] + r * np.sin(a)])


def _extend(path: np.ndarray, length: float) -> np.ndarray:
    """`path`, continued straight along its last segment to arc `length`."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    have = float(seg.sum())
    if have >= length:
        return path
    nonzero = np.nonzero(seg > 1e-9)[0]
    if len(nonzero) == 0:
        return path
    k = nonzero[-1]
    direction = (path[k + 1] - path[k]) / seg[k]
    return np.vstack([path, path[-1] + direction * (length - have)])


def clearance_along_path(points: np.ndarray, path: np.ndarray, half_width: float,
                         length: float) -> float:
    """[m] Arc length along `path` to the first point within `half_width` of it.

    The corridor follows the path round a corner, so the outside wall of a
    bend the car is taking is not in it, while a wall the plan itself runs
    into is. The path is continued straight past its end to `length`; with
    nothing in the corridor the clearance is `length`.
    """
    path = _extend(np.asarray(path, dtype=np.float64), length)
    a, b = path[:-1], path[1:]
    ab = b - a
    seg_sq = np.einsum('ij,ij->i', ab, ab)
    keep = seg_sq > 1e-18
    a, ab, seg_sq = a[keep], ab[keep], seg_sq[keep]
    if len(points) == 0 or len(a) == 0:
        return float(length)
    s0 = np.concatenate([[0.0], np.cumsum(np.sqrt(seg_sq))])[:-1]
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum('mkj,kj->mk', rel, ab) / seg_sq, 0.0, 1.0)
    gap = rel - t[..., None] * ab[None, :, :]
    inside = np.einsum('mkj,mkj->mk', gap, gap) <= half_width * half_width
    arc = np.where(inside, s0[None, :] + t * np.sqrt(seg_sq)[None, :], np.inf).min(axis=1)
    arc = arc[np.isfinite(arc)]
    if len(arc) == 0:
        return float(length)
    return float(min(np.percentile(arc, CLEARANCE_PERCENTILE, method='lower'), length))


class ClosedPath:
    """Points, and optionally a value, along a closed polyline by arc length.

    The arc is the MPC's: cumulative segment lengths, plus the closing segment
    from the last point back to the first, and positions wrap at its length.
    """

    def __init__(self, xy: np.ndarray, values: Optional[np.ndarray] = None):
        p = np.asarray(xy, dtype=np.float64)[:, :2]
        if len(p) < 2:
            raise ValueError(f'a path needs at least 2 points, got {len(p)}')
        if not np.all(np.isfinite(p)):
            raise ValueError('a path with non-finite points')
        v = None if values is None else np.asarray(values, dtype=np.float64)
        if v is not None:
            if v.shape != (len(p),):
                raise ValueError(f'values of shape {v.shape} for a path of {len(p)} points')
            if not np.all(np.isfinite(v)):
                raise ValueError('non-finite values along the path')
        closed = np.vstack([p, p[:1]])
        self._arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0),
                                                                    axis=1))])
        self.length = float(self._arc[-1])
        if not self.length > 0.0:
            raise ValueError('a path of zero length')
        self._x = closed[:, 0]
        self._y = closed[:, 1]
        # One per point plus the first again, for the closing segment.
        self._values = None if v is None else np.append(v, v[0])

    def _wrap(self, s: float) -> float:
        if not math.isfinite(s):
            raise ValueError(f'arc position {s} is not finite')
        return s % self.length

    def point_at(self, s: float) -> np.ndarray:
        """(x, y) at arc position `s`, wrapped onto the lap."""
        s = self._wrap(s)
        return np.array([np.interp(s, self._arc, self._x), np.interp(s, self._arc, self._y)])

    def points_at(self, s: np.ndarray) -> np.ndarray:
        """(n, 2) points at arc positions `s`, each wrapped onto the lap."""
        s = np.asarray(s, dtype=np.float64)
        if not np.all(np.isfinite(s)):
            raise ValueError('non-finite arc positions')
        s = s % self.length
        return np.column_stack([np.interp(s, self._arc, self._x),
                                np.interp(s, self._arc, self._y)])

    def value_at(self, s: float) -> float:
        """The value at arc position `s`, interpolated and wrapped onto the lap."""
        if self._values is None:
            raise ValueError('this path carries no values')
        return float(np.interp(self._wrap(s), self._arc, self._values))
