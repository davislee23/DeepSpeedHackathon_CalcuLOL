#!/usr/bin/env python3
"""A minimum-curvature racing line, built from the occupancy grid at runtime.

Pipeline (Heilmeier et al. 2019, docs/04 section 4.6):

1. Load `maps/<track>.yaml` + pgm, read-only.
2. Seal the cone rows. A cone row is a line of small islands to the grid, ~0.7-1.0 m
   apart centre to centre, with drivable gaps of up to ~0.55 m. Neighbouring
   cones are joined by a drawn barrier (the idea behind `track_tool --link-cones`,
   reimplemented here because `scripts/` is not a runtime dependency), then a
   morphological closing of ~0.3 m radius fills what is left. No line can thread
   a row (rule 11); a corridor wider than twice the radius stays open.
3. Per point of the centreline, find how far the car may move along the normal
   before its centre is closer to a wall than `half_width + margin`.
4. Minimise the sum of squared second differences of the path over the lateral
   offsets, subject to those bounds. The path is linear in the offsets, so this
   is a box-bounded linear least-squares problem: `scipy.optimize.lsq_linear`.
5. Re-linearise (resample, recompute the normals and bounds), because second
   differences only approximate curvature on an even spacing. Each round weights
   the rows by the previous round's curvature (`KAPPA_REF`), so that a tight apex
   is not left sharper than the steering can follow; if that solve fails, the
   plain two-or-three-round solve takes over.
6. Offline only (`--refine`, ~60 s): refine that line for lap time. Lateral offsets at
   `REFINE_CONTROL_POINTS` points, spline-interpolated, minimise the ideal lap of the
   speed profile (`REFINE_PROFILE`) inside the same clearance bounds. Minimum
   curvature is only a proxy for lap time; this trades it where the profile says
   the time is. The result ships as `config/icra26_raceline.csv`, which the driver
   checks against the map and drives first. Regenerate it with OMP_NUM_THREADS=1:
   threaded BLAS moves the QP line by ~1e-6 m, which the search turns into ~9 cm.

Pure numpy / scipy, no ROS. Only reads files it is told to read.
"""

import argparse
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np
from scipy import ndimage, sparse
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear, minimize

import speed_profile as sp

HALF_WIDTH = 0.155            # [m] half the car's width (VehicleParameters.width / 2)
DEFAULT_MARGIN = 0.10         # [m] extra clearance from a sealed wall
SEAL_RADIUS = 0.3             # [m] closing radius; seals gaps up to twice this
CONE_LINK_DISTANCE = 1.1      # [m] cones closer than this are joined (between row spacing and lane width)
CONE_MIN_SPAN = 0.1           # [m] an obstacle this small is noise, not a cone
CONE_MAX_SPAN = 0.9           # [m] one this large is a wall
CONE_MAX_CELLS = 400
OPT_SPACING = 0.2             # [m] spacing the QP is solved on
OUTPUT_SPACING = 0.1          # [m] spacing of the delivered line
ITERATIONS = 3                # re-linearisations of the QP
MARCH_STEP_CELLS = 0.5        # march along a normal in half-cell steps
MAX_OFFSET = 3.0              # [m] no lateral bound is searched past this
SAFETY_SLACKS = (0.02, 0.05, 0.10)   # [m] extra clearance aimed for, widened if the line falls short
SOLVER_MAX_ITER = 30
REWEIGHT_ITERATIONS = 6       # re-linearisations of the curvature-reweighted QP
KAPPA_REF = 0.6               # [1/m] rows where the line is tighter than this are weighted up
REWEIGHT_MAX_ITER = 200       # the weighted operator is worse conditioned and needs more steps
MIN_CACHED_POINTS = 50
MAX_CLOSING_GAP = 1.0         # [m] a cached lap must close to within this
CLEARANCE_EPS = 1e-3          # [m] tolerance on the delivered clearance
REFINE_CONTROL_POINTS = 50    # lateral offsets the lap-time refinement moves
REFINE_ROUNDS = 3             # re-linearisations: normals and bounds round the new line
REFINE_MAX_ITER = 100
REFINE_FD_STEP = 1e-3         # [m] finite-difference step of the lap-time gradient
REFINE_BOUND_SLACK = 0.02     # [m] past `required` for the control points' box bounds
REFINE_PENALTY = 1e3          # [s/m^2] on the squared clearance shortfall between them
REFINE_PENALTY_SLACK = 0.005  # [m] past `required` where that penalty starts
# The profile the shipped line is refined against: the sub-10 tune (a_lat 22 with the
# g-g brake limit, and its one kink sector). Fixed here, not read from the driver's
# config, so the CSV stays reproducible from the map when the driver's sectors are
# later re-tuned to this line.
REFINE_PROFILE = dict(a_lat_max=22.0, lock_margin=0.95, a_accel_max=9.51, a_brake_max=6.0,
                      v_max=10.0, sector_scale=((43.0, 47.0, 0.85),), gg_brake=True)
MIN_CURVATURE_METHOD = 'min-curvature'
MIN_TIME_METHOD = 'min-time'


@dataclass(frozen=True)
class OccupancyGrid:
    """Boolean occupancy in image order (row 0 is the top), with its geometry."""

    occupied: np.ndarray
    resolution: float
    origin: tuple

    def world_to_cell(self, xy: np.ndarray):
        """(rows, cols) of the cells holding the (N, 2) world points."""
        xy = np.atleast_2d(xy)
        cols = np.floor((xy[:, 0] - self.origin[0]) / self.resolution).astype(int)
        rows = self.occupied.shape[0] - 1 - np.floor(
            (xy[:, 1] - self.origin[1]) / self.resolution).astype(int)
        return rows, cols

    def cell_to_world(self, rows, cols) -> np.ndarray:
        """World coordinates of the centres of the given cells, shape (N, 2)."""
        rows, cols = np.asarray(rows), np.asarray(cols)
        x = self.origin[0] + (cols + 0.5) * self.resolution
        y = self.origin[1] + (self.occupied.shape[0] - 1 - rows + 0.5) * self.resolution
        return np.column_stack([x, y])

    def fractional_cells(self, xy: np.ndarray):
        """Row / column in cell-centre coordinates, for interpolation."""
        col = (xy[:, 0] - self.origin[0]) / self.resolution - 0.5
        row = self.occupied.shape[0] - 1 - ((xy[:, 1] - self.origin[1]) / self.resolution - 0.5)
        return row, col


class RacelineError(RuntimeError):
    """The QP did not converge, or the line could not keep its clearance. The
    reweighted build falls back to the uniform solve on this and on nothing else."""


class Raceline(NamedTuple):
    xy: np.ndarray            # (N, 2) the line, at OUTPUT_SPACING
    sealed: OccupancyGrid     # the grid the line was bounded by
    build_seconds: float
    reweighted: bool = False  # False: the uniform solve (asked for, or the fallback)


# ----------------------------------------------------------------------------
# Map loading
# ----------------------------------------------------------------------------
def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf'^\s*{key}\s*:\s*(.+?)\s*(?:#.*)?$', text, re.MULTILINE)
    if match is None:
        raise ValueError(f'map yaml has no `{key}`')
    return match.group(1).strip().strip('\'"')


def _read_pgm(path: str) -> np.ndarray:
    """8-bit binary (P5) PGM as a (rows, cols) uint8 array."""
    with open(path, 'rb') as handle:
        data = handle.read()
    if data[:2] != b'P5':
        raise ValueError(f'{path}: only binary P5 PGM is supported')
    fields, pos = [], 2
    while len(fields) < 3:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            while data[pos:pos + 1] != b'\n':
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    width, height, maxval = fields
    if maxval > 255:
        raise ValueError(f'{path}: only 8-bit PGM is supported')
    pos += 1
    return np.frombuffer(data, dtype=np.uint8, count=width * height, offset=pos
                         ).reshape(height, width)


def load_occupancy(yaml_path: str) -> OccupancyGrid:
    """The occupancy grid a map yaml describes. Read-only."""
    with open(yaml_path) as handle:
        text = handle.read()
    image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)),
                         _yaml_scalar(text, 'image'))
    resolution = float(_yaml_scalar(text, 'resolution'))
    origin = [float(v) for v in re.search(
        r'^\s*origin\s*:\s*\[([^\]]+)\]', text, re.MULTILINE).group(1).split(',')]
    negate = int(_yaml_scalar(text, 'negate'))
    threshold = float(_yaml_scalar(text, 'occupied_thresh'))
    pixels = _read_pgm(image).astype(np.float64)
    probability = pixels / 255.0 if negate else (255.0 - pixels) / 255.0
    return OccupancyGrid(occupied=probability > threshold, resolution=resolution,
                         origin=(origin[0], origin[1]))


# ----------------------------------------------------------------------------
# Sealing and clearance
# ----------------------------------------------------------------------------
def _disk(radius_cells: int) -> np.ndarray:
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    return (x * x + y * y) <= radius_cells * radius_cells


def _cone_centres(grid: OccupancyGrid) -> np.ndarray:
    """(N, 2) world centres of the small free-standing obstacles, i.e. the cones."""
    labels, count = ndimage.label(grid.occupied, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros((0, 2))
    sizes = ndimage.sum(grid.occupied, labels, np.arange(1, count + 1))
    rows, cols = [], []
    for box, size in zip(ndimage.find_objects(labels), sizes):
        height, width = box[0].stop - box[0].start, box[1].stop - box[1].start
        span = max(height, width) * grid.resolution
        if size > CONE_MAX_CELLS or not CONE_MIN_SPAN < span < CONE_MAX_SPAN:
            continue
        rows.append((box[0].start + box[0].stop - 1) / 2.0)
        cols.append((box[1].start + box[1].stop - 1) / 2.0)
    return grid.cell_to_world(np.array(rows), np.array(cols)) if rows else np.zeros((0, 2))


def cone_links(grid: OccupancyGrid, max_distance: float = CONE_LINK_DISTANCE) -> np.ndarray:
    """(L, 2, 2) segments joining every pair of cones closer than `max_distance`."""
    centres = _cone_centres(grid)
    segments = [(a, b) for i, a in enumerate(centres) for b in centres[i + 1:]
                if np.linalg.norm(a - b) <= max_distance]
    return np.array(segments).reshape(-1, 2, 2)


def _draw_barriers(grid: OccupancyGrid, segments: np.ndarray) -> OccupancyGrid:
    """The grid with each segment drawn three cells thick."""
    occupied = grid.occupied.copy()
    steps = max(2, int(np.ceil(CONE_LINK_DISTANCE / (grid.resolution / 2.0))))
    t = np.linspace(0.0, 1.0, steps)[:, None]
    for a, b in segments:
        rows, cols = grid.world_to_cell(a + t * (b - a))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = rows + dr, cols + dc
                ok = (r >= 0) & (r < occupied.shape[0]) & (c >= 0) & (c < occupied.shape[1])
                occupied[r[ok], c[ok]] = True
    return OccupancyGrid(occupied=occupied, resolution=grid.resolution, origin=grid.origin)


def seal_cones(grid: OccupancyGrid, radius: float = SEAL_RADIUS) -> OccupancyGrid:
    """The grid with the cone rows sealed: neighbouring cones linked, then every
    obstacle gap narrower than `2 * radius` filled in."""
    linked = _draw_barriers(grid, cone_links(grid))
    cells = int(round(radius / grid.resolution))
    if cells < 1:
        return linked
    pad = cells + 2
    closed = ndimage.binary_closing(np.pad(linked.occupied, pad, mode='edge'),
                                    structure=_disk(cells))[pad:-pad, pad:-pad]
    return OccupancyGrid(occupied=closed | linked.occupied, resolution=grid.resolution,
                         origin=grid.origin)


def _clearance_map(grid: OccupancyGrid) -> np.ndarray:
    """Distance in metres from every cell centre to the nearest occupied cell."""
    return ndimage.distance_transform_edt(~grid.occupied) * grid.resolution


def _sample(values: np.ndarray, grid: OccupancyGrid, xy: np.ndarray) -> np.ndarray:
    row, col = grid.fractional_cells(xy)
    return ndimage.map_coordinates(values, [row, col], order=1, mode='nearest')


def wall_clearance(grid: OccupancyGrid, xy: np.ndarray) -> np.ndarray:
    """Distance from each (N, 2) point to the nearest occupied cell, in metres."""
    return _sample(_clearance_map(grid), grid, np.atleast_2d(xy))


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Evenly spaced points round a closed polyline, starting at the first point."""
    pts = np.asarray(points, dtype=np.float64)[:, :2]
    loop = np.vstack([pts, pts[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    count = max(3, int(round(arc[-1] / spacing)))
    s = np.linspace(0.0, arc[-1], count, endpoint=False)
    return np.column_stack([np.interp(s, arc, loop[:, 0]), np.interp(s, arc, loop[:, 1])])


def _left_normals(points: np.ndarray) -> np.ndarray:
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _offset_bounds(grid: OccupancyGrid, clearance_map: np.ndarray, ref: np.ndarray,
                   normals: np.ndarray, required: float):
    """How far left / right of `ref` the car's centre may go, keeping `required`.

    Marches each normal outward until the centre would sit closer than
    `required` to a wall. A reference point already too close gets a zero bound
    on that side, never a negative one.
    """
    step = grid.resolution * MARCH_STEP_CELLS
    distances = np.arange(0.0, MAX_OFFSET + step, step)
    bounds = []
    for sign in (1.0, -1.0):
        pos = ref[:, None, :] + sign * distances[None, :, None] * normals[:, None, :]
        clear = _sample(clearance_map, grid, pos.reshape(-1, 2)).reshape(
            len(ref), len(distances))
        blocked = clear < required
        blocked[:, 0] = False
        first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), len(distances))
        bounds.append(distances[np.maximum(first - 1, 0)])
    return -bounds[1], bounds[0]                # (lower, upper) lateral offsets


def _second_difference(n: int) -> sparse.csr_matrix:
    """Cyclic (p[i-1] - 2 p[i] + p[i+1]) as an n x n sparse operator."""
    rows = np.arange(n)
    ones = np.ones(n)
    up = sparse.csr_matrix((ones, (rows, (rows + 1) % n)), shape=(n, n))
    down = sparse.csr_matrix((ones, (rows, (rows - 1) % n)), shape=(n, n))
    return (up + down - 2.0 * sparse.identity(n, format='csr')).tocsr()


def _min_curvature_offsets(ref: np.ndarray, normals: np.ndarray,
                           lower: np.ndarray, upper: np.ndarray, *,
                           weights: Optional[np.ndarray] = None,
                           max_iter: int = SOLVER_MAX_ITER) -> np.ndarray:
    """Lateral offsets minimising the squared second differences of ref + a * n.

    `weights` (one positive value per point) multiplies that point's second
    difference before it is squared, so its share of the sum scales as w squared
    and a stretch weighted up is straightened at the expense of the rest. None,
    or all ones, is the plain minimum-curvature solve.
    """
    n = len(ref)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(f'weights must be {n} finite positive values')
    d2 = _second_difference(n)
    a = sparse.vstack([d2 @ sparse.diags(normals[:, 0]),
                       d2 @ sparse.diags(normals[:, 1])]).tocsr()
    b = -np.concatenate([d2 @ ref[:, 0], d2 @ ref[:, 1]])
    if weights is not None:
        rows = np.concatenate([weights, weights])
        a = (sparse.diags(rows) @ a).tocsr()
        b = rows * b
    upper = np.maximum(upper, lower + 1e-9)
    # Dense exact trust-region steps: a few hundred unknowns solve in ~0.5 s.
    # `bvls` took 16 s per solve here and the iterative sparse solver 4 s; both
    # crawl on this ill-conditioned operator.
    result = lsq_linear(a.toarray(), b, bounds=(lower, upper), method='trf',
                        lsq_solver='exact', max_iter=max_iter)
    if result.status <= 0:
        raise RacelineError(f'raceline QP did not converge: {result.message}')
    return np.clip(result.x, lower, upper)


# ----------------------------------------------------------------------------
# The line
# ----------------------------------------------------------------------------
def _relinearise(centerline: np.ndarray, sealed: OccupancyGrid, clearance_map: np.ndarray,
                 required: float, *, opt_spacing: float, iterations: int,
                 kappa_ref: Optional[float]) -> np.ndarray:
    """`iterations` rounds of resample, bound, solve, on the QP's own spacing.

    With `kappa_ref`, each round weights a point's second difference by
    `max(1, kappa / kappa_ref)`, kappa being the previous round's curvature
    there. Plain minimum curvature spreads the turn evenly in the sum of
    squares, which leaves one tight apex sharper than the steering can follow;
    the weights trade a little length elsewhere for a gentler apex.
    """
    ref = resample_closed(centerline, opt_spacing)
    weights = None if kappa_ref is None else np.ones(len(ref))
    for _ in range(iterations):
        normals = _left_normals(ref)
        lower, upper = _offset_bounds(sealed, clearance_map, ref, normals, required)
        if weights is None:
            offsets = _min_curvature_offsets(ref, normals, lower, upper)
        else:
            offsets = _min_curvature_offsets(ref, normals, lower, upper, weights=weights,
                                             max_iter=REWEIGHT_MAX_ITER)
        ref = resample_closed(ref + offsets[:, None] * normals, opt_spacing)
        if kappa_ref is not None:
            weights = np.maximum(1.0, sp.closed_curvature(ref) / kappa_ref)
    return ref


def _solve_line(centerline: np.ndarray, sealed: OccupancyGrid, clearance_map: np.ndarray,
                required: float, *, opt_spacing: float, spacing: float, iterations: int,
                kappa_ref: Optional[float]) -> np.ndarray:
    """The delivered line, at least `required` from every wall."""
    # Resampling and re-interpolating shave a few millimetres off the clearance
    # the bounds guaranteed, so aim a little past it and widen the slack if the
    # delivered line still comes up short.
    for slack in SAFETY_SLACKS:
        ref = _relinearise(centerline, sealed, clearance_map, required + slack,
                           opt_spacing=opt_spacing, iterations=iterations, kappa_ref=kappa_ref)
        line = resample_closed(ref, spacing)
        if float(_sample(clearance_map, sealed, line).min()) >= required - CLEARANCE_EPS:
            return line
    raise RacelineError(f'raceline keeps closer than {required:.3f} m to a wall '
                        f'even with {SAFETY_SLACKS[-1]} m of slack')


def build_raceline(centerline: np.ndarray, grid: OccupancyGrid, *,
                   margin: float = DEFAULT_MARGIN, half_width: float = HALF_WIDTH,
                   seal_radius: float = SEAL_RADIUS, opt_spacing: float = OPT_SPACING,
                   spacing: float = OUTPUT_SPACING, iterations: int = ITERATIONS,
                   reweight: bool = True, reweight_iterations: int = REWEIGHT_ITERATIONS,
                   kappa_ref: float = KAPPA_REF) -> Raceline:
    """Minimum-curvature line inside the sealed track, at least `half_width + margin`
    from every wall. `centerline` is a closed lap; its first point stays the first.

    `reweight` straightens the tightest corners (see `_relinearise`). If that solve
    fails to converge, or cannot keep the clearance, the plain solve of `iterations`
    rounds is used instead and `Raceline.reweighted` says so: a failed build must
    not drop the car to the centreline at startup. A build that falls back pays for
    both attempts, several times the normal ~3 s.
    """
    if kappa_ref <= 0.0 or reweight_iterations < 1:
        raise ValueError(f'kappa_ref ({kappa_ref}) must be positive and '
                         f'reweight_iterations ({reweight_iterations}) at least 1')
    started = time.perf_counter()
    sealed = seal_cones(grid, seal_radius)
    clearance_map = _clearance_map(sealed)
    common = dict(opt_spacing=opt_spacing, spacing=spacing)
    required = half_width + margin
    if reweight:
        try:
            line = _solve_line(centerline, sealed, clearance_map, required, **common,
                               iterations=reweight_iterations, kappa_ref=kappa_ref)
            return Raceline(line, sealed, time.perf_counter() - started, reweighted=True)
        except RacelineError as error:
            warnings.warn(f'reweighted raceline failed ({error}); using the uniform solve',
                          RuntimeWarning, stacklevel=2)
    line = _solve_line(centerline, sealed, clearance_map, required, **common,
                       iterations=iterations, kappa_ref=None)
    return Raceline(line, sealed, time.perf_counter() - started, reweighted=False)


def _control_bounds(lower: np.ndarray, upper: np.ndarray, count: int):
    """(centres, lower, upper) for `count` evenly spaced control points.

    Each control point takes the tightest bound within half an interval either
    side, so the spline between two points starts out inside the walls; the
    clearance penalty catches what overshoot is left.
    """
    n = len(lower)
    centres = np.linspace(0.0, n, count, endpoint=False)
    half = n / count / 2.0
    windows = [np.arange(int(np.floor(c - half)), int(np.ceil(c + half))) % n for c in centres]
    # lower <= 0 <= upper everywhere (`_offset_bounds`), so each box holds 0.
    return (centres, np.array([lower[w].max() for w in windows]),
            np.array([upper[w].min() for w in windows]))


def _refine_round(line: np.ndarray, sealed: OccupancyGrid, clearance_map: np.ndarray,
                  required: float, profile: dict, control_points: int,
                  max_iter: int) -> np.ndarray:
    """One round: offsets along this line's normals that minimise the ideal lap."""
    n = len(line)
    normals = _left_normals(line)
    lower, upper = _offset_bounds(sealed, clearance_map, line, normals,
                                  required + REFINE_BOUND_SLACK)
    centres, lo, hi = _control_bounds(lower, upper, control_points)
    knots, at = np.append(centres, n), np.arange(n)

    def shaped(u):
        offsets = CubicSpline(knots, np.append(u, u[0]), bc_type='periodic')(at)
        return line + offsets[:, None] * normals

    def cost(u):
        xy = shaped(u)
        short = np.maximum(0.0, required + REFINE_PENALTY_SLACK
                           - _sample(clearance_map, sealed, xy))
        return (sp.ideal_lap_time(xy, sp.speed_profile(xy, **profile))
                + REFINE_PENALTY * float(np.sum(short ** 2)))

    # The profile is piecewise (min of limits, wrapped passes), so the gradient is a
    # finite difference; L-BFGS-B keeps every control point inside its box.
    result = minimize(cost, np.zeros(control_points), method='L-BFGS-B',
                      bounds=list(zip(lo, hi)),
                      options=dict(maxiter=max_iter, eps=REFINE_FD_STEP))
    return resample_closed(shaped(result.x), OUTPUT_SPACING)


def refine_min_time(xy: np.ndarray, sealed: OccupancyGrid, required: float, *,
                    profile: Optional[dict] = None,
                    control_points: int = REFINE_CONTROL_POINTS, rounds: int = REFINE_ROUNDS,
                    max_iter: int = REFINE_MAX_ITER) -> np.ndarray:
    """`xy` (at OUTPUT_SPACING) moved sideways to cut the ideal lap at `profile`.

    Every point stays at least `required` from the sealed walls, or RacelineError.
    `profile` is `speed_profile` keyword arguments, `REFINE_PROFILE` by default.
    Offline: the shipped settings take ~60 s, too long for the 60 s startup budget.
    """
    profile = REFINE_PROFILE if profile is None else profile
    clearance_map = _clearance_map(sealed)
    start = np.asarray(xy, dtype=np.float64)[:, :2]
    line = start
    for _ in range(rounds):
        line = _refine_round(line, sealed, clearance_map, required, profile,
                             control_points, max_iter)
    closest = float(_sample(clearance_map, sealed, line).min())
    if closest < required - CLEARANCE_EPS:
        raise RacelineError(f'refined line comes within {closest:.3f} m of a wall, '
                            f'{required:.3f} m required')
    # The finite-difference search can stop early on this piecewise cost; a line
    # that gained nothing is not worth shipping over the one it started from.
    before, after = (sp.ideal_lap_time(p, sp.speed_profile(p, **profile)) for p in (start, line))
    if after >= before:
        raise RacelineError(f'refinement did not shorten the ideal lap ({before:.3f} -> '
                            f'{after:.3f} s)')
    return line


def raceline_table(xy: np.ndarray, *, a_lat_max: float, lock_margin: float,
                   a_accel_max: float, a_brake_max: float, v_max: float) -> np.ndarray:
    """(N, 7): s, x, y, psi, kappa (signed, left positive), vx, ax along the line."""
    pts = np.asarray(xy, dtype=np.float64)[:, :2]
    ds = sp.segment_lengths(pts, True)
    s = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    psi = np.arctan2(tangent[:, 1], tangent[:, 0])
    kappa = sp.closed_curvature(pts, signed=True)
    vx = sp.speed_profile(pts, a_lat_max=a_lat_max, lock_margin=lock_margin,
                          a_accel_max=a_accel_max, a_brake_max=a_brake_max, v_max=v_max,
                          kappa=kappa)
    ax = (np.roll(vx, -1) ** 2 - vx ** 2) / (2.0 * ds)
    return np.column_stack([s, pts[:, 0], pts[:, 1], psi, kappa, vx, ax])


def read_cached(csv_path: str):
    """(xy, margin) from a cached line, or ValueError if it is not a usable lap."""
    with open(csv_path) as handle:
        margin = next((float(m.group(1)) for line in handle
                       if (m := re.match(r'#\s*margin_m:\s*([0-9.eE+-]+)', line))), None)
    xy = np.atleast_2d(np.loadtxt(csv_path, delimiter=',', comments='#'))
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < MIN_CACHED_POINTS:
        raise ValueError(f'{csv_path}: expected at least {MIN_CACHED_POINTS} x,y rows')
    xy = xy[:, :2]
    if not np.all(np.isfinite(xy)):
        raise ValueError(f'{csv_path}: non-finite coordinates')
    if margin is None:
        raise ValueError(f'{csv_path}: no `margin_m:` header, cannot tell how safe it is')
    if np.linalg.norm(xy[0] - xy[-1]) > MAX_CLOSING_GAP:
        raise ValueError(f'{csv_path}: not a closed lap (gap {np.linalg.norm(xy[0] - xy[-1]):.2f} m)')
    return xy, margin


def cached_method(csv_path: str) -> str:
    """How a cached line was made, from its `method:` header. Caches written before
    the header existed are the minimum-curvature line."""
    with open(csv_path) as handle:
        return next((m.group(1) for line in handle
                     if (m := re.match(r'#\s*method:\s*(\S+)', line))), MIN_CURVATURE_METHOD)


def find_map_yaml(centerline_csv: str, name: str = 'icra26.yaml') -> Optional[str]:
    """The map yaml that ships beside the centreline CSV, if it is there."""
    candidate = os.path.join(os.path.dirname(os.path.abspath(centerline_csv)), name)
    return candidate if os.path.exists(candidate) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Build the icra26 racing line from the map.')
    parser.add_argument('--centerline', help='centreline CSV (default: the shipped one)')
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN)
    parser.add_argument('--write', metavar='CSV',
                        help='write x,y to this CSV (config/icra26_raceline.csv for the driver)')
    parser.add_argument('--refine', action='store_true',
                        help=f'refine the line for lap time at REFINE_PROFILE (~60 s); the '
                             f'driver drives a {MIN_TIME_METHOD} CSV before its runtime build')
    args = parser.parse_args(argv)

    import lap_floor
    center_csv = args.centerline or lap_floor.default_centerline_path()
    yaml_path = find_map_yaml(center_csv)
    if yaml_path is None:
        print(f'no map yaml beside {center_csv}', file=sys.stderr)
        return 1
    center = lap_floor.load_path(center_csv)
    line = build_raceline(center, load_occupancy(yaml_path), margin=args.margin)

    print(f'built in {line.build_seconds:.2f} s; {len(line.xy)} points; '
          f'reweighted: {line.reweighted}')
    if args.write and not line.reweighted:
        print('refusing to write: the reweighted solve failed and this is the uniform line, '
              'which the tuned speed profile does not expect', file=sys.stderr)
        return 1
    xy, method = line.xy, MIN_CURVATURE_METHOD
    if args.refine:
        started = time.perf_counter()
        xy, method = refine_min_time(xy, line.sealed, HALF_WIDTH + args.margin), MIN_TIME_METHOD
        ideal = [sp.ideal_lap_time(p, sp.speed_profile(p, **REFINE_PROFILE)) for p in (line.xy, xy)]
        print(f'refined in {time.perf_counter() - started:.1f} s; ideal lap at REFINE_PROFILE '
              f'{ideal[0]:.3f} -> {ideal[1]:.3f} s')
    clearance = wall_clearance(line.sealed, xy)
    print(f'min wall clearance {clearance.min():.3f} m (required {HALF_WIDTH + args.margin:.3f})')
    for name, path in (('centreline', center), ('raceline', xy)):
        row = lap_floor.floor_table(path, (10.0,))[0]
        length = float(np.sum(sp.segment_lengths(path, True)))
        print(f'{name:11s} length {length:6.2f} m  floor@10 {row.lap_time:6.2f} s')
    if args.write:
        np.savetxt(args.write, xy, delimiter=',', fmt='%.4f',
                   header=f'x_m, y_m\nmargin_m: {args.margin}\nmethod: {method}\n'
                          f'{method} racing line of icra26, built from the map by '
                          'team_driver/raceline.py.')
        print(f'wrote {args.write}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
