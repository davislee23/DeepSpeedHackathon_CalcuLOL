#!/usr/bin/env python3
"""LiDAR and wall contact for the offline plant, ROS-free.

`plant_sim.GymPlant` moves the car; this moves the sensor and the walls. Both
exist so the offline harness drives the same `DriverCore` the judge does,
against the same geometry -- `lap_sim.py` today has no LiDAR at all, so
`preprocess()` and the whole reactive safety layer have never been exercised
offline.

WHAT IS EXACT
    Wall geometry. `extract_walls` (marching squares, Douglas-Peucker, contour
    winding normals) and the tile helpers are the judge's own code, imported
    verbatim -- so the harness sees the same 3333 segments the judge does.
    Ray/segment intersection is a line-for-line numpy port of the JAX kernel in
    `lidar/kernels.py`, and the sensor and body frames are taken from
    `simulator._lidar_pose_from_cog` and `_collision_pose_from_base`.
    Collision DETECTION is exact: the judge's `get_vertices` rectangle against
    those same segments, which is the boolean `/ego_racecar/collision` carries
    and the only collision signal the referee scores.

WHAT IS APPROXIMATE
    Collision RESPONSE. The judge runs a 64-iteration JAX impulse solver with
    Coulomb friction and Baumgarte stabilisation. This applies the physically
    matched simplification: the judged config sets `restitution = 0.0`, so a
    wall contact IS a dead stop along the normal -- the normal velocity
    component is removed and the body pushed out of penetration. Tangential
    friction (0.6) and the angular impulse are NOT modelled, so the trajectory
    after a hit is close but not bit-identical. `ContactResolver.approximate`
    is True so a harness can say so in its output.
"""

import math
import os
import sys
import types

import numpy as np

import plant_sim

# Matches the JAX kernel's PARALLEL_EPS (lidar/kernels.py:6).
PARALLEL_EPS = 1e-12

# The scored sweep: +/-135 degrees, from the bridge's sim.yaml. Named here
# rather than computed in a default argument so the value is evaluated once and
# the degrees it came from stay visible.
SCAN_ANGLE_MIN = math.radians(-135.0)
SCAN_ANGLE_MAX = math.radians(135.0)


def _install_requests_stub():
    """`track/utils.py` imports requests to DOWNLOAD tracks. We load from disk.

    Stubbed for the same reason as numba in `plant_sim`: it stands between us
    and code we need, and nothing we call touches it. Installed only when the
    real package is absent.
    """
    if 'requests' in sys.modules:
        return False
    try:
        import requests  # noqa: F401
        return False
    except ImportError:
        sys.modules['requests'] = types.ModuleType('requests')
        return True


def _load_gym_geometry():
    """Import the judge's wall extraction, tile index and body geometry.

    `plant_sim` has already registered the synthetic `f1tenth_gym` /
    `f1tenth_gym.envs` packages and stubbed numba, so this only adds the
    requests stub and pulls the geometry modules in through the same path.
    """
    _install_requests_stub()
    import importlib
    return (
        importlib.import_module('f1tenth_gym.envs.track.track'),
        importlib.import_module('f1tenth_gym.envs.track.walls'),
        importlib.import_module('f1tenth_gym.envs.track.ray_tiles'),
        importlib.import_module('f1tenth_gym.envs.collision_models'),
    )


_TRACK, _WALLS, _RAY_TILES, _COLLISION = _load_gym_geometry()

TrackSpec = _TRACK.TrackSpec
extract_walls = _WALLS.extract_walls
wall_segments = _WALLS.wall_segments
point_segment_distance = _RAY_TILES.point_segment_distance
get_vertices = _COLLISION.get_vertices


class MapTrack:
    """The three attributes `wall_segments` reads off a `Track`.

    The real `Track` cannot be imported: `track/__init__.py` pulls in
    `Raceline` -> `CubicSplineND`, which we do not need and which drags in more
    of the package. `wall_segments(track)` only ever touches `spec`,
    `occupancy_map` and `occupancy_grey`, so this supplies exactly those --
    built by the judge's own `_occupancy_from_image`, not by a reimplementation
    of the ROS thresholding rules.
    """

    def __init__(self, spec, occupancy_map, occupancy_grey=None):
        self.spec = spec
        self.occupancy_map = occupancy_map
        self.occupancy_grey = occupancy_grey


def load_map(map_yaml_path):
    """Read a ROS map (`.yaml` + image) into a `MapTrack`.

    The image is flipped top-to-bottom, matching `track.py:202` and `:292`:
    image row 0 is the top of the picture but row 0 of an occupancy grid is the
    smallest world y. Getting this wrong mirrors the whole track.
    """
    import yaml
    from PIL import Image
    from PIL.Image import Transpose

    map_yaml_path = os.path.abspath(map_yaml_path)
    with open(map_yaml_path) as handle:
        raw = yaml.safe_load(handle)

    spec = TrackSpec(
        name=os.path.splitext(os.path.basename(map_yaml_path))[0],
        image=raw['image'],
        resolution=float(raw['resolution']),
        origin=tuple(float(v) for v in raw['origin']),
        negate=int(raw.get('negate', 0)),
        occupied_thresh=float(raw.get('occupied_thresh', 0.65)),
        free_thresh=float(raw.get('free_thresh', 0.196)),
    )

    image_path = os.path.join(os.path.dirname(map_yaml_path), str(raw['image']))
    image = Image.open(image_path).transpose(Transpose.FLIP_TOP_BOTTOM)
    return MapTrack(spec,
                    _TRACK._occupancy_from_image(image, spec),
                    _TRACK._grayscale_from_image(image))


def ray_segment_range(origin, direction, seg_a, seg_b, max_range):
    """Distance along each ray to its nearest valid segment intersection.

    A line-for-line port of `lidar/kernels.py:9-37`, numpy for jax.numpy. The
    degenerate-segment guard is the same: a near-zero denominator means the ray
    is parallel to the segment, `safe` keeps the division finite, and the `hit`
    mask throws the result away.

    Args:
        origin: (2,) sensor position.
        direction: (B, 2) unit direction per beam.
        seg_a: (S, 2) segment starts.
        seg_b: (S, 2) segment ends.
        max_range: Reported range for a beam that hits nothing.

    Returns:
        (B,) ranges in metres.
    """
    edge = seg_b - seg_a
    offset = seg_a[None, :, :] - origin[None, None, :]

    denominator = (
        direction[:, None, 0] * edge[None, :, 1]
        - direction[:, None, 1] * edge[None, :, 0]
    )
    ray_fraction = (
        offset[..., 0] * edge[None, :, 1]
        - offset[..., 1] * edge[None, :, 0]
    )
    segment_fraction = (
        offset[..., 0] * direction[:, None, 1]
        - offset[..., 1] * direction[:, None, 0]
    )

    parallel = np.abs(denominator) > PARALLEL_EPS
    safe = np.where(parallel, denominator, denominator.dtype.type(1.0))
    ray_fraction = ray_fraction / safe
    segment_fraction = segment_fraction / safe
    hit = (parallel
           & (ray_fraction >= 0.0)
           & (segment_fraction >= 0.0)
           & (segment_fraction <= 1.0))
    ranges = np.where(hit, ray_fraction, max_range)
    return np.minimum(np.min(ranges, axis=1), max_range)


class ScanSimulator:
    """The judged LiDAR: 819 beams over +/-135 deg, exact against wall segments.

    Defaults are the ones the scored run actually uses -- the bridge's
    `sim.yaml` (819 beams, +/-135 deg, 0.05-25 m) with the referee's
    `lidar_noise_std = 0.0` override from `simulator.launch.py:45`, which is set
    so a submission scores the same on every attempt.

    Attributes:
        angles: (num_beams,) beam bearings relative to the sensor heading.
        walls: The judge's extracted `WallSegments`.
    """

    def __init__(self, track, num_beams=819, angle_min=SCAN_ANGLE_MIN,
                 angle_max=SCAN_ANGLE_MAX, range_min=0.05, range_max=25.0,
                 noise_std=0.0, base_link_to_lidar_tf=(0.275, 0.0, 0.0),
                 lr=0.17145, cull_reach=None, seed=None):
        self.track = track
        self.num_beams = int(num_beams)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.noise_std = float(noise_std)
        self.tf = tuple(float(v) for v in base_link_to_lidar_tf)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)

        # A segment farther than `cull_reach` from the sensor cannot be hit
        # within it, so restricting the segment set is EXACT for ranges up to
        # that distance -- a broad phase, not an approximation. Beyond it the
        # scan saturates early, which is why it defaults to the full range.
        self.cull_reach = float(cull_reach) if cull_reach is not None else self.range_max
        if self.cull_reach < self.range_min:
            raise ValueError(
                f'cull_reach ({self.cull_reach}) is below range_min ({self.range_min})')

        self.angle_increment = ((self.angle_max - self.angle_min)
                                / max(self.num_beams - 1, 1))
        self.angles = (self.angle_min
                       + np.arange(self.num_beams, dtype=np.float64) * self.angle_increment)

        self.walls = wall_segments(track)
        if self.walls.is_empty:
            raise ValueError('the map produced no wall segments')
        # float64 for the cull (the judge's own helper), float32 for the sweep:
        # 819 x 3333 in float64 is 22 MB per temporary and measurably slower,
        # and the difference against a float64 sweep is ~1e-6 m.
        self._a64 = self.walls.a.astype(np.float64)
        self._b64 = self.walls.b.astype(np.float64)
        self._a32 = self.walls.a.astype(np.float32)
        self._b32 = self.walls.b.astype(np.float32)
        self._cull_cache = None

    def lidar_pose(self, cog_pose):
        """CoG pose -> sensor pose, per `simulator._lidar_pose_from_cog`."""
        x, y, yaw = float(cog_pose[0]), float(cog_pose[1]), float(cog_pose[2])
        dx = self.tf[0] - self.lr
        dy, dtheta = self.tf[1], self.tf[2]
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        return (x + dx * cos_yaw - dy * sin_yaw,
                y + dx * sin_yaw + dy * cos_yaw,
                yaw + dtheta)

    def _candidates(self, origin):
        """Segments within `cull_reach` of the sensor. Exact broad phase."""
        key = (round(float(origin[0]), 3), round(float(origin[1]), 3))
        if self._cull_cache is not None and self._cull_cache[0] == key:
            return self._cull_cache[1]
        keep = point_segment_distance(np.asarray(origin, dtype=np.float64),
                                      self._a64, self._b64) <= self.cull_reach
        subset = (self._a32[keep], self._b32[keep])
        self._cull_cache = (key, subset)
        return subset

    def scan_from_lidar_pose(self, sensor_pose):
        """One sweep from a sensor pose already in the LiDAR frame."""
        origin = np.asarray(sensor_pose[:2], dtype=np.float32)
        bearings = float(sensor_pose[2]) + self.angles
        directions = np.stack([np.cos(bearings), np.sin(bearings)],
                              axis=1).astype(np.float32)

        seg_a, seg_b = self._candidates(sensor_pose[:2])
        if len(seg_a) == 0:
            ranges = np.full(self.num_beams, self.range_max, dtype=np.float64)
        else:
            ranges = ray_segment_range(origin, directions, seg_a, seg_b,
                                       np.float32(self.cull_reach)).astype(np.float64)

        # simulator._update_scans:491-495, in that order. The noise term is
        # skipped rather than drawn at std 0 so the RNG stream is untouched.
        if self.noise_std > 0.0:
            ranges = ranges + self.rng.normal(0.0, self.noise_std, self.num_beams)
        ranges = np.minimum(ranges, self.range_max)
        ranges[ranges < self.range_min] = 0.0
        return ranges.astype(np.float32)

    def scan(self, cog_pose):
        """One sweep for a car whose CoG is at `cog_pose`."""
        return self.scan_from_lidar_pose(self.lidar_pose(cog_pose))


def _points_in_convex(points, polygon):
    """Boolean per point: inside the convex `polygon` (cyclic vertex order)."""
    edge = np.roll(polygon, -1, axis=0) - polygon
    rel = points[:, None, :] - polygon[None, :, :]
    cross = edge[None, :, 0] * rel[..., 1] - edge[None, :, 1] * rel[..., 0]
    return np.all(cross >= 0.0, axis=1) | np.all(cross <= 0.0, axis=1)


def _segments_intersect(seg_a, seg_b, other_a, other_b):
    """Proper segment intersection, vectorised over the first pair."""
    r = seg_b - seg_a
    s = other_b - other_a
    denominator = r[:, 0] * s[1] - r[:, 1] * s[0]
    delta = other_a - seg_a
    parallel = np.abs(denominator) <= PARALLEL_EPS
    safe = np.where(parallel, 1.0, denominator)
    t = (delta[:, 0] * s[1] - delta[:, 1] * s[0]) / safe
    u = (delta[:, 0] * r[:, 1] - delta[:, 1] * r[:, 0]) / safe
    return (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)


def _segments_cross_polygon(seg_a, seg_b, polygon):
    """Boolean per segment: does it cross any edge of `polygon`?"""
    poly_b = np.roll(polygon, -1, axis=0)
    result = np.zeros(len(seg_a), dtype=bool)
    for start, end in zip(polygon, poly_b):
        result |= _segments_intersect(seg_a, seg_b, start, end)
    return result


class ContactResolver:
    """Wall contact: exact detection, restitution-0 response.

    Detection answers the question the referee scores -- is the body rectangle
    overlapping a wall? -- using the judge's own body geometry. Response is the
    simplification documented at the top of this module.

    Attributes:
        approximate: Always True. The response is not the judge's solver, and a
            harness reporting these numbers should say so.
    """

    approximate = True

    def __init__(self, track, params=None, slop=0.002, cull_reach=1.5):
        self.track = track
        self.params = (params if params is not None
                       else plant_sim.F1TENTH_VEHICLE_PARAMETERS)
        self.slop = float(slop)
        self.cull_reach = float(cull_reach)

        self.length = float(self.params.length)
        self.width = float(self.params.width)
        # simulator._compute_collision_body_offset: the body box is centred on
        # the wheelbase midpoint, not on the CoG.
        base_dx = -float(self.params.lr)
        if not math.isfinite(base_dx):
            base_dx = 0.0
        self.body_dx = base_dx + float(self.params.collision_body_center_x)
        self.body_dy = float(self.params.collision_body_center_y)

        walls = wall_segments(track)
        self.walls = walls
        self._a = walls.a.astype(np.float64)
        self._b = walls.b.astype(np.float64)
        self._n = walls.n.astype(np.float64)

    def body_pose(self, cog_pose):
        """CoG pose -> collision-body pose, per `_collision_pose_from_base`."""
        x, y, yaw = float(cog_pose[0]), float(cog_pose[1]), float(cog_pose[2])
        if self.body_dx == 0.0 and self.body_dy == 0.0:
            return (x, y, yaw)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        return (x + self.body_dx * cos_yaw - self.body_dy * sin_yaw,
                y + self.body_dx * sin_yaw + self.body_dy * cos_yaw,
                yaw)

    def vertices(self, cog_pose):
        """The four body corners, from the judge's `get_vertices`."""
        pose = np.asarray(self.body_pose(cog_pose), dtype=np.float64)
        return get_vertices(pose, self.length, self.width)

    def _nearby(self, centre):
        """Segments that could touch a body of this size. Exact broad phase."""
        reach = self.cull_reach + 0.5 * math.hypot(self.length, self.width)
        return point_segment_distance(np.asarray(centre, dtype=np.float64),
                                      self._a, self._b) <= reach

    def contacts(self, cog_pose):
        """`(normals, penetrations)` for every wall the body overlaps.

        A wall overlaps the body when one of its endpoints is inside the
        rectangle, or when it crosses one of the four edges. Both cases are
        tested: short segments can sit entirely within the box, and a long one
        can cut straight through without either endpoint inside.

        All of them are returned, not just the deepest. Wall extraction splits
        a corner into a fan of short segments whose normals disagree by 90
        degrees or more, so a body wedged into one touches a dozen walls at
        once and no single normal describes the constraint. Resolving against
        one of them leaves the car driving through the others.
        """
        verts = self.vertices(cog_pose)
        centre = np.asarray(self.body_pose(cog_pose)[:2], dtype=np.float64)
        keep = self._nearby(centre)
        if not keep.any():
            return np.empty((0, 2)), np.empty(0)

        seg_a, seg_b, normals = self._a[keep], self._b[keep], self._n[keep]
        inside = _points_in_convex(seg_a, verts) | _points_in_convex(seg_b, verts)
        crossing = _segments_cross_polygon(seg_a, seg_b, verts)
        touching = inside | crossing
        if not touching.any():
            return np.empty((0, 2)), np.empty(0)

        # Penetration: how far the deepest corner sits behind each wall plane,
        # measured along that wall's outward normal.
        hit_a, hit_n = seg_a[touching], normals[touching]
        offsets = verts[None, :, :] - hit_a[:, None, :]
        depths = np.einsum('ijk,ik->ij', offsets, hit_n).min(axis=1)
        return hit_n, np.maximum(-depths, 0.0)

    def detect(self, cog_pose):
        """`(hit, normal, penetration)` for the deepest overlapping wall."""
        normals, penetrations = self.contacts(cog_pose)
        if len(normals) == 0:
            return False, None, 0.0
        worst = int(np.argmax(penetrations))
        return True, normals[worst].copy(), float(penetrations[worst])

    def resolve(self, state, iterations=8):
        """Push a penetrating body out and kill its inward velocity.

        `state` is `plant_sim`'s 7-vector and is modified in place. Returns
        whether contact was detected.

        Every touching wall is resolved, not just the deepest, and the velocity
        pass is repeated until no constraint is still being violated -- the
        same reason the judge runs Jacobi sweeps rather than one pass. Removing
        the inward component against one normal can push the velocity into
        another wall of the same corner, so a single pass leaves the car
        sliding through geometry it is provably touching.

        Restitution is 0.0 in the judged config, so there is no bounce to
        model; Coulomb friction and the angular impulse are not modelled, which
        is what `approximate` is warning about.
        """
        cog_pose = (float(state[plant_sim.X]), float(state[plant_sim.Y]),
                    float(state[plant_sim.PSI]))
        normals, penetrations = self.contacts(cog_pose)
        if len(normals) == 0:
            return False

        # Positional correction, once per distinct normal so a corner is pushed
        # out of both of its walls rather than only the deeper one.
        offset = np.zeros(2)
        for normal, penetration in _deepest_per_normal(normals, penetrations):
            if penetration > self.slop:
                offset = offset + normal * (penetration - self.slop)
        state[plant_sim.X] = float(state[plant_sim.X]) + offset[0]
        state[plant_sim.Y] = float(state[plant_sim.Y]) + offset[1]

        speed = float(state[plant_sim.V])
        psi = float(state[plant_sim.PSI])
        course = psi + float(state[plant_sim.BETA])
        velocity = np.array([speed * math.cos(course), speed * math.sin(course)])

        for _ in range(iterations):
            approach = normals @ velocity
            inward = approach < -1e-12
            if not inward.any():
                break
            worst = int(np.argmin(approach))
            velocity = velocity - approach[worst] * normals[worst]

        new_speed = float(np.hypot(velocity[0], velocity[1]))
        state[plant_sim.V] = new_speed
        state[plant_sim.BETA] = (
            math.atan2(velocity[1], velocity[0]) - psi if new_speed > 1e-6 else 0.0)
        return True


def _deepest_per_normal(normals, penetrations, decimals=3):
    """The worst penetration for each distinct outward normal.

    Wall extraction gives a straight run many collinear segments sharing one
    normal; summing a push over every one of them would fling the car off the
    track. Grouping collapses that to a single constraint per direction.
    """
    seen = {}
    for normal, penetration in zip(normals, penetrations):
        key = (round(float(normal[0]), decimals), round(float(normal[1]), decimals))
        if key not in seen or penetration > seen[key][1]:
            seen[key] = (normal, float(penetration))
    return list(seen.values())


class CollisionCounter:
    """The referee's scoring rule for collisions.

    `session.register_collision`: the simulator raises its flag on every
    physics step the car is touching, ~100 times a second, so the referee
    counts at most one per `interval_s`. The clock runs from the last COUNTED
    collision, not the last pulse -- so parking against a barrier keeps earning
    a collision every second rather than being cheaper than one clean hit.

    Attributes:
        events: Collisions counted so far, as the referee would score them.
    """

    def __init__(self, interval_s=1.0, penalty_s=10.0, max_collisions=10):
        self.interval_s = float(interval_s)
        self.penalty_s = float(penalty_s)
        self.max_collisions = int(max_collisions)
        self.events = 0
        self.steps_in_contact = 0
        self._last_counted = None

    def update(self, time_s, in_contact):
        """Feed one step's raw flag. Returns True if it scored a new collision."""
        if not in_contact:
            return False
        self.steps_in_contact += 1
        if (self._last_counted is not None
                and time_s - self._last_counted < self.interval_s):
            return False
        self._last_counted = float(time_s)
        self.events += 1
        return True

    @property
    def penalty(self):
        """Seconds added to the run so far."""
        return self.events * self.penalty_s

    @property
    def disqualified(self):
        """The referee DQs strictly above `max_collisions`."""
        return self.events > self.max_collisions
