#!/usr/bin/env python3
"""YOUR DRIVER GOES HERE.

This is the node the judges run. Keep `driver` as the executable name and
`/drive` as the output topic and everything else is yours to change - rewrite
this file completely if you want to.

--------------------------------------------------------------------------
THIS TEMPLATE DOES NOT DRIVE
--------------------------------------------------------------------------
It is wiring, not a driver. It connects to the simulator, subscribes to the
sensors, and then asks for a slow constant speed with the wheels straight. It
will set off from the grid and into the first thing in front of it. That is
deliberate and it is the whole point: **there is no algorithm here and no
algorithm is shipped anywhere else in this repository.** Writing one is the
hackathon.

What the template is good for is proving your setup works. If the car moves
when you run it, then the image, the bridge, the workspace, the topics and
your commands are all correct, and every problem left is yours.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

`docs/04-algorithms.md` lists the approaches worth starting from - reactive
ones that need nothing but the LiDAR, planners that follow a line, model-based
control, and learned policies - with what each needs and where each breaks.
Pick one and replace `plan()` below.

What you are allowed to read (see docs/06-rules.md):
    /scan               LiDAR, 819 beams over 270 degrees
    /ego_racecar/odom   ground-truth pose and velocity - ALLOWED and RECOMMENDED
    TF, /map            the static map
What you publish:
    /drive              AckermannDriveStamped - you ask for a SPEED, and the
                        simulator closes that loop for you
    /driver/...         anything of your own, for visualisation
"""

import math
import os

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # Declared parameters can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p crawl_speed:=2.0
        # Can add our own as we go; config/driver_params.yaml loads them.
        # ADJUSTABLE-> EXPERIMENT WITH IT!!
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('crawl_speed', 1.0)        # [m/s]
        self.declare_parameter('max_range', 8.0)          # [m] clip the scan here
        self.declare_parameter('raceline_csv', '/hackathon/maps/icra26_centerline.csv')
        # Tuned with results/tuning/tune.sh (0 collisions; ~17.8 s laps, was ~27.8 s)
        self.declare_parameter('v_max', 6.5)              # [m/s] top speed on straights (was 4.0)
        self.declare_parameter('a_lat', 9.5)              # [m/s^2] sideways grip we trust in corners (was 3.0)
        self.declare_parameter('a_accel', 3.5)            # [m/s^2] how hard we speed up (was 3.0)
        self.declare_parameter('a_brake', 6.0)            # [m/s^2] how hard we slow down (was 5.0)
        self.declare_parameter('lookahead_base', 0.8)     # [m] lookahead at standstill
        self.declare_parameter('lookahead_gain', 0.325)   # [s] extra lookahead per m/s of speed (was 0.3)
        self.declare_parameter('lookahead_min', 1.0)      # [m] lookahead never shorter than this
        self.declare_parameter('lookahead_max', 3.0)      # [m] lookahead never longer than this
        self.declare_parameter('max_steer', 0.4)          # [rad] steering clip; sim limit is 0.4189

        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.v_max = self.get_parameter('v_max').value
        self.a_lat = self.get_parameter('a_lat').value
        self.a_accel = self.get_parameter('a_accel').value
        self.a_brake = self.get_parameter('a_brake').value
        self.lookahead_base = self.get_parameter('lookahead_base').value
        self.lookahead_gain = self.get_parameter('lookahead_gain').value
        self.lookahead_min = self.get_parameter('lookahead_min').value
        self.lookahead_max = self.get_parameter('lookahead_max').value
        self.max_steer = self.get_parameter('max_steer').value
        self.wheelbase = 0.33     # [m] distance between front and rear axle

        # Latest known pose and speed. Ground truth from the simulator, as guided by the repodity
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        # Pure pursuit path: (x array, y array, speed array), filled by load_path()
        self._path = None
        self._ds = 0.1            # [m] spacing between path points after resampling
        self._last_idx = None     # index of the path point nearest the car last time

        # Logging state: which algorithm is active, scans spent in each, laps seen
        self._mode = None
        self._mode_scans = {'pure_pursuit': 0, 'follow_the_gap': 0}
        self._laps = 0

        # Lap timing on the simulator's clock (odom timestamps), so times match the referee's
        self._sim_time = 0.0      # [s] stamp of the latest odometry message
        self._lap_start = None    # [s] sim time the current lap started
        self._best_lap = None     # [s] fastest full lap so far

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        # Read the track centreline and build the speed profile once at start-up
        self.load_path(self.get_parameter('raceline_csv').value)

        self._marker_divisor = 0

    
    # Odometry: scalable for custom nitpicking for best path/control
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self._sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    # The main control loop, once per LiDAR scan (about 40 Hz).
    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering)

    def preprocess(self, scan):
        """AI suggested: Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # Counts scans per algorithm and prints one line only when the active algorithm changes.
    def log_mode(self, mode, reason):
        self._mode_scans[mode] += 1
        if mode != self._mode:
            self._mode = mode
            self.get_logger().info(
                f'MODE -> {mode} ({reason}). Scans so far: '
                f'pure_pursuit={self._mode_scans["pure_pursuit"]}, '
                f'follow_the_gap={self._mode_scans["follow_the_gap"]}')

    # Prints the lap just finished with its sim-time duration and the best lap so far.
    # The first lap starts from the grid, so it is partial and is not timed.
    def log_lap(self):
        now = self._sim_time
        scans = (f'pure_pursuit={self._mode_scans["pure_pursuit"]}, '
                 f'follow_the_gap={self._mode_scans["follow_the_gap"]}')
        if self._lap_start is None:
            self.get_logger().info(
                f'LAP {self._laps} of path completed (partial, from the grid; not timed). '
                f'Scans so far: {scans}')
        else:
            lap_time = now - self._lap_start
            if self._best_lap is None or lap_time < self._best_lap:
                self._best_lap = lap_time
            self.get_logger().info(
                f'LAP {self._laps} of path completed in {lap_time:.2f} s '
                f'(best {self._best_lap:.2f} s). Scans so far: {scans}')
        self._lap_start = now

    # Picks pure pursuit when the path and pose are ready, OTHERWSIE follow-the-gap.
    def plan(self, ranges, angles):
        # No path or no odometry yet, so use the LiDAR-only method
        if self._path is None or self.position is None:
            self.log_mode('follow_the_gap', 'waiting for path/odometry')
            return self.follow_the_gap(ranges, angles)

        return self.pure_pursuit(ranges, angles)

    # Reads the centreline CSV, spaces it evenly and computes a target speed per point.
    def load_path(self, csv_path):
        if not os.path.exists(csv_path):
            self.get_logger().error(f'Path file not found: {csv_path}. Using follow-the-gap.')
            return

        # Read the file, accepting either ',' or ';' as the separator
        with open(csv_path, 'r') as f:
            text = f.read().replace(';', ',')
        rows = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(',') if p.strip() != '']
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue  # skip header or comment lines
        rows = [r for r in rows if len(r) >= 2]
        if len(rows) < 10:
            self.get_logger().error('Path file has too few numeric rows. Using follow-the-gap.')
            return

        # 7 columns means s;x;y;psi;kappa;vx;ax so x,y are columns 1,2; otherwise columns 0,1
        width = min(len(r) for r in rows)
        cols = (1, 2) if width >= 7 else (0, 1)
        data = np.array([[r[cols[0]], r[cols[1]]] for r in rows])

        # Make sure the path runs counter-clockwise (the racing direction)
        x, y = data[:, 0], data[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        if area < 0:
            data = data[::-1]

        # Resample so points are evenly spaced (~0.1 m), which makes curvature reliable
        closed = np.vstack([data, data[0]])
        seg = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = int(s[-1] / 0.1)
        ds = s[-1] / n
        s_new = np.arange(n) * ds
        px = np.interp(s_new, s, closed[:, 0])
        py = np.interp(s_new, s, closed[:, 1])

        # Curvature = how fast the path's heading changes per metre (bigger = tighter)
        heading = np.unwrap(np.arctan2(np.gradient(py), np.gradient(px)))
        kappa = np.abs(np.gradient(heading, ds))
        w = 9
        padded = np.concatenate([kappa[-w:], kappa, kappa[:w]])
        kappa = np.convolve(padded, np.ones(w) / w, mode='same')[w:-w]

        # Corner speed limit from grip: v = sqrt(a_lat / curvature), capped at v_max
        v = np.minimum(self.v_max, np.sqrt(self.a_lat / np.maximum(kappa, 1e-3)))

        # Propagate braking backwards and acceleration forwards (twice round the loop)
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2 * self.a_brake * ds))
            for i in range(n):
                v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * self.a_accel * ds))

        self._path = (px, py, v)
        self._ds = ds
        self.get_logger().info(
            f'Loaded {n} path points, {s[-1]:.1f} m lap, speeds {v.min():.1f}-{v.max():.1f} m/s.')

    # Steers along an arc towards a path point a set distance ahead of the car.
    def pure_pursuit(self, ranges, angles):
        px, py, v = self._path
        n = len(px)
        x, y = self.position
        dists = np.hypot(px - x, py - y)

        # 1. Nearest path point: search everywhere at first, then only around the last one
        prev_idx = self._last_idx
        if self._last_idx is None:
            nearest = int(np.argmin(dists))
        else:
            cand = (self._last_idx + np.arange(-10, 50)) % n
            nearest = int(cand[np.argmin(dists[cand])])
        self._last_idx = nearest

        # Lap counter: the nearest index wrapping from the end of the path back to the start
        if prev_idx is not None and prev_idx > 0.9 * n and nearest < 0.1 * n:
            self._laps += 1
            self.log_lap()

        # Far off the path (spun out or pushed away): recover with follow-the-gap
        if dists[nearest] > 1.5:
            self.log_mode('follow_the_gap', f'off path by {dists[nearest]:.1f} m')
            self._last_idx = None
            return self.follow_the_gap(ranges, angles)
        self.log_mode('pure_pursuit', 'on path')

        # 2. Lookahead grows with speed (faster = look further ahead)
        #ADJUSTABLE: CAN CHANGE GAIN
        #Original: 0.8 + 0.3 * self.speed, 1.0, 3.0
        lookahead = float(np.clip(self.lookahead_base + self.lookahead_gain * self.speed,
                                  self.lookahead_min, self.lookahead_max))
        target_idx = (nearest + int(round(lookahead / self._ds))) % n

        # 3. Convert the target from the map frame into the car's own frame
        dx = px[target_idx] - x
        dy = py[target_idx] - y
        local_x = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        local_y = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy

        # 4. Pure pursuit: curvature = 2*y / L^2, steering = atan(wheelbase * curvature)
        dist_sq = max(local_x ** 2 + local_y ** 2, 0.01)
        curvature = 2.0 * local_y / dist_sq

        #ADJUSTABLE: CAN CHANGE 
        steering = float(np.clip(math.atan(self.wheelbase * curvature),
                                 -self.max_steer, self.max_steer))

        # 5. Speed comes straight from the precomputed profile
        speed = float(v[nearest])

        # Status line every 2 s: progress round the path, distance from it, steering, speeds
        self.get_logger().info(
            f'[PP] {100.0 * nearest / n:.0f}% round | off_path={dists[nearest]:.2f}m | '
            f'steer={steering:.2f} | v_target={speed:.1f} v_actual={self.speed:.1f}',
            throttle_duration_sec=2.0)

        return steering, speed

    # Your original follow-the-gap algorithm, unchanged (now the fallback).
    def follow_the_gap(self, ranges, angles):
        # 1. Restrict view to forward field of view [-100 deg, +100 deg] (~ -1.75 to +1.75 rad)
        fov_mask = (angles >= -1.75) & (angles <= 1.75)
        fov_ranges = ranges[fov_mask]
        fov_angles = angles[fov_mask]

        if len(fov_ranges) == 0:
            return 0.0, self.crawl_speed

        # 2. Locate nearest obstacle point inside FOV
        min_idx = np.argmin(fov_ranges)
        min_dist = fov_ranges[min_idx]

        # 3. Create a safety bubble around nearest obstacle (radius ~0.35m)
        car_radius = 0.35
        bubble_rad = math.atan2(car_radius, max(min_dist, 0.1))

        proc_ranges = fov_ranges.copy()
        bubble_mask = np.abs(fov_angles - fov_angles[min_idx]) <= bubble_rad
        proc_ranges[bubble_mask] = 0.0

        # 4. Find the largest continuous gap beyond threshold distance
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

        # 5. Compute steering angle with gain & clipping
        steering = 0.85 * target_angle
        steering = float(np.clip(steering, -0.4, 0.4))

        # 6. Dynamic speed scaling based on steering angle & clearance ahead
        front_mask = (angles >= -0.2) & (angles <= 0.2)
        front_dist = np.mean(ranges[front_mask]) if np.any(front_mask) else 1.0

        if abs(steering) > 0.25 or front_dist < 2.0:
            # Slow down for sharp corners or tight walls
            speed = 2.0
        elif front_dist > 5.0 and abs(steering) < 0.1:
            # Full throttle on open straights
            speed = 4.5
        else:
            # Cruising speed
            speed = 3.0

        return steering, speed

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    # Sends the steering angle and speed to the simulator on /drive.
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    # Draws an arrow in RViz showing where the car is steering.
    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


# Starts ROS, runs the driver node until you press Ctrl+C, then shuts down cleanly.
def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()