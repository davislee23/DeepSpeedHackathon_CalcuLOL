#!/usr/bin/env python3
"""ROS wiring for the team driver. The algorithm lives in `driver_core.py`.

This node does four things and nothing else: declare parameters, subscribe to
the sensors, hand each message to `DriverCore`, and publish whatever comes back
on `/drive`. Every control decision is made in `DriverCore`, so the offline
harness can drive exactly the same code the judge runs.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

What you are allowed to read (see docs/06-rules.md):
    /scan                LiDAR, 819 beams over 270 degrees
    /ego_racecar/odom    ground-truth pose and velocity - ALLOWED and RECOMMENDED
    TF, /map             the static map
What you publish:
    /drive               AckermannDriveStamped - you ask for a SPEED, and the
                         simulator closes that loop for you
    /driver/...          anything of your own, for visualisation
"""

import os
import csv
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray


class TrackProcessor:
    """Handles offline path loading, closed-loop smoothing, curvature calculation,
    and forward/backward speed profile generation.
    """
    def __init__(self, csv_path: str, max_speed: float, max_lat_accel: float, 
                 max_accel: float, max_decel: float):
        self.csv_path = csv_path
        self.max_speed = max_speed
        self.max_lat_accel = max_lat_accel
        self.max_accel = max_accel
        self.max_decel = max_decel
        
        self.x = np.array([])
        self.y = np.array([])
        self.s = np.array([])
        self.kappa = np.array([])
        self.v_target = np.array([])
        self.N = 0

        self._process_track()

    def _process_track(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Raceline CSV not found at: {self.csv_path}")

        raw_x, raw_y = [], []
        with open(self.csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                delimiter = ';' if ';' in line else ','
                parts = [p.strip() for p in line.split(delimiter)]
                
                if len(parts) >= 6:  # Full raceline format (s; x; y; psi; kappa; vx; ax)
                    try:
                        raw_x.append(float(parts[1]))
                        raw_y.append(float(parts[2]))
                    except ValueError:
                        continue
                elif len(parts) >= 2:  # Simple centerline format (x, y)
                    try:
                        raw_x.append(float(parts[0]))
                        raw_y.append(float(parts[1]))
                    except ValueError:
                        continue

        if len(raw_x) < 2:
            raise ValueError(f"Not enough valid coordinate pairs in CSV: {self.csv_path}")

        x = np.array(raw_x, dtype=np.float64)
        y = np.array(raw_y, dtype=np.float64)

        # 1. Close loop explicitly if endpoints differ
        if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 0.1:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        # 2. Closed-loop path smoothing
        pad_len = 15
        x_padded = np.pad(x, pad_len, mode='wrap')
        y_padded = np.pad(y, pad_len, mode='wrap')
        
        sigma = 1.0
        x_smooth = gaussian_filter1d(x_padded, sigma=sigma)[pad_len:-pad_len]
        y_smooth = gaussian_filter1d(y_padded, sigma=sigma)[pad_len:-pad_len]
        
        self.N = len(x_smooth)
        self.x = x_smooth
        self.y = y_smooth

        # 3. Calculate arc length (s)
        dx = np.diff(self.x, prepend=self.x[-1])
        dy = np.diff(self.y, prepend=self.y[-1])
        ds = np.hypot(dx, dy)
        ds[ds < 1e-6] = 1e-6
        self.s = np.cumsum(ds)

        # 4. Smooth gradient-based curvature calculation
        dx_ds = np.gradient(self.x, self.s)
        dy_ds = np.gradient(self.y, self.s)
        ddx_ds = np.gradient(dx_ds, self.s)
        ddy_ds = np.gradient(dy_ds, self.s)

        num = dx_ds * ddy_ds - dy_ds * ddx_ds
        den = (dx_ds**2 + dy_ds**2)**1.5
        den[den < 1e-6] = 1e-6
        
        self.kappa = gaussian_filter1d(np.pad(num / den, pad_len, mode='wrap'), sigma=1.5)[pad_len:-pad_len]

        # 5. Speed profile generation with realistic lateral friction limits
        v_lat = np.sqrt(self.max_lat_accel / (np.abs(self.kappa) + 1e-3))
        v_profile = np.minimum(self.max_speed, v_lat)

        v_ext = np.tile(v_profile, 2)
        ds_ext = np.tile(ds, 2)
        N_ext = len(v_ext)

        # Backward pass (Braking limit)
        for i in range(N_ext - 2, -1, -1):
            v_max_brake = np.sqrt(v_ext[i + 1] ** 2 + 2.0 * self.max_decel * ds_ext[i + 1])
            v_ext[i] = min(v_ext[i], v_max_brake)

        # Forward pass (Acceleration limit)
        for i in range(1, N_ext):
            v_max_accel = np.sqrt(v_ext[i - 1] ** 2 + 2.0 * self.max_accel * ds_ext[i])
            v_ext[i] = min(v_ext[i], v_max_accel)

        self.v_target = np.clip(v_ext[self.N:2 * self.N], 1.5, self.max_speed)


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('csv_path', '/hackathon/race_ws/src/team_driver/team_driver/maps/icra26_raceline.csv')
        self.declare_parameter('crawl_speed', 1.5)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steering_angle', 0.4189)
        self.declare_parameter('max_speed', 6.5)
        self.declare_parameter('max_lat_accel', 3.8)      # Reduced from 5.5 to prevent understeer wall strikes
        self.declare_parameter('max_accel', 3.0)          
        self.declare_parameter('max_decel', 4.5)          
        self.declare_parameter('min_lookahead', 0.8)      # Reduced min lookahead to prevent corner-cutting
        self.declare_parameter('max_lookahead', 2.2)      
        self.declare_parameter('lookahead_gain', 0.20)    
        self.declare_parameter('search_window', 80)
        self.declare_parameter('obstacle_corridor_width', 0.50)
        self.declare_parameter('debounce_threshold', 3)

        # Retrieve parameters
        self.crawl_speed = self.get_parameter('crawl_speed').value
        self.max_range = self.get_parameter('max_range').value
        self.wb = self.get_parameter('wheelbase').value
        self.max_steer = self.get_parameter('max_steering_angle').value
        self.max_speed = self.get_parameter('max_speed').value
        self.min_lh = self.get_parameter('min_lookahead').value
        self.max_lh = self.get_parameter('max_lookahead').value
        self.lh_gain = self.get_parameter('lookahead_gain').value
        self.search_window = self.get_parameter('search_window').value
        self.corridor_w = self.get_parameter('obstacle_corridor_width').value
        self.debounce_thresh = self.get_parameter('debounce_threshold').value

        # Real-time State Variables
        self.position = None
        self.yaw = 0.0
        self.speed = 0.0
        self.prev_nearest_idx = 0
        self.prev_steering = 0.0
        
        # Debounce State Machine Variables
        self.blocked_counter = 0
        self.clear_counter = 0
        self.state_avoidance = False

        # Load Track Processing Data
        csv_path = self.get_parameter('csv_path').value
        try:
            self.track = TrackProcessor(
                csv_path=csv_path,
                max_speed=self.max_speed,
                max_lat_accel=self.get_parameter('max_lat_accel').value,
                max_accel=self.get_parameter('max_accel').value,
                max_decel=self.get_parameter('max_decel').value
            )
            self.get_logger().info(f"Loaded track raceline with {self.track.N} points")
        except Exception as e:
            self.get_logger().error(f"Failed to load track processor: {e}")
            self.track = None

        # Publishers & Subscribers
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        self._marker_divisor = 0

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering)

    def preprocess(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    def _find_nearest_index(self, car_x: float, car_y: float) -> int:
        N = self.track.N
        indices = [(self.prev_nearest_idx + i) % N for i in range(-self.search_window, self.search_window)]
        
        pts_x = self.track.x[indices]
        pts_y = self.track.y[indices]
        
        dists = (pts_x - car_x) ** 2 + (pts_y - car_y) ** 2
        min_local_idx = np.argmin(dists)
        
        if dists[min_local_idx] > 4.0:
            all_dists = (self.track.x - car_x) ** 2 + (self.track.y - car_y) ** 2
            nearest_idx = int(np.argmin(all_dists))
        else:
            nearest_idx = indices[min_local_idx]

        self.prev_nearest_idx = nearest_idx
        return nearest_idx

    def _find_target_point_arclength(self, car_x: float, car_y: float, car_yaw: float, lookahead_dist: float) -> tuple[float, float, int]:
        """Find lookahead point along raceline path arc-length (prevents jumping across track walls)."""
        N = self.track.N
        nearest_idx = self._find_nearest_index(car_x, car_y)

        accum_dist = 0.0
        for i in range(1, N):
            idx = (nearest_idx + i) % N
            prev_idx = (nearest_idx + i - 1) % N

            dx = self.track.x[idx] - self.track.x[prev_idx]
            dy = self.track.y[idx] - self.track.y[prev_idx]
            accum_dist += math.hypot(dx, dy)

            if accum_dist >= lookahead_dist:
                tx, ty = self.track.x[idx], self.track.y[idx]
                dx_c = tx - car_x
                dy_c = ty - car_y
                x_c = dx_c * math.cos(car_yaw) + dy_c * math.sin(car_yaw)
                if x_c > 0.0:
                    return tx, ty, idx

        return self.track.x[nearest_idx], self.track.y[nearest_idx], nearest_idx

    def _check_obstacle(self, ranges: np.ndarray, angles: np.ndarray) -> bool:
        """Check dynamic forward corridor scaled with current speed."""
        forward_look = max(1.2, 0.4 * self.speed + 0.6)
        valid = (ranges > 0.1) & (ranges < forward_look)
        r = ranges[valid]
        a = angles[valid]

        x_c = r * np.cos(a)
        y_c = r * np.sin(a)

        in_corridor = (x_c > 0.15) & (x_c < forward_look) & (np.abs(y_c) < self.corridor_w / 2.0)
        return bool(np.any(in_corridor))

    def _plan_fallback_ftg(self, ranges: np.ndarray, angles: np.ndarray) -> tuple[float, float]:
        fov_mask = (angles >= -1.2) & (angles <= 1.2)
        fov_ranges = ranges[fov_mask]
        fov_angles = angles[fov_mask]

        if len(fov_ranges) == 0:
            return 0.0, self.crawl_speed

        min_idx = np.argmin(fov_ranges)
        min_dist = fov_ranges[min_idx]

        car_radius = 0.40
        bubble_rad = math.atan2(car_radius, max(min_dist, 0.1))
        
        proc_ranges = fov_ranges.copy()
        bubble_mask = np.abs(fov_angles - fov_angles[min_idx]) <= bubble_rad
        proc_ranges[bubble_mask] = 0.0

        gap_threshold = 1.0
        free_space = proc_ranges > gap_threshold

        if not np.any(free_space):
            best_idx = np.argmax(proc_ranges)
            target_angle = fov_angles[best_idx]
        else:
            bounded = np.pad(free_space.astype(int), (1, 1), 'constant')
            diffs = np.diff(bounded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]

            gap_lengths = ends - starts
            max_gap_idx = np.argmax(gap_lengths)

            start_i = starts[max_gap_idx]
            end_i = ends[max_gap_idx]

            gap_angles = fov_angles[start_i:end_i]
            target_angle = (gap_angles[0] + gap_angles[-1]) / 2.0

        steering = float(np.clip(0.85 * target_angle, -self.max_steer, self.max_steer))
        speed = 2.0
        return steering, speed

    def _apply_wall_safety_guard(self, ranges: np.ndarray, angles: np.ndarray, 
                                 raw_speed: float, raw_steering: float) -> tuple[float, float]:
        """Scans front/side LIDAR to actively prevent collisions into nearby walls."""
        # Front wall emergency check
        front_mask = (angles >= -0.6) & (angles <= 0.6) & (ranges > 0.05)
        if np.any(front_mask):
            front_min = np.min(ranges[front_mask])
            safe_dist = 0.4 + 0.15 * self.speed
            if front_min < safe_dist:
                speed_scale = max(0.2, front_min / safe_dist)
                raw_speed *= speed_scale

        # Left / Right Wall Repulsion
        left_mask = (angles > 0.35) & (angles < 1.4) & (ranges > 0.05)
        right_mask = (angles < -0.35) & (angles > -1.4) & (ranges > 0.05)

        min_left = np.min(ranges[left_mask]) if np.any(left_mask) else 2.0
        min_right = np.min(ranges[right_mask]) if np.any(right_mask) else 2.0

        wall_margin = 0.38  # Minimum desired clearance (meters)
        
        if min_left < wall_margin:
            nudge = -0.12 * (wall_margin - min_left) / wall_margin
            raw_steering += nudge
        elif min_right < wall_margin:
            nudge = 0.12 * (wall_margin - min_right) / wall_margin
            raw_steering += nudge

        # High steering angle speed reduction (prevents high-speed friction loss / understeer)
        steer_factor = 1.0 - 0.45 * (abs(raw_steering) / self.max_steer)
        adjusted_speed = raw_speed * steer_factor

        return float(np.clip(raw_steering, -self.max_steer, self.max_steer)), float(max(1.2, adjusted_speed))

    def plan(self, ranges, angles):
        if self.position is None or self.track is None:
            return 0.0, self.crawl_speed

        car_x, car_y = self.position
        car_yaw = self.yaw

        # 1. Dynamic Lookahead scaled by speed & track curvature
        nearest_idx = self._find_nearest_index(car_x, car_y)
        curr_kappa = abs(self.track.kappa[nearest_idx])
        
        # Scale lookahead shorter in tight turns (high kappa) to prevent cutting corners into inner walls
        curv_scale = 1.0 / (1.0 + 3.0 * curr_kappa)
        base_lh = self.min_lh + self.lh_gain * self.speed
        lookahead_dist = float(np.clip(base_lh * curv_scale, self.min_lh, self.max_lh))

        # 2. Debounced obstacle detection
        is_blocked = self._check_obstacle(ranges, angles)
        if is_blocked:
            self.blocked_counter += 1
            self.clear_counter = 0
        else:
            self.clear_counter += 1
            self.blocked_counter = 0

        if self.blocked_counter >= self.debounce_thresh:
            self.state_avoidance = True
        elif self.clear_counter >= self.debounce_thresh:
            self.state_avoidance = False

        # 3. Path Tracking vs Obstacle Avoidance Mode
        if self.state_avoidance:
            raw_steer, raw_speed = self._plan_fallback_ftg(ranges, angles)
            steering = 0.3 * self.prev_steering + 0.7 * raw_steer
            self.prev_steering = steering
            return steering, raw_speed

        # 4. Pure Pursuit Target Search via Path Arc Length
        gx, gy, target_idx = self._find_target_point_arclength(car_x, car_y, car_yaw, lookahead_dist)

        # Map target point into vehicle frame (x-forward, y-left)
        dx, dy = gx - car_x, gy - car_y
        x_c = dx * math.cos(car_yaw) + dy * math.sin(car_yaw)
        y_c = -dx * math.sin(car_yaw) + dy * math.cos(car_yaw)

        # Pure Pursuit geometry calculation
        L_sq = x_c**2 + y_c**2
        if L_sq < 1e-4:
            L_sq = 1e-4
            
        gamma = (2.0 * y_c) / L_sq
        raw_steering = math.atan(gamma * self.wb)

        # Target Speed from raceline profile
        target_speed = float(self.track.v_target[target_idx])

        # 5. Apply Active LIDAR Wall Safety Guard
        safe_steering, safe_speed = self._apply_wall_safety_guard(ranges, angles, target_speed, raw_steering)

        # 6. Fast-response Low-Pass Filter (0.15 prev / 0.85 new) to prevent steering phase lag
        steering = 0.15 * self.prev_steering + 0.85 * safe_steering
        self.prev_steering = steering

        return steering, safe_speed

    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_marker(self, target_angle):
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        
        if self.state_avoidance:
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.2, 0.2, 0.9
        else:
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 1.0, 0.4, 0.9
            
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        
        array = MarkerArray()
        array.markers.append(marker)
        self.marker_pub.publish(array)


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
