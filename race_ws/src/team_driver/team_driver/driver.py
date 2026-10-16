#!/usr/bin/env python3
"""ROS wiring for the team driver. The algorithm lives in `driver_core.py`.

This node does four things and nothing else: declare parameters, subscribe to
the sensors, hand each message to `DriverCore`, and publish whatever comes back
on `/drive`. Every control decision is made in `DriverCore`, so the offline
harness can drive exactly the same code the judge runs.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

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
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

import driver_params
from driver_core import DriverCore, LatchClock
from mpc_controller import MPCController

DT_REPORT_PERIOD = 5.0   # [s] how often to log dt_measured statistics


def _stamp_s(stamp):
    """A ROS time message as float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


class Driver(Node):

    def __init__(self):
        # The judge runs a bare `ros2 run team_driver driver`, so sim time is
        # set here rather than trusted to a params file.
        super().__init__('driver', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Every parameter is declared from one table (driver_params.py), so it
        # can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p mpc.q_pos:=3.0
        # config/driver_params.yaml loads them, nested under this node's name.
        driver_params.declare_all(self.declare_parameter)
        params = driver_params.read_all(self.get_parameter)
        driver_params.validate(params)
        self.control_period = params['control_period']

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, params['drive_topic'], 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        # Depth-1 best-effort: the queue never holds a stale scan, so the
        # callbacks are pure setters and the timer always reads the freshest data.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            LaserScan, params['scan_topic'], self.scan_callback, sensor_qos)
        self.create_subscription(
            Odometry, params['odom_topic'], self.odom_callback, sensor_qos)

        mpc_controller = MPCController(
            vehicle_model=driver_params.vehicle_model(params),
            **driver_params.mpc_kwargs(params))

        self.core = DriverCore(
            cfg=driver_params.driver_config(params),
            mpc_controller=mpc_controller,
            logger=self.get_logger(),
        )

        # What actually arrived. `params_source = CODE_DEFAULTS` means the
        # params file was not loaded and the table's defaults are running.
        for line in driver_params.echo_lines(params):
            self.get_logger().info(f'param {line}')

        self._marker_divisor = 0
        self._last_dt_report = self.get_clock().now()
        self._latch_clock = LatchClock()

        # A regular hold period is what mpc.dt assumes. With use_sim_time this
        # is 25 ms of plant time; on the wall clock it would drift with RTF.
        self.create_timer(self.control_period, self._control_step)
        self.get_logger().info(
            f'team_driver is up with MPC controller! control_period='
            f'{self.control_period * 1e3:.1f} ms')

    # ------------------------------------------------------------------
    # Odometry: where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # The bridge publishes (v cos beta, v sin beta, yaw_rate), so the
        # sideslip angle and the yaw rate come free alongside the speed.
        lx = msg.twist.twist.linear.x
        ly = msg.twist.twist.linear.y
        self.core.set_odom(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
            math.hypot(lx, ly),
            beta=math.atan2(ly, lx),
            yaw_rate=msg.twist.twist.angular.z,
            t_sim=_stamp_s(msg.header.stamp),
        )

    # ------------------------------------------------------------------
    # The control loop, on a fixed timer at mpc.dt. Callbacks only store.
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        self.core.set_scan(scan.ranges, scan.angle_min, scan.angle_increment,
                           t_sim=_stamp_s(scan.header.stamp))

    def _control_step(self):
        # With use_sim_time this clock is plant time, but it stands still while
        # this callback runs; the bridge does not. LatchClock turns the solve's
        # wall time into the plant time the bridge will latch the command at,
        # which is where the steering estimate replays the servo from.
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        wall0 = time.perf_counter()
        self._latch_clock.update(now_s, wall0)

        t = self.core.t_scan
        cmd = self.core.step(t_sim=t)
        self.publish(cmd, now)
        self.core.note_published(
            cmd, t_sim=self._latch_clock.latch_time(now_s, time.perf_counter() - wall0))

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(cmd.steering_angle)
        self._report_dt()

    def _report_dt(self):
        now = self.get_clock().now()
        if (now - self._last_dt_report).nanoseconds * 1e-9 < DT_REPORT_PERIOD:
            return
        self._last_dt_report = now
        mean, p95, worst = self.core.dt_summary()
        self.get_logger().info(
            f'dt_measured mean={mean * 1e3:.1f} p95={p95 * 1e3:.1f} '
            f'max={worst * 1e3:.1f} ms')
        # How far the servo replay sits from the wheel angle the yaw rate
        # implies. Small and steady is healthy; growing means drift.
        rtf = self._latch_clock.rtf
        self.get_logger().info(
            f'delta_hat={self.core.delta_hat:.3f} rad, '
            f'p95|delta_hat - delta_meas|={self.core.steering_residual_summary():.4f} rad, '
            f'rtf={rtf if rtf is not None else float("nan"):.2f}')
        # Healthy: the limiter almost never binds, coasting never happens, and
        # the emergency cap binds on under 1 % of commands.
        stats = self.core.conditioning_stats()
        self.get_logger().info(
            f'conditioning: slew bound {stats.slew_binds}/{stats.steps} '
            f'({stats.slew_bind_rate:.1%}), coast steps {stats.coast_count}, '
            f'emergency cap {stats.emergency_binds} ({stats.emergency_bind_rate:.1%})')

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, cmd, now=None):
        msg = AckermannDriveStamped()
        msg.header.stamp = (now if now is not None else self.get_clock().now()).to_msg()
        msg.drive.steering_angle = float(cmd.steering_angle)
        msg.drive.speed = float(cmd.speed)
        msg.drive.steering_angle_velocity = float(cmd.steering_angle_velocity)
        self.drive_pub.publish(msg)

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
