#!/usr/bin/env python3
"""Sim2Real autonomy node (self-contained) - Tier 1: published waypoint coordinates + dead reckoning.

Only consumes contestant-facing topics. In particular, this module does not
subscribe to /evaluator/* or /gz/* topics.
"""
from __future__ import annotations

import heapq
import itertools
import math
from typing import Optional, Sequence

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from tf2_ros import TransformBroadcaster


# Simulation geometry from the organizer's robot/world descriptions.
WHEEL_RADIUS = 0.0225
WHEEL_SEPARATION = 0.159
MAX_LINEAR = 0.30
MAX_ANGULAR = 1.50

# The walls' inside faces are approximately [-0.125, 2.875] on both axes.
CAMERA_X = 1.50
CAMERA_Y = 1.50
CAMERA_Z = 4.00
WAYPOINT_TOP_Z = 0.02
OBSTACLE_TOP_Z = 0.20
EXPECTED_TARGETS = 7
WP_CAPTURE_RADIUS = 0.20  # actual evaluator threshold
WP_STOP_DWELL = 2.35       # margin above the required 2.00 simulation seconds
HOME_STOP_DWELL = 2.35
STOP_SPEED_EST = 0.020

# A conservative planning inflation for the robot's asymmetric footprint.
# The evaluator's largest base_link corner radius is about 0.219 m.
OBSTACLE_INFLATION = 0.245
GRID_RESOLUTION = 0.04
GRID_MAX = 2.60
HOME_POINT = (0.02, 0.02)  # Tier 1/3 home target near the origin
TIER2_HOME_POINT = (0.14, 0.08)  # User-validated: preserves wall clearance and passes home dwell on seeds 42/456


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


class Target:
    __slots__ = ("x", "y", "hits", "completed", "last_seen", "source")

    def __init__(self, x: float, y: float, source: str, now: float):
        self.x = float(x)
        self.y = float(y)
        self.hits = 1
        self.completed = False
        self.last_seen = now
        self.source = source

    @property
    def stable(self) -> bool:
        return self.source != "vision" or self.hits >= 2


class Sim2RealAutonomyNode(Node):
    """Shared node; each tier supplies a different target/obstacle source."""

    def __init__(self, tier: int):
        super().__init__("solution_node")
        if tier not in (1, 2, 3):
            raise ValueError("tier must be 1, 2, or 3")
        self.tier = tier
        # Tier 2's home leg clipped a perimeter while the robot was facing west.
        # The base_link front extends 0.20 m, so a home target too close to x=0
        # leaves insufficient clearance for the forward footprint. TIER2_HOME_POINT
        # is offset east but remains within the evaluator's 0.20 m home radius.
        # Keep Tier 1/3 goals unchanged until independently tested.
        self.home_goal = TIER2_HOME_POINT if self.tier == 2 else HOME_POINT
        self.bridge = CvBridge()

        # ROS 2 Jazzy declares use_sim_time automatically. Do not redeclare it.
        self.declare_parameter("debug_perception", True)
        self.debug_perception = bool(self.get_parameter("debug_perception").value)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Estimator state. Joint positions are available from the public
        # /joint_states stream; we use their increments preferentially and use
        # noisy velocity values only as a fallback/consistency signal.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear_speed = 0.0
        self.yaw_rate = 0.0
        self._last_joint_pos: dict[str, float] = {}
        self._last_joint_stamp: Optional[float] = None
        self._last_imu_stamp: Optional[float] = None
        self._last_imu_z: Optional[float] = None
        self._gyro_bias_sum = 0.0
        self._gyro_bias_n = 0
        self._gyro_bias = 0.0
        self._gyro_rate = 0.0
        self._motion_started = False
        # Encoder polarities are learned independently and only once. A single
        # shared sign is unsafe because forward and differential-yaw conventions
        # need not have the same sign in every simulator/URDF.
        self._linear_encoder_sign: Optional[float] = None
        self._yaw_encoder_sign: Optional[float] = None
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0
        self._last_pose_stamp: Optional[float] = None
        self._last_publish_stamp: Optional[float] = None
        self._last_status_log = -1e9
        self._odom_ready = False

        # Target and obstacle state.
        self.targets: list[Target] = []
        self.obstacles: list[dict[str, float]] = []
        self._obstacle_hits: list[int] = []
        self._camera_frame_count = 0
        self._camera_dirs: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._camera_axis_origin_px: Optional[np.ndarray] = None
        self._fx = 800.0 / (2.0 * math.tan(math.radians(70.0) / 2.0))
        self._fy = self._fx
        self._cx = 399.5
        self._cy = 399.5
        self._camera_info_received = False
        self._last_perception_log = 0.0

        # Tier 2 visual self-localization. The overhead image contains the robot's
        # blue-gray chassis, whose centroid is offset forward from base_link.
        # Calibrate that offset from the known initial pose, then softly correct
        # wheel/IMU dead reckoning from image observations. If no confident robot
        # detection is available, the existing odometry-only estimator remains in use.
        self._robot_offset_base: Optional[np.ndarray] = None
        self._robot_offset_samples: list[tuple[float, float]] = []
        self._robot_visual_updates = 0
        self._robot_last_residual: Optional[float] = None
        self._robot_last_seen = -1e9
        self._robot_last_log = -1e9
        self._robot_detection_misses = 0

        # Navigation state.
        self.phase = "WAIT"  # WAIT, NAVIGATE, DWELL_WP, DWELL_HOME, COMPLETE
        self.current_target_index: Optional[int] = None
        self.current_goal: Optional[tuple[float, float]] = None
        self.current_goal_is_home = False
        self.path_points: list[tuple[float, float]] = []
        self.path_index = 0
        self.dwell_start: Optional[float] = None
        self.stationary_since: Optional[float] = None
        self._last_plan_target_count = -1
        self._home_started = False
        self._route_failed_logged = False
        self._last_route_error_time = -1e9

        # Tier-specific subscriptions. Reliable transient-local QoS is needed
        # for Tier 1's latched target arrays.
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 30)
        self.create_subscription(Imu, "/imu", self._on_imu, 30)
        self.create_subscription(CameraInfo, "/overhead_camera/camera_info", self._on_camera_info, 5)
        self.create_subscription(Image, "/overhead_camera/image_raw", self._on_image, 5)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        if self.tier == 1:
            self.create_subscription(PoseArray, "/waypoints", self._on_waypoints, latched_qos)
            self.create_subscription(PoseArray, "/dynamic_waypoints", self._on_dynamic_waypoints, latched_qos)

        self.create_timer(0.05, self._control_loop)  # 20 Hz controller / odometry publisher
        self.get_logger().info(f"Sim2Real autonomy core ready (tier={tier}).")

    # ------------------------------------------------------------------
    # Time / ROS helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _publish_cmd(self, v: float, w: float) -> None:
        cmd = Twist()
        cmd.linear.x = clamp(float(v), 0.0, MAX_LINEAR)
        cmd.angular.z = clamp(float(w), -MAX_ANGULAR, MAX_ANGULAR)
        # Explicitly keep every unused Twist component zero.
        self.cmd_pub.publish(cmd)
        self._last_cmd_v = cmd.linear.x
        self._last_cmd_w = cmd.angular.z
        if abs(cmd.linear.x) > 1e-4 or abs(cmd.angular.z) > 1e-4:
            self._motion_started = True

    def _publish_odom_and_tf(self, stamp_msg) -> None:
        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self.linear_speed)
        odom.twist.twist.angular.z = float(self.yaw_rate)
        # Diagonal covariance: modest wheel/IMU uncertainty, growing by use.
        odom.pose.covariance[0] = 0.02
        odom.pose.covariance[7] = 0.02
        odom.pose.covariance[35] = 0.05
        odom.twist.covariance[0] = 0.03
        odom.twist.covariance[35] = 0.08
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp_msg
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = float(self.x)
        tf.transform.translation.y = float(self.y)
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    # ------------------------------------------------------------------
    # State estimation
    # ------------------------------------------------------------------
    def _on_imu(self, msg: Imu) -> None:
        """Cache gyro rate and estimate its stationary bias; integrate pose once per encoder sample.

        Integrating yaw here and then blending it again in the wheel callback made
        the pose update depend on callback ordering. The encoder callback below is
        now the single place where x/y/yaw advance, using one common dt.
        """
        stamp = self._stamp_seconds(msg.header.stamp)
        gz = float(msg.angular_velocity.z)
        cmd_is_stopped = abs(self._last_cmd_v) < 1e-3 and abs(self._last_cmd_w) < 1e-3
        if not self._motion_started and cmd_is_stopped:
            self._gyro_bias_sum += gz
            self._gyro_bias_n += 1
            self._gyro_bias = self._gyro_bias_sum / max(1, self._gyro_bias_n)

        corrected = gz - self._gyro_bias
        self._gyro_rate = corrected
        self._last_imu_stamp = stamp
        self._last_imu_z = gz

    def _on_joint_states(self, msg: JointState) -> None:
        stamp = self._stamp_seconds(msg.header.stamp)
        names = list(msg.name)
        try:
            li = names.index("left_wheel_joint")
            ri = names.index("right_wheel_joint")
        except ValueError:
            self.get_logger().warn(
                "/joint_states is missing left_wheel_joint/right_wheel_joint",
                throttle_duration_sec=5.0,
            )
            return
        if len(msg.position) <= max(li, ri):
            return

        positions = {"left": float(msg.position[li]), "right": float(msg.position[ri])}
        prev_stamp = self._last_joint_stamp
        prev_positions = self._last_joint_pos

        # Save the newest sample even after a discontinuity, so one bad timestamp
        # cannot poison every subsequent delta.
        self._last_joint_pos = positions
        self._last_joint_stamp = stamp
        self._odom_ready = True

        if prev_stamp is None or not prev_positions:
            if len(msg.velocity) > max(li, ri):
                vl = float(msg.velocity[li]) * WHEEL_RADIUS
                vr = float(msg.velocity[ri]) * WHEEL_RADIUS
                self.linear_speed = 0.5 * (vl + vr)
            return

        dt = stamp - prev_stamp
        if not (1e-4 < dt <= 0.15):
            # Never integrate duplicate, out-of-order, or very stale samples.
            return

        # Position is passed through unchanged by the organizer's encoder-noise
        # node. Use angular-position increments for displacement; the noisy
        # velocity fields are only a fallback/initialization signal.
        dl_angle = positions["left"] - prev_positions["left"]
        dr_angle = positions["right"] - prev_positions["right"]
        dl_angle = wrap_angle(dl_angle)
        dr_angle = wrap_angle(dr_angle)

        v_raw = WHEEL_RADIUS * (dl_angle + dr_angle) / (2.0 * dt)
        w_raw = WHEEL_RADIUS * (dr_angle - dl_angle) / (WHEEL_SEPARATION * dt)

        # Learn linear sign during a predominantly-forward command. Learn yaw
        # sign independently by matching differential encoder yaw to gyro yaw.
        # Lock each sign after its first meaningful observation to prevent it
        # oscillating between turns and straight segments.
        if (self._linear_encoder_sign is None and self._last_cmd_v > 0.06
                and abs(self._last_cmd_w) < 0.95 and abs(v_raw) > 0.025):
            self._linear_encoder_sign = 1.0 if v_raw >= 0.0 else -1.0
            self.get_logger().info(
                f"Encoder linear polarity calibrated: {self._linear_encoder_sign:+.0f}"
            )

        gyro_rate = self._gyro_rate
        if (self._yaw_encoder_sign is None and abs(self._last_cmd_w) > 0.15
                and abs(gyro_rate) > 0.10 and abs(w_raw) > 0.10):
            self._yaw_encoder_sign = 1.0 if w_raw * gyro_rate >= 0.0 else -1.0
            self.get_logger().info(
                f"Encoder yaw polarity calibrated: {self._yaw_encoder_sign:+.0f}"
            )

        linear_sign = self._linear_encoder_sign if self._linear_encoder_sign is not None else 1.0
        ds = linear_sign * WHEEL_RADIUS * (dl_angle + dr_angle) * 0.5
        dtheta_enc_raw = WHEEL_RADIUS * (dr_angle - dl_angle) / WHEEL_SEPARATION

        # The clean encoder positions provide the primary motion estimate. Once
        # yaw polarity is calibrated, blend predominantly encoder yaw with a
        # small gyro contribution. Before then, integrate the bias-corrected gyro
        # while turning to avoid using an uncalibrated sign.
        gyro_delta = gyro_rate * dt
        if self._yaw_encoder_sign is not None:
            dtheta_enc = self._yaw_encoder_sign * dtheta_enc_raw
            dtheta = 0.85 * dtheta_enc + 0.15 * gyro_delta
        else:
            dtheta = gyro_delta

        theta_mid = wrap_angle(self.yaw + 0.5 * dtheta)
        self.x += ds * math.cos(theta_mid)
        self.y += ds * math.sin(theta_mid)
        self.yaw = wrap_angle(self.yaw + dtheta)
        self.linear_speed = ds / dt
        # Use fused yaw rate for dwell gating and odometry twist.
        self.yaw_rate = dtheta / dt

    # ------------------------------------------------------------------
    # Camera calibration and perception (Tiers 2 / 3)
    # ------------------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 9 and msg.k[0] > 1.0 and msg.k[4] > 1.0:
            self._fx = float(msg.k[0])
            self._fy = float(msg.k[4])
            self._cx = float(msg.k[2])
            self._cy = float(msg.k[5])
            self._camera_info_received = True

    @staticmethod
    def _dominant_line(mask: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return midpoint/direction of the longest long line in a binary mask."""
        lines = cv2.HoughLinesP(
            mask,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=70,
            minLineLength=160,
            maxLineGap=70,
        )
        if lines is None:
            return None
        best = None
        best_len = 0.0
        for row in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, row)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length > best_len:
                best_len = length
                best = (np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64),
                        np.array([dx / max(length, 1e-9), dy / max(length, 1e-9)], dtype=np.float64))
        return best

    def _calibrate_camera_axes(self, bgr: np.ndarray) -> bool:
        """Find the red world-X and green world-Y axes visible on the arena floor.

        The source world draws these two axes through (0,0). Their image directions
        let us determine signs/order instead of relying solely on camera optical-frame
        convention. Intrinsics and the known static camera pose then provide scale.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 90, 45]), np.array([12, 255, 255]))
        red_mask |= cv2.inRange(hsv, np.array([168, 90, 45]), np.array([179, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 70, 35]), np.array([90, 255, 255]))
        red_line = self._dominant_line(red_mask)
        green_line = self._dominant_line(green_mask)
        if red_line is None or green_line is None:
            return False
        p_red, d_red = red_line
        p_green, d_green = green_line
        determinant = d_red[0] * (-d_green[1]) - (-d_green[0]) * d_red[1]
        if abs(determinant) < 0.65:
            return False
        # Solve p_red + t*d_red = p_green + s*d_green.
        matrix = np.column_stack((d_red, -d_green))
        try:
            t, _s = np.linalg.solve(matrix, p_green - p_red)
        except np.linalg.LinAlgError:
            return False
        origin_px = p_red + t * d_red
        principal = np.array([self._cx, self._cy], dtype=np.float64)
        # The camera projects world (1.5,1.5) to the principal point and the
        # axes' crossing (0,0) lies in the negative direction along both axes.
        if float(np.dot(origin_px - principal, d_red)) > 0.0:
            d_red = -d_red
        if float(np.dot(origin_px - principal, d_green)) > 0.0:
            d_green = -d_green
        # Enforce orthogonality against small Hough-angle noise. The direction
        # closer to horizontal/vertical is retained; the perpendicular is built
        # from it while preserving the measured green-axis orientation.
        d_red = d_red / max(np.linalg.norm(d_red), 1e-9)
        d_green = d_green / max(np.linalg.norm(d_green), 1e-9)
        if abs(float(np.dot(d_red, d_green))) > 0.15:
            return False
        self._camera_dirs = (d_red, d_green)
        self._camera_axis_origin_px = origin_px
        return True

    def _pixel_to_world(self, u: float, v: float, object_z: float) -> tuple[float, float]:
        # Fallback direction for this repository's Gazebo camera pose (pitch +90°).
        # Runtime axis detection replaces this if the image's visible axes prove otherwise.
        if self._camera_dirs is None:
            dir_x = np.array([0.0, -1.0], dtype=np.float64)
            dir_y = np.array([-1.0, 0.0], dtype=np.float64)
        else:
            dir_x, dir_y = self._camera_dirs
        pixel = np.array([float(u), float(v)], dtype=np.float64)
        principal = np.array([self._cx, self._cy], dtype=np.float64)
        delta = pixel - principal
        fx_dir = math.sqrt((float(dir_x[0]) * self._fx) ** 2 + (float(dir_x[1]) * self._fy) ** 2)
        fy_dir = math.sqrt((float(dir_y[0]) * self._fx) ** 2 + (float(dir_y[1]) * self._fy) ** 2)
        scale_h = max(0.5, CAMERA_Z - float(object_z))
        wx = CAMERA_X + float(np.dot(delta, dir_x)) * scale_h / max(fx_dir, 1.0)
        wy = CAMERA_Y + float(np.dot(delta, dir_y)) * scale_h / max(fy_dir, 1.0)
        return wx, wy

    def _target_centroids(self, bgr: np.ndarray) -> list[tuple[float, float]]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 65, 35]), np.array([92, 255, 255]))
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[tuple[float, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 180.0 or area > 5000.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 16 or max(w, h) > 80:
                continue
            aspect = float(w) / max(float(h), 1.0)
            if not 0.62 <= aspect <= 1.62:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-9)
            if circularity < 0.45:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 1e-6:
                continue
            u = moments["m10"] / moments["m00"]
            v = moments["m01"] / moments["m00"]
            wx, wy = self._pixel_to_world(u, v, WAYPOINT_TOP_Z)
            # Sampling bounds are 0.45..2.40; a margin tolerates pixel/contour error.
            if not (0.25 <= wx <= 2.60 and 0.25 <= wy <= 2.60):
                continue
            detections.append((wx, wy))
        return detections

    def _obstacle_detections(self, bgr: np.ndarray) -> list[dict[str, float]]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([132, 75, 35]), np.array([179, 255, 255]))
        # Include magenta hues that wrap near zero in case the renderer shifts hue.
        mask |= cv2.inRange(hsv, np.array([0, 75, 35]), np.array([5, 255, 255]))
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result: list[dict[str, float]] = []
        for contour in contours:
            area_px = float(cv2.contourArea(contour))
            if area_px < 35.0:
                continue
            rect_px = cv2.minAreaRect(contour)
            w_px, h_px = rect_px[1]
            long_px, short_px = max(w_px, h_px), min(w_px, h_px)
            if long_px < 22.0 or short_px < 2.0 or long_px / max(short_px, 1.0) < 2.8:
                continue
            raw_points = contour.reshape(-1, 2)
            world_points = np.array(
                [self._pixel_to_world(float(u), float(v), OBSTACLE_TOP_Z) for u, v in raw_points],
                dtype=np.float32,
            )
            if len(world_points) < 4:
                continue
            rect = cv2.minAreaRect(world_points.reshape(-1, 1, 2))
            (cx, cy), (rw, rh), angle_deg = rect
            if min(rw, rh) <= 1e-4:
                continue
            if rw >= rh:
                length, width = float(rw), float(rh)
                yaw = math.radians(float(angle_deg))
            else:
                length, width = float(rh), float(rw)
                yaw = math.radians(float(angle_deg) + 90.0)
            yaw = wrap_angle(yaw)
            if length < 0.25 or length > 0.85 or width < 0.02 or width > 0.18:
                continue
            if not (0.20 <= cx <= 2.75 and 0.20 <= cy <= 2.75):
                continue
            result.append({"x": float(cx), "y": float(cy), "yaw": yaw,
                           "length": length, "width": width})
        return result

    def _register_vision_target(self, x: float, y: float, now: float) -> None:
        for target in self.targets:
            if math.hypot(target.x - x, target.y - y) < 0.13:
                if not target.completed:
                    # Smooth centroid jitter without allowing one bad frame to move a target.
                    target.x = 0.75 * target.x + 0.25 * float(x)
                    target.y = 0.75 * target.y + 0.25 * float(y)
                target.hits += 1
                target.last_seen = now
                return
        if len(self.targets) >= EXPECTED_TARGETS:
            return
        self.targets.append(Target(x, y, "vision", now))

    def _register_obstacle(self, detected: dict[str, float]) -> None:
        for idx, old in enumerate(self.obstacles):
            if math.hypot(old["x"] - detected["x"], old["y"] - detected["y"]) < 0.13:
                # A line rectangle's angle is equivalent modulo pi.
                angle_error = abs(wrap_angle(old["yaw"] - detected["yaw"]))
                angle_error = min(angle_error, abs(math.pi - angle_error))
                if angle_error < math.radians(25.0):
                    for key in ("x", "y", "length", "width"):
                        old[key] = 0.8 * old[key] + 0.2 * detected[key]
                    # Rectangle orientation is pi-periodic: +89 degrees and
                    # -91 degrees describe the same wall. Interpolate modulo pi.
                    yaw_delta = (detected["yaw"] - old["yaw"] + 0.5 * math.pi) % math.pi - 0.5 * math.pi
                    old["yaw"] = wrap_angle(old["yaw"] + 0.2 * yaw_delta)
                    self._obstacle_hits[idx] += 1
                    return
        if len(self.obstacles) < 4:
            self.obstacles.append(dict(detected))
            self._obstacle_hits.append(1)

    def _robot_visual_measurement(self, bgr: np.ndarray) -> Optional[tuple[float, float, float, float, float]]:
        """Detect the robot chassis and return (centroid_x, centroid_y, yaw, area, score).

        The organizer's robot chassis/lid are blue-gray while the arena floor is
        nearly neutral gray. The colour gate plus shape/pose gates suppress the
        red/green axes and the green/magenta task objects. Coordinates are derived
        exclusively from /overhead_camera/image_raw and the public camera mapping.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Body/lid material is blue-gray (OpenCV hue roughly 95-120), unlike the
        # low-saturation floor. Keep a broad saturation/value range for lighting.
        mask = cv2.inRange(
            hsv,
            np.array([88, 18, 45], dtype=np.uint8),
            np.array([132, 205, 245], dtype=np.uint8),
        )
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Predict the visible chassis centre from the base_link pose. Before the
        # initial offset calibration, use the known approximate visual offset.
        offset = self._robot_offset_base if self._robot_offset_base is not None else np.array([0.095, 0.0])
        cyaw, syaw = math.cos(self.yaw), math.sin(self.yaw)
        expected_cx = self.x + offset[0] * cyaw - offset[1] * syaw
        expected_cy = self.y + offset[0] * syaw + offset[1] * cyaw

        best = None
        best_score = float("inf")
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (90.0 <= area <= 2200.0):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 10 or max(w, h) > 72:
                continue
            # Use rotated-box aspect ratio rather than the axis-aligned bounding
            # box: a diagonally oriented robot can have a nearly square AABB.
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if min(rw, rh) < 1.0 or max(rw, rh) / max(min(rw, rh), 1.0) < 1.12:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 1e-6:
                continue
            u = moments["m10"] / moments["m00"]
            v = moments["m01"] / moments["m00"]
            cx_world, cy_world = self._pixel_to_world(u, v, 0.055)
            world_dist = math.hypot(cx_world - expected_cx, cy_world - expected_cy)
            gate = 0.40 if self._robot_offset_base is not None else 0.50
            if world_dist > gate:
                continue

            # Find the long axis of the rotated rectangle in world coordinates.
            box = cv2.boxPoints(rect)
            best_edge = None
            best_edge_len = -1.0
            for idx in range(4):
                p0 = box[idx]
                p1 = box[(idx + 1) % 4]
                edge_len = float(np.linalg.norm(p1 - p0))
                if edge_len > best_edge_len:
                    best_edge_len = edge_len
                    best_edge = (p0, p1)
            if best_edge is None:
                continue
            w0 = self._pixel_to_world(float(best_edge[0][0]), float(best_edge[0][1]), 0.055)
            w1 = self._pixel_to_world(float(best_edge[1][0]), float(best_edge[1][1]), 0.055)
            axis_yaw = math.atan2(w1[1] - w0[1], w1[0] - w0[0])
            # A rectangle's major axis is pi-periodic. Choose the heading closest
            # to the motion-integrated yaw; the asymmetric body helps keep it stable.
            yaw_a = wrap_angle(axis_yaw)
            yaw_b = wrap_angle(axis_yaw + math.pi)
            vis_yaw = min((yaw_a, yaw_b), key=lambda a: abs(wrap_angle(a - self.yaw)))
            yaw_error = abs(wrap_angle(vis_yaw - self.yaw))
            if yaw_error > 0.90:
                continue
            score = world_dist + 0.06 * yaw_error - 0.00001 * area
            if score < best_score:
                best_score = score
                best = (float(cx_world), float(cy_world), float(vis_yaw), area, score)
        return best

    def _update_visual_localization(self, bgr: np.ndarray, now: float) -> None:
        """Use confident overhead robot observations to bound accumulated odometry drift.

        The robot spawns at (0,0,0), which provides a calibration opportunity for
        the visible-chassis-centroid offset relative to base_link. Thereafter the
        visual pose is fused conservatively with wheel/IMU dead reckoning.
        """
        if self.tier != 2 or self._camera_dirs is None:
            return
        measurement = self._robot_visual_measurement(bgr)
        if measurement is None:
            self._robot_detection_misses += 1
            return
        center_x, center_y, visual_yaw, _area, _score = measurement
        self._robot_last_seen = now

        # Calibrate the chassis-centroid offset while the robot is still at its
        # known spawn pose. Several observations reduce pixel/contour jitter.
        if (self._robot_offset_base is None and not self._motion_started
                and math.hypot(self.x, self.y) < 0.08 and abs(self.yaw) < 0.35):
            c, si = math.cos(visual_yaw), math.sin(visual_yaw)
            dx, dy = center_x - self.x, center_y - self.y
            offset_x = c * dx + si * dy
            offset_y = -si * dx + c * dy
            if 0.035 <= offset_x <= 0.19 and abs(offset_y) <= 0.07:
                self._robot_offset_samples.append((offset_x, offset_y))
                if len(self._robot_offset_samples) >= 5:
                    samples = np.asarray(self._robot_offset_samples[-9:], dtype=np.float64)
                    self._robot_offset_base = np.median(samples, axis=0)
                    self.get_logger().info(
                        "Camera robot-localization offset calibrated: "
                        f"({self._robot_offset_base[0]:.3f}, {self._robot_offset_base[1]:.3f}) m"
                    )
            # Do not correct the state from the body-centroid measurement until
            # its offset has been calibrated against the known spawn pose.
            if self._robot_offset_base is None:
                return

        if self._robot_offset_base is None:
            return
        ox, oy = map(float, self._robot_offset_base)
        c, si = math.cos(visual_yaw), math.sin(visual_yaw)
        base_x = center_x - (ox * c - oy * si)
        base_y = center_y - (ox * si + oy * c)
        residual = math.hypot(base_x - self.x, base_y - self.y)
        self._robot_last_residual = residual
        # Reject mismatched contours rather than snapping the estimate to them.
        if residual > 0.28:
            return

        # Thirty-Hz visual correction with modest gains reduces multi-metre
        # accumulated wheel drift without injecting raw pixel noise into the pose.
        self.x += 0.28 * (base_x - self.x)
        self.y += 0.28 * (base_y - self.y)
        yaw_error = wrap_angle(visual_yaw - self.yaw)
        if abs(yaw_error) < 0.55:
            self.yaw = wrap_angle(self.yaw + 0.12 * yaw_error)
        self._robot_visual_updates += 1

    def _on_image(self, msg: Image) -> None:
        if self.tier == 1:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # encoding / bridge errors should be visible, not fatal
            self.get_logger().warn(f"Cannot decode overhead image: {exc}", throttle_duration_sec=5.0)
            return
        self._camera_frame_count += 1
        now = self._now()
        if self._camera_dirs is None:
            self._calibrate_camera_axes(bgr)
            # The exact SDF camera pose gives a valid fallback if axis masks are
            # temporarily obscured. Once calibrated, image-derived axis direction wins.
            if self._camera_dirs is None and self._camera_frame_count >= 30:
                self._camera_dirs = (np.array([0.0, -1.0]), np.array([-1.0, 0.0]))
                self.get_logger().warn("Arena-axis image calibration unavailable; using the SDF camera-pose fallback.")

        # Do not register detections while the mapping orientation is unknown: doing
        # so could fill the seven-target table using fallback coordinates and later
        # leave no room for the correctly calibrated detections.
        if self._camera_dirs is None:
            return
        self._update_visual_localization(bgr, now)
        for wx, wy in self._target_centroids(bgr):
            self._register_vision_target(wx, wy, now)
        if self.tier == 3:
            for obstacle in self._obstacle_detections(bgr):
                self._register_obstacle(obstacle)

        if self.debug_perception and now - self._last_perception_log >= 1.0:
            self._last_perception_log = now
            self.get_logger().info(
                f"perception targets={sum(t.stable for t in self.targets)}/{EXPECTED_TARGETS}, "
                f"obstacles={sum(h >= 3 for h in self._obstacle_hits) if self.tier == 3 else 0}/4, "
                f"axis_calibrated={self._camera_dirs is not None}, "
                f"robot_vis_updates={self._robot_visual_updates}, "
                f"robot_residual={self._robot_last_residual if self._robot_last_residual is not None else 'n/a'}, "
                f"robot_offset={tuple(np.round(self._robot_offset_base, 3)) if self._robot_offset_base is not None else 'uncalibrated'}"
            )

    # ------------------------------------------------------------------
    # Tier 1 waypoint subscriptions
    # ------------------------------------------------------------------
    def _register_topic_target(self, x: float, y: float, now: float) -> None:
        for target in self.targets:
            if math.hypot(target.x - x, target.y - y) < 0.10:
                # Topic coordinates are exact for the purposes of this simulation.
                target.x = float(x)
                target.y = float(y)
                target.last_seen = now
                target.hits += 1
                return
        if len(self.targets) < EXPECTED_TARGETS:
            self.targets.append(Target(x, y, "topic", now))

    def _on_waypoints(self, msg: PoseArray) -> None:
        now = self._now()
        for pose in msg.poses:
            self._register_topic_target(float(pose.position.x), float(pose.position.y), now)

    def _on_dynamic_waypoints(self, msg: PoseArray) -> None:
        now = self._now()
        for pose in msg.poses:
            self._register_topic_target(float(pose.position.x), float(pose.position.y), now)

    # ------------------------------------------------------------------
    # Navigation / route optimization
    # ------------------------------------------------------------------
    def _stable_target_indices(self) -> list[int]:
        return [i for i, target in enumerate(self.targets) if target.stable]

    def _targets_ready(self) -> bool:
        stable_count = len(self._stable_target_indices())
        if self.tier == 1:
            return stable_count >= 5
        if self.tier == 2:
            return stable_count >= 5 and self._camera_dirs is not None
        # Avoid launching into Tier 3 until all 4 obstacle rectangles have been
        # independently observed in multiple camera frames.
        return (stable_count >= 5 and self._camera_dirs is not None
                and sum(h >= 3 for h in self._obstacle_hits) >= 4)

    def _all_targets_known(self) -> bool:
        return len(self._stable_target_indices()) >= EXPECTED_TARGETS

    def _free_target_indices(self) -> list[int]:
        return [i for i, target in enumerate(self.targets) if target.stable and not target.completed]

    def _build_occupancy(self) -> Optional[np.ndarray]:
        if self.tier != 3 or sum(h >= 3 for h in self._obstacle_hits) < 4:
            return None
        n = int(round(GRID_MAX / GRID_RESOLUTION)) + 1
        coords = np.arange(n, dtype=np.float64) * GRID_RESOLUTION
        X, Y = np.meshgrid(coords, coords, indexing="ij")
        occupied = np.zeros((n, n), dtype=bool)
        for obs, hits in zip(self.obstacles, self._obstacle_hits):
            if hits < 3:
                continue
            dx = X - obs["x"]
            dy = Y - obs["y"]
            c = math.cos(obs["yaw"])
            s = math.sin(obs["yaw"])
            local_x = dx * c + dy * s
            local_y = -dx * s + dy * c
            out_x = np.maximum(np.abs(local_x) - 0.5 * obs["length"], 0.0)
            out_y = np.maximum(np.abs(local_y) - 0.5 * obs["width"], 0.0)
            occupied |= (out_x * out_x + out_y * out_y) <= (OBSTACLE_INFLATION ** 2)

        # Outer-boundary clearance: keep routes away from the perimeter except
        # for the intentional start/home pocket. A hard 0.22 m margin here would
        # disconnect (0,0) from the rest of the grid, so only the first 8 cm near
        # either boundary is excluded outside the home pocket.
        radius_from_home = np.hypot(X, Y)
        unsafe_low_edge = ((X < 0.08) | (Y < 0.08)) & (radius_from_home > 0.25)
        occupied |= unsafe_low_edge
        occupied |= (X > 2.60) | (Y > 2.60)
        home_pocket = (X < 0.30) & (Y < 0.30) & (radius_from_home <= 0.30)
        occupied[home_pocket] = False
        return occupied

    @staticmethod
    def _grid_cell(point: tuple[float, float], shape: tuple[int, int]) -> tuple[int, int]:
        ix = int(round(point[0] / GRID_RESOLUTION))
        iy = int(round(point[1] / GRID_RESOLUTION))
        return max(0, min(shape[0] - 1, ix)), max(0, min(shape[1] - 1, iy))

    @staticmethod
    def _cell_world(cell: tuple[int, int]) -> tuple[float, float]:
        return cell[0] * GRID_RESOLUTION, cell[1] * GRID_RESOLUTION

    @staticmethod
    def _nearest_free(cell: tuple[int, int], occupancy: np.ndarray) -> tuple[int, int]:
        if not occupancy[cell[0], cell[1]]:
            return cell
        nx, ny = occupancy.shape
        for radius in range(1, 12):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    xx, yy = cell[0] + dx, cell[1] + dy
                    if 0 <= xx < nx and 0 <= yy < ny and not occupancy[xx, yy]:
                        candidates.append((xx, yy))
            if candidates:
                return min(candidates, key=lambda p: (p[0]-cell[0])**2 + (p[1]-cell[1])**2)
        return cell

    def _astar(self, start: tuple[float, float], goal: tuple[float, float],
               occupancy: np.ndarray) -> tuple[list[tuple[float, float]], float]:
        shape = occupancy.shape
        s = self._nearest_free(self._grid_cell(start, shape), occupancy)
        g = self._nearest_free(self._grid_cell(goal, shape), occupancy)
        if occupancy[s[0], s[1]] or occupancy[g[0], g[1]]:
            return [], math.inf
        directions = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                      (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
                      (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]
        g_score = {s: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        heap = [(math.hypot(g[0]-s[0], g[1]-s[1]), 0.0, s)]
        visited: set[tuple[int, int]] = set()
        while heap:
            _f, cost, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            if current == g:
                cells = [current]
                while cells[-1] != s:
                    cells.append(came_from[cells[-1]])
                cells.reverse()
                points = [self._cell_world(c) for c in cells]
                # Replace snapped endpoints with the exact requested coordinates.
                if points:
                    points[0] = (float(start[0]), float(start[1]))
                    points[-1] = (float(goal[0]), float(goal[1]))
                return self._smooth_grid_path(points, occupancy), cost * GRID_RESOLUTION
            for dx, dy, step in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < shape[0] and 0 <= ny < shape[1]) or occupancy[nx, ny]:
                    continue
                # Do not cut diagonally through the corner of two blocked cells.
                if dx != 0 and dy != 0 and (occupancy[current[0] + dx, current[1]] or occupancy[current[0], current[1] + dy]):
                    continue
                nxt = (nx, ny)
                new_cost = cost + step
                if new_cost < g_score.get(nxt, math.inf):
                    g_score[nxt] = new_cost
                    came_from[nxt] = current
                    heuristic = math.hypot(g[0] - nx, g[1] - ny)
                    heapq.heappush(heap, (new_cost + heuristic, new_cost, nxt))
        return [], math.inf

    def _line_free(self, a: tuple[float, float], b: tuple[float, float], occupancy: np.ndarray) -> bool:
        distance = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(math.ceil(distance / (GRID_RESOLUTION * 0.35))))
        for i in range(steps + 1):
            t = i / steps
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            cell = self._grid_cell(p, occupancy.shape)
            if occupancy[cell[0], cell[1]]:
                return False
        return True

    def _smooth_grid_path(self, points: list[tuple[float, float]], occupancy: np.ndarray) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return points
        smoothed = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._line_free(points[i], points[j], occupancy):
                j -= 1
            smoothed.append(points[j])
            i = j
        return smoothed

    def _route_next_target(self) -> Optional[int]:
        remaining = self._free_target_indices()
        if not remaining:
            return None
        current = (self.x, self.y)
        home_in_route = self._all_targets_known() and all(t.completed for t in self.targets if t.stable)
        # Tier 1/2: exact Euclidean TSP over at most seven points. Keep the
        # current target fixed until collection, then re-optimize remaining stops.
        if self.tier < 3:
            best_order = None
            best_cost = math.inf
            for order in itertools.permutations(remaining):
                cost = math.hypot(current[0] - self.targets[order[0]].x,
                                  current[1] - self.targets[order[0]].y)
                for a, b in zip(order, order[1:]):
                    ta, tb = self.targets[a], self.targets[b]
                    cost += math.hypot(ta.x - tb.x, ta.y - tb.y)
                if self._all_targets_known():
                    last = self.targets[order[-1]]
                    cost += math.hypot(last.x - self.home_goal[0], last.y - self.home_goal[1])
                if cost < best_cost:
                    best_cost, best_order = cost, order
            return best_order[0] if best_order else remaining[0]

        occupancy = self._build_occupancy()
        if occupancy is None:
            return None
        points = [current] + [(self.targets[i].x, self.targets[i].y) for i in remaining]
        if self._all_targets_known():
            points.append(self.home_goal)
        matrix: dict[tuple[int, int], float] = {}
        for i, p in enumerate(points):
            for j, q in enumerate(points):
                if i == j:
                    matrix[(i, j)] = 0.0
                elif (j, i) in matrix:
                    matrix[(i, j)] = matrix[(j, i)]
                else:
                    _path, dist = self._astar(p, q, occupancy)
                    matrix[(i, j)] = dist
        best_order = None
        best_cost = math.inf
        n_remaining = len(remaining)
        for order in itertools.permutations(range(1, n_remaining + 1)):
            cost = matrix[(0, order[0])]
            for a, b in zip(order, order[1:]):
                cost += matrix[(a, b)]
            if self._all_targets_known():
                cost += matrix[(order[-1], n_remaining + 1)]
            if cost < best_cost:
                best_cost, best_order = cost, order
        if best_order is None or not math.isfinite(best_cost):
            return None
        return remaining[best_order[0] - 1]

    def _plan_to(self, goal: tuple[float, float], is_home: bool = False) -> bool:
        self.current_goal = (float(goal[0]), float(goal[1]))
        self.current_goal_is_home = bool(is_home)
        if self.tier < 3:
            self.path_points = [self.current_goal]
        else:
            occupancy = self._build_occupancy()
            if occupancy is None:
                return False
            path, _distance = self._astar((self.x, self.y), self.current_goal, occupancy)
            if not path:
                return False
            self.path_points = path
            if not self.path_points or math.hypot(self.path_points[-1][0] - self.current_goal[0],
                                                  self.path_points[-1][1] - self.current_goal[1]) > 0.03:
                self.path_points.append(self.current_goal)
        self.path_index = 0
        self.phase = "NAVIGATE"
        self.dwell_start = None
        return True

    def _begin_next_leg_if_needed(self) -> None:
        if self.current_target_index is not None or self.phase in ("DWELL_HOME", "COMPLETE"):
            return
        target_idx = self._route_next_target()
        if target_idx is not None:
            target = self.targets[target_idx]
            self.current_target_index = target_idx
            if not self._plan_to((target.x, target.y), is_home=False):
                if self._now() - self._last_route_error_time >= 5.0:
                    self.get_logger().error(f"No safe path to waypoint {target_idx + 1}; remaining stopped.")
                    self._last_route_error_time = self._now()
                self.current_target_index = None
                self.current_goal = None
                self.path_points = []
                self._route_failed_logged = True
            return
        # Do not declare home completion until the two dynamic targets have
        # appeared and every one of the seven targets has completed dwell.
        if self._all_targets_known() and all(t.completed for t in self.targets if t.stable):
            if not self._home_started:
                self._home_started = True
                if not self._plan_to(self.home_goal, is_home=True):
                    if self._now() - self._last_route_error_time >= 5.0:
                        self.get_logger().error("No safe path to home; remaining stopped.")
                        self._last_route_error_time = self._now()
                    self._home_started = False
                    self._route_failed_logged = True
        elif self._free_target_indices() and self.tier == 3:
            if self._now() - self._last_route_error_time >= 5.0:
                self.get_logger().error("No collision-free target ordering found; remaining stopped.")
                self._last_route_error_time = self._now()
        # If only the five static targets have been seen and they are all done,
        # wait at a safe stop for dynamic targets to appear.

    def _ready_to_move(self) -> bool:
        return self._odom_ready and self._targets_ready()

    def _control_loop(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        now = self._now()
        self._publish_odom_and_tf(now_msg)

        # Compact diagnostics help distinguish localization failure from target
        # detection/route-state problems without accessing any evaluator topics.
        if now - self._last_status_log >= 3.0:
            self._last_status_log = now
            if self.current_target_index is not None and self.current_target_index < len(self.targets):
                _t = self.targets[self.current_target_index]
                _target_text = f"wp{self.current_target_index + 1}=({_t.x:.2f},{_t.y:.2f}) err={math.hypot(self.x-_t.x, self.y-_t.y):.2f}m"
            elif self.current_goal is not None:
                _target_text = f"goal=({self.current_goal[0]:.2f},{self.current_goal[1]:.2f}) err={math.hypot(self.x-self.current_goal[0], self.y-self.current_goal[1]):.2f}m"
            else:
                _target_text = "goal=none"
            self.get_logger().info(
                f"diag phase={self.phase} pose=({self.x:.2f},{self.y:.2f},{self.yaw:.2f}) "
                f"v={self.linear_speed:.3f} w={self.yaw_rate:.3f} {_target_text} "
                f"enc_signs=({self._linear_encoder_sign},{self._yaw_encoder_sign}) "
                f"targets={sum(t.completed for t in self.targets)}/{len(self.targets)}"
            )

        if self.phase == "COMPLETE":
            self._publish_cmd(0.0, 0.0)
            return
        if not self._ready_to_move():
            self._publish_cmd(0.0, 0.0)
            return

        # Waiting for late targets: stay stopped, but keep estimator and camera
        # callbacks alive; the dynamic-spawn clock has already started if we moved.
        if self.current_target_index is None and self.phase != "DWELL_HOME":
            self._begin_next_leg_if_needed()
            if self.current_target_index is None and self.phase == "WAIT":
                self._publish_cmd(0.0, 0.0)
                return
            if self.current_goal is None and self.phase != "DWELL_HOME":
                self._publish_cmd(0.0, 0.0)
                return

        if self.phase == "DWELL_WP":
            self._publish_cmd(0.0, 0.0)
            target = self.targets[self.current_target_index] if self.current_target_index is not None else None
            if target is None:
                self.phase = "WAIT"
                self.dwell_start = None
                return
            distance = math.hypot(self.x - target.x, self.y - target.y)
            if distance > 0.16:
                self.phase = "NAVIGATE"
                self.dwell_start = None
                self._plan_to((target.x, target.y), is_home=False)
                return
            if abs(self.linear_speed) <= STOP_SPEED_EST:
                if self.dwell_start is None:
                    self.dwell_start = now
                elif now - self.dwell_start >= WP_STOP_DWELL:
                    target.completed = True
                    self.get_logger().info(
                        f"Waypoint {self.current_target_index + 1} dwell complete at "
                        f"({target.x:.3f}, {target.y:.3f}); {sum(t.completed for t in self.targets)}/{len(self.targets)} collected."
                    )
                    self.current_target_index = None
                    self.current_goal = None
                    self.path_points = []
                    self.path_index = 0
                    self.dwell_start = None
                    self.phase = "WAIT"
            else:
                self.dwell_start = None
            return

        if self.phase == "DWELL_HOME":
            self._publish_cmd(0.0, 0.0)
            home_goal_error = math.hypot(self.x - self.home_goal[0], self.y - self.home_goal[1])
            if home_goal_error > 0.055:
                # Return toward the selected home approach point if odometry drifts,
                # while preserving margin inside the evaluator's 0.20 m home radius.
                if self._plan_to(self.home_goal, is_home=True):
                    self._home_started = True
                return
            if abs(self.linear_speed) <= STOP_SPEED_EST and abs(self.yaw_rate) < 0.10:
                if self.dwell_start is None:
                    self.dwell_start = now
                elif now - self.dwell_start >= HOME_STOP_DWELL:
                    self.phase = "COMPLETE"
                    self.get_logger().info("All seven waypoint dwells complete; home dwell complete.")
            else:
                self.dwell_start = None
            return

        # NAVIGATE phase.
        if self.current_goal is None or not self.path_points:
            self._publish_cmd(0.0, 0.0)
            self.phase = "WAIT"
            return

        final_distance = math.hypot(self.x - self.current_goal[0], self.y - self.current_goal[1])
        if self.current_goal_is_home and final_distance <= 0.040:
            # Home scoring requires a stationary dwell, not a final heading.
            # Avoid an unnecessary in-place rotation near the perimeter wall.
            self._publish_cmd(0.0, 0.0)
            self.phase = "DWELL_HOME"
            self.dwell_start = None
            return
        if not self.current_goal_is_home and final_distance <= 0.105:
            self._publish_cmd(0.0, 0.0)
            self.phase = "DWELL_WP"
            self.dwell_start = None
            return

        # Advance intermediate path points. The last point is the exact waypoint.
        while self.path_index < len(self.path_points) - 1:
            p = self.path_points[self.path_index]
            if math.hypot(self.x - p[0], self.y - p[1]) <= 0.07:
                self.path_index += 1
            else:
                break
        subgoal = self.path_points[min(self.path_index, len(self.path_points) - 1)]
        dx, dy = subgoal[0] - self.x, subgoal[1] - self.y
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_yaw - self.yaw)

        # Rotate in place for large heading errors. This obeys v >= 0 and avoids
        # forward arcs that could clip inflated obstacles during alignment.
        if abs(heading_error) > 0.40:
            angular = clamp(1.8 * heading_error, -1.20, 1.20)
            self._publish_cmd(0.0, angular)
            return

        # Proportional speed with braking near every subgoal; never command reverse.
        speed = min(0.24, 0.85 * distance)
        speed *= max(0.25, math.cos(heading_error))
        if distance < 0.16:
            speed = min(speed, 0.075)
        angular = clamp(2.2 * heading_error, -1.0, 1.0)
        self._publish_cmd(clamp(speed, 0.035, MAX_LINEAR), angular)


def spin_tier(tier: int) -> None:
    rclpy.init()
    node = Sim2RealAutonomyNode(tier)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# Import lazily to keep startup imports simple on systems whose rclpy version
# exports ExternalShutdownException from rclpy.executors instead of rclpy.
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # pragma: no cover - compatibility fallback
    class ExternalShutdownException(Exception):
        pass


def main() -> None:
    spin_tier(1)


if __name__ == "__main__":
    main()
