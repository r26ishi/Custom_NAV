#!/usr/bin/env python3

import math
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist

import tf2_ros
from tf2_ros import TransformException


# ================================================================
# Utility functions
# ================================================================

def normalize_angle(angle: float) -> float:

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def quaternion_to_yaw(q) -> float:

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)

    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

    return math.atan2(siny_cosp, cosy_cosp)


# ================================================================
# Controller
# ================================================================

class Nav2StyleController(Node):

    def __init__(self):

        super().__init__('nav2_style_controller')

        # ========================================================
        # Parameters
        # ========================================================

        self.declare_parameter(
            'plan_topic',
            '/plan'
        )

        self.declare_parameter(
            'odom_topic',
            '/odom'
        )

        self.declare_parameter(
            'cmd_vel_topic',
            '/cmd_vel'
        )

        self.declare_parameter(
            'global_frame',
            'map'
        )

        self.declare_parameter(
            'robot_frame',
            'base_link'
        )

        # Controller frequency

        self.declare_parameter(
            'controller_frequency',
            30.0
        )

        # --------------------------------------------------------
        # Lookahead
        # --------------------------------------------------------

        self.declare_parameter(
            'lookahead_distance',
            0.50
        )

        self.declare_parameter(
            'min_lookahead_distance',
            0.30
        )

        self.declare_parameter(
            'max_lookahead_distance',
            0.80
        )

        # --------------------------------------------------------
        # Velocity
        # --------------------------------------------------------

        self.declare_parameter(
            'max_linear_velocity',
            0.60
        )

        self.declare_parameter(
            'min_linear_velocity',
            0.05
        )

        self.declare_parameter(
            'max_angular_velocity',
            0.40
        )

        # --------------------------------------------------------
        # Acceleration limits
        # --------------------------------------------------------

        self.declare_parameter(
            'max_linear_acceleration',
            0.60
        )

        self.declare_parameter(
            'max_angular_acceleration',
            0.40
        )

        # --------------------------------------------------------
        # Goal tolerances
        # --------------------------------------------------------

        self.declare_parameter(
            'goal_position_tolerance',
            0.15
        )

        self.declare_parameter(
            'goal_yaw_tolerance',
            0.30
        )

        # --------------------------------------------------------
        # Curvature regulation
        # --------------------------------------------------------

        self.declare_parameter(
            'curvature_velocity_scale',
            1.0
        )

        # --------------------------------------------------------
        # Heading control
        # --------------------------------------------------------

        self.declare_parameter(
            'heading_gain',
            2.0
        )

        # ========================================================
        # Get parameters
        # ========================================================

        self.plan_topic = self.get_parameter(
            'plan_topic'
        ).value

        self.odom_topic = self.get_parameter(
            'odom_topic'
        ).value

        self.cmd_vel_topic = self.get_parameter(
            'cmd_vel_topic'
        ).value

        self.global_frame = self.get_parameter(
            'global_frame'
        ).value

        self.robot_frame = self.get_parameter(
            'robot_frame'
        ).value

        self.controller_frequency = self.get_parameter(
            'controller_frequency'
        ).value

        self.lookahead_distance = self.get_parameter(
            'lookahead_distance'
        ).value

        self.min_lookahead_distance = self.get_parameter(
            'min_lookahead_distance'
        ).value

        self.max_lookahead_distance = self.get_parameter(
            'max_lookahead_distance'
        ).value

        self.max_linear_velocity = self.get_parameter(
            'max_linear_velocity'
        ).value

        self.min_linear_velocity = self.get_parameter(
            'min_linear_velocity'
        ).value

        self.max_angular_velocity = self.get_parameter(
            'max_angular_velocity'
        ).value

        self.max_linear_acceleration = self.get_parameter(
            'max_linear_acceleration'
        ).value

        self.max_angular_acceleration = self.get_parameter(
            'max_angular_acceleration'
        ).value

        self.goal_position_tolerance = self.get_parameter(
            'goal_position_tolerance'
        ).value

        self.goal_yaw_tolerance = self.get_parameter(
            'goal_yaw_tolerance'
        ).value

        self.curvature_velocity_scale = self.get_parameter(
            'curvature_velocity_scale'
        ).value

        self.heading_gain = self.get_parameter(
            'heading_gain'
        ).value

        # ========================================================
        # State
        # ========================================================

        self.plan: Optional[Path] = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.robot_linear_velocity = 0.0
        self.robot_angular_velocity = 0.0

        self.have_odom = False

        self.current_linear_cmd = 0.0
        self.current_angular_cmd = 0.0

        self.goal_reached = False
        self.debug_counter = 0  # Print detailed controller state periodically

        # ========================================================
        # TF
        # ========================================================

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # ========================================================
        # Subscribers
        # ========================================================

        self.plan_sub = self.create_subscription(
            Path,
            self.plan_topic,
            self.plan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # ========================================================
        # Publisher
        # ========================================================

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )

        # ========================================================
        # Controller timer
        # ========================================================

        period = 1.0 / self.controller_frequency

        self.timer = self.create_timer(
            period,
            self.control_loop
        )

        self.get_logger().info(
            'Nav2-style controller started'
        )

    # ============================================================
    # PLAN CALLBACK
    # ============================================================

    def plan_callback(self, msg: Path):

        if len(msg.poses) == 0:

            self.plan = None

            self.stop_robot()

            self.get_logger().warn(
                'Received empty plan'
            )

            return

        self.plan = msg

        self.goal_reached = False

        start = msg.poses[0].pose.position
        goal = msg.poses[-1].pose.position
        self.get_logger().info(
            f'Received plan: poses={len(msg.poses)}, '
            f'frame="{msg.header.frame_id}", '
            f'start=({start.x:.3f}, {start.y:.3f}), '
            f'goal=({goal.x:.3f}, {goal.y:.3f})'
        )

    # ============================================================
    # ODOM CALLBACK
    # ============================================================

    def odom_callback(self, msg: Odometry):

        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        self.robot_yaw = quaternion_to_yaw(
            msg.pose.pose.orientation
        )

        self.robot_linear_velocity = (
            msg.twist.twist.linear.x
        )

        self.robot_angular_velocity = (
            msg.twist.twist.angular.z
        )

        self.have_odom = True
        # Periodic odometry diagnostics (avoid flooding the terminal).
        if not hasattr(self, '_odom_debug_counter'):
            self._odom_debug_counter = 0
        self._odom_debug_counter += 1
        if self._odom_debug_counter % 50 == 0:
            self.get_logger().info(
                f'[ODOM] x={self.robot_x:.3f}, y={self.robot_y:.3f}, '
                f'yaw={self.robot_yaw:.3f} rad, '
                f'v={self.robot_linear_velocity:.3f} m/s, '
                f'w={self.robot_angular_velocity:.3f} rad/s'
            )

    # ============================================================
    # CONTROL LOOP
    # ============================================================

    def control_loop(self):
        self.debug_counter += 1
        debug_now = (self.debug_counter % 20 == 0)

        if self.plan is None:
            if debug_now:
                self.get_logger().warn('[DEBUG] No plan received; publishing stop command')

            self.stop_robot()

            return

        if not self.have_odom:
            if debug_now:
                self.get_logger().warn('[DEBUG] No odometry received yet; publishing stop command')

            self.stop_robot()

            return

        # --------------------------------------------------------
        # Get robot pose in plan/global frame
        # --------------------------------------------------------

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            if debug_now:
                self.get_logger().warn(
                    f'[DEBUG] Cannot get robot pose in frame "{self.global_frame}" '
                    f'from "{self.robot_frame}"; check TF tree'
                )

            self.stop_robot()

            return

        robot_x, robot_y, robot_yaw = robot_pose
        if debug_now:
            self.get_logger().info(
                f'[POSE] global x={robot_x:.3f}, y={robot_y:.3f}, '
                f'yaw={robot_yaw:.3f} rad'
            )

        # --------------------------------------------------------
        # Check goal
        # --------------------------------------------------------

        if self.check_goal_reached(
            robot_x,
            robot_y,
            robot_yaw
        ):

            self.stop_robot()

            if not self.goal_reached:

                self.get_logger().info(
                    'Goal reached'
                )

            self.goal_reached = True

            return

        # --------------------------------------------------------
        # Transform plan into robot frame
        # --------------------------------------------------------

        local_plan = self.transform_plan_to_robot_frame()

        if local_plan is None:
            if debug_now:
                self.get_logger().warn(
                    f'[DEBUG] Plan transform failed: plan frame="{self.plan.header.frame_id}", '
                    f'robot frame="{self.robot_frame}"'
                )

            self.stop_robot()

            return

        # --------------------------------------------------------
        # Find closest point
        # --------------------------------------------------------

        closest_index = self.find_closest_point(
            local_plan
        )

        if debug_now:
            cx, cy = local_plan[closest_index]
            self.get_logger().info(
                f'[PATH] points={len(local_plan)}, closest_index={closest_index}, '
                f'closest_local=({cx:.3f}, {cy:.3f})'
            )

        # --------------------------------------------------------
        # Dynamic lookahead
        # --------------------------------------------------------

        lookahead_distance = self.compute_lookahead_distance()

        # --------------------------------------------------------
        # Find lookahead point
        # --------------------------------------------------------

        target = self.find_lookahead_point(
            local_plan,
            closest_index,
            lookahead_distance
        )

        if target is None:
            if debug_now:
                self.get_logger().warn('[DEBUG] No lookahead target found; stopping')

            self.stop_robot()

            return

        target_x, target_y = target
        if debug_now:
            self.get_logger().info(
                f'[TARGET] local=({target_x:.3f}, {target_y:.3f}), '
                f'distance={math.hypot(target_x, target_y):.3f} m, '
                f'lookahead={lookahead_distance:.3f} m'
            )

        # --------------------------------------------------------
        # Calculate curvature
        #
        # Pure pursuit:
        #
        # curvature = 2*y / L²
        #
        # where:
        #
        # y = lateral position of target
        # L = distance to target
        # --------------------------------------------------------

        distance = math.hypot(
            target_x,
            target_y
        )

        if distance < 0.001:
            if debug_now:
                self.get_logger().warn('[DEBUG] Target is almost at robot origin; stopping')

            self.stop_robot()

            return

        curvature = (
            2.0 * target_y
            /
            (distance * distance)
        )

        # --------------------------------------------------------
        # Generate velocity
        # --------------------------------------------------------

        desired_linear_velocity = (
            self.max_linear_velocity
        )

        # --------------------------------------------------------
        # Curvature-based velocity regulation
        # --------------------------------------------------------

        curvature_factor = (
            1.0 /
            (
                1.0
                +
                self.curvature_velocity_scale
                *
                abs(curvature)
            )
        )

        desired_linear_velocity *= curvature_factor

        # --------------------------------------------------------
        # If target is strongly to the side,
        # slow down.
        # --------------------------------------------------------

        heading_error = math.atan2(
            target_y,
            target_x
        )

        heading_factor = max(
            0.1,
            math.cos(heading_error)
        )

        desired_linear_velocity *= heading_factor

        # --------------------------------------------------------
        # Don't move backwards
        # --------------------------------------------------------

        desired_linear_velocity = max(
            self.min_linear_velocity,
            desired_linear_velocity
        )

        desired_linear_velocity = min(
            desired_linear_velocity,
            self.max_linear_velocity
        )

        # --------------------------------------------------------
        # Angular velocity
        # --------------------------------------------------------

        desired_angular_velocity = (
            desired_linear_velocity
            *
            curvature
        )

        # --------------------------------------------------------
        # Heading correction
        # --------------------------------------------------------

        desired_angular_velocity += (
            self.heading_gain
            *
            heading_error
        )

        # --------------------------------------------------------
        # Limit angular velocity
        # --------------------------------------------------------

        desired_angular_velocity = max(
            -self.max_angular_velocity,
            min(
                desired_angular_velocity,
                self.max_angular_velocity
            )
        )

        # --------------------------------------------------------
        # Slow down close to goal
        # --------------------------------------------------------

        goal_distance = self.distance_to_goal(
            robot_x,
            robot_y
        )

        if goal_distance < 1.0:

            goal_factor = max(
                0.15,
                goal_distance / 1.0
            )

            desired_linear_velocity *= goal_factor

        # --------------------------------------------------------
        # Apply acceleration limits
        # --------------------------------------------------------

        dt = 1.0 / self.controller_frequency

        linear_velocity = self.acceleration_limit(
            self.current_linear_cmd,
            desired_linear_velocity,
            self.max_linear_acceleration,
            dt
        )

        angular_velocity = self.acceleration_limit(
            self.current_angular_cmd,
            desired_angular_velocity,
            self.max_angular_acceleration,
            dt
        )

        # --------------------------------------------------------
        # Save commands
        # --------------------------------------------------------

        self.current_linear_cmd = linear_velocity
        self.current_angular_cmd = angular_velocity

        # --------------------------------------------------------
        # Publish
        # --------------------------------------------------------

        cmd = Twist()

        cmd.linear.x = linear_velocity

        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = angular_velocity

        self.cmd_vel_pub.publish(cmd)

        if debug_now:
            self.get_logger().info(
                f'[CONTROL] heading_error={heading_error:.3f} rad, '
                f'curvature={curvature:.3f} 1/m, '
                f'v_desired={desired_linear_velocity:.3f} m/s, '
                f'w_desired={desired_angular_velocity:.3f} rad/s, '
                f'v_cmd={linear_velocity:.3f} m/s, '
                f'w_cmd={angular_velocity:.3f} rad/s'
            )
            self.get_logger().info(
                f'[CMD_VEL] publishing linear.x={cmd.linear.x:.3f}, '
                f'angular.z={cmd.angular.z:.3f}'
            )

    # ============================================================
    # GET ROBOT POSE
    # ============================================================

    def get_robot_pose(self) -> Optional[
        Tuple[float, float, float]
    ]:

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    self.global_frame,
                    self.robot_frame,
                    rclpy.time.Time()
                )
            )

            x = transform.transform.translation.x

            y = transform.transform.translation.y

            yaw = quaternion_to_yaw(
                transform.transform.rotation
            )

            return x, y, yaw

        except TransformException as ex:

            self.get_logger().warn(
                f'[TF ERROR] lookup_transform(target="{self.global_frame}", '
                f'source="{self.robot_frame}") failed: {ex}'
            )

            return None

    # ============================================================
    # TRANSFORM PLAN TO ROBOT FRAME
    # ============================================================

    def transform_plan_to_robot_frame(self):

        if self.plan is None:

            return None

        local_points = []

        # --------------------------------------------------------
        # Get map -> base_link
        # --------------------------------------------------------

        try:

            transform = (
                self.tf_buffer.lookup_transform(
                    self.robot_frame,
                    self.plan.header.frame_id,
                    rclpy.time.Time()
                )
            )

        except TransformException as ex:

            self.get_logger().warn(
                f'[TF ERROR] lookup_transform(target="{self.robot_frame}", '
                f'source="{self.plan.header.frame_id}") failed: {ex}'
            )

            return None

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y

        yaw = quaternion_to_yaw(
            transform.transform.rotation
        )

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # --------------------------------------------------------
        # Transform every path point
        # --------------------------------------------------------

        for pose in self.plan.poses:

            x = pose.pose.position.x
            y = pose.pose.position.y

            # lookup_transform(robot_frame, plan_frame) already gives
            # the transform from the plan/global frame into robot_frame:
            # p_robot = R * p_plan + translation.
            local_x = cos_yaw * x - sin_yaw * y + tx
            local_y = sin_yaw * x + cos_yaw * y + ty

            local_points.append(
                (local_x, local_y)
            )

        return local_points

    # ============================================================
    # FIND CLOSEST POINT
    # ============================================================

    def find_closest_point(
        self,
        points: List[Tuple[float, float]]
    ):

        if not points:

            return 0

        closest_index = 0
        closest_distance = float('inf')

        for i, (x, y) in enumerate(points):

            distance = math.hypot(
                x,
                y
            )

            if distance < closest_distance:

                closest_distance = distance
                closest_index = i

        return closest_index

    # ============================================================
    # LOOKAHEAD
    # ============================================================

    def compute_lookahead_distance(self):

        velocity = abs(
            self.robot_linear_velocity
        )

        lookahead = (
            self.lookahead_distance
            +
            0.5 * velocity
        )

        lookahead = max(
            self.min_lookahead_distance,
            lookahead
        )

        lookahead = min(
            self.max_lookahead_distance,
            lookahead
        )

        return lookahead

    # ============================================================
    # FIND LOOKAHEAD POINT
    # ============================================================

    def find_lookahead_point(
        self,
        points,
        closest_index,
        lookahead_distance
    ):

        for i in range(
            closest_index,
            len(points)
        ):

            x, y = points[i]

            distance = math.hypot(
                x,
                y
            )

            if distance >= lookahead_distance:

                return x, y

        # --------------------------------------------------------
        # If no lookahead point exists,
        # use final point.
        # --------------------------------------------------------

        if points:

            return points[-1]

        return None

    # ============================================================
    # GOAL CHECK
    # ============================================================

    def check_goal_reached(
        self,
        robot_x,
        robot_y,
        robot_yaw
    ):

        if self.plan is None:
            return False

        goal_pose = self.plan.poses[-1]

        goal_x = goal_pose.pose.position.x
        goal_y = goal_pose.pose.position.y

        distance = math.hypot(
            goal_x - robot_x,
            goal_y - robot_y
        )

        if distance > self.goal_position_tolerance:

            return False

        goal_yaw = quaternion_to_yaw(
            goal_pose.pose.orientation
        )

        yaw_error = normalize_angle(
            goal_yaw - robot_yaw
        )

        return abs(yaw_error) <= (
            self.goal_yaw_tolerance
        )

    # ============================================================
    # DISTANCE TO GOAL
    # ============================================================

    def distance_to_goal(
        self,
        robot_x,
        robot_y
    ):

        if self.plan is None:

            return float('inf')

        goal = self.plan.poses[-1]

        return math.hypot(
            goal.pose.position.x - robot_x,
            goal.pose.position.y - robot_y
        )

    # ============================================================
    # ACCELERATION LIMITER
    # ============================================================

    def acceleration_limit(
        self,
        current,
        desired,
        max_acceleration,
        dt
    ):

        max_delta = (
            max_acceleration * dt
        )

        delta = desired - current

        if delta > max_delta:

            delta = max_delta

        elif delta < -max_delta:

            delta = -max_delta

        return current + delta

    # ============================================================
    # STOP
    # ============================================================

    def stop_robot(self):

        dt = 1.0 / self.controller_frequency

        self.current_linear_cmd = (
            self.acceleration_limit(
                self.current_linear_cmd,
                0.0,
                self.max_linear_acceleration,
                dt
            )
        )

        self.current_angular_cmd = (
            self.acceleration_limit(
                self.current_angular_cmd,
                0.0,
                self.max_angular_acceleration,
                dt
            )
        )

        cmd = Twist()

        cmd.linear.x = self.current_linear_cmd
        cmd.angular.z = self.current_angular_cmd

        self.cmd_vel_pub.publish(cmd)


# ================================================================
# MAIN
# ================================================================

def main(args=None):

    rclpy.init(args=args)

    node = Nav2StyleController()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.stop_robot()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
