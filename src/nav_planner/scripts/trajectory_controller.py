#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class WaypointCmdVel(Node):
    def __init__(self):
        super().__init__('waypoint_cmd_vel')

        # Waypoints in the odometry frame
        self.waypoints = [
            (-4.0938318487768, -0.2799419895182391),
            (-4.454426771869557, -4.990935608597384),
            (-3.452892190885935, -8.522662917514998),
        ]

        # Controller parameters
        self.max_linear_speed = 0.5
        self.max_angular_speed = 0.19
        self.k_linear = 0.5
        self.k_angular = 1.5
        self.position_tolerance = 0.10
        self.angle_tolerance = 0.12

        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.initialized = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.index = 0
        self.finished = False

        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Waiting for odometry...')

    def quaternion_to_yaw(self, q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.current_x = p.x
        self.current_y = p.y
        self.current_yaw = self.quaternion_to_yaw(q)

        if not self.initialized:
            self.initialized = True
            self.get_logger().info(
                f'Odometry received: '
                f'x={self.current_x:.3f}, '
                f'y={self.current_y:.3f}'
            )

    def control_loop(self):

        cmd = Twist()

        if not self.initialized or self.finished:
            self.cmd_pub.publish(cmd)
            return

        # ============================================================
        # FINISHED
        # ============================================================

        if self.index >= len(self.waypoints):

            self.finished = True

            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                'All waypoints reached!'
            )

            return

        # ============================================================
        # CURRENT ROBOT POSE
        # ============================================================

        x = self.current_x
        y = self.current_y
        yaw = self.current_yaw

        # ============================================================
        # SKIP WAYPOINTS WITHOUT STOPPING
        # ============================================================

        goal_x, goal_y = self.waypoints[self.index]

        ex = goal_x - x
        ey = goal_y - y

        distance = math.hypot(ex, ey)

        # If close to current waypoint,
        # immediately switch to the next waypoint.
        if distance < self.position_tolerance:

            self.get_logger().info(
                f'Waypoint {self.index} reached'
            )

            self.index += 1

            # If this was the final waypoint
            if self.index >= len(self.waypoints):

                self.finished = True
                self.cmd_pub.publish(Twist())

                self.get_logger().info(
                    'Final waypoint reached!'
                )

                return

            # IMPORTANT:
            # Do NOT publish zero velocity here.
            # Continue directly toward the next waypoint.

            goal_x, goal_y = self.waypoints[self.index]

            ex = goal_x - x
            ey = goal_y - y

            distance = math.hypot(ex, ey)

        # ============================================================
        # TARGET HEADING
        # ============================================================

        target_yaw = math.atan2(
            ey,
            ex
        )

        yaw_error = self.normalize_angle(
            target_yaw - yaw
        )

        # ============================================================
        # ANGULAR VELOCITY
        # ============================================================

        cmd.angular.z = max(
            -self.max_angular_speed,
            min(
                self.max_angular_speed,
                self.k_angular * yaw_error
            )
        )

        # ============================================================
        # LINEAR VELOCITY
        # ============================================================

        if abs(yaw_error) > self.angle_tolerance:

            # Don't completely stop.
            # Slow down while turning.
            cmd.linear.x = 0.10

        else:

            cmd.linear.x = min(
                self.max_linear_speed,
                self.k_linear * distance
            )

            # Reduce speed according to heading error
            cmd.linear.x *= max(
                0.0,
                math.cos(yaw_error)
            )

        # ============================================================
        # PUBLISH
        # ============================================================

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = WaypointCmdVel()

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
