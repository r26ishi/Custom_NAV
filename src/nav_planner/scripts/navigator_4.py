#!/usr/bin/env python3

import math
import sys
import time
import select

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool

from enum import Enum


class RobotState(Enum):
    IDLE = 0
    GOING_TO_KITCHEN = 1
    WAITING_AT_KITCHEN = 2
    GOING_TO_TABLE = 3
    WAITING_AT_TABLE = 4
    RETURNING_HOME = 5


# Sentinel result codes returned by goto()
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"


class RestaurantNavigator(Node):

    def __init__(self):
        super().__init__('restaurant_navigator')

        self.client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        self.create_subscription(
            Bool,
            "/cancel_task",
            self.cancel_callback,
            10
        )

        # Named waypoints
        self.waypoints = {
            "home":    (0.0, 0.0, 0.0),
            "kitchen": (-8.08, -4.66, 0.0),
            "table1":  (1.20, -3.63, 0.0),
            "table2":  (0.83, -11.69, 0.0),
            "table3":  (-8.83, -11.75, 0.0),
        }

        self.state = RobotState.IDLE
        self.cancel_requested = False

    def cancel_callback(self, msg):

        if msg.data:
            self.get_logger().warn("Task cancellation requested")
            self.cancel_requested = True
        else:
            self.cancel_requested = False

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()

        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = 0.006
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685
        ]
        msg.pose.covariance = covariance

        self.initial_pose_pub.publish(msg)

        self.get_logger().info("Published initial pose (HOME)")

    def wait_for_confirmation(self, location, timeout):

        self.get_logger().info(
            f"Waiting for confirmation at {location} ({timeout} seconds)"
        )

        print("Type 'ok' and press Enter")

        end_time = time.time() + timeout

        while time.time() < end_time:

            if self.cancel_requested:
                self.get_logger().warn(
                    f"Cancellation received while waiting at {location}"
                )
                self.cancel_requested = False
                return "CANCELLED"

            remaining = max(0.0, min(0.1, end_time - time.time()))

            ready, _, _ = select.select([sys.stdin], [], [], remaining)

            if ready:
                user_input = sys.stdin.readline().strip().lower()

                if user_input == "ok":
                    self.get_logger().info(
                        f"Confirmation received at {location}"
                    )
                    return "CONFIRMED"

            rclpy.spin_once(self, timeout_sec=0.0)

        self.get_logger().warn(
            f"No confirmation at {location}. Timeout."
        )

        return "TIMEOUT"

    def execute_order(self, table):

        self.cancel_requested = False

        self.state = RobotState.GOING_TO_KITCHEN

        result = self.goto("kitchen")

        if result == CANCELLED:
            self.get_logger().warn("Order cancelled en route to kitchen.")
            self.goto("home")
            self.state = RobotState.IDLE
            return

        if result != SUCCEEDED:
            self.get_logger().error(
                f"Failed to reach kitchen (result={result}). Aborting order."
            )
            self.goto("home")
            self.state = RobotState.IDLE
            return

        self.state = RobotState.WAITING_AT_KITCHEN

        confirmation = self.wait_for_confirmation("kitchen", 10)

        if confirmation != "CONFIRMED":
            self.get_logger().warn(
                f"No pickup confirmation at kitchen ({confirmation}). Returning home."
            )
            self.goto("home")
            self.state = RobotState.IDLE
            return

        self.state = RobotState.GOING_TO_TABLE

        result = self.goto(table)

        if result == CANCELLED:
            self.get_logger().warn("Order cancelled en route to table; returning food.")
            # carrying food
            self.goto("kitchen")
            self.goto("home")
            self.state = RobotState.IDLE
            return

        if result != SUCCEEDED:
            self.get_logger().error(
                f"Failed to reach {table} (result={result}). Returning food."
            )
            self.goto("kitchen")
            self.goto("home")
            self.state = RobotState.IDLE
            return

        self.state = RobotState.WAITING_AT_TABLE

        confirmation = self.wait_for_confirmation(table, 15)

        if confirmation != "CONFIRMED":
            self.get_logger().warn(
                f"No delivery confirmation at {table} ({confirmation}). "
                f"Returning undelivered food."
            )
            # return undelivered food
            self.goto("kitchen")
            self.goto("home")
            self.state = RobotState.IDLE
            return

        self.state = RobotState.RETURNING_HOME

        result = self.goto("home")

        if result != SUCCEEDED:
            self.get_logger().error(
                f"Delivery succeeded but failed to return home (result={result})."
            )

        self.state = RobotState.IDLE

    def goto(self, label):

        if label not in self.waypoints:
            self.get_logger().error(f"Unknown waypoint '{label}'")
            return FAILED

        x, y, yaw = self.waypoints[label]

        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "NavigateToPose action server not available."
            )
            return FAILED

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Going to {label} ({x:.2f}, {y:.2f})")

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"Goal to '{label}' was rejected.")
            return FAILED

        result_future = goal_handle.get_result_async()

        while not result_future.done():

            rclpy.spin_once(self, timeout_sec=0.1)

            if self.cancel_requested:

                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)

                self.cancel_requested = False

                return CANCELLED

        result = result_future.result()

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            return SUCCEEDED

        return FAILED


def main():

    rclpy.init()

    navigator = RestaurantNavigator()

    navigator.publish_initial_pose()
    time.sleep(3)

    try:
        while rclpy.ok():

            rclpy.spin_once(navigator, timeout_sec=0.1)

            table = input("Enter order (table1/table2/table3 or q): ").strip()

            if table == "q":
                break

            if table not in ["table1", "table2", "table3"]:
                print("Invalid table")
                continue

            navigator.execute_order(table)

    except KeyboardInterrupt:
        navigator.get_logger().info("Interrupted by user.")

    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()