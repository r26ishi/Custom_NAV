#!/usr/bin/env python3

import math
import sys
import time
import select


import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseWithCovarianceStamped



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

        # Named waypoints
        self.waypoints = {
            "home":    (0.0, 0.0, 0.0),
            "kitchen": (-8.08, -4.66, 0.0),
            "table1":  (1.20, -3.63, 0.0),
            "table2":  (0.83, -11.69, 0.0),
            "table3":  (-8.83, -11.75, 0.0),
        }
        
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

        self.get_logger().info(
            "Published initial pose (HOME)"
        )

    def wait_for_confirmation(self, location, timeout):
        self.get_logger().info(
            f"Waiting for confirmation at {location} ({timeout} seconds)"
        )

        print("Type 'ok' and press Enter")

        end_time = time.time() + timeout

        while time.time() < end_time:
            remaining = end_time - time.time()

            ready, _, _ = select.select([sys.stdin], [], [], remaining)

            if ready:
                user_input = sys.stdin.readline().strip().lower()

                if user_input == "ok":
                    self.get_logger().info(
                        f"Confirmation received at {location}"
                    )
                    return True

            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().warn(
            f"No confirmation at {location}. Timeout."
        )

        return False
    
    def execute_order(self, table_name):

        self.get_logger().info(
            f"Received order for {table_name}"
        )

        if not self.goto("kitchen"):
            return False

        kitchen_confirmed = self.wait_for_confirmation(
            "kitchen",
            timeout=10
        )

        if not kitchen_confirmed:
            self.get_logger().warn(
                "Kitchen timeout. Returning home."
            )

            self.goto("home")
            return False


        if not self.goto(table_name):
            self.goto("home")
            return False
        
        table_confirmed = self.wait_for_confirmation(
            table_name,
            timeout=15
        )

        if not table_confirmed:
            self.get_logger().warn(
                "Table timeout. Returning food to kitchen."
            )

            self.goto("kitchen")

            self.goto("home")

            return False

        self.get_logger().info(
            f"Delivery completed for {table_name}"
        )

        self.goto("home")

        return True

    def goto(self, label):
        """Navigate to a named waypoint."""

        if label not in self.waypoints:
            self.get_logger().error(
                f"Unknown waypoint '{label}'"
            )
            return False

        x, y, yaw = self.waypoints[label]

        self.client.wait_for_server()

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f"Going to {label} ({x:.2f}, {y:.2f})"
        )

        future = self.client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                f"{label} goal rejected"
            )
            return False

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result.status == 4:
            self.get_logger().info(
                f"Reached {label}"
            )
            return True

        self.get_logger().warn(
            f"Failed to reach {label}. Status={result.status}"
        )
        return False


def main():

    rclpy.init()

    navigator = RestaurantNavigator()

    navigator.publish_initial_pose()
    time.sleep(3)

    while True:

        table = input(
            "Enter order (table1/table2/table3 or q): "
        )

        if table == "q":
            break

        if table not in ["table1", "table2", "table3"]:
            print("Invalid table")
            continue

        navigator.execute_order(table)

    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
