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
from std_msgs.msg import Bool, String

from enum import Enum


class RobotState(Enum):
    IDLE = 0
    GOING_TO_KITCHEN = 1
    WAITING_AT_KITCHEN = 2
    GOING_TO_TABLE = 3
    WAITING_AT_TABLE = 4
    RETURNING_TO_KITCHEN = 5
    RETURNING_HOME = 6


class DeliveryOutcome(Enum):
    DELIVERED = 0                 # confirmed at the table
    SKIPPED_NAV = 1               # could not reach the table (failed/cancelled goal)
    SKIPPED_NO_CONFIRM = 2        # reached the table but no confirmation in time
    SKIPPED_ORDER_CANCELLED = 3   # that table's order was explicitly cancelled
    BATCH_ABORTED = 4             # whole run aborted (e.g. cancelled at kitchen)


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

        self.create_subscription(
            String,
            "/cancel_order",
            self.cancel_order_callback,
            10
        )

        self.cancelled_tables = set()

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
        self.current_table = None

    def cancel_callback(self, msg):

        if msg.data:
            self.get_logger().warn("Task cancellation requested")
            self.cancel_requested = True
        else:
            self.cancel_requested = False

    def cancel_order_callback(self, msg):

        text = (msg.data or "").strip().lower()

        if not text:
            return

        if text.endswith(":clear"):
            table = text[: -len(":clear")].strip()
            if table in self.cancelled_tables:
                self.cancelled_tables.discard(table)
                self.get_logger().info(f"Cancellation cleared for {table}.")
            return

        table = text

        if table not in self.waypoints or table in ("home", "kitchen"):
            self.get_logger().warn(
                f"Ignoring /cancel_order for invalid table '{table}'."
            )
            return

        self.cancelled_tables.add(table)
        self.get_logger().warn(f"Order for {table} has been cancelled.")

        if self.current_table == table:
            self.cancel_requested = True

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

    def execute_batch_order(self, tables):

        self.cancel_requested = False
        outcomes = {}

        self.state = RobotState.GOING_TO_KITCHEN

        result = self.goto("kitchen")

        if result != SUCCEEDED:
            reason = "cancelled" if result == CANCELLED else f"failed ({result})"
            self.get_logger().error(
                f"Could not reach kitchen ({reason}). Aborting entire batch."
            )
            self.goto("home")
            self.state = RobotState.IDLE
            for table in tables:
                outcomes[table] = DeliveryOutcome.BATCH_ABORTED
            return outcomes


        self.state = RobotState.WAITING_AT_KITCHEN

        confirmation = self.wait_for_confirmation("kitchen", 15)

        if confirmation != "CONFIRMED":
            self.get_logger().warn(
                f"No pickup confirmation at kitchen ({confirmation}). "
                f"Aborting entire batch."
            )
            self.goto("home")
            self.state = RobotState.IDLE
            for table in tables:
                outcomes[table] = DeliveryOutcome.BATCH_ABORTED
            return outcomes

        for table in tables:

            self.current_table = table

            if table in self.cancelled_tables:
                self.get_logger().warn(
                    f"Order for {table} was cancelled before departure. Skipping."
                )
                outcomes[table] = DeliveryOutcome.SKIPPED_ORDER_CANCELLED
                self.cancelled_tables.discard(table)
                continue

            self.cancel_requested = False  # don't carry over a stale flag

            self.state = RobotState.GOING_TO_TABLE

            result = self.goto(table)

            if table in self.cancelled_tables:
                self.get_logger().warn(
                    f"Order for {table} was cancelled en route. Skipping."
                )
                outcomes[table] = DeliveryOutcome.SKIPPED_ORDER_CANCELLED
                self.cancelled_tables.discard(table)
                continue

            if result != SUCCEEDED:
                reason = "cancelled" if result == CANCELLED else f"failed ({result})"
                self.get_logger().warn(
                    f"Could not reach {table} ({reason}). "
                    f"Skipping this table and continuing with the rest of the batch."
                )
                outcomes[table] = DeliveryOutcome.SKIPPED_NAV
                continue

            self.state = RobotState.WAITING_AT_TABLE

            confirmation = self.wait_for_confirmation(table, 15)

            if table in self.cancelled_tables:
                self.get_logger().warn(
                    f"Order for {table} was cancelled while awaiting confirmation. Skipping."
                )
                outcomes[table] = DeliveryOutcome.SKIPPED_ORDER_CANCELLED
                self.cancelled_tables.discard(table)
                continue

            if confirmation != "CONFIRMED":
                self.get_logger().warn(
                    f"No delivery confirmation at {table} ({confirmation}). "
                    f"Skipping this table and continuing with the rest of the batch."
                )
                outcomes[table] = DeliveryOutcome.SKIPPED_NO_CONFIRM
                continue

            self.get_logger().info(f"Delivery confirmed at {table}.")
            outcomes[table] = DeliveryOutcome.DELIVERED

        self.current_table = None

        self.state = RobotState.RETURNING_TO_KITCHEN

        result = self.goto("kitchen")

        if result != SUCCEEDED:
            self.get_logger().error(
                f"Could not return to kitchen after deliveries (result={result}). "
                f"Heading home directly instead."
            )

        self.state = RobotState.RETURNING_HOME

        result = self.goto("home")

        if result != SUCCEEDED:
            self.get_logger().error(
                f"Batch finished but failed to return home (result={result})."
            )

        self.state = RobotState.IDLE

        delivered = [t for t, o in outcomes.items() if o == DeliveryOutcome.DELIVERED]
        skipped = [t for t, o in outcomes.items() if o != DeliveryOutcome.DELIVERED]
        self.get_logger().info(
            f"Batch complete. Delivered: {delivered or 'none'}. "
            f"Skipped: {skipped or 'none'}."
        )

        return outcomes

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

            raw_input = input(
                "Enter order(s) - comma separated "
                "(e.g. table1,table2,table3) or q: "
            ).strip()

            if raw_input.lower() == "q":
                break

            if not raw_input:
                continue

            requested = [t.strip().lower() for t in raw_input.split(",") if t.strip()]

            valid_tables = {"table1", "table2", "table3"}
            tables = []
            invalid = []

            for t in requested:
                if t not in valid_tables:
                    invalid.append(t)
                elif t not in tables:  # dedupe, preserve order
                    tables.append(t)

            if invalid:
                print(f"Invalid table(s) ignored: {', '.join(invalid)}")

            if not tables:
                print("No valid tables in that order. Try again.")
                continue

            navigator.execute_batch_order(tables)

    except KeyboardInterrupt:
        navigator.get_logger().info("Interrupted by user.")

    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()