#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path

import matplotlib.pyplot as plt


class CoarsePathNode(Node):

    def __init__(self):
        super().__init__('coarse_path_node')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        self.declare_parameter('spacing', 5.5)

        self.spacing = self.get_parameter(
            'spacing'
        ).value

        # --------------------------------------------------
        # Store paths
        # --------------------------------------------------

        self.original_path = None
        self.coarse_path = None

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------

        self.plan_sub = self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            10
        )

        # --------------------------------------------------
        # Publisher
        # --------------------------------------------------

        self.coarse_plan_pub = self.create_publisher(
            Path,
            '/coarse_plan',
            10
        )

        # --------------------------------------------------
        # Matplotlib
        # --------------------------------------------------

        plt.ion()

        self.fig, self.ax = plt.subplots(
            figsize=(10, 7)
        )

        self.get_logger().info(
            f'Coarse path node started. '
            f'Spacing = {self.spacing:.2f} m'
        )

    # ======================================================
    # Nav2 plan callback
    # ======================================================

    def plan_callback(self, msg):

        if len(msg.poses) < 2:
            return

        # Store original Nav2 path
        self.original_path = msg

        # Generate coarse path
        self.coarse_path = self.create_coarse_path(msg)

        # Publish coarse path
        self.coarse_plan_pub.publish(
            self.coarse_path
        )

        # Plot both paths
        self.plot_paths()

        self.get_logger().info(
            f'Nav2 path: {len(msg.poses)} points | '
            f'Coarse path: {len(self.coarse_path.poses)} points'
        )

    # ======================================================
    # Create coarse path
    # ======================================================

    def create_coarse_path(self, msg):

        coarse_path = Path()

        coarse_path.header = msg.header

        # Always keep first point
        coarse_path.poses.append(
            msg.poses[0]
        )

        last_x = msg.poses[0].pose.position.x
        last_y = msg.poses[0].pose.position.y

        # ----------------------------------------------
        # Select points based on distance
        # ----------------------------------------------

        for pose in msg.poses[1:-1]:

            x = pose.pose.position.x
            y = pose.pose.position.y

            distance = math.sqrt(
                (x - last_x) ** 2 +
                (y - last_y) ** 2
            )

            if distance >= self.spacing:

                coarse_path.poses.append(
                    pose
                )

                last_x = x
                last_y = y

        # Always keep final point
        coarse_path.poses.append(
            msg.poses[-1]
        )

        return coarse_path

    # ======================================================
    # Plot
    # ======================================================

    def plot_paths(self):

        if self.original_path is None:
            return

        if self.coarse_path is None:
            return

        # Clear previous plot
        self.ax.clear()

        # --------------------------------------------------
        # Extract original path X/Y
        # --------------------------------------------------

        original_x = []
        original_y = []

        for pose in self.original_path.poses:

            original_x.append(
                pose.pose.position.x
            )

            original_y.append(
                pose.pose.position.y
            )

        # --------------------------------------------------
        # Extract coarse path X/Y
        # --------------------------------------------------

        coarse_x = []
        coarse_y = []

        for pose in self.coarse_path.poses:

            coarse_x.append(
                pose.pose.position.x
            )

            coarse_y.append(
                pose.pose.position.y
            )

        # --------------------------------------------------
        # Plot original Nav2 path
        # --------------------------------------------------

        self.ax.plot(
            original_x,
            original_y,
            '-',
            linewidth=1.5,
            label='Nav2 Global Plan'
        )

        # --------------------------------------------------
        # Plot coarse path
        # --------------------------------------------------

        self.ax.plot(
            coarse_x,
            coarse_y,
            'o-',
            markersize=5,
            linewidth=2,
            label='Coarse Plan'
        )

        # --------------------------------------------------
        # Mark start
        # --------------------------------------------------

        self.ax.plot(
            original_x[0],
            original_y[0],
            's',
            markersize=8,
            label='Start'
        )

        # --------------------------------------------------
        # Mark goal
        # --------------------------------------------------

        self.ax.plot(
            original_x[-1],
            original_y[-1],
            '*',
            markersize=12,
            label='Goal'
        )

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------

        self.ax.set_xlabel(
            'X [m]'
        )

        self.ax.set_ylabel(
            'Y [m]'
        )

        self.ax.set_title(
            'Nav2 Global Plan vs Coarse Plan'
        )

        self.ax.grid(
            True
        )

        self.ax.axis(
            'equal'
        )

        self.ax.legend()

        # --------------------------------------------------
        # Update figure
        # --------------------------------------------------

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):

    rclpy.init(args=args)

    node = CoarsePathNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        plt.close(
            node.fig
        )

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
