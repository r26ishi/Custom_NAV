#!/usr/bin/env python3

import csv
import os

import numpy as np
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class SmoothPlanner(Node):

    def __init__(self):
        super().__init__('smooth_planner')

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('points_between', 10)

        self.declare_parameter('wd', 0.1)
        self.declare_parameter('ws', 10.0)
        self.declare_parameter('wc', 1.0)

        self.declare_parameter('maxiter', 5000)
        self.declare_parameter('gtol', 1e-8)

        self.declare_parameter(
            'csv_file',
            os.path.abspath('smoothed_plan.csv')
        )

        self.frame_id = self.get_parameter('frame_id').value
        self.points_between = self.get_parameter(
            'points_between'
        ).value

        self.wd = self.get_parameter('wd').value
        self.ws = self.get_parameter('ws').value
        self.wc = self.get_parameter('wc').value

        self.maxiter = self.get_parameter('maxiter').value
        self.gtol = self.get_parameter('gtol').value

        self.csv_file = self.get_parameter('csv_file').value

        if self.points_between < 0:
            raise ValueError('points_between must be >= 0')

        # Same QoS as the raw path publisher
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ====================================================
        # ROS INTERFACES
        # ====================================================

        self.subscription = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            qos
        )

        self.publisher = self.create_publisher(
            Path,
            '/smooth_plan',
            qos
        )

        self.get_logger().info(
            'Smooth planner started. Waiting for /coarse_plan'
        )

    # ========================================================
    # INTERPOLATION
    # ========================================================

    def interpolate_path(self, raw_path):

        path = []

        for i in range(len(raw_path) - 1):

            start = raw_path[i]
            goal = raw_path[i + 1]

            for j in range(self.points_between + 1):

                t = j / float(self.points_between + 1)

                point = (1.0 - t) * start + t * goal

                path.append(point)

        path.append(raw_path[-1])

        return np.array(path, dtype=float)

    # ========================================================
    # ORIGINAL WAYPOINT INDICES
    # ========================================================

    def get_waypoint_indices(self, raw_path, initial_path):

        spacing = self.points_between + 1

        indices = [
            i * spacing for i in range(len(raw_path) - 1)
        ]

        indices.append(len(initial_path) - 1)

        return np.array(indices, dtype=int)

    # ========================================================
    # COST FUNCTIONS
    # ========================================================

    def data_cost(self, path, initial_path):

        difference = path - initial_path

        return np.sum(difference ** 2)

    def smoothness_cost(self, path):

        second_difference = (
            path[:-2]
            - 2.0 * path[1:-1]
            + path[2:]
        )

        return np.sum(second_difference ** 2)

    def curvature_cost(self, path):

        dx = np.diff(path[:, 0])
        dy = np.diff(path[:, 1])

        heading = np.arctan2(dy, dx)

        heading_difference = np.diff(heading)

        # Normalize angular differences to [-pi, pi]
        heading_difference = np.arctan2(
            np.sin(heading_difference),
            np.cos(heading_difference)
        )

        return np.sum(heading_difference ** 2)

    def total_cost(self, path, initial_path):

        Jdata = self.data_cost(path, initial_path)
        Jsmooth = self.smoothness_cost(path)
        Jcurvature = self.curvature_cost(path)

        return (
            self.wd * Jdata
            + self.ws * Jsmooth
            + self.wc * Jcurvature
        )

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    def optimize_path(self, raw_path):

        initial_path = self.interpolate_path(raw_path)

        waypoint_indices = self.get_waypoint_indices(
            raw_path,
            initial_path
        )

        optimize_mask = np.ones(
            len(initial_path),
            dtype=bool
        )

        # Keep original global waypoints fixed
        optimize_mask[waypoint_indices] = False

        optimized_indices = np.where(optimize_mask)[0]

        # If there are no interpolated points, return as is
        if len(optimized_indices) == 0:
            return initial_path, None, waypoint_indices

        # Initial optimization variables
        initial_guess = initial_path[
            optimized_indices
        ].flatten()

        # Optimization objective
        def optimization_function(x):

            path = initial_path.copy()

            path[optimized_indices] = x.reshape((-1, 2))

            return self.total_cost(path, initial_path)

        self.get_logger().info(
            f'Optimizing {len(optimized_indices)} inserted points'
        )

        result = minimize(
            optimization_function,
            initial_guess,
            method='BFGS',
            options={
                'maxiter': self.maxiter,
                'gtol': self.gtol
            }
        )

        smoothed_path = initial_path.copy()

        smoothed_path[optimized_indices] = result.x.reshape((-1, 2))

        # Explicitly restore original waypoints
        for i, index in enumerate(waypoint_indices):
            smoothed_path[index] = raw_path[i]

        return smoothed_path, result, waypoint_indices

    # ========================================================
    # SAVE CSV
    # ========================================================

    def save_path_to_csv(self, path, filename):

        directory = os.path.dirname(os.path.abspath(filename))

        os.makedirs(directory, exist_ok=True)

        with open(filename, 'w', newline='') as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(['index', 'x', 'y'])

            for i, point in enumerate(path):

                writer.writerow([
                    i,
                    f'{point[0]:.9f}',
                    f'{point[1]:.9f}'
                ])

    # ========================================================
    # BUILD ROS PATH MESSAGE
    # ========================================================

    def create_path_message(self, path):

        now = self.get_clock().now().to_msg()

        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = self.frame_id

        for i, point in enumerate(path):

            pose = PoseStamped()

            pose.header.stamp = now
            pose.header.frame_id = self.frame_id

            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0

            # Calculate orientation from path tangent
            if i < len(path) - 1:
                dx = path[i + 1, 0] - path[i, 0]
                dy = path[i + 1, 1] - path[i, 1]
            elif i > 0:
                dx = path[i, 0] - path[i - 1, 0]
                dy = path[i, 1] - path[i - 1, 1]
            else:
                dx = 1.0
                dy = 0.0

            yaw = np.arctan2(dy, dx)

            # Planar yaw to quaternion
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))

            path_msg.poses.append(pose)

        return path_msg

    # ========================================================
    # VERIFY WAYPOINTS
    # ========================================================

    def verify_waypoints(
        self,
        raw_path,
        smoothed_path,
        waypoint_indices
    ):

        all_correct = True

        self.get_logger().info('Waypoint verification:')

        for i, index in enumerate(waypoint_indices):

            error = np.linalg.norm(
                smoothed_path[index] - raw_path[i]
            )

            self.get_logger().info(
                f'Waypoint {i}: index={index}, error={error:.12f}'
            )

            if error > 1e-10:
                all_correct = False

        self.get_logger().info(
            f'All original waypoints preserved: {all_correct}'
        )

    # ========================================================
    # PATH CALLBACK
    # ========================================================

    def path_callback(self, msg):

        if len(msg.poses) < 2:
            self.get_logger().warning(
                'Received path with fewer than 2 waypoints'
            )
            return

        # Convert ROS Path to NumPy array
        raw_path = np.array([
            [
                pose.pose.position.x,
                pose.pose.position.y
            ]
            for pose in msg.poses
        ], dtype=float)

        self.get_logger().info(
            f'Received coarse plan with {len(raw_path)} waypoints'
        )

        # Run smoothing
        try:
            smoothed_path, result, waypoint_indices = (
                self.optimize_path(raw_path)
            )

            # Save verification CSV
            self.save_path_to_csv(
                smoothed_path,
                self.csv_file
            )

            # Verify fixed original waypoints
            self.verify_waypoints(
                raw_path,
                smoothed_path,
                waypoint_indices
            )

            # Publish smoothed path
            smooth_msg = self.create_path_message(smoothed_path)

            self.publisher.publish(smooth_msg)

            if result is not None:
                self.get_logger().info(
                    f'Optimization success: {result.success}'
                )
                self.get_logger().info(
                    f'Optimization message: {result.message}'
                )
                self.get_logger().info(
                    f'Iterations: {result.nit}'
                )
                self.get_logger().info(
                    f'Final cost: {self.total_cost(smoothed_path, self.interpolate_path(raw_path)):.9f}'
                )

            self.get_logger().info(
                f'Published {len(smoothed_path)} points on /smooth_plan'
            )

            self.get_logger().info(
                f'Smoothed CSV saved to: {self.csv_file}'
            )

        except Exception as e:
            self.get_logger().error(
                f'Path smoothing failed: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = SmoothPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
