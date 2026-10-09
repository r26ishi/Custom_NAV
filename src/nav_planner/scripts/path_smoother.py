#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline


class PathSmoother(Node):

    def __init__(self):

        super().__init__('path_smoother')

        # ====================================================
        # PARAMETERS
        # ====================================================

        # Number of points generated between coarse points
        self.points_per_segment = 20

        # Final trajectory resolution
        self.min_points = 300

        # ====================================================
        # SUBSCRIBER
        # ====================================================

        self.plan_subscriber = self.create_subscription(
            Path,
            '/plan',
            self.plan_callback,
            10
        )

        # ====================================================
        # PUBLISHER
        # ====================================================

        self.smooth_publisher = self.create_publisher(
            Path,
            '/smoothed_plan',
            10
        )

        # ====================================================
        # STARTUP LOGS
        # ====================================================

        self.get_logger().info(
            '=========================================='
        )

        self.get_logger().info(
            'Path smoother started'
        )

        self.get_logger().info(
            'Subscribed to: /plan'
        )

        self.get_logger().info(
            'Publishing to: /smoothed_plan'
        )

        self.get_logger().info(
            f'points_per_segment: {self.points_per_segment}'
        )

        self.get_logger().info(
            f'min_points: {self.min_points}'
        )

        self.get_logger().info(
            '=========================================='
        )

    # ========================================================
    # CALLBACK
    # ========================================================

    def plan_callback(self, msg):

        self.get_logger().info(
            ''
        )

        self.get_logger().info(
            '========== PLAN CALLBACK START =========='
        )

        # ----------------------------------------------------
        # Check received path
        # ----------------------------------------------------

        num_poses = len(msg.poses)

        self.get_logger().info(
            f'Received /plan message'
        )

        self.get_logger().info(
            f'Number of poses: {num_poses}'
        )

        if num_poses < 2:

            self.get_logger().warn(
                f'Received path contains only {num_poses} points.'
            )

            self.get_logger().info(
                '========== PLAN CALLBACK END =========='
            )

            return

        # ----------------------------------------------------
        # Print frame
        # ----------------------------------------------------

        self.get_logger().info(
            f'Path frame_id: "{msg.header.frame_id}"'
        )

        self.get_logger().info(
            f'Path timestamp: {msg.header.stamp.sec}.'
            f'{msg.header.stamp.nanosec}'
        )

        # ----------------------------------------------------
        # Extract all subscribed points
        # ----------------------------------------------------

        self.get_logger().info(
            'Extracting coarse path points...'
        )

        coarse_path = []

        for i, pose_stamped in enumerate(msg.poses):

            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y

            coarse_path.append(
                [x, y]
            )

            # Print every point
            self.get_logger().info(
                f'  Point {i}: '
                f'x={x:.4f}, '
                f'y={y:.4f}'
            )

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        coarse_path = np.array(
            coarse_path,
            dtype=float
        )

        self.get_logger().info(
            f'Coarse numpy shape: {coarse_path.shape}'
        )

        self.get_logger().info(
            f'Coarse path dtype: {coarse_path.dtype}'
        )

        # ----------------------------------------------------
        # Check NaN / Inf
        # ----------------------------------------------------

        if not np.all(np.isfinite(coarse_path)):

            self.get_logger().error(
                'Coarse path contains NaN or Inf values!'
            )

            self.get_logger().error(
                f'Coarse path:\n{coarse_path}'
            )

            return

        # ----------------------------------------------------
        # First / Last point
        # ----------------------------------------------------

        self.get_logger().info(
            f'First coarse point: '
            f'x={coarse_path[0, 0]:.4f}, '
            f'y={coarse_path[0, 1]:.4f}'
        )

        self.get_logger().info(
            f'Last coarse point: '
            f'x={coarse_path[-1, 0]:.4f}, '
            f'y={coarse_path[-1, 1]:.4f}'
        )

        # ----------------------------------------------------
        # Remove duplicate points
        # ----------------------------------------------------

        self.get_logger().info(
            'Removing duplicate points...'
        )

        original_count = len(coarse_path)

        coarse_path = self.remove_duplicate_points(
            coarse_path
        )

        new_count = len(coarse_path)

        self.get_logger().info(
            f'Duplicate removal: '
            f'{original_count} -> {new_count} points'
        )

        if len(coarse_path) < 2:

            self.get_logger().warn(
                'Not enough unique points after duplicate removal.'
            )

            return

        # ----------------------------------------------------
        # Generate smooth trajectory
        # ----------------------------------------------------

        self.get_logger().info(
            'Generating smooth trajectory...'
        )

        try:

            smooth_path = self.generate_smooth_trajectory(
                coarse_path
            )

        except Exception as e:

            self.get_logger().error(
                '=========================================='
            )

            self.get_logger().error(
                f'SMOOTHING FAILED'
            )

            self.get_logger().error(
                f'Exception type: {type(e).__name__}'
            )

            self.get_logger().error(
                f'Exception: {e}'
            )

            self.get_logger().error(
                '=========================================='
            )

            import traceback
            traceback.print_exc()

            return

        # ----------------------------------------------------
        # Check smooth trajectory
        # ----------------------------------------------------

        self.get_logger().info(
            f'Smooth trajectory generated.'
        )

        self.get_logger().info(
            f'Smooth path shape: {smooth_path.shape}'
        )

        self.get_logger().info(
            f'Smooth path points: {len(smooth_path)}'
        )

        if not np.all(np.isfinite(smooth_path)):

            self.get_logger().error(
                'Smooth path contains NaN or Inf!'
            )

            return

        self.get_logger().info(
            f'Smooth first point: '
            f'x={smooth_path[0, 0]:.4f}, '
            f'y={smooth_path[0, 1]:.4f}'
        )

        self.get_logger().info(
            f'Smooth last point: '
            f'x={smooth_path[-1, 0]:.4f}, '
            f'y={smooth_path[-1, 1]:.4f}'
        )

        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        self.get_logger().info(
            'Publishing smooth path...'
        )

        try:

            self.publish_path(
                smooth_path,
                msg.header
            )

        except Exception as e:

            self.get_logger().error(
                f'PUBLISH FAILED: '
                f'{type(e).__name__}: {e}'
            )

            import traceback
            traceback.print_exc()

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        self.get_logger().info(
            'Calling plot_path()...'
        )

        try:

            self.plot_path(
                coarse_path,
                smooth_path
            )

            self.get_logger().info(
                'plot_path() returned successfully.'
            )

        except Exception as e:

            self.get_logger().error(
                '=========================================='
            )

            self.get_logger().error(
                'PLOT FAILED'
            )

            self.get_logger().error(
                f'Exception type: {type(e).__name__}'
            )

            self.get_logger().error(
                f'Exception: {e}'
            )

            self.get_logger().error(
                '=========================================='
            )

            import traceback
            traceback.print_exc()

        self.get_logger().info(
            '========== PLAN CALLBACK END =========='
        )

    # ========================================================
    # REMOVE DUPLICATE POINTS
    # ========================================================

    def remove_duplicate_points(
        self,
        path
    ):

        self.get_logger().info(
            'Checking for duplicate points...'
        )

        if len(path) < 2:

            return path

        distance = np.sqrt(
            np.sum(
                np.diff(
                    path,
                    axis=0
                ) ** 2,
                axis=1
            )
        )

        self.get_logger().info(
            f'Min segment distance: '
            f'{np.min(distance):.8f} m'
        )

        self.get_logger().info(
            f'Max segment distance: '
            f'{np.max(distance):.8f} m'
        )

        keep = np.concatenate(
            (
                [True],
                distance > 1e-6
            )
        )

        removed = np.sum(~keep)

        self.get_logger().info(
            f'Removing {removed} duplicate points.'
        )

        return path[keep]

    # ========================================================
    # GENERATE SMOOTH TRAJECTORY
    # ========================================================

    def generate_smooth_trajectory(
        self,
        coarse_path
    ):

        self.get_logger().info(
            '----- generate_smooth_trajectory() -----'
        )

        # ----------------------------------------------------
        # Calculate cumulative distance
        # ----------------------------------------------------

        dx = np.diff(
            coarse_path[:, 0]
        )

        dy = np.diff(
            coarse_path[:, 1]
        )

        segment_length = np.sqrt(
            dx ** 2 + dy ** 2
        )

        distance = np.zeros(
            len(coarse_path)
        )

        distance[1:] = np.cumsum(
            segment_length
        )

        # ----------------------------------------------------
        # Print distances
        # ----------------------------------------------------

        self.get_logger().info(
            f'Segment lengths: {segment_length}'
        )

        self.get_logger().info(
            f'Cumulative distance: {distance}'
        )

        # ----------------------------------------------------
        # Check total distance
        # ----------------------------------------------------

        total_distance = distance[-1]

        self.get_logger().info(
            f'Total path distance: '
            f'{total_distance:.4f} m'
        )

        if total_distance < 1e-6:

            self.get_logger().warn(
                'Total path distance is almost zero.'
            )

            return coarse_path.copy()

        # ----------------------------------------------------
        # Cubic spline X
        # ----------------------------------------------------

        self.get_logger().info(
            'Creating cubic spline for X...'
        )

        spline_x = CubicSpline(
            distance,
            coarse_path[:, 0],
            bc_type='natural'
        )

        self.get_logger().info(
            'X spline created.'
        )

        # ----------------------------------------------------
        # Cubic spline Y
        # ----------------------------------------------------

        self.get_logger().info(
            'Creating cubic spline for Y...'
        )

        spline_y = CubicSpline(
            distance,
            coarse_path[:, 1],
            bc_type='natural'
        )

        self.get_logger().info(
            'Y spline created.'
        )

        # ----------------------------------------------------
        # Number of points
        # ----------------------------------------------------

        num_points = max(
            self.min_points,
            (len(coarse_path) - 1)
            * self.points_per_segment
        )

        self.get_logger().info(
            f'Generating {num_points} smooth points.'
        )

        # ----------------------------------------------------
        # Generate distance values
        # ----------------------------------------------------

        smooth_distance = np.linspace(
            0.0,
            total_distance,
            num_points
        )

        # ----------------------------------------------------
        # Evaluate spline
        # ----------------------------------------------------

        smooth_x = spline_x(
            smooth_distance
        )

        smooth_y = spline_y(
            smooth_distance
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        smooth_path = np.column_stack(
            (
                smooth_x,
                smooth_y
            )
        )

        self.get_logger().info(
            f'Smooth trajectory shape: '
            f'{smooth_path.shape}'
        )

        return smooth_path

    # ========================================================
    # PUBLISH PATH
    # ========================================================

    def publish_path(
        self,
        smooth_path,
        header
    ):

        self.get_logger().info(
            '----- publish_path() -----'
        )

        path_msg = Path()

        path_msg.header = header

        for i in range(
            len(smooth_path)
        ):

            pose = PoseStamped()

            pose.header = header

            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            pose.pose.position.x = float(
                smooth_path[i, 0]
            )

            pose.pose.position.y = float(
                smooth_path[i, 1]
            )

            pose.pose.position.z = 0.0

            # ------------------------------------------------
            # Calculate yaw
            # ------------------------------------------------

            if i < len(smooth_path) - 1:

                dx = (
                    smooth_path[i + 1, 0]
                    - smooth_path[i, 0]
                )

                dy = (
                    smooth_path[i + 1, 1]
                    - smooth_path[i, 1]
                )

            else:

                dx = (
                    smooth_path[i, 0]
                    - smooth_path[i - 1, 0]
                )

                dy = (
                    smooth_path[i, 1]
                    - smooth_path[i - 1, 1]
                )

            yaw = np.arctan2(
                dy,
                dx
            )

            # ------------------------------------------------
            # Quaternion
            # ------------------------------------------------

            pose.pose.orientation.x = 0.0

            pose.pose.orientation.y = 0.0

            pose.pose.orientation.z = float(
                np.sin(yaw / 2.0)
            )

            pose.pose.orientation.w = float(
                np.cos(yaw / 2.0)
            )

            path_msg.poses.append(
                pose
            )

        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        self.smooth_publisher.publish(
            path_msg
        )

        self.get_logger().info(
            f'Published {len(path_msg.poses)} '
            f'smooth trajectory points'
        )

    # ========================================================
    # PLOT
    # ========================================================

    def plot_path(
        self,
        coarse_path,
        smooth_path
    ):

        self.get_logger().info(
            '=========================================='
        )

        self.get_logger().info(
            'ENTERED plot_path()'
        )

        self.get_logger().info(
            f'Coarse path shape: '
            f'{coarse_path.shape}'
        )

        self.get_logger().info(
            f'Smooth path shape: '
            f'{smooth_path.shape}'
        )

        # ----------------------------------------------------
        # Matplotlib backend
        # ----------------------------------------------------

        self.get_logger().info(
            f'Matplotlib backend: '
            f'{plt.get_backend()}'
        )

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        self.get_logger().info(
            'Creating matplotlib figure...'
        )

        fig = plt.figure(
            'ROS2 Path Smoother',
            figsize=(10, 8)
        )

        self.get_logger().info(
            f'Figure created: {fig}'
        )

        # ----------------------------------------------------
        # Clear figure
        # ----------------------------------------------------

        plt.clf()

        self.get_logger().info(
            'Figure cleared.'
        )

        # ----------------------------------------------------
        # Coarse plan
        # ----------------------------------------------------

        self.get_logger().info(
            'Plotting coarse path...'
        )

        plt.plot(
            coarse_path[:, 0],
            coarse_path[:, 1],
            'o--',
            label='Coarse Plan'
        )

        # ----------------------------------------------------
        # Smooth trajectory
        # ----------------------------------------------------

        self.get_logger().info(
            'Plotting smooth trajectory...'
        )

        plt.plot(
            smooth_path[:, 0],
            smooth_path[:, 1],
            '-',
            linewidth=2.5,
            label='Smooth Trajectory'
        )

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        self.get_logger().info(
            'Plotting start point...'
        )

        plt.scatter(
            coarse_path[0, 0],
            coarse_path[0, 1],
            s=120,
            label='Start'
        )

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        self.get_logger().info(
            'Plotting goal point...'
        )

        plt.scatter(
            coarse_path[-1, 0],
            coarse_path[-1, 1],
            s=120,
            label='Goal'
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        plt.xlabel(
            'X Position (m)'
        )

        plt.ylabel(
            'Y Position (m)'
        )

        plt.title(
            'Fixed Coarse Points vs Smooth Trajectory'
        )

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        plt.grid(
            True
        )

        # ----------------------------------------------------
        # Equal aspect ratio
        # ----------------------------------------------------

        plt.axis(
            'equal'
        )

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        plt.legend()

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        plt.tight_layout()

        # ----------------------------------------------------
        # SHOW
        # ----------------------------------------------------

        self.get_logger().info(
            'Calling plt.show(block=False)...'
        )

        plt.show(
            block=False
        )

        self.get_logger().info(
            'plt.show() returned.'
        )

        # ----------------------------------------------------
        # Process GUI events
        # ----------------------------------------------------

        self.get_logger().info(
            'Calling plt.pause(0.1)...'
        )

        plt.pause(
            0.1
        )

        self.get_logger().info(
            'plt.pause() returned.'
        )

        # ----------------------------------------------------
        # Check figure
        # ----------------------------------------------------

        self.get_logger().info(
            f'Figure number: {fig.number}'
        )

        self.get_logger().info(
            f'Figure exists: '
            f'{plt.fignum_exists(fig.number)}'
        )

        self.get_logger().info(
            'Plot window should now be visible.'
        )

        self.get_logger().info(
            'EXITED plot_path()'
        )

        self.get_logger().info(
            '=========================================='
        )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    # --------------------------------------------------------
    # ROS init
    # --------------------------------------------------------

    rclpy.init(
        args=args
    )

    # --------------------------------------------------------
    # Matplotlib interactive mode
    # --------------------------------------------------------

    plt.ion()

    print(
        "=============================================="
    )

    print(
        "MATPLOTLIB DEBUG"
    )

    print(
        f"Backend: {plt.get_backend()}"
    )

    print(
        f"Interactive mode: {plt.isinteractive()}"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Create node
    # --------------------------------------------------------

    node = PathSmoother()

    try:

        node.get_logger().info(
            'Starting rclpy.spin()...'
        )

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        node.get_logger().info(
            'KeyboardInterrupt received.'
        )

    except Exception as e:

        node.get_logger().error(
            f'ROS spin failed: '
            f'{type(e).__name__}: {e}'
        )

        import traceback
        traceback.print_exc()

    finally:

        node.get_logger().info(
            'Closing matplotlib...'
        )

        plt.close(
            'all'
        )

        node.get_logger().info(
            'Destroying ROS node...'
        )

        node.destroy_node()

        node.get_logger().info(
            'Shutting down ROS...'
        )

        rclpy.shutdown()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':

    main()
