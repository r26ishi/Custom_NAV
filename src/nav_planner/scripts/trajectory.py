#!/usr/bin/env python3

import math
import csv
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy
)

from nav_msgs.msg import Path
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class TrajectoryGeneratorNode(Node):

    def __init__(self):
        super().__init__('trajectory_generator')

        # ====================================================
        # PARAMETERS
        # ====================================================

        self.declare_parameter('sample_spacing', 0.1)
        self.declare_parameter('max_velocity', 0.5)
        self.declare_parameter('max_acceleration', 0.2)
        self.declare_parameter('max_deceleration', 0.2)
        self.declare_parameter('start_velocity', 0.0)
        self.declare_parameter('end_velocity', 0.0)
        self.declare_parameter(
            'output_csv',
            os.path.abspath('time_parameterized_trajectory.csv')
        )
        self.declare_parameter('frame_id', 'map')

        self.ds = float(
            self.get_parameter('sample_spacing').value
        )
        self.v_max = float(
            self.get_parameter('max_velocity').value
        )
        self.a_max = float(
            self.get_parameter('max_acceleration').value
        )
        self.d_max = float(
            self.get_parameter('max_deceleration').value
        )
        self.v_start = float(
            self.get_parameter('start_velocity').value
        )
        self.v_end = float(
            self.get_parameter('end_velocity').value
        )
        self.output_csv = self.get_parameter(
            'output_csv'
        ).value
        self.frame_id = self.get_parameter('frame_id').value

        if self.ds <= 0.0:
            raise ValueError('sample_spacing must be positive')

        if self.v_max <= 0.0:
            raise ValueError('max_velocity must be positive')

        if self.a_max <= 0.0 or self.d_max <= 0.0:
            raise ValueError(
                'Acceleration and deceleration must be positive'
            )

        # Prevent duplicate processing when the upstream node
        # periodically republishes an unchanged plan.
        self.last_path = None

        # ====================================================
        # ROS INTERFACES
        # ====================================================

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.subscription = self.create_subscription(
            Path,
            '/smooth_plan',
            self.path_callback,
            qos
        )

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/trajectory',
            qos
        )

        self.get_logger().info(
            'Trajectory generator started. Waiting for /smooth_plan'
        )

    # ========================================================
    # DISTANCE BETWEEN TWO POINTS
    # ========================================================

    @staticmethod
    def distance(p1, p2):
        return math.hypot(
            p2[0] - p1[0],
            p2[1] - p1[1]
        )

    # ========================================================
    # SAMPLE PATH
    # ========================================================

    def sample_path(self, waypoints):

        points = waypoints

        cumulative = [0.0]

        for i in range(1, len(points)):
            segment_distance = self.distance(
                points[i - 1],
                points[i]
            )
            cumulative.append(
                cumulative[-1] + segment_distance
            )

        total_length = cumulative[-1]

        if total_length <= 0.0:
            raise ValueError('Path length must be positive')

        self.get_logger().info(
            f'Path length: {total_length:.4f} m'
        )

        self.get_logger().info(
            f'Sampling spacing: {self.ds:.4f} m'
        )

        # Generate sampling distances
        distances = []
        s = 0.0

        while s < total_length:
            distances.append(s)
            s += self.ds

        if not distances or distances[-1] < total_length:
            distances.append(total_length)

        # Interpolate along the input smooth path
        sampled = []
        segment = 0

        for s in distances:

            while (
                segment < len(cumulative) - 2
                and cumulative[segment + 1] < s
            ):
                segment += 1

            seg_length = (
                cumulative[segment + 1]
                - cumulative[segment]
            )

            if seg_length <= 1e-12:
                x = points[segment + 1][0]
                y = points[segment + 1][1]

            else:
                ratio = (
                    s - cumulative[segment]
                ) / seg_length

                x = (
                    points[segment][0]
                    + ratio * (
                        points[segment + 1][0]
                        - points[segment][0]
                    )
                )

                y = (
                    points[segment][1]
                    + ratio * (
                        points[segment + 1][1]
                        - points[segment][1]
                    )
                )

            sampled.append({
                's': s,
                'x': x,
                'y': y
            })

        self.get_logger().info(
            f'Sampled points: {len(sampled)}'
        )

        return sampled

    # ========================================================
    # GENERATE TIME-PARAMETERIZED TRAJECTORY
    # ========================================================

    def generate(self, waypoints):

        trajectory = self.sample_path(waypoints)

        n = len(trajectory)

        if n < 2:
            raise ValueError(
                'At least two sampled points are required'
            )

        # ====================================================
        # 1. CALCULATE YAW
        # ====================================================

        for i in range(n - 1):

            dx = (
                trajectory[i + 1]['x']
                - trajectory[i]['x']
            )

            dy = (
                trajectory[i + 1]['y']
                - trajectory[i]['y']
            )

            trajectory[i]['yaw'] = math.atan2(dy, dx)

        trajectory[-1]['yaw'] = trajectory[-2]['yaw']

        # ====================================================
        # 2. INITIAL VELOCITY PROFILE
        # ====================================================

        v = [self.v_max for _ in range(n)]

        v[0] = min(self.v_start, self.v_max)
        v[-1] = min(self.v_end, self.v_max)

        # ====================================================
        # 3. FORWARD PASS
        # ====================================================

        for i in range(1, n):

            ds = (
                trajectory[i]['s']
                - trajectory[i - 1]['s']
            )

            max_reachable_velocity = math.sqrt(
                max(
                    0.0,
                    v[i - 1] ** 2
                    + 2.0 * self.a_max * ds
                )
            )

            v[i] = min(
                v[i],
                max_reachable_velocity
            )

        # ====================================================
        # 4. BACKWARD PASS
        # ====================================================

        for i in range(n - 2, -1, -1):

            ds = (
                trajectory[i + 1]['s']
                - trajectory[i]['s']
            )

            max_velocity_from_deceleration = math.sqrt(
                max(
                    0.0,
                    v[i + 1] ** 2
                    + 2.0 * self.d_max * ds
                )
            )

            v[i] = min(
                v[i],
                max_velocity_from_deceleration
            )

        # ====================================================
        # 5. CALCULATE TIME
        # ====================================================

        t = [0.0] * n

        for i in range(1, n):

            ds = (
                trajectory[i]['s']
                - trajectory[i - 1]['s']
            )

            v_avg = (v[i - 1] + v[i]) / 2.0

            if v_avg <= 1e-9:
                raise ValueError(
                    f'Zero average velocity between points {i-1} and {i}'
                )

            dt = ds / v_avg
            t[i] = t[i - 1] + dt

        # ====================================================
        # 6. STORE TIME + VELOCITY
        # ====================================================

        for i in range(n):

            trajectory[i]['time'] = t[i]
            trajectory[i]['v'] = v[i]

            trajectory[i]['vx'] = (
                v[i] * math.cos(trajectory[i]['yaw'])
            )

            trajectory[i]['vy'] = (
                v[i] * math.sin(trajectory[i]['yaw'])
            )

        # ====================================================
        # 7. ANGULAR VELOCITY
        # ====================================================

        trajectory[0]['omega'] = 0.0

        for i in range(1, n):

            dyaw = math.atan2(
                math.sin(
                    trajectory[i]['yaw']
                    - trajectory[i - 1]['yaw']
                ),
                math.cos(
                    trajectory[i]['yaw']
                    - trajectory[i - 1]['yaw']
                )
            )

            dt = t[i] - t[i - 1]

            if dt <= 1e-12:
                trajectory[i]['omega'] = 0.0
            else:
                trajectory[i]['omega'] = dyaw / dt

        # ====================================================
        # 8. TANGENTIAL ACCELERATION
        # ====================================================

        trajectory[0]['ax'] = 0.0

        for i in range(1, n):

            dt = t[i] - t[i - 1]

            if dt <= 1e-12:
                trajectory[i]['ax'] = 0.0
            else:
                trajectory[i]['ax'] = (
                    v[i] - v[i - 1]
                ) / dt

        return trajectory

    # ========================================================
    # SAVE CSV (SAME FORMAT AS ORIGINAL)
    # ========================================================

    def save_csv(self, trajectory, filename):

        fieldnames = [
            'time',
            's',
            'x',
            'y',
            'yaw',
            'v',
            'vx',
            'vy',
            'omega',
            'ax'
        ]

        directory = os.path.dirname(
            os.path.abspath(filename)
        )
        os.makedirs(directory, exist_ok=True)

        with open(
            filename,
            mode='w',
            newline=''
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for point in trajectory:
                writer.writerow({
                    key: point[key]
                    for key in fieldnames
                })

        self.get_logger().info(
            f'Trajectory saved to {os.path.abspath(filename)}'
        )

        self.get_logger().info(
            f'Number of trajectory points: {len(trajectory)}'
        )

        self.get_logger().info(
            f'Total trajectory time: {trajectory[-1]["time"]:.3f} s'
        )

    # ========================================================
    # PUBLISH TRAJECTORY
    # ========================================================

    def publish_trajectory(self, trajectory):

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # These are planar trajectory coordinates represented
        # as named variables, not physical robot joint names.
        msg.joint_names = ['x', 'y', 'yaw']

        for point in trajectory:

            traj_point = JointTrajectoryPoint()

            traj_point.positions = [
                float(point['x']),
                float(point['y']),
                float(point['yaw'])
            ]

            traj_point.velocities = [
                float(point['vx']),
                float(point['vy']),
                float(point['omega'])
            ]

            # Convert relative trajectory time to ROS Duration
            time_sec = float(point['time'])
            sec = int(time_sec)
            nanosec = int(round((time_sec - sec) * 1e9))

            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000

            traj_point.time_from_start.sec = sec
            traj_point.time_from_start.nanosec = nanosec

            msg.points.append(traj_point)

        self.trajectory_publisher.publish(msg)

        self.get_logger().info(
            f'Published {len(msg.points)} trajectory points'
        )

    # ========================================================
    # PATH CALLBACK
    # ========================================================

    def path_callback(self, msg):

        if len(msg.poses) < 2:
            self.get_logger().warning(
                'Received path with fewer than 2 points'
            )
            return

        # Convert ROS Path into (x, y) waypoints
        waypoints = [
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y)
            )
            for pose in msg.poses
        ]

        # Skip unchanged plans if the upstream publisher
        # periodically republishes the same path.
        path_signature = tuple(waypoints)

        if path_signature == self.last_path:
            return

        self.last_path = path_signature

        self.get_logger().info(
            f'Received smooth plan: {len(waypoints)} waypoints'
        )

        try:
            trajectory = self.generate(waypoints)

            # Save the full time-parameterized CSV
            self.save_csv(
                trajectory,
                self.output_csv
            )

            # Publish time-parameterized trajectory
            self.publish_trajectory(trajectory)

            # Print summary
            self.get_logger().info(
                f'Path length: {trajectory[-1]["s"]:.3f} m'
            )

            self.get_logger().info(
                f'Duration: {trajectory[-1]["time"]:.3f} s'
            )

            self.get_logger().info(
                f'Maximum velocity: '
                f'{max(p["v"] for p in trajectory):.3f} m/s'
            )

            self.get_logger().info(
                f'Maximum |omega|: '
                f'{max(abs(p["omega"]) for p in trajectory):.3f} rad/s'
            )

            self.get_logger().info(
                f'Maximum |ax|: '
                f'{max(abs(p["ax"]) for p in trajectory):.3f} m/s^2'
            )

        except Exception as e:
            self.get_logger().error(
                f'Trajectory generation failed: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = TrajectoryGeneratorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
