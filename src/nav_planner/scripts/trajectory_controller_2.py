#!/usr/bin/env python3

import math
import time
import bisect

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# ============================================================
# YOUR SMOOTHED WAYPOINTS
# ============================================================

WAYPOINTS = [
    ( 0.000000,  0.000000),
    (-0.413972,  0.047758),
    (-0.828126,  0.094034),
    (-1.242202,  0.136864),
    (-1.655059,  0.173341),
    (-2.064253,  0.199221),
    (-2.465661,  0.208672),
    (-2.853230,  0.194273),
    (-3.218988,  0.147389),
    (-3.553599,  0.058937),
    (-3.847750, -0.080059),
    (-4.093832, -0.279942),
    (-4.284539, -0.557585),
    (-4.425333, -0.903591),
    (-4.521565, -1.304583),
    (-4.580532, -1.745737),
    (-4.610224, -2.213091),
    (-4.617832, -2.694460),
    (-4.608881, -3.179491),
    (-4.586954, -3.659309),
    (-4.553816, -4.126034),
    (-4.509804, -4.572304),
    (-4.454427, -4.990936),
    (-4.387106, -5.374797),
    (-4.310702, -5.731620),
    (-4.227454, -6.068382),
    (-4.139050, -6.390979),
    (-4.046749, -6.704128),
    (-3.951508, -7.011404),
    (-3.854072, -7.315367),
    (-3.755038, -7.617710),
    (-3.654897, -7.919438),
    (-3.554062, -8.221039),
    (-3.452892, -8.522663),
]


# ============================================================
# TRAJECTORY GENERATOR
# ============================================================

class TrajectoryGenerator:

    def __init__(
        self,
        waypoints,
        sample_spacing=0.1,
        max_velocity=0.5,
        max_acceleration=0.2,
        max_deceleration=0.2,
        start_velocity=0.0,
        end_velocity=0.0
    ):
        self.waypoints = waypoints
        self.ds = sample_spacing
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.d_max = max_deceleration
        self.v_start = start_velocity
        self.v_end = end_velocity

        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required")
        if self.ds <= 0 or self.v_max <= 0:
            raise ValueError("Spacing and maximum velocity must be positive")
        if self.a_max <= 0 or self.d_max <= 0:
            raise ValueError("Acceleration limits must be positive")
        if not (0 <= self.v_start <= self.v_max):
            raise ValueError("Invalid start velocity")
        if not (0 <= self.v_end <= self.v_max):
            raise ValueError("Invalid end velocity")

    @staticmethod
    def distance(p1, p2):
        return math.hypot(
            p2[0] - p1[0],
            p2[1] - p1[1]
        )

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def sample_path(self):
        cumulative = [0.0]

        for i in range(1, len(self.waypoints)):
            ds = self.distance(
                self.waypoints[i - 1],
                self.waypoints[i]
            )
            cumulative.append(cumulative[-1] + ds)

        total_length = cumulative[-1]

        if total_length <= 0:
            raise ValueError("Path has zero length")

        # Regular spatial sampling, including the endpoint
        distances = []
        s = 0.0

        while s < total_length:
            distances.append(s)
            s += self.ds

        if not distances or total_length - distances[-1] > 1e-9:
            distances.append(total_length)
        else:
            distances[-1] = total_length

        sampled = []
        segment = 0

        for s in distances:
            while (
                segment < len(cumulative) - 2
                and cumulative[segment + 1] < s
            ):
                segment += 1

            seg_len = cumulative[segment + 1] - cumulative[segment]

            if seg_len <= 1e-12:
                x, y = self.waypoints[segment + 1]
            else:
                alpha = (
                    (s - cumulative[segment]) / seg_len
                )

                p1 = self.waypoints[segment]
                p2 = self.waypoints[segment + 1]

                x = p1[0] + alpha * (p2[0] - p1[0])
                y = p1[1] + alpha * (p2[1] - p1[1])

            sampled.append({
                's': s,
                'x': x,
                'y': y
            })

        return sampled

    def generate(self):
        trajectory = self.sample_path()
        n = len(trajectory)

        # Heading along the sampled path
        for i in range(n - 1):
            dx = trajectory[i + 1]['x'] - trajectory[i]['x']
            dy = trajectory[i + 1]['y'] - trajectory[i]['y']

            trajectory[i]['yaw'] = math.atan2(dy, dx)

        trajectory[-1]['yaw'] = trajectory[-2]['yaw']

        # Initialize velocity limits
        v = [self.v_max] * n
        v[0] = self.v_start
        v[-1] = self.v_end

        # Forward pass: acceleration constraint
        for i in range(1, n):
            ds = trajectory[i]['s'] - trajectory[i - 1]['s']

            v[i] = min(
                v[i],
                math.sqrt(
                    v[i - 1]**2 + 2.0 * self.a_max * ds
                )
            )

        # Backward pass: braking constraint
        for i in range(n - 2, -1, -1):
            ds = trajectory[i + 1]['s'] - trajectory[i]['s']

            v[i] = min(
                v[i],
                math.sqrt(
                    v[i + 1]**2 + 2.0 * self.d_max * ds
                )
            )

        # Time parameterization
        t = [0.0] * n

        for i in range(1, n):
            ds = trajectory[i]['s'] - trajectory[i - 1]['s']
            v_avg = (v[i - 1] + v[i]) / 2.0

            if v_avg <= 1e-9:
                raise ValueError("Cannot time-parameterize zero-speed segment")

            dt = ds / v_avg
            t[i] = t[i - 1] + dt

        # Fill trajectory state
        for i in range(n):
            yaw = trajectory[i]['yaw']

            trajectory[i]['time'] = t[i]
            trajectory[i]['v'] = v[i]
            trajectory[i]['vx'] = v[i] * math.cos(yaw)
            trajectory[i]['vy'] = v[i] * math.sin(yaw)

        # Angular velocity from unwrapped shortest heading differences
        trajectory[0]['omega'] = 0.0

        for i in range(1, n):
            dyaw = self.normalize_angle(
                trajectory[i]['yaw'] - trajectory[i - 1]['yaw']
            )
            dt = t[i] - t[i - 1]

            trajectory[i]['omega'] = dyaw / dt if dt > 0 else 0.0

        # Tangential acceleration
        trajectory[0]['ax'] = 0.0

        for i in range(1, n):
            dt = t[i] - t[i - 1]
            trajectory[i]['ax'] = (
                (v[i] - v[i - 1]) / dt if dt > 0 else 0.0
            )

        return trajectory


# ============================================================
# ROS 2 TRAJECTORY TRACKER
# ============================================================

class TrajectoryTracker(Node):

    def __init__(self, trajectory):
        super().__init__('trajectory_tracker')

        self.trajectory = trajectory
        self.times = [p['time'] for p in trajectory]

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Controller frequency = 20 Hz
        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )

        # Actual robot pose
        self.x = None
        self.y = None
        self.yaw = None
        self.last_odom_time = None

        # Trajectory timing
        self.start_time = None
        self.finished = False

        # Feedback gains
        self.kx = 0.8
        self.ky = 2.0
        self.ktheta = 2.0

        # Command limits
        self.max_linear = 0.5
        self.max_angular = 1.0

        # Completion tolerances
        self.position_tolerance = 0.08
        self.yaw_tolerance = 0.15

        self.get_logger().info(
            f'Trajectory loaded: {len(trajectory)} points, '
            f'duration={self.times[-1]:.2f} s'
        )

        self.get_logger().info(
            'Waiting for odometry on /odom'
        )

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def quaternion_to_yaw(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        self.yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation
        )

        self.last_odom_time = time.monotonic()

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def interpolate_trajectory(self, elapsed):
        if elapsed <= self.times[0]:
            return self.trajectory[0]

        if elapsed >= self.times[-1]:
            return self.trajectory[-1]

        i = bisect.bisect_right(self.times, elapsed) - 1

        p1 = self.trajectory[i]
        p2 = self.trajectory[i + 1]

        dt = p2['time'] - p1['time']
        alpha = (
            (elapsed - p1['time']) / dt
            if dt > 0 else 0.0
        )

        dyaw = self.normalize_angle(
            p2['yaw'] - p1['yaw']
        )

        return {
            'x': p1['x'] + alpha * (p2['x'] - p1['x']),
            'y': p1['y'] + alpha * (p2['y'] - p1['y']),
            'yaw': self.normalize_angle(
                p1['yaw'] + alpha * dyaw
            ),
            'v': p1['v'] + alpha * (p2['v'] - p1['v']),
            'omega': p1['omega'] + alpha * (
                p2['omega'] - p1['omega']
            )
        }

    def control_loop(self):

        if self.finished:
            self.publish_stop()
            return

        # Wait for first odometry
        if self.x is None or self.last_odom_time is None:
            self.publish_stop()
            return

        # Abort safely if odometry is lost
        if time.monotonic() - self.last_odom_time > 0.5:
            self.publish_stop()
            self.finished = True

            self.get_logger().error(
                'Odometry timeout. Trajectory aborted. '
                'Restart the node after fixing odometry.'
            )
            return

        # Start trajectory clock when odometry is available
        if self.start_time is None:
            self.start_time = time.monotonic()
            self.get_logger().info('Trajectory tracking started')

        elapsed = time.monotonic() - self.start_time

        # Desired final state
        if elapsed >= self.times[-1]:
            final = self.trajectory[-1]

            position_error = math.hypot(
                final['x'] - self.x,
                final['y'] - self.y
            )

            heading_error = abs(
                self.normalize_angle(final['yaw'] - self.yaw)
            )

            self.publish_stop()
            self.finished = True

            self.get_logger().info(
                f'Trajectory time complete. '
                f'Final position error={position_error:.3f} m, '
                f'heading error={heading_error:.3f} rad. '
                f'Within tolerances: '
                f'{position_error <= self.position_tolerance and heading_error <= self.yaw_tolerance}'
            )
            return

        # Desired trajectory state
        ref = self.interpolate_trajectory(elapsed)

        xd = ref['x']
        yd = ref['y']
        yawd = ref['yaw']
        vd = ref['v']
        omegad = ref['omega']

        # Global position error
        dx = xd - self.x
        dy = yd - self.y

        # Transform error into robot frame
        ex = (
            math.cos(self.yaw) * dx
            + math.sin(self.yaw) * dy
        )

        ey = (
            -math.sin(self.yaw) * dx
            + math.cos(self.yaw) * dy
        )

        # Heading error
        etheta = self.normalize_angle(
            yawd - self.yaw
        )

        # Feedback tracking controller
        v_cmd = (
            vd * math.cos(etheta)
            + self.kx * ex
        )

        omega_cmd = (
            omegad
            + self.ky * vd * ey
            + self.ktheta * math.sin(etheta)
        )

        # Limit output commands
        v_cmd = max(
            -self.max_linear,
            min(self.max_linear, v_cmd)
        )

        omega_cmd = max(
            -self.max_angular,
            min(self.max_angular, omega_cmd)
        )

        # Publish cmd_vel
        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(omega_cmd)

        self.cmd_pub.publish(cmd)

        # Periodic diagnostic log
        self.get_logger().info(
            f't={elapsed:.2f}s | '
            f'actual=({self.x:.2f},{self.y:.2f}) | '
            f'desired=({xd:.2f},{yd:.2f}) | '
            f'v={v_cmd:.2f} | '
            f'omega={omega_cmd:.2f} | '
            f'error={math.hypot(dx, dy):.3f}m',
            throttle_duration_sec=1.0
        )


# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    # Generate time-parameterized trajectory
    generator = TrajectoryGenerator(
        waypoints=WAYPOINTS,
        sample_spacing=0.1,
        max_velocity=0.5,
        max_acceleration=0.2,
        max_deceleration=0.2,
        start_velocity=0.0,
        end_velocity=0.0
    )

    trajectory = generator.generate()

    print('\nGenerated trajectory:')
    print(f'Number of points: {len(trajectory)}')
    print(f'Total duration: {trajectory[-1]["time"]:.2f} s')
    print(
        f'Start: ({trajectory[0]["x"]:.3f}, '
        f'{trajectory[0]["y"]:.3f})'
    )
    print(
        f'End: ({trajectory[-1]["x"]:.3f}, '
        f'{trajectory[-1]["y"]:.3f})'
    )

    # Save trajectory as CSV for later analysis
    import csv

    with open('generated_trajectory.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'time', 'x', 'y', 'yaw',
                'v', 'vx', 'vy', 'omega', 'ax', 's'
            ]
        )
        writer.writeheader()
        writer.writerows(trajectory)

    print('Trajectory saved to generated_trajectory.csv')

    # Pass trajectory directly to tracker
    node = TrajectoryTracker(trajectory)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('Stopping trajectory tracker')

    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
