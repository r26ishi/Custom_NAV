#!/usr/bin/env python3

import csv
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class RawPathPublisher(Node):

    def __init__(self):
        super().__init__('coarse_publisher')

        # Parameters
        self.declare_parameter('csv_file', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate', 1.0)

        self.csv_file = self.get_parameter('csv_file').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate').value

        if not self.csv_file:
            self.get_logger().fatal(
                'Please provide the CSV file path using csv_file parameter.'
            )
            raise ValueError('csv_file parameter is empty')

        if not os.path.isfile(self.csv_file):
            self.get_logger().fatal(
                f'CSV file does not exist: {self.csv_file}'
            )
            raise FileNotFoundError(self.csv_file)

        self.raw_points = self.load_csv(self.csv_file)

        # QoS: retain the most recent path for late subscribers
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.publisher = self.create_publisher(
            Path,
            '/coarse_plan',
            qos
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_path
        )

        self.get_logger().info(
            f'Loaded {len(self.raw_points)} waypoints from {self.csv_file}'
        )

        self.get_logger().info(
            'Publishing raw path on /coarse_plan'
        )

    def load_csv(self, filename):

        points = []

        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            if not reader.fieldnames or not {'x', 'y'}.issubset(
                set(reader.fieldnames)
            ):
                raise ValueError(
                    'CSV must contain x and y columns'
                )

            for row in reader:
                x = float(row['x'])
                y = float(row['y'])

                points.append((x, y))

        if len(points) < 2:
            raise ValueError(
                'CSV must contain at least two waypoints'
            )

        return points

    def publish_path(self):

        now = self.get_clock().now().to_msg()

        path_msg = Path()
        path_msg.header.stamp = now
        path_msg.header.frame_id = self.frame_id

        for x, y in self.raw_points:

            pose = PoseStamped()

            pose.header.stamp = now
            pose.header.frame_id = self.frame_id

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # Orientation is not specified for coarse waypoints.
            # Identity quaternion.
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.publisher.publish(path_msg)

        self.get_logger().info(
            f'Published {len(path_msg.poses)} raw waypoints',
            throttle_duration_sec=5.0
        )


def main(args=None):

    rclpy.init(args=args)

    node = RawPathPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
