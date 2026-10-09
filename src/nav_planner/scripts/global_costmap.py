#!/usr/bin/env python3

import math
import heapq

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy
)

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32


# ================================================================
# COST VALUES
# ================================================================

FREE_SPACE = 0

# Nav2-style internal cost values
INSCRIBED_INFLATED_OBSTACLE = 253
LETHAL_OBSTACLE = 254
NO_INFORMATION = 255


class GlobalCostmapNode(Node):

    def __init__(self):

        super().__init__('global_costmap')

        # ========================================================
        # PARAMETERS
        # ========================================================

        self.declare_parameter('map_topic', '/map')

        self.declare_parameter('global_frame', 'map')

        self.declare_parameter('inflation_radius', 0.55)

        self.declare_parameter('cost_scaling_factor', 1.0)

        self.declare_parameter('robot_radius', 0.22)

        self.declare_parameter('publish_frequency', 1.0)

        self.declare_parameter('inflate_unknown', False)

        # --------------------------------------------------------

        self.map_topic = self.get_parameter(
            'map_topic'
        ).value

        self.global_frame = self.get_parameter(
            'global_frame'
        ).value

        self.inflation_radius = self.get_parameter(
            'inflation_radius'
        ).value

        self.cost_scaling_factor = self.get_parameter(
            'cost_scaling_factor'
        ).value

        self.robot_radius = self.get_parameter(
            'robot_radius'
        ).value

        self.publish_frequency = self.get_parameter(
            'publish_frequency'
        ).value

        self.inflate_unknown = self.get_parameter(
            'inflate_unknown'
        ).value

        # ========================================================
        # MAP STORAGE
        # ========================================================

        self.map_received = False

        self.width = 0
        self.height = 0
        self.resolution = 0.0

        self.origin_x = 0.0
        self.origin_y = 0.0

        self.static_layer = []
        self.inflation_layer = []
        self.master_costmap = []

        # ========================================================
        # MAP QoS
        #
        # map_server publishes /map with transient_local durability.
        # ========================================================

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ========================================================
        # SUBSCRIBER
        # ========================================================

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos
        )

        # ========================================================
        # PUBLISHERS
        # ========================================================

        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/global_costmap/costmap',
            10
        )

        self.raw_costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/global_costmap/costmap_raw',
            10
        )

        self.static_pub = self.create_publisher(
            OccupancyGrid,
            '/global_costmap/static_layer/costmap',
            10
        )

        self.inflation_pub = self.create_publisher(
            OccupancyGrid,
            '/global_costmap/inflation_layer/costmap',
            10
        )

        self.footprint_pub = self.create_publisher(
            PolygonStamped,
            '/global_costmap/footprint',
            10
        )

        # ========================================================
        # TIMER
        # ========================================================

        period = 1.0 / self.publish_frequency

        self.timer = self.create_timer(
            period,
            self.publish_costmaps
        )

        self.get_logger().info(
            'Global Costmap node started'
        )

        self.get_logger().info(
            f'Inflation radius: {self.inflation_radius:.2f} m'
        )

        self.get_logger().info(
            f'Robot radius: {self.robot_radius:.2f} m'
        )

        self.get_logger().info(
            f'Cost scaling factor: {self.cost_scaling_factor:.2f}'
        )

    # ================================================================
    # MAP CALLBACK
    # ================================================================

    def map_callback(self, msg):

        self.get_logger().info(
            'Received map from map_server'
        )

        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution

        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.get_logger().info(
            f'Map size: {self.width} x {self.height}'
        )

        self.get_logger().info(
            f'Resolution: {self.resolution:.3f} m'
        )

        self.get_logger().info(
            f'Origin: ({self.origin_x:.3f}, '
            f'{self.origin_y:.3f})'
        )

        # ========================================================
        # CREATE STATIC LAYER
        # ========================================================

        self.create_static_layer(msg)

        # ========================================================
        # CREATE INFLATION LAYER
        # ========================================================

        self.create_inflation_layer()

        # ========================================================
        # COMBINE LAYERS
        # ========================================================

        self.combine_layers()

        self.map_received = True

        self.get_logger().info(
            'Global costmap successfully generated'
        )

    # ================================================================
    # STATIC LAYER
    # ================================================================

    def create_static_layer(self, map_msg):

        size = self.width * self.height

        self.static_layer = [FREE_SPACE] * size

        occupied_count = 0
        unknown_count = 0

        for i, value in enumerate(map_msg.data):

            if value == -1:

                # Unknown cell
                self.static_layer[i] = NO_INFORMATION

                unknown_count += 1

            elif value >= 65:

                # Occupied
                self.static_layer[i] = LETHAL_OBSTACLE

                occupied_count += 1

            else:

                # Free
                self.static_layer[i] = FREE_SPACE

        self.get_logger().info(
            f'Static layer created: '
            f'{occupied_count} occupied, '
            f'{unknown_count} unknown'
        )

    # ================================================================
    # INFLATION LAYER
    # ================================================================

    def create_inflation_layer(self):

        size = self.width * self.height

        self.inflation_layer = [FREE_SPACE] * size

        # ------------------------------------------------------------
        # Number of cells corresponding to inflation radius
        # ------------------------------------------------------------

        inflation_cells = int(
            math.ceil(
                self.inflation_radius /
                self.resolution
            )
        )

        self.get_logger().info(
            f'Inflation radius = '
            f'{self.inflation_radius:.2f} m '
            f'({inflation_cells} cells)'
        )

        # ------------------------------------------------------------
        # Priority queue
        #
        # Each entry:
        #
        # (distance, cell_index)
        # ------------------------------------------------------------

        queue = []

        visited = [False] * size

        # ------------------------------------------------------------
        # Insert all lethal cells
        # ------------------------------------------------------------

        for y in range(self.height):

            for x in range(self.width):

                index = self.index(x, y)

                if self.static_layer[index] == LETHAL_OBSTACLE:

                    self.inflation_layer[index] = LETHAL_OBSTACLE

                    heapq.heappush(
                        queue,
                        (0.0, index)
                    )

                    visited[index] = True

        # ------------------------------------------------------------
        # 8-connected neighbors
        # ------------------------------------------------------------

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),

            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0))
        ]

        # ------------------------------------------------------------
        # Dijkstra-like propagation
        # ------------------------------------------------------------

        while queue:

            distance_cells, index = heapq.heappop(queue)

            x, y = self.coordinates(index)

            distance_m = (
                distance_cells *
                self.resolution
            )

            # --------------------------------------------------------
            # Don't propagate beyond inflation radius
            # --------------------------------------------------------

            if distance_m > self.inflation_radius:

                continue

            # --------------------------------------------------------
            # Calculate inflation cost
            # --------------------------------------------------------

            if distance_m == 0.0:

                cost = LETHAL_OBSTACLE

            elif distance_m <= self.robot_radius:

                cost = INSCRIBED_INFLATED_OBSTACLE

            else:

                # Nav2-style exponential decay
                #
                # cost =
                #   (INSCRIBED - 1)
                #   * exp(
                #       -cost_scaling_factor
                #       * (distance - inscribed_radius)
                #     )
                #   + 1

                cost = int(
                    (INSCRIBED_INFLATED_OBSTACLE - 1)
                    *
                    math.exp(
                        -self.cost_scaling_factor
                        *
                        (
                            distance_m
                            - self.robot_radius
                        )
                    )
                ) + 1

                cost = max(
                    1,
                    min(
                        cost,
                        INSCRIBED_INFLATED_OBSTACLE
                    )
                )

            # --------------------------------------------------------
            # Update cell
            # --------------------------------------------------------

            if cost > self.inflation_layer[index]:

                self.inflation_layer[index] = cost

            # --------------------------------------------------------
            # Explore neighbors
            # --------------------------------------------------------

            for dx, dy, step_cost in neighbors:

                nx = x + dx
                ny = y + dy

                if nx < 0 or nx >= self.width:
                    continue

                if ny < 0 or ny >= self.height:
                    continue

                neighbor_index = self.index(nx, ny)

                if visited[neighbor_index]:

                    continue

                # Don't inflate through unknown cells unless enabled
                if (
                    self.static_layer[neighbor_index]
                    == NO_INFORMATION
                    and not self.inflate_unknown
                ):

                    continue

                new_distance = (
                    distance_cells + step_cost
                )

                if (
                    new_distance * self.resolution
                    <= self.inflation_radius
                ):

                    visited[neighbor_index] = True

                    heapq.heappush(
                        queue,
                        (
                            new_distance,
                            neighbor_index
                        )
                    )

        # ------------------------------------------------------------
        # Keep unknown cells unknown
        # ------------------------------------------------------------

        if not self.inflate_unknown:

            for i in range(size):

                if self.static_layer[i] == NO_INFORMATION:

                    self.inflation_layer[i] = NO_INFORMATION

    # ================================================================
    # COMBINE STATIC + INFLATION
    # ================================================================

    def combine_layers(self):

        size = self.width * self.height

        self.master_costmap = [FREE_SPACE] * size

        for i in range(size):

            static_cost = self.static_layer[i]

            inflation_cost = self.inflation_layer[i]

            # --------------------------------------------------------
            # Unknown
            # --------------------------------------------------------

            if static_cost == NO_INFORMATION:

                self.master_costmap[i] = NO_INFORMATION

            # --------------------------------------------------------
            # Static obstacle
            # --------------------------------------------------------

            elif static_cost == LETHAL_OBSTACLE:

                self.master_costmap[i] = LETHAL_OBSTACLE

            # --------------------------------------------------------
            # Inflation
            # --------------------------------------------------------

            else:

                self.master_costmap[i] = max(
                    static_cost,
                    inflation_cost
                )

    # ================================================================
    # PUBLISH COSTMAPS
    # ================================================================

    def publish_costmaps(self):

        if not self.map_received:

            return

        # ------------------------------------------------------------
        # Static layer
        # ------------------------------------------------------------

        static_msg = self.create_occupancy_grid(
            self.static_layer
        )

        self.static_pub.publish(static_msg)

        # ------------------------------------------------------------
        # Inflation layer
        # ------------------------------------------------------------

        inflation_msg = self.create_occupancy_grid(
            self.inflation_layer
        )

        self.inflation_pub.publish(inflation_msg)

        # ------------------------------------------------------------
        # Master costmap
        # ------------------------------------------------------------

        costmap_msg = self.create_occupancy_grid(
            self.master_costmap
        )

        self.costmap_pub.publish(costmap_msg)

        # ------------------------------------------------------------
        # Raw costmap
        # ------------------------------------------------------------

        raw_msg = self.create_occupancy_grid(
            self.master_costmap
        )

        self.raw_costmap_pub.publish(raw_msg)

        # ------------------------------------------------------------
        # Footprint
        # ------------------------------------------------------------

        self.publish_footprint()

    # ================================================================
    # CREATE OCCUPANCY GRID
    # ================================================================

    def create_occupancy_grid(self, costs):

        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = self.global_frame

        msg.info.map_load_time = msg.header.stamp

        msg.info.resolution = self.resolution

        msg.info.width = self.width

        msg.info.height = self.height

        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0

        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # ------------------------------------------------------------
        # Convert internal costmap values
        #
        # Internal:
        #
        # 0   = free
        # 1-252 = inflated
        # 253 = inscribed
        # 254 = lethal
        # 255 = unknown
        #
        # OccupancyGrid:
        #
        # 0-100
        # -1 = unknown
        # ------------------------------------------------------------

        occupancy_data = []

        for cost in costs:

            if cost == NO_INFORMATION:

                occupancy_data.append(-1)

            elif cost >= LETHAL_OBSTACLE:

                occupancy_data.append(100)

            elif cost == INSCRIBED_INFLATED_OBSTACLE:

                occupancy_data.append(99)

            else:

                # Convert 0-252 to approximately 0-98

                occupancy = int(
                    cost * 98 / 252
                )

                occupancy = max(
                    0,
                    min(
                        occupancy,
                        98
                    )
                )

                occupancy_data.append(
                    occupancy
                )

        msg.data = occupancy_data

        return msg

    # ================================================================
    # FOOTPRINT
    # ================================================================

    def publish_footprint(self):

        msg = PolygonStamped()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = self.global_frame

        # Circular approximation of robot footprint

        num_points = 32

        for i in range(num_points):

            angle = (
                2.0 *
                math.pi *
                i /
                num_points
            )

            point = Point32()

            point.x = (
                self.robot_radius *
                math.cos(angle)
            )

            point.y = (
                self.robot_radius *
                math.sin(angle)
            )

            point.z = 0.0

            msg.polygon.points.append(point)

        self.footprint_pub.publish(msg)

    # ================================================================
    # INDEX
    # ================================================================

    def index(self, x, y):

        return y * self.width + x

    # ================================================================
    # COORDINATES
    # ================================================================

    def coordinates(self, index):

        x = index % self.width
        y = index // self.width

        return x, y


# ====================================================================
# MAIN
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = GlobalCostmapNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()


