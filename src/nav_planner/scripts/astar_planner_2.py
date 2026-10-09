#!/usr/bin/env python3

import math
import heapq
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

import tf2_ros
from tf2_ros import TransformException


class AStarPlanner(Node):

    def __init__(self):
        super().__init__('astar_planner')

        # ============================================================
        # PARAMETERS
        # ============================================================

        self.declare_parameter(
            'costmap_topic',
            '/global_costmap/costmap'
        )

        self.declare_parameter(
            'goal_topic',
            '/goal_pose'
        )

        self.declare_parameter(
            'plan_topic',
            '/plan'
        )

        self.declare_parameter(
            'global_frame',
            'map'
        )

        self.declare_parameter(
            'robot_frame',
            'base_link'
        )

        # ------------------------------------------------------------
        # OBSTACLE PARAMETERS
        # ------------------------------------------------------------

        self.declare_parameter(
            'obstacle_cost',
            90
        )

        self.declare_parameter(
            'allow_unknown',
            False
        )

        # ------------------------------------------------------------
        # ROBOT FOOTPRINT
        # ------------------------------------------------------------

        # Robot radius in meters.
        #
        # Example:
        # TurtleBot3 Burger ~= 0.105 m radius
        # Your actual robot may be larger.
        #
        self.declare_parameter(
            'robot_radius',
            0.3
        )

        # Extra safety distance from obstacles.
        #
        # Example:
        # robot radius = 0.22 m
        # safety margin = 0.05 m
        #
        # effective radius = 0.27 m
        #
        self.declare_parameter(
            'safety_margin',
            0.05
        )

        # ------------------------------------------------------------
        # A* COST PARAMETERS
        # ------------------------------------------------------------

        self.declare_parameter(
            'cost_weight',
            2.0
        )

        self.declare_parameter(
            'allow_diagonal',
            True
        )

        # ------------------------------------------------------------
        # PATH SAMPLING
        # ------------------------------------------------------------

        self.declare_parameter(
            'path_sample_distance',
            0.0
        )

        # ============================================================
        # GET PARAMETERS
        # ============================================================

        self.costmap_topic = self.get_parameter(
            'costmap_topic'
        ).value

        self.goal_topic = self.get_parameter(
            'goal_topic'
        ).value

        self.plan_topic = self.get_parameter(
            'plan_topic'
        ).value

        self.global_frame = self.get_parameter(
            'global_frame'
        ).value

        self.robot_frame = self.get_parameter(
            'robot_frame'
        ).value

        self.obstacle_cost = self.get_parameter(
            'obstacle_cost'
        ).value

        self.allow_unknown = self.get_parameter(
            'allow_unknown'
        ).value

        self.robot_radius = self.get_parameter(
            'robot_radius'
        ).value

        self.safety_margin = self.get_parameter(
            'safety_margin'
        ).value

        self.cost_weight = self.get_parameter(
            'cost_weight'
        ).value

        self.allow_diagonal = self.get_parameter(
            'allow_diagonal'
        ).value

        self.path_sample_distance = self.get_parameter(
            'path_sample_distance'
        ).value

        # ============================================================
        # EFFECTIVE ROBOT RADIUS
        # ============================================================

        self.effective_radius = (
            self.robot_radius
            + self.safety_margin
        )

        # Number of cells required for inflation.
        #
        # This is calculated after receiving the costmap because
        # the costmap resolution is contained inside OccupancyGrid.
        #
        self.inflation_cells = 0

        # ============================================================
        # DATA
        # ============================================================

        self.costmap: Optional[OccupancyGrid] = None

        self.goal: Optional[PoseStamped] = None

        # Inflated obstacle grid.
        #
        # True  = obstacle / unsafe
        # False = free
        #
        self.inflated_obstacles: Optional[List[bool]] = None

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # SUBSCRIBERS
        # ============================================================

        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.costmap_topic,
            self.costmap_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self.goal_callback,
            10
        )

        # ============================================================
        # PUBLISHER
        # ============================================================

        self.plan_pub = self.create_publisher(
            Path,
            self.plan_topic,
            10
        )

        # ============================================================
        # TIMER
        # ============================================================

        self.timer = self.create_timer(
            0.2,
            self.planning_loop
        )

        # Used to avoid rebuilding the inflation grid unnecessarily.
        self.last_costmap_stamp = None

        self.get_logger().info(
            '============================================'
        )

        self.get_logger().info(
            'A* Planner started'
        )

        self.get_logger().info(
            f'Costmap: {self.costmap_topic}'
        )

        self.get_logger().info(
            f'Goal: {self.goal_topic}'
        )

        self.get_logger().info(
            f'Plan: {self.plan_topic}'
        )

        self.get_logger().info(
            f'Robot radius: {self.robot_radius:.3f} m'
        )

        self.get_logger().info(
            f'Safety margin: {self.safety_margin:.3f} m'
        )

        self.get_logger().info(
            f'Effective radius: '
            f'{self.effective_radius:.3f} m'
        )

        self.get_logger().info(
            '============================================'
        )

    # ================================================================
    # COSTMAP CALLBACK
    # ================================================================

    def costmap_callback(
        self,
        msg: OccupancyGrid
    ):

        self.costmap = msg

        resolution = msg.info.resolution

        # Calculate how many grid cells correspond to the
        # effective robot radius.
        self.inflation_cells = int(
            math.ceil(
                self.effective_radius / resolution
            )
        )

        # Rebuild inflated obstacle grid.
        self.build_inflated_obstacle_grid()

        self.get_logger().info(
            f'Costmap received: '
            f'{msg.info.width} x {msg.info.height}, '
            f'resolution={resolution:.3f}, '
            f'inflation={self.inflation_cells} cells '
            f'({self.inflation_cells * resolution:.3f} m)'
        )

    # ================================================================
    # GOAL CALLBACK
    # ================================================================

    def goal_callback(
        self,
        msg: PoseStamped
    ):

        self.goal = msg

        self.get_logger().info(
            f'New goal received: '
            f'x={msg.pose.position.x:.3f}, '
            f'y={msg.pose.position.y:.3f}, '
            f'frame={msg.header.frame_id}'
        )

    # ================================================================
    # BUILD INFLATED OBSTACLE GRID
    # ================================================================

    def build_inflated_obstacle_grid(self):

        if self.costmap is None:
            return

        width = self.costmap.info.width
        height = self.costmap.info.height

        total_cells = width * height

        # Start with everything free.
        inflated = [
            False
            for _ in range(total_cells)
        ]

        # ------------------------------------------------------------
        # Find all obstacle cells
        # ------------------------------------------------------------

        obstacle_cells = []

        for y in range(height):

            for x in range(width):

                index = self.grid_index(
                    x,
                    y
                )

                cost = self.costmap.data[index]

                # Unknown cells
                if cost < 0:

                    if not self.allow_unknown:

                        obstacle_cells.append(
                            (x, y)
                        )

                    continue

                # Occupied cells
                if cost >= self.obstacle_cost:

                    obstacle_cells.append(
                        (x, y)
                    )

        # ------------------------------------------------------------
        # Inflate every obstacle
        # ------------------------------------------------------------

        for obstacle_x, obstacle_y in obstacle_cells:

            min_x = max(
                0,
                obstacle_x - self.inflation_cells
            )

            max_x = min(
                width - 1,
                obstacle_x + self.inflation_cells
            )

            min_y = max(
                0,
                obstacle_y - self.inflation_cells
            )

            max_y = min(
                height - 1,
                obstacle_y + self.inflation_cells
            )

            for y in range(
                min_y,
                max_y + 1
            ):

                for x in range(
                    min_x,
                    max_x + 1
                ):

                    dx = (
                        x - obstacle_x
                    )

                    dy = (
                        y - obstacle_y
                    )

                    distance = math.sqrt(
                        dx * dx
                        + dy * dy
                    )

                    # Circular robot footprint.
                    if distance <= self.inflation_cells:

                        index = self.grid_index(
                            x,
                            y
                        )

                        inflated[index] = True

        self.inflated_obstacles = inflated

        self.get_logger().info(
            f'Inflated obstacle grid created: '
            f'{len(obstacle_cells)} original obstacle cells'
        )

    # ================================================================
    # MAIN PLANNING LOOP
    # ================================================================

    def planning_loop(self):

        if self.costmap is None:
            return

        if self.goal is None:
            return

        if self.inflated_obstacles is None:
            return

        # ------------------------------------------------------------
        # Get robot pose
        # ------------------------------------------------------------

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            return

        # ------------------------------------------------------------
        # Check goal frame
        # ------------------------------------------------------------

        goal = self.goal

        if goal.header.frame_id != self.global_frame:

            self.get_logger().warn(
                f'Goal frame is '
                f'{goal.header.frame_id}, expected '
                f'{self.global_frame}'
            )

            return

        # ------------------------------------------------------------
        # World -> Grid
        # ------------------------------------------------------------

        start = self.world_to_grid(
            robot_pose[0],
            robot_pose[1]
        )

        goal_cell = self.world_to_grid(
            goal.pose.position.x,
            goal.pose.position.y
        )

        if start is None:

            self.get_logger().error(
                'Robot is outside costmap'
            )

            return

        if goal_cell is None:

            self.get_logger().error(
                'Goal is outside costmap'
            )

            return

        self.get_logger().info(
            f'Planning: '
            f'start={start}, '
            f'goal={goal_cell}'
        )

        # ------------------------------------------------------------
        # Check start
        # ------------------------------------------------------------

        if not self.is_valid_cell(
            start[0],
            start[1]
        ):

            self.get_logger().error(
                'Robot start cell is inside inflated obstacle'
            )

            return

        # ------------------------------------------------------------
        # Check goal
        # ------------------------------------------------------------

        if not self.is_valid_cell(
            goal_cell[0],
            goal_cell[1]
        ):

            self.get_logger().error(
                'Goal cell is inside inflated obstacle'
            )

            return

        # ------------------------------------------------------------
        # A*
        # ------------------------------------------------------------

        path_cells = self.a_star(
            start,
            goal_cell
        )

        if path_cells is None:

            self.get_logger().warn(
                'A* could not find a path'
            )

            empty_path = Path()

            empty_path.header.frame_id = (
                self.global_frame
            )

            empty_path.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            self.plan_pub.publish(
                empty_path
            )

            return

        # ------------------------------------------------------------
        # Cells -> Path
        # ------------------------------------------------------------

        path_msg = self.cells_to_path(
            path_cells
        )

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.plan_pub.publish(
            path_msg
        )

        self.get_logger().info(
            f'Path published: '
            f'{len(path_cells)} cells'
        )

    # ================================================================
    # GET ROBOT POSE FROM TF
    # ================================================================

    def get_robot_pose(
        self
    ) -> Optional[Tuple[float, float]]:

        try:

            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time()
            )

            x = (transform.transform.translation.x)

            y = (transform.transform.translation.y)

            return x, y

        except TransformException as ex:

            self.get_logger().info(
                f'Waiting for TF '
                f'{self.global_frame} -> '
                f'{self.robot_frame}: {ex}'
            )

            return None

    # ================================================================
    # WORLD -> GRID
    # ================================================================

    def world_to_grid(
        self,
        x: float,
        y: float
    ) -> Optional[Tuple[int, int]]:

        info = self.costmap.info

        resolution = info.resolution

        origin_x = (
            info.origin.position.x
        )

        origin_y = (
            info.origin.position.y
        )

        gx = int(
            math.floor(
                (x - origin_x)
                / resolution
            )
        )

        gy = int(
            math.floor(
                (y - origin_y)
                / resolution
            )
        )

        if gx < 0 or gx >= info.width:
            return None

        if gy < 0 or gy >= info.height:
            return None

        return gx, gy

    # ================================================================
    # GRID -> WORLD
    # ================================================================

    def grid_to_world(
        self,
        gx: int,
        gy: int
    ) -> Tuple[float, float]:

        info = self.costmap.info

        resolution = info.resolution

        origin_x = (
            info.origin.position.x
        )

        origin_y = (
            info.origin.position.y
        )

        x = (
            origin_x
            + (gx + 0.5)
            * resolution
        )

        y = (
            origin_y
            + (gy + 0.5)
            * resolution
        )

        return x, y

    # ================================================================
    # GRID INDEX
    # ================================================================

    def grid_index(
        self,
        x: int,
        y: int
    ) -> int:

        return (
            y * self.costmap.info.width
            + x
        )

    # ================================================================
    # CELL COST
    # ================================================================

    def cell_cost(
        self,
        x: int,
        y: int
    ) -> int:

        index = self.grid_index(
            x,
            y
        )

        return self.costmap.data[index]

    # ================================================================
    # VALID CELL
    # ================================================================

    def is_valid_cell(
        self,
        x: int,
        y: int
    ) -> bool:

        if self.costmap is None:
            return False

        width = self.costmap.info.width
        height = self.costmap.info.height

        if x < 0 or x >= width:
            return False

        if y < 0 or y >= height:
            return False

        # ------------------------------------------------------------
        # Check inflated obstacle grid
        # ------------------------------------------------------------

        if self.inflated_obstacles is not None:

            index = self.grid_index(
                x,
                y
            )

            if self.inflated_obstacles[index]:

                return False

        return True

    # ================================================================
    # MOVEMENT COST
    # ================================================================

    def movement_cost(
        self,
        current: Tuple[int, int],
        neighbor: Tuple[int, int]
    ) -> float:

        dx = abs(
            neighbor[0]
            - current[0]
        )

        dy = abs(
            neighbor[1]
            - current[1]
        )

        # Straight
        if dx + dy == 1:

            distance_cost = 1.0

        # Diagonal
        else:

            distance_cost = math.sqrt(
                2.0
            )

        # ------------------------------------------------------------
        # Costmap penalty
        # ------------------------------------------------------------

        cell_cost = self.cell_cost(
            neighbor[0],
            neighbor[1]
        )

        if cell_cost < 0:
            cell_cost = 0

        cost_penalty = (
            self.cost_weight
            * (cell_cost / 100.0)
        )

        return (
            distance_cost
            + cost_penalty
        )

    # ================================================================
    # HEURISTIC
    # ================================================================

    def heuristic(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int]
    ) -> float:

        dx = abs(
            a[0] - b[0]
        )

        dy = abs(
            a[1] - b[1]
        )

        if self.allow_diagonal:

            # Octile distance
            return (
                max(dx, dy)
                + (
                    math.sqrt(2.0) - 1.0
                )
                * min(dx, dy)
            )

        else:

            # Manhattan distance
            return (
                dx + dy
            )

    # ================================================================
    # NEIGHBORS
    # ================================================================

    def get_neighbors(
        self,
        node: Tuple[int, int]
    ) -> List[
        Tuple[
            Tuple[int, int],
            float
        ]
    ]:

        x, y = node

        neighbors = []

        # ------------------------------------------------------------
        # 4-connected movement
        # ------------------------------------------------------------

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ]

        # ------------------------------------------------------------
        # 8-connected movement
        # ------------------------------------------------------------

        if self.allow_diagonal:

            directions.extend([
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1)
            ])

        # ------------------------------------------------------------
        # Explore
        # ------------------------------------------------------------

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            # --------------------------------------------------------
            # Check bounds + inflated obstacle
            # --------------------------------------------------------

            if not self.is_valid_cell(
                nx,
                ny
            ):

                continue

            # --------------------------------------------------------
            # Prevent diagonal corner cutting
            # --------------------------------------------------------

            if dx != 0 and dy != 0:

                if not self.is_valid_cell(
                    x + dx,
                    y
                ):

                    continue

                if not self.is_valid_cell(
                    x,
                    y + dy
                ):

                    continue

            # --------------------------------------------------------
            # Movement cost
            # --------------------------------------------------------

            cost = self.movement_cost(
                node,
                (nx, ny)
            )

            neighbors.append(
                (
                    (nx, ny),
                    cost
                )
            )

        return neighbors

    # ================================================================
    # A* ALGORITHM
    # ================================================================

    def a_star(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[
        List[
            Tuple[int, int]
        ]
    ]:

        # ------------------------------------------------------------
        # Priority queue
        #
        # (f_score, counter, node)
        # ------------------------------------------------------------

        open_set = []

        counter = 0

        start_h = self.heuristic(
            start,
            goal
        )

        heapq.heappush(
            open_set,
            (
                start_h,
                counter,
                start
            )
        )

        # ------------------------------------------------------------
        # Cost from start
        # ------------------------------------------------------------

        g_score = {
            start: 0.0
        }

        # ------------------------------------------------------------
        # Parent relationship
        # ------------------------------------------------------------

        came_from = {}

        # ------------------------------------------------------------
        # Processed nodes
        # ------------------------------------------------------------

        closed_set = set()

        # ============================================================
        # SEARCH
        # ============================================================

        while open_set:

            _, _, current = (
                heapq.heappop(
                    open_set
                )
            )

            # --------------------------------------------------------
            # Already processed
            # --------------------------------------------------------

            if current in closed_set:

                continue

            # --------------------------------------------------------
            # Goal reached
            # --------------------------------------------------------

            if current == goal:

                return self.reconstruct_path(
                    came_from,
                    current
                )

            closed_set.add(
                current
            )

            # --------------------------------------------------------
            # Explore neighbors
            # --------------------------------------------------------

            for neighbor, move_cost in (
                self.get_neighbors(
                    current
                )
            ):

                if neighbor in closed_set:

                    continue

                tentative_g = (
                    g_score[current]
                    + move_cost
                )

                if (
                    neighbor not in g_score
                    or tentative_g
                    < g_score[neighbor]
                ):

                    came_from[
                        neighbor
                    ] = current

                    g_score[
                        neighbor
                    ] = tentative_g

                    h = self.heuristic(
                        neighbor,
                        goal
                    )

                    f = (
                        tentative_g
                        + h
                    )

                    counter += 1

                    heapq.heappush(
                        open_set,
                        (
                            f,
                            counter,
                            neighbor
                        )
                    )

        return None

    # ================================================================
    # RECONSTRUCT PATH
    # ================================================================

    def reconstruct_path(
        self,
        came_from,
        current
    ) -> List[
        Tuple[int, int]
    ]:

        path = [
            current
        ]

        while current in came_from:

            current = came_from[
                current
            ]

            path.append(
                current
            )

        path.reverse()

        return path

    # ================================================================
    # CELLS -> NAV_MSGS/PATH
    # ================================================================

    def cells_to_path(
        self,
        cells: List[
            Tuple[int, int]
        ]
    ) -> Path:

        path = Path()

        path.header.frame_id = (
            self.global_frame
        )

        path.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        previous_x = None
        previous_y = None

        # ------------------------------------------------------------
        # Convert every grid cell
        # ------------------------------------------------------------

        for i, (gx, gy) in enumerate(
            cells
        ):

            x, y = self.grid_to_world(
                gx,
                gy
            )

            # --------------------------------------------------------
            # Optional path downsampling
            # --------------------------------------------------------

            if (
                self.path_sample_distance
                > 0.0
            ):

                if (
                    previous_x is not None
                    and previous_y is not None
                ):

                    distance = math.hypot(
                        x - previous_x,
                        y - previous_y
                    )

                    if (
                        distance
                        < self.path_sample_distance
                        and i != len(cells) - 1
                    ):

                        continue

            pose = PoseStamped()

            pose.header.frame_id = (
                self.global_frame
            )

            pose.header.stamp = (
                path.header.stamp
            )

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # --------------------------------------------------------
            # Calculate yaw
            # --------------------------------------------------------

            if i < len(cells) - 1:

                nx, ny = self.grid_to_world(
                    cells[i + 1][0],
                    cells[i + 1][1]
                )

                yaw = math.atan2(
                    ny - y,
                    nx - x
                )

            elif i > 0:

                px, py = self.grid_to_world(
                    cells[i - 1][0],
                    cells[i - 1][1]
                )

                yaw = math.atan2(
                    y - py,
                    x - px
                )

            else:

                yaw = 0.0

            # --------------------------------------------------------
            # Yaw -> Quaternion
            # --------------------------------------------------------

            pose.pose.orientation.x = 0.0

            pose.pose.orientation.y = 0.0

            pose.pose.orientation.z = math.sin(
                yaw / 2.0
            )

            pose.pose.orientation.w = math.cos(
                yaw / 2.0
            )

            path.poses.append(
                pose
            )

            previous_x = x
            previous_y = y

        return path


# ====================================================================
# MAIN
# ====================================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = AStarPlanner()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()
