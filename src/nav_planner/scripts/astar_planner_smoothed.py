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

        # Cells with cost >= this value are considered obstacles.
        #
        # OccupancyGrid:
        #   0      = free
        #   1-99   = occupied probability
        #   100    = occupied
        #   -1     = unknown
        #
        self.declare_parameter(
            'obstacle_cost',
            90
        )

        self.declare_parameter(
            'allow_unknown',
            False
        )

        # Cost weighting.
        #
        # If > 0, cells with higher cost become more expensive.
        self.declare_parameter(
            'cost_weight',
            2.0
        )

        # Whether diagonal movement is allowed.
        self.declare_parameter(
            'allow_diagonal',
            True
        )

        # Minimum distance between path points.
        self.declare_parameter(
            'path_sample_distance',
            0.0
        )

        # Path smoothing parameters
        self.declare_parameter('enable_smoothing', True)
        self.declare_parameter('smoothing_iterations', 100)
        self.declare_parameter('smoothing_data_weight', 0.20)
        self.declare_parameter('smoothing_smooth_weight', 0.30)
        self.declare_parameter('enable_path_shortcutting', True)
        # Number of interpolated curve samples between each pair of waypoints.
        self.declare_parameter('spline_samples_per_segment', 12)

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

        self.cost_weight = self.get_parameter(
            'cost_weight'
        ).value

        self.allow_diagonal = self.get_parameter(
            'allow_diagonal'
        ).value

        self.path_sample_distance = self.get_parameter(
            'path_sample_distance'
        ).value

        self.enable_smoothing = self.get_parameter(
            'enable_smoothing'
        ).value
        self.smoothing_iterations = self.get_parameter(
            'smoothing_iterations'
        ).value
        self.smoothing_data_weight = self.get_parameter(
            'smoothing_data_weight'
        ).value
        self.smoothing_smooth_weight = self.get_parameter(
            'smoothing_smooth_weight'
        ).value
        self.enable_path_shortcutting = self.get_parameter(
            'enable_path_shortcutting'
        ).value
        self.spline_samples_per_segment = int(
            self.get_parameter('spline_samples_per_segment').value
        )

        # ============================================================
        # DATA
        # ============================================================

        self.costmap: Optional[OccupancyGrid] = None
        self.goal: Optional[PoseStamped] = None

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

        self.last_goal = None

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

    # ================================================================
    # COSTMAP CALLBACK
    # ================================================================

    def costmap_callback(
        self,
        msg: OccupancyGrid
    ):

        self.costmap = msg

        # Useful debugging information
        self.get_logger().info(
            f'Costmap received: '
            f'{msg.info.width} x {msg.info.height}, '
            f'resolution={msg.info.resolution:.3f}'
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
            f'y={msg.pose.position.y:.3f}'
        )

    # ================================================================
    # MAIN PLANNING LOOP
    # ================================================================

    def planning_loop(self):

        if self.costmap is None:
            return

        if self.goal is None:
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
        # Convert world coordinates -> grid cells
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
            f'start={start}, goal={goal_cell}'
        )

        # ------------------------------------------------------------
        # Check cells
        # ------------------------------------------------------------

        if not self.is_valid_cell(
            start[0],
            start[1]
        ):
            self.get_logger().error(
                'Robot start cell is occupied'
            )
            return

        if not self.is_valid_cell(
            goal_cell[0],
            goal_cell[1]
        ):
            self.get_logger().error(
                'Goal cell is occupied'
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

            # Publish empty path
            empty_path = Path()

            empty_path.header.frame_id = self.global_frame
            empty_path.header.stamp = self.get_clock().now().to_msg()

            self.plan_pub.publish(empty_path)

            return

        # ------------------------------------------------------------
        # Smooth the A* path before converting it to nav_msgs/Path
        # ------------------------------------------------------------

        raw_path_count = len(path_cells)
        smoothed_path = self.smooth_path(path_cells)

        path_msg = self.cells_to_path(smoothed_path)

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.plan_pub.publish(
            path_msg
        )

        self.get_logger().info(
            f'Path published: raw={raw_path_count} points, '
            f'output={len(smoothed_path)} points'
        )

        # ------------------------------------------------------------
        # Prevent repeatedly planning the exact same goal
        # ------------------------------------------------------------

        self.last_goal = (
            goal.pose.position.x,
            goal.pose.position.y
        )

        # Keep goal so the planner can still be replanned if
        # the costmap changes. We don't disable planning here.

    # ================================================================
    # PATH SMOOTHING
    # ================================================================

    def line_is_free(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float]
    ) -> bool:
        """Check a line segment against the current costmap.

        Coordinates are continuous grid coordinates (gx, gy), not meters.
        Sampling at half-cell intervals helps detect obstacle crossings.
        """
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        distance = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(distance * 2.0)))

        previous = None
        for i in range(steps + 1):
            t = i / float(steps)
            gx_f = a[0] + t * dx
            gy_f = a[1] + t * dy
            gx = int(round(gx_f))
            gy = int(round(gy_f))

            if not self.is_valid_cell(gx, gy):
                return False

            # Prevent a line from squeezing diagonally between two blocked
            # cells when it changes both grid coordinates at once.
            if previous is not None:
                px, py = previous
                if gx != px and gy != py:
                    if (not self.is_valid_cell(gx, py)
                            or not self.is_valid_cell(px, gy)):
                        return False

            previous = (gx, gy)

        return True

    def path_is_free(
        self,
        points: List[Tuple[float, float]]
    ) -> bool:
        if not points:
            return False

        for point in points:
            if not self.is_valid_cell(
                int(round(point[0])), int(round(point[1]))
            ):
                return False

        for i in range(len(points) - 1):
            if not self.line_is_free(points[i], points[i + 1]):
                return False

        return True

    def shortcut_path(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Remove unnecessary waypoints while preserving line-of-sight safety."""
        if len(points) <= 2:
            return list(points)

        result = [points[0]]
        current_index = 0

        while current_index < len(points) - 1:
            # Try the furthest remaining point first.
            next_index = len(points) - 1
            while next_index > current_index + 1:
                if self.line_is_free(points[current_index], points[next_index]):
                    break
                next_index -= 1

            result.append(points[next_index])
            current_index = next_index

        return result

    def smooth_path(
        self,
        cells: List[Tuple[int, int]]
    ) -> List[Tuple[float, float]]:
        """Shortcut and iteratively smooth A* grid coordinates.

        The first and last points remain fixed. Every update is accepted only
        if the complete candidate path remains collision-free on the costmap.
        If smoothing is disabled, the original A* path is returned unchanged.
        """
        points = [(float(x), float(y)) for x, y in cells]

        if not self.enable_smoothing or len(points) < 3:
            return points

        if self.enable_path_shortcutting:
            points = self.shortcut_path(points)

        if len(points) < 3:
            return points

        original = list(points)
        smoothed = list(points)
        data_weight = float(self.smoothing_data_weight)
        smooth_weight = float(self.smoothing_smooth_weight)

        # Keep the start and goal fixed. Reject a whole iteration if its
        # changes would cause any segment to intersect an obstacle.
        for _ in range(max(0, int(self.smoothing_iterations))):
            candidate = list(smoothed)
            max_change = 0.0

            for i in range(1, len(smoothed) - 1):
                x, y = smoothed[i]
                ox, oy = original[i]
                px, py = smoothed[i - 1]
                nx, ny = smoothed[i + 1]

                new_x = x + data_weight * (ox - x) + smooth_weight * (
                    px + nx - 2.0 * x
                )
                new_y = y + data_weight * (oy - y) + smooth_weight * (
                    py + ny - 2.0 * y
                )

                candidate[i] = (new_x, new_y)
                max_change = max(
                    max_change, math.hypot(new_x - x, new_y - y)
                )

            if not self.path_is_free(candidate):
                # Keep the last known safe path; do not force smoothing.
                break

            smoothed = candidate
            if max_change < 1e-3:
                break

        # Convert the smoothed polyline into a true curved path.
        # Catmull-Rom interpolation adds intermediate poses between corners.
        curved = self.catmull_rom_path(
            smoothed, max(2, self.spline_samples_per_segment)
        )

        # Never publish a curve that intersects blocked/unknown cells.
        # If the curve is unsafe, keep the last collision-checked polyline.
        if self.path_is_free(curved):
            return curved

        self.get_logger().warn(
            'Spline would cross a blocked costmap cell; using safe smoothed polyline'
        )
        return smoothed

    def catmull_rom_path(
        self,
        points: List[Tuple[float, float]],
        samples_per_segment: int
    ) -> List[Tuple[float, float]]:
        """Interpolate a polyline as a Catmull-Rom curve in grid coordinates."""
        if len(points) < 3:
            return list(points)

        result: List[Tuple[float, float]] = [points[0]]
        n = len(points)

        # Duplicate the endpoint control points so the curve begins and ends
        # exactly at the first and last path waypoints.
        for i in range(n - 1):
            p0 = points[max(0, i - 1)]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[min(n - 1, i + 2)]

            for sample in range(1, samples_per_segment + 1):
                t = sample / float(samples_per_segment)
                t2 = t * t
                t3 = t2 * t

                x = 0.5 * (
                    2.0 * p1[0]
                    + (-p0[0] + p2[0]) * t
                    + (2.0 * p0[0] - 5.0 * p1[0]
                       + 4.0 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3.0 * p1[0]
                       - 3.0 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    2.0 * p1[1]
                    + (-p0[1] + p2[1]) * t
                    + (2.0 * p0[1] - 5.0 * p1[1]
                       + 4.0 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3.0 * p1[1]
                       - 3.0 * p2[1] + p3[1]) * t3
                )
                result.append((x, y))

        return result

    # ================================================================
    # GET ROBOT POSE FROM TF
    # ================================================================

    def get_robot_pose(self) -> Optional[Tuple[float, float]]:

        try:

            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time()
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y

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
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        gx = int(
            math.floor(
                (x - origin_x) / resolution
            )
        )

        gy = int(
            math.floor(
                (y - origin_y) / resolution
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

        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        x = (
            origin_x
            + (gx + 0.5) * resolution
        )

        y = (
            origin_y
            + (gy + 0.5) * resolution
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

        index = self.grid_index(x, y)

        return self.costmap.data[index]

    # ================================================================
    # VALID CELL
    # ================================================================

    def is_valid_cell(
        self,
        x: int,
        y: int
    ) -> bool:

        width = self.costmap.info.width
        height = self.costmap.info.height

        if x < 0 or x >= width:
            return False

        if y < 0 or y >= height:
            return False

        cost = self.cell_cost(x, y)

        # Unknown cell
        if cost < 0:

            return self.allow_unknown

        # Obstacle
        if cost >= self.obstacle_cost:

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
            neighbor[0] - current[0]
        )

        dy = abs(
            neighbor[1] - current[1]
        )

        # Straight
        if dx + dy == 1:

            distance_cost = 1.0

        # Diagonal
        else:

            distance_cost = math.sqrt(2.0)

        # Add costmap cost.
        #
        # Example:
        #
        # free cell = 0
        # near obstacle = 50
        #
        # Higher cost means A* prefers safer cells.

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

        return distance_cost + cost_penalty

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
                + (math.sqrt(2.0) - 1.0)
                * min(dx, dy)
            )

        else:

            # Manhattan distance

            return dx + dy

    # ================================================================
    # NEIGHBORS
    # ================================================================

    def get_neighbors(
        self,
        node: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:

        x, y = node

        neighbors = []

        # ------------------------------------------------------------
        # 4-connected
        # ------------------------------------------------------------

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        ]

        # ------------------------------------------------------------
        # 8-connected
        # ------------------------------------------------------------

        if self.allow_diagonal:

            directions.extend([
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1)
            ])

        for dx, dy in directions:

            nx = x + dx
            ny = y + dy

            if not self.is_valid_cell(
                nx,
                ny
            ):
                continue

            # Prevent diagonal corner cutting.
            #
            # Example:
            #
            #   X .
            #   . X
            #
            # Don't allow diagonal movement between
            # two obstacles.

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

            cost = self.movement_cost(
                node,
                (nx, ny)
            )

            neighbors.append(
                ((nx, ny), cost)
            )

        return neighbors

    # ================================================================
    # A* ALGORITHM
    # ================================================================

    def a_star(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:

        # Priority queue:
        #
        # (f_score, counter, node)
        #
        # counter avoids comparison issues when f_score is equal.

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

        # Cost from start
        g_score = {
            start: 0.0
        }

        # Parent relationship
        came_from = {}

        closed_set = set()

        while open_set:

            _, _, current = heapq.heappop(
                open_set
            )

            # Already processed
            if current in closed_set:
                continue

            # Goal reached
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

            for neighbor, move_cost in self.get_neighbors(
                current
            ):

                if neighbor in closed_set:
                    continue

                tentative_g = (
                    g_score[current]
                    + move_cost
                )

                if (
                    neighbor not in g_score
                    or tentative_g < g_score[neighbor]
                ):

                    came_from[neighbor] = current

                    g_score[neighbor] = tentative_g

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
    ) -> List[Tuple[int, int]]:

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
        cells: List[Tuple[float, float]]
    ) -> Path:

        path = Path()

        path.header.frame_id = self.global_frame

        path.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        previous_x = None
        previous_y = None

        for i, (gx, gy) in enumerate(cells):

            x, y = self.grid_to_world(
                gx,
                gy
            )

            # Optional path downsampling
            if self.path_sample_distance > 0.0:

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
            # Calculate orientation from path direction
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

