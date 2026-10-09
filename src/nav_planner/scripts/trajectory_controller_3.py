#!/usr/bin/env python3

import csv
import math
import bisect
import time

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "generated_trajectory.csv"

ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"

CONTROL_RATE = 20.0


# ============================================================
# CSV TRAJECTORY LOADER
# ============================================================

def load_trajectory_csv(filename):

    trajectory = []

    required_columns = [
        "time",
        "x",
        "y",
        "yaw",
        "v",
        "vx",
        "vy",
        "omega",
        "ax"
    ]

    with open(filename, "r") as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        # Check required columns
        missing = [
            c for c in required_columns
            if c not in reader.fieldnames
        ]

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

        for row in reader:

            point = {
                "time": float(row["time"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "yaw": float(row["yaw"]),
                "v": float(row["v"]),
                "vx": float(row["vx"]),
                "vy": float(row["vy"]),
                "omega": float(row["omega"]),
                "ax": float(row["ax"]),
            }

            trajectory.append(point)

    if len(trajectory) < 2:
        raise ValueError(
            "Trajectory must contain at least 2 points"
        )

    # --------------------------------------------------------
    # Check time ordering
    # --------------------------------------------------------

    for i in range(1, len(trajectory)):

        if trajectory[i]["time"] <= trajectory[i - 1]["time"]:

            raise ValueError(
                f"Trajectory time is not strictly increasing "
                f"at row {i}"
            )

    return trajectory


# ============================================================
# TRAJECTORY TRACKER
# ============================================================

class CSVTrajectoryTracker(Node):

    def __init__(self):

        super().__init__(
            "csv_trajectory_tracker"
        )

        # ----------------------------------------------------
        # Load trajectory
        # ----------------------------------------------------

        self.trajectory = load_trajectory_csv(
            CSV_FILE
        )

        self.times = [
            p["time"]
            for p in self.trajectory
        ]

        self.duration = self.times[-1]

        # ----------------------------------------------------
        # ROS interfaces
        # ----------------------------------------------------

        self.odom_sub = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.odom_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            CMD_VEL_TOPIC,
            10
        )

        # ----------------------------------------------------
        # Control timer
        # ----------------------------------------------------

        self.timer = self.create_timer(
            1.0 / CONTROL_RATE,
            self.control_loop
        )

        # ----------------------------------------------------
        # Actual robot state
        # ----------------------------------------------------

        self.x = None
        self.y = None
        self.yaw = None

        self.last_odom_time = None

        # ----------------------------------------------------
        # Trajectory clock
        # ----------------------------------------------------

        self.start_time = None

        self.finished = False

        # ----------------------------------------------------
        # Feedback gains
        # ----------------------------------------------------

        self.kx = 0.8
        self.ky = 2.0
        self.ktheta = 2.0

        # ----------------------------------------------------
        # Command limits
        # ----------------------------------------------------

        self.max_linear = 0.5
        self.max_angular = 1.0

        # ----------------------------------------------------
        # Completion tolerances
        # ----------------------------------------------------

        self.position_tolerance = 0.08
        self.yaw_tolerance = 0.15

        # ----------------------------------------------------
        # Debug counter
        # ----------------------------------------------------

        self.last_debug_time = 0.0

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        self.get_logger().info(
            "============================================"
        )

        self.get_logger().info(
            "CSV trajectory loaded"
        )

        self.get_logger().info(
            f"File: {CSV_FILE}"
        )

        self.get_logger().info(
            f"Points: {len(self.trajectory)}"
        )

        self.get_logger().info(
            f"Duration: {self.duration:.3f} s"
        )

        self.get_logger().info(
            f"Start: "
            f"({self.trajectory[0]['x']:.3f}, "
            f"{self.trajectory[0]['y']:.3f})"
        )

        self.get_logger().info(
            f"End: "
            f"({self.trajectory[-1]['x']:.3f}, "
            f"{self.trajectory[-1]['y']:.3f})"
        )

        self.get_logger().info(
            "Waiting for /odom..."
        )

    # ========================================================
    # ANGLE UTILITIES
    # ========================================================

    @staticmethod
    def normalize_angle(angle):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    @staticmethod
    def quaternion_to_yaw(q):

        return math.atan2(
            2.0 * (
                q.w * q.z +
                q.x * q.y
            ),
            1.0 - 2.0 * (
                q.y * q.y +
                q.z * q.z
            )
        )

    # ========================================================
    # ODOM CALLBACK
    # ========================================================

    def odom_callback(self, msg):

        self.x = msg.pose.pose.position.x

        self.y = msg.pose.pose.position.y

        self.yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation
        )

        self.last_odom_time = time.monotonic()

    # ========================================================
    # STOP ROBOT
    # ========================================================

    def publish_stop(self):

        cmd = Twist()

        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    # ========================================================
    # INTERPOLATE CSV TRAJECTORY
    # ========================================================

    def interpolate_trajectory(self, elapsed):

        # ----------------------------------------------------
        # Before trajectory
        # ----------------------------------------------------

        if elapsed <= self.times[0]:

            return self.trajectory[0]

        # ----------------------------------------------------
        # After trajectory
        # ----------------------------------------------------

        if elapsed >= self.times[-1]:

            return self.trajectory[-1]

        # ----------------------------------------------------
        # Find surrounding CSV points
        # ----------------------------------------------------

        i = bisect.bisect_right(
            self.times,
            elapsed
        ) - 1

        p1 = self.trajectory[i]
        p2 = self.trajectory[i + 1]

        t1 = p1["time"]
        t2 = p2["time"]

        dt = t2 - t1

        if dt <= 0.0:

            return p1

        alpha = (
            (elapsed - t1) / dt
        )

        # ----------------------------------------------------
        # Position interpolation
        # ----------------------------------------------------

        x = (
            p1["x"] +
            alpha * (
                p2["x"] -
                p1["x"]
            )
        )

        y = (
            p1["y"] +
            alpha * (
                p2["y"] -
                p1["y"]
            )
        )

        # ----------------------------------------------------
        # Yaw interpolation
        #
        # IMPORTANT:
        #
        # Handles +pi -> -pi transition.
        # ----------------------------------------------------

        yaw_difference = self.normalize_angle(
            p2["yaw"] -
            p1["yaw"]
        )

        yaw = self.normalize_angle(
            p1["yaw"] +
            alpha * yaw_difference
        )

        # ----------------------------------------------------
        # Velocity interpolation
        # ----------------------------------------------------

        v = (
            p1["v"] +
            alpha * (
                p2["v"] -
                p1["v"]
            )
        )

        # ----------------------------------------------------
        # Angular velocity interpolation
        # ----------------------------------------------------

        omega = (
            p1["omega"] +
            alpha * (
                p2["omega"] -
                p1["omega"]
            )
        )

        # ----------------------------------------------------
        # World-frame velocities
        # ----------------------------------------------------

        vx = (
            p1["vx"] +
            alpha * (
                p2["vx"] -
                p1["vx"]
            )
        )

        vy = (
            p1["vy"] +
            alpha * (
                p2["vy"] -
                p1["vy"]
            )
        )

        # ----------------------------------------------------
        # Acceleration
        # ----------------------------------------------------

        ax = (
            p1["ax"] +
            alpha * (
                p2["ax"] -
                p1["ax"]
            )
        )

        return {
            "time": elapsed,
            "x": x,
            "y": y,
            "yaw": yaw,
            "v": v,
            "vx": vx,
            "vy": vy,
            "omega": omega,
            "ax": ax
        }

    # ========================================================
    # CONTROL LOOP
    # ========================================================

    def control_loop(self):

        # ----------------------------------------------------
        # Already finished
        # ----------------------------------------------------

        if self.finished:

            self.publish_stop()

            return

        # ----------------------------------------------------
        # Wait for odometry
        # ----------------------------------------------------

        if (
            self.x is None or
            self.y is None or
            self.yaw is None
        ):

            self.publish_stop()

            return

        # ----------------------------------------------------
        # Check odometry timeout
        # ----------------------------------------------------

        if self.last_odom_time is None:

            self.publish_stop()

            return

        odom_age = (
            time.monotonic() -
            self.last_odom_time
        )

        if odom_age > 0.5:

            self.publish_stop()

            self.finished = True

            self.get_logger().error(
                "Odometry timeout. "
                "Trajectory aborted."
            )

            return

        # ----------------------------------------------------
        # Start trajectory clock
        # ----------------------------------------------------

        if self.start_time is None:

            self.start_time = time.monotonic()

            self.get_logger().info(
                "============================================"
            )

            self.get_logger().info(
                "Trajectory tracking STARTED"
            )

        # ----------------------------------------------------
        # Current trajectory time
        # ----------------------------------------------------

        elapsed = (
            time.monotonic() -
            self.start_time
        )

        # ----------------------------------------------------
        # Check trajectory completion
        # ----------------------------------------------------

        if elapsed >= self.duration:

            final = self.trajectory[-1]

            dx_final = (
                final["x"] -
                self.x
            )

            dy_final = (
                final["y"] -
                self.y
            )

            position_error = math.hypot(
                dx_final,
                dy_final
            )

            heading_error = abs(
                self.normalize_angle(
                    final["yaw"] -
                    self.yaw
                )
            )

            self.publish_stop()

            self.finished = True

            self.get_logger().info(
                "============================================"
            )

            self.get_logger().info(
                "Trajectory COMPLETE"
            )

            self.get_logger().info(
                f"Final position error: "
                f"{position_error:.3f} m"
            )

            self.get_logger().info(
                f"Final heading error: "
                f"{heading_error:.3f} rad"
            )

            return

        # ----------------------------------------------------
        # Get desired trajectory state
        # ----------------------------------------------------

        ref = self.interpolate_trajectory(
            elapsed
        )

        xd = ref["x"]
        yd = ref["y"]
        yawd = ref["yaw"]

        vd = ref["v"]
        omegad = ref["omega"]

        # ----------------------------------------------------
        # Position error in WORLD frame
        # ----------------------------------------------------

        dx = xd - self.x
        dy = yd - self.y

        # ----------------------------------------------------
        # Convert position error to ROBOT frame
        # ----------------------------------------------------

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        ex = (
            cos_yaw * dx +
            sin_yaw * dy
        )

        ey = (
            -sin_yaw * dx +
            cos_yaw * dy
        )

        # ----------------------------------------------------
        # Heading error
        # ----------------------------------------------------

        etheta = self.normalize_angle(
            yawd -
            self.yaw
        )

        # ----------------------------------------------------
        # Feed-forward + feedback controller
        # ----------------------------------------------------

        v_cmd = (
            vd * math.cos(etheta)
            +
            self.kx * ex
        )

        omega_cmd = (
            omegad
            +
            self.ky * vd * ey
            +
            self.ktheta *
            math.sin(etheta)
        )

        # ----------------------------------------------------
        # Command saturation
        # ----------------------------------------------------

        v_cmd = max(
            -self.max_linear,
            min(
                self.max_linear,
                v_cmd
            )
        )

        omega_cmd = max(
            -self.max_angular,
            min(
                self.max_angular,
                omega_cmd
            )
        )

        # ----------------------------------------------------
        # Publish cmd_vel
        # ----------------------------------------------------

        cmd = Twist()

        cmd.linear.x = float(
            v_cmd
        )

        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0

        cmd.angular.z = float(
            omega_cmd
        )

        self.cmd_pub.publish(cmd)

        # ----------------------------------------------------
        # Diagnostic information
        # ----------------------------------------------------

        now = time.monotonic()

        if now - self.last_debug_time > 1.0:

            self.last_debug_time = now

            position_error = math.hypot(
                dx,
                dy
            )

            self.get_logger().info(
                f"t={elapsed:6.2f} | "
                f"actual=({self.x:7.3f}, "
                f"{self.y:7.3f}) | "
                f"desired=({xd:7.3f}, "
                f"{yd:7.3f}) | "
                f"ex={ex:6.3f} | "
                f"ey={ey:6.3f} | "
                f"v_ref={vd:5.3f} | "
                f"v_cmd={v_cmd:5.3f} | "
                f"w_ref={omegad:5.3f} | "
                f"w_cmd={omega_cmd:5.3f} | "
                f"err={position_error:.3f}"
            )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    try:

        node = CSVTrajectoryTracker()

        rclpy.spin(node)

    except KeyboardInterrupt:

        print(
            "\nKeyboard interrupt. "
            "Stopping robot."
        )

    except Exception as e:

        print(
            f"\nERROR: {e}"
        )

    finally:

        try:

            node.publish_stop()
            node.destroy_node()

        except Exception:
            pass

        rclpy.shutdown()


if __name__ == "__main__":

    main()
