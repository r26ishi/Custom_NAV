from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    planner_params = (
        "/home/hild/nav_ws/src/nav_planner/config/planner.yaml"
    )

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[planner_params],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_planner",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "autostart": True,
                "node_names": [
                    "planner_server"
                ],
            }
        ],
    )

    return LaunchDescription([
        planner_server,
        lifecycle_manager,
    ])
