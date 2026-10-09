#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ============================================================
    # 1. TurtleBot3 Gazebo World
    # ============================================================

    tb3_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'turtlebot3_world.launch.py'
            ])
        )
    )

    # ============================================================
    # 2. Coarse Path Publisher
    # ============================================================

    coarse_path = Node(
        package='nav_planner',
        executable='coarse_path.py',
        parameters=[
            {
                'csv_file': '/home/hild/nav_ws/src/nav_planner/include/raw_path.csv',
                'frame_id': 'map',
            }
        ],
        output='screen'
    )

    # ============================================================
    # 3. Path Smoother
    # ============================================================

    smoother = Node(
        package='nav_planner',
        executable='smoother_path.py',
        output='screen'
    )

    # ============================================================
    # 4. Trajectory Generator
    # ============================================================

    trajectory = Node(
        package='nav_planner',
        executable='trajectory.py',
        output='screen'
    )

    # ============================================================
    # 5. Trajectory Controller
    # ============================================================

    controller = Node(
        package='nav_planner',
        executable='trajectory_controller.py',
        output='screen'
    )
    
       # ============================================================
    # Start Controller After Delay
    # ============================================================

    delayed_controller = TimerAction(
        period=10.0,          # Delay in seconds
        actions=[controller]
    )

    # ============================================================
    # Launch all nodes
    # ============================================================

    return LaunchDescription([
        tb3_world,
        coarse_path,
        smoother,
        trajectory,
        delayed_controller,
    ])
