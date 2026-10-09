from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    map_file = '/home/hild/nav_ws/src/nav_planner/maps/my_map.yaml'

    # 1) TurtleBot3 Gazebo world
    tb3_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'turtlebot3_world.launch.py'
            ])
        )
    )

    # 2) Nav2 localization
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'localization_launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'autostart': 'True',
            'map': map_file
        }.items()
    )

    # 3) Nav2 navigation
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'autostart': 'True',
            'map': map_file
        }.items()
    )

    return LaunchDescription([
        tb3_world,
        localization,
        navigation
    ])
