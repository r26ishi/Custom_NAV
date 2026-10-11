cmake_minimum_required(VERSION 3.8)
project(nav_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)



install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY maps
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/navigator_final.py
  scripts/path_smoother.py
  scripts/discrete_path.py
  scripts/trajectory_controller.py
  scripts/trajectory_controller_2.py
  scripts/trajectory_controller_3.py
  scripts/astar_planner.py
  scripts/astar_planner_2.py
  scripts/astar_planner_smoothed.py
  scripts/global_costmap.py
  scripts/custom_controller.py
  
  DESTINATION lib/${PROJECT_NAME}
)

set(dependencies
  rclpy
  nav_msgs
  tf2_ros
  geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
