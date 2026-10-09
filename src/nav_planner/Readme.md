### Start Gazebo

```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Start Localization

```bash
ros2 launch nav2_bringup localization_launch.py \
    use_sim_time:=True \
    autostart:=True \
    map:=/home/rishi/nav_ws/src/nav_planner/maps/my_map.yaml
```

### Start Navigation Stack

```bash
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=True \
    autostart:=True \
    map:=/home/rishi/nav_ws/src/nav_planner/maps/my_map.yaml
```

---

## Start Navigation

To navigate the AMR within the environment, run:

```bash
ros2 run nav_planner navigator_final.py
```

---

## Gazebo GUI Follow Camera Offset

To adjust the Gazebo GUI follow camera offset:

```bash
gz service -s /gui/follow/offset \
    --reqtype gz.msgs.Vector3d \
    --reptype gz.msgs.Boolean \
    --timeout 1000 \
    --req 'x: -1.5, y: 0.0, z: 1.0'
```


ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0685]}}"


ros2 action send_goal --feedback /compute_path_to_pose nav2_msgs/action/ComputePathToPose "{goal: {header: {frame_id: 'map'}, pose: {position: {x: -8.83, y: -11.75, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}, use_start: false, planner_id: 'GridBased'}"

 ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: -8.83, y: -11.75, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"


