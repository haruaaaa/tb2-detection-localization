# tb2-detection-localization

Detection and localization Turtlebot 2 based on Livox lidar point cloud.

## How detection works

Two robots (TB2) drive in a maze. The lidar sees walls and the opponent robot. Detection separates them by **shape**:

- **Wall** = long/elongated cluster (segment), size > 0.6m
- **Robot** = compact round cluster (circle), size 0.12..0.55m

This works even when robot is **static** because shape doesn't depend on motion.

For **moving robots** we also track trajectory span (bbox diagonal of positions over time). Robot has large span (>1m), static clutter has span ~0.

Livox lidar has non-repetitive scan pattern causing wall flicker. To filter false positives, we require **temporal persistence**: a cluster must survive N consecutive frames to be confirmed as robot.

## Detection pipeline

```
/livox/lidar (PointCloud2)
    │
    ▼
clustering/cluster_node
    │  DBSCAN 3D (eps=0.1, min_samples=15)
    │  z filter (keep z <= 0.33)
    ▼
/livox/lidar/clust (PointCloud2 with cluster_id in intensity)
    │
    ▼
obstacle_detector/detector_node
    │  classify each cluster: wall (large) vs robot candidate (compact)
    │  track candidates with NN + gate
    │  confirm after min_hits consecutive frames
    ▼
/detection/robot    (PoseStamped)
/detection/robots   (PoseArray)
/detection/markers  (MarkerArray for RViz)
```

## Parameters

| param | default | description |
|-------|---------|-------------|
| robot_min_size | 0.12 | min cluster size to be robot candidate |
| robot_max_size | 0.55 | max cluster size to be robot candidate |
| seg_len_min | 0.60 | cluster larger than this = wall |
| elong_max | 2.5 | elongation threshold (major/minor axis) |
| gate | 0.40 | max distance to associate cluster with track |
| min_hits | 5 | frames to confirm track as robot |
| max_miss | 6 | frames without update before track drops |

## Tests

Tests validate detection on recorded rosbags:

**Shape detection tests** (static/rot bags):
- Robot detected by shape alone at expected distance
- Works for static robot

**Motion detection tests** (mov bags):
- Robot detected AND has large trajectory span (>2m)
- Confirms motion tracking works

**Wall rejection test**:
- Walls classified as segments, not robot candidates
- Robot candidates are compact

Run tests:
```bash
cd src/obstacle_detector
pytest test/test_detection.py -v
```

Requires rosbags in `../../rosbags/` relative to package.

## Dependencies

```bash
pip install scikit-learn
```

## Bags

Put rosbag2/mcap in `src/bags/` (gitignored, dockerignored).
In container: `/root/ws/src/bags`.

Current: `rosbag2_2026_06_27-18_43-mov_01`

## Build and run

```bash
colcon build
source install/setup.bash

ros2 launch obstacle_detector detect_from_bag.launch.py
ros2 launch obstacle_detector detect_from_bag.launch.py bag:=/path/to/bag rviz:=true
```
In tmux:

```
rviz2
```
```
ros2 run clustering cluster_node 
```
```
ros2 run obstacle_detector detector_node
```
```
cd src/bags
ros2 bag play /path/to/your/bag
```


Run detector alone:
```bash
ros2 run obstacle_detector detector_node
ros2 run obstacle_detector detector_node --ros-args -p robot_max_size:=0.5 -p min_hits:=5
```

## File structure

```
src/
├── clustering/
│   └── clustering/
│       └── cluster_node.py
├── obstacle_detector/
│   ├── obstacle_detector/
│   │   ├── core.py          # detection logic (no ROS, testable)
│   │   └── detector_node.py # ROS2 node
│   ├── launch/
│   │   └── detect_from_bag.launch.py
│   └── test/
│       └── test_detection.py
└── bags/
    └── rosbag2_...-mov_01/
```
