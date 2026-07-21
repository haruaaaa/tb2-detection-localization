# tb2-detection-localization
Detection and localization Turtlebot 2 based on lidar point cloud.

## Пакеты (src/)
- **clustering** — `/livox/lidar` → DBSCAN в 3D → `/livox/lidar/clust`
  (id кластера в поле intensity).
- **obstacle_detector** — распознаёт робота среди кластеров: каждый кластер по
  форме относит к **стене** (длинный/вытянутый сегмент) или к **роботу**
  (компактный кружок робот-размерного габарита), ведёт кружки во времени
  (NN + гейт, подтверждение за N кадров против мерцания Livox). Работает и когда
  робот стоит. Публикует:
  - `/detection/robot` (PoseStamped) — первичный робот;
  - `/detection/robots` (PoseArray) — все подтверждённые;
  - `/detection/markers` (MarkerArray) — визуализация для RViz
    (зелёные сферы = роботы, серые кубы = стены).

## Зависимости
```
pip install scikit-learn
```

## Бэги
Rosbag2/mcap кладутся локально в `src/bags/` (в git и в docker-образ не входят,
см. `.gitignore` / `.dockerignore`; в контейнере путь `/root/ws/src/bags`).
Сейчас там лежит `rosbag2_2026_06_27-18_43-mov_01` (движущийся робот).

## Сборка и запуск
```bash
colcon build
source install/setup.bash
ros2 launch obstacle_detector detect_from_bag.launch.py

ros2 launch obstacle_detector detect_from_bag.launch.py \
    bag:=/root/ws/src/bags/<имя_бэга> loop:=false rviz:=true
```

Отдельно только детектор (если clustering уже публикует `/livox/lidar/clust`):
```bash
ros2 run obstacle_detector detector_node
ros2 run obstacle_detector detector_node --ros-args \
    -p robot_max_size:=0.5 -p min_hits:=5 -p seg_len_min:=0.6
```
