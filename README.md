# tb2-detection-localization

Единый сабмодуль детекции и локализации мобильного робота TurtleBot 2 по данным 3D-лидара Livox MID-360 для Хакатона StarLine 2026.

## Принцип работы и возможности

Сабмодуль реализует конвейер 3D-детекции и трекинга робота-соперника:

1. **Кластеризация 3D-облака точек (пакет `clustering`, узел `point_cloud_cluster`)**:
   - Вход: 3D PointCloud2 лидара Livox MID-360 (`/livox/lidar`).
   - Фильтрация по высоте $Z$: отсечение пола (`z_min = -0.15` м) и потолка (`z_max = 0.325` м).
   - Фильтрация по дистанции: отсечение собственного шасси (`min_dist_xy = 0.35` м) и дальних шумов (`max_dist_xy = 4.5` м).
   - Адаптивный алгоритм DBSCAN с динамическим масштабированием эпсилон по дальности.
   - Выход: облако точек `/livox/lidar/clust`, где поле `intensity` содержит идентификатор кластера.

2. **Единый универсальный детектор (пакет `obstacle_detector`, узел `detector_node`)**:
   - **Универсальные входные данные**: поддерживает как 3D PointCloud2 (`/livox/lidar/clust` или `/livox/lidar`), так и 2D LaserScan (`/scan`).
   - **Фильтрация известных стен по карте SLAM (`/map`)**:
     - Метод `filter_known_map_points`: удаление точек препятствий, совпадающих с известными стенами OccupancyGrid.
     - Метод `is_pos_inside_wall`: отсечение центроидов кластеров, находящихся внутри статичных стен лабиринта.
   - **Геометрическая классификация**:
     - Метод главных компонент (PCA) для отсечения вытянутых плоских сегментов стен.
     - Алгебраическая аппроксимация окружности методом Каса (Kasa circle fitting) под цилиндрический корпус TurtleBot 2 ($R \approx 0.177\text{ м}$) с контролем среднеквадратичной ошибки (RMS $\le 0.035\text{ м}$).
   - **Продвинутый трекинг**:
     - Отслеживание траектории и различение движущихся и статичных объектов (`SPAN_MOVING`).
     - Публикация цели в `/detection/opponent_robot` (для автономного контроллера миссии), `/detection/robot`, `/detection/robots` и маркеров в `/detection/markers`.
     - Вещание TF-фрейма `opponent_robot`.

---

## Сборка и запуск

### Сборка пакетов
```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select clustering obstacle_detector
source install/setup.bash
```

### Запуск полной 3D-детекции (кластеризация + классификация)
```bash
# Терминал 1: кластеризация 3D-облака Livox
ros2 run clustering point_cloud_cluster --ros-args -p use_sim_time:=true

# Терминал 2: классификатор и трекер
ros2 run obstacle_detector detector_node --ros-args -p use_sim_time:=true -p use_scan:=false -p cloud_topic:=/livox/lidar/clust -p map_topic:=/map -p target_topic:=/detection/opponent_robot
```

### Запуск по 2D LaserScan (резервный режим)
```bash
ros2 run obstacle_detector detector_node --ros-args -p use_sim_time:=true -p use_scan:=true -p scan_topic:=/scan -p map_topic:=/map -p target_topic:=/detection/opponent_robot
```

### Квалификационный запуск из rosbag
```bash
ros2 launch obstacle_detector detect_from_bag.launch.py bag:=/path/to/rosbag loop:=true rviz:=true
```
