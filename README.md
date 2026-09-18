# tb2-detection-localization

Единый сабмодуль детекции и локализации мобильного робота TurtleBot 2 по данным лидара Livox для Хакатона StarLine 2026.

## Принцип работы и возможности

Сабмодуль объединяет в себе полный конвейер детекции:
1. **Кластеризация точек (пакет `clustering`)**:
   - Адаптивный DBSCAN по 3D облаку точек лидара Livox (`/livox/lidar`).
   - Фильтрация по высоте и дистанции.
   - Публикация кластеризованного облака в `/livox/lidar/clust`.

2. **Единый универсальный детектор (пакет `obstacle_detector`)**:
   - **Универсальные входные данные**: поддерживает как 2D LaserScan (`/scan`), так и 3D PointCloud2 (`/livox/lidar/clust` или `/livox/lidar`).
   - **Фильтрация стен по карте SLAM (`/map`)**: при наличии карты автоматически отсекает точки известных стен лабиринта, выделяя только динамические объекты в свободном пространстве.
   - **Геометрическая классификация**:
     - Метод главных компонент (PCA) для отсечения протяженных сегментов стен.
     - Алгебраическая подгонка окружности методом Каса (Kasa circle fitting) под цилиндрический корпус TurtleBot 2 ($R \approx 0.177\text{ м}$).
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

### Запуск детектора в составе системы
```bash
ros2 run obstacle_detector detector_node --ros-args -p use_scan:=true -p map_topic:=/map -p target_topic:=/detection/opponent_robot
```

### Квалификационный запуск из rosbag
```bash
ros2 launch obstacle_detector detect_from_bag.launch.py bag:=/path/to/rosbag loop:=true rviz:=true
```
