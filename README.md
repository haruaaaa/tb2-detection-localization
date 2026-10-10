# tb2-detection-localization

Сабмодуль 3D-детекции и локализации мобильного робота TurtleBot 2 по данным лидара Livox MID-360 для соревнований StarLine 2026.

---

## 1. Архитектура пайплайна

Система выполняет детекцию робота-соперника в реальном времени через связку двух независимых ROS 2 узлов:

1. **`clustering` (`cluster_node`)**:
   - Вход: 3D PointCloud2 из топика `/livox/lidar`.
   - Пространственная фильтрация: отсечение пола ($z > +0.325$ м) и потолка ($z < -0.16$ м) с учетом ориентации лидара.
   - Быстрая кластеризация DBSCAN (SciPy/NumPy).
   - Выход: облако точек `/livox/lidar/clust` с закодированным в поле `intensity` идентификатором кластера.

2. **`o3d_detector` (`detector`)**:
   - Вход: `/livox/lidar/clust`.
   - Геометрический анализ: фильтрация плоских стен и крупных препятствий по габаритам и PCA.
   - Аппроксимация формы цилиндра методом Kasa Circle Fit (радиус базы $R \approx 0.177$ м).
   - Трекинг положения с фильтрацией ложных выбросов и режимом удержания трека (coasting).
   - Выход:
     - `/detection/robot` (`geometry_msgs/PoseStamped`) — поза обнаруженного соперника в СК лидара/карты.
     - `/detected_robot`, `/detected_robot_pose` (`visualization_msgs/MarkerArray`) — маркеры для RViz2.

Подробная математическая модель и калибровка параметров описаны в [DETECTION_AND_CLUSTERING.md](DETECTION_AND_CLUSTERING.md).

---

## 2. Сборка и запуск

### Сборка
```bash
colcon build --packages-select clustering o3d_detector --symlink-install
source install/setup.bash
```

### Запуск в составе общей системы
В `full_system.launch.py` и `role.launch.py` детекция запускается автоматически при аргументе `detector:=o3d` (по умолчанию).

### Автономный запуск узлов для отладки
```bash
# Терминал 1: кластеризация
ros2 run clustering cluster_node

# Терминал 2: детекция и трекинг
ros2 run o3d_detector detector
```
