# tb2-detection-localization
Detection and localization TurtleBot 2 based on Livox MID-360 LiDAR point cloud.

## Документация по алгоритмам и настройке
Подробное описание архитектуры, математических моделей (DBSCAN, Kasa Circle Fit, PCA-фильтры) и инструкции по тюнингу параметров находятся в файле:
- [DETECTION_AND_CLUSTERING.md](DETECTION_AND_CLUSTERING.md)

## Быстрый запуск

1. Сборка пакетов:
```bash
colcon build --packages-select clustering o3d_detector --symlink-install
source install/setup.bash
```

2. Запуск кластеризации:
```bash
ros2 run clustering cluster_node
```

3. Запуск детектора робота:
```bash
ros2 run o3d_detector detector
```
