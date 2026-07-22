# tb2-detection-localization
Detection and localization Turtlebot 2 based on lidar point cloud

```
pip install scikit-learn
```

Запуск детектора с нужной URDF

```
ros2 run o3d_detector detector --ros-args -p xacro_path:=src/turtlebot_description/robots/kobuki_hexagons_kinect.urdf.xacro
```