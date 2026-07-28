# tb2-detection-localization

Detection and localization Turtlebot 2 based on Livox lidar point cloud.

## Принцип работы

- Лидар видит окружающую среду и объекты вокруг.
- Узел кластеризации разбивает облако точек на отдельные группы.
- Затем узел детектора оценивает каждую группу по форме и размеру.


### 1. Соберите контейнер

```bash
docker compose build hac
```

### 2. Разовое разрешение на использование GUI на устройстве 
Для демонcтрации решения в RVIZ

```
xhost +local:docker
```

### 3. Проверьте наличие rosbag

Rosbag'и должны лежать в папке:

```bash
/root/ws/src/bags
```

Если нужного файла нет, сначала положите его в эту папку.

## Запуск решения
1. Запуск docker-контейнера 
```bash
docker compose run --rm hac 
```
2. Построение пакета
```bash
colcon build
source install/setup.bash
```

### Базовый запуск
3. Запускаем launch, содержащий все необходимые ноды
```bash
ros2 launch obstacle_detector detect_from_bag.launch.py
```

Этот вариант использует дефолтный rosbag, который задан внутри launch-файла.

### Дополнительные аргументы

Launch-файл поддерживает аргументы:

- bag — путь к rosbag-папке;
- loop — повторять воспроизведение по кругу (`true`/`false`);
- rate — скорость воспроизведения;
- rviz — запускать ли RViz (`true`/`false`).

Пример:

```bash
ros2 launch obstacle_detector detect_from_bag.launch.py bag:=/root/ws/src/bags/rosbag2_2026_06_27-18_43-mov_01 loop:=false rate:=1.0 rviz:=true
```


## Что происходит после запуска

После запуска launch-файла:

- начинает воспроизводиться rosbag;
- запускаются узлы кластеризации и детекции;
- публикуются результаты в topic `/detection/robot`, `/detection/robots` и `/detection/markers`.
