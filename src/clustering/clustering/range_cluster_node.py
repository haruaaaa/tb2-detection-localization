import rclpy
from rclpy.node import Node
import numpy as np
import cv2

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py import point_cloud2

class RangeImageSegmenter:
    """
    Класс для преобразования 3D облака точек в 2D Range Image
    и быстрой сегментации объектов.
    """
    def __init__(self, 
                 width=1024, 
                 height=64, 
                 fov_up=52.0, 
                 fov_down=-7.0, 
                 edge_thresh=0.4, 
                 min_samples=20):
        
        self.W = width
        self.H = height
        
        # Переводим FOV Livox Mid-360 из градусов в радианы
        self.fov_up = np.radians(fov_up)
        self.fov_down = np.radians(fov_down)
        self.fov_range = self.fov_up - self.fov_down
        
        # Порог (в метрах) скачка глубины, который считается границей объекта
        self.edge_thresh = edge_thresh
        self.min_samples = min_samples

    def process(self, points):
        """
        Принимает NX3 numpy массив точек (x, y, z).
        Возвращает:
        1. labels - 1D массив меток кластеров для каждой точки (-1 если шум/фон).
        2. depth_image - 2D numpy массив для визуализации.
        """
        N = points.shape[0]
        if N == 0:
            return np.array([]), np.zeros((self.H, self.W), dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        r_xy = np.hypot(x, y)
        r = np.hypot(r_xy, z)

        # 1. Сферическая проекция
        yaw = np.arctan2(y, x)  # Горизонтальный угол [-pi, pi]
        pitch = np.arctan2(z, r_xy)  # Вертикальный угол

        # Маппинг в пиксели (u, v)
        u = np.clip(((yaw + np.pi) / (2 * np.pi) * self.W).astype(int), 0, self.W - 1)
        v = np.clip(self.H - 1 - ((pitch - self.fov_down) / self.fov_range * self.H).astype(int), 0, self.H - 1)

        # 2. Создаем пустые матрицы для глубины и индексов
        depth_img = np.zeros((self.H, self.W), dtype=np.float32)
        idx_img = -np.ones((self.H, self.W), dtype=int)

        # Z-buffer: сортируем точки так, чтобы сначала шли дальние, а затем ближние.
        # Таким образом ближние объекты перекроют дальние в одних и тех же пикселях.
        order = np.argsort(r)[::-1]
        u_sorted, v_sorted, r_sorted, idx_sorted = u[order], v[order], r[order], order

        depth_img[v_sorted, u_sorted] = r_sorted
        idx_img[v_sorted, u_sorted] = idx_sorted

        # 3. Морфология (закрытие дыр от паттерна Livox)
        # Livox оставляет пустоты между лучами, немного расширяем пиксели
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        depth_filled = cv2.dilate(depth_img, kernel)

        # 4. Поиск границ (Edge Detection)
        # Находим резкие скачки глубины
        depth_erosion = cv2.erode(depth_filled, kernel)
        depth_dilation = cv2.dilate(depth_filled, kernel)
        
        # Если разница между локальным максимумом и минимумом глубины больше порога - это граница
        edges = (depth_dilation - depth_erosion) > self.edge_thresh
        
        # 5. Сегментация
        # Выделяем маску "тела" объектов (есть глубина И нет границы)
        valid_mask = (depth_filled > 0) & (~edges)
        valid_mask_uint8 = valid_mask.astype(np.uint8) * 255

        # Быстрый аналог DBSCAN на картинке
        num_labels, labels_2d = cv2.connectedComponents(valid_mask_uint8, connectivity=4)

        # 6. Обратный маппинг меток на 3D точки
        point_labels = np.full(N, -1, dtype=int)
        
        # Берем только те пиксели, куда попала реальная 3D точка
        valid_indices = idx_img != -1
        flat_indices = idx_img[valid_indices]
        flat_labels = labels_2d[valid_indices]
        
        point_labels[flat_indices] = flat_labels

        # 7. Фильтрация мелких кластеров (эквивалент min_samples)
        # 0 - это фон в connectedComponents, его игнорируем
        unique_labels, counts = np.unique(point_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label == 0 or label == -1:
                point_labels[point_labels == label] = -1
            elif count < self.min_samples:
                point_labels[point_labels == label] = -1

        return point_labels, depth_filled


class RangeClusterNode(Node):
    def __init__(self):
        super().__init__('point_cloud_range_cluster')
        
        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback,
            10
        )
        
        # Паблишер 3D кластеров (старый формат, для совместимости с детектором)
        self.clust_pub = self.create_publisher(PointCloud2, '/livox/lidar/clust', 10)
        
        # Паблишер 2D картинки глубины (для отладки в rqt_image_view)
        self.image_pub = self.create_publisher(Image, '/livox/lidar/range_image', 10)

        # Инициализация сегментатора
        self.segmenter = RangeImageSegmenter(
            width=1024,         # Горизонтальное разрешение 360 град
            height=64,          # Вертикальное разрешение (~59 град Livox)
            fov_up=52.0,        # Верхний угол Livox
            fov_down=-7.0,      # Нижний угол Livox
            edge_thresh=0.3,    # Разница в глубине 30 см = разрыв объектов (аналог eps=0.15)
            min_samples=30   # Минимальное кол-во точек для робота
        )

        # Фильтры самого лидара (земля и сам робот)
        self.z_min = 0.325
        self.min_dist_xy = 0.35

    def lidar_callback(self, msg):
        # 1. Читаем точки
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        
        if points.shape[0] == 0:
            return

        # 2. Базовая фильтрация (земля и корпус)
        z_mask = points[:, 2] <= self.z_min
        dist_xy = np.linalg.norm(points[:, :2], axis=1)
        dist_mask = dist_xy >= self.min_dist_xy
        points = points[z_mask & dist_mask]

        if points.shape[0] == 0:
            return
        
        # 3. ОСНОВНАЯ МАГИЯ: Получаем лейблы и картинку за доли секунды
        labels, depth_img = self.segmenter.process(points)

        # Оставляем только те точки, которые попали в любой кластер
        mask = labels != -1
        clustered_points = points[mask]
        clustered_labels = labels[mask]
        
        if len(clustered_points) > 0:
            # 4. Публикуем кластеризованное облако точек
            output_data = np.hstack((clustered_points, clustered_labels.reshape(-1, 1)))

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            header = Header()
            header.stamp = msg.header.stamp
            header.frame_id = msg.header.frame_id 

            clust_msg = point_cloud2.create_cloud(header, fields, output_data.tolist())
            self.clust_pub.publish(clust_msg)
        
        # 5. Публикуем Range Image
        self.publish_range_image(depth_img, msg.header)

    def publish_range_image(self, depth_img, header):
        # Нормализуем глубину для визуализации (от 0 до 10 метров -> 0 до 255 пикселей)
        MAX_DIST = 5.0
        img_norm = np.clip(depth_img / MAX_DIST * 255, 0, 255).astype(np.uint8)
        
        # Применяем цветовую карту, чтобы выглядело красиво (близко - красно, далеко - сине)
        # Если cv2 не найдет colormap, можно просто оставить mono8
        color_img = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)
        
        # СОХРАНИТЬ В ФАЙЛ:
        cv2.imwrite('/root/ws/src/clustering/range_image_debug.jpg', color_img)
        
        img_msg = Image()
        img_msg.header = header
        img_msg.height = color_img.shape[0]
        img_msg.width = color_img.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.is_bigendian = False
        img_msg.step = color_img.shape[1] * 3
        img_msg.data = color_img.tobytes()
        
        self.image_pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RangeClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()