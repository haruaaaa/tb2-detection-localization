import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray


class RobotDetector(Node):
    """
    Распознаёт робота (TB2) среди готовых кластеров.

    На вход берёт кластеризованное облако от пакета clustering
    (/livox/lidar/clust, id кластера лежит в поле intensity).
    Логика obstacle_detector в сути: каждый кластер по форме относим либо
    к СТЕНЕ (длинный/вытянутый сегмент), либо к РОБОТУ (компактный кружок
    робот-размерного габарита). Кружки ведём во времени (NN + гейт) и
    подтверждаем только те, что живут несколько кадров подряд -- это гасит
    мерцание Livox. Работает независимо от того, движется робот или стоит.
    """

    def __init__(self):
        super().__init__('robot_detector')

        # --- параметры (можно переопределить из launch/CLI) ---
        self.declare_parameter('input_topic', '/livox/lidar/clust')
        self.declare_parameter('robot_min_size', 0.12)   # мин. габарит робота, м
        self.declare_parameter('robot_max_size', 0.55)   # макс. габарит робота, м
        self.declare_parameter('seg_len_min', 0.60)      # длиннее -> это стена
        self.declare_parameter('elong_max', 2.5)         # major/minor выше -> вытянутый (стена)
        self.declare_parameter('gate', 0.40)             # радиус связи трека между кадрами, м
        self.declare_parameter('min_hits', 5)            # столько кадров подряд -> робот подтверждён
        self.declare_parameter('max_miss', 6)            # столько кадров без обновления -> трек закрыт

        gp = self.get_parameter
        self.robot_min = gp('robot_min_size').value
        self.robot_max = gp('robot_max_size').value
        self.seg_len_min = gp('seg_len_min').value
        self.elong_max = gp('elong_max').value
        self.gate = gp('gate').value
        self.min_hits = gp('min_hits').value
        self.max_miss = gp('max_miss').value
        topic = gp('input_topic').value

        self.tracks = []       # [{pos, hits, miss, size, tid}]
        self._next_tid = 0

        self.sub = self.create_subscription(PointCloud2, topic, self.on_cloud, 10)
        self.pub_robot = self.create_publisher(PoseStamped, '/detection/robot', 10)
        self.pub_robots = self.create_publisher(PoseArray, '/detection/robots', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/detection/markers', 10)

        self.get_logger().info(
            f"robot_detector слушает {topic}; робот = кружок "
            f"{self.robot_min:.2f}..{self.robot_max:.2f} м, подтверждение за {self.min_hits} кадров")

    # ------------------------------------------------------------------
    def on_cloud(self, msg):
        gen = point_cloud2.read_points(
            msg, field_names=("x", "y", "intensity"), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        if pts.shape[0] == 0:
            return

        walls, candidates = self._classify_clusters(pts)
        confirmed = self._update_tracks(candidates)

        frame = msg.header.frame_id
        stamp = msg.header.stamp
        self._publish(confirmed, walls, frame, stamp)

    # ------------------------------------------------------------------
    def _classify_clusters(self, pts):
        """Каждый кластер -> ('wall', центр, размер) либо кандидат-робот."""
        walls, candidates = [], []
        ids = pts[:, 2]
        for cid in np.unique(ids):
            xy = pts[ids == cid, :2]
            if xy.shape[0] < 3:
                continue
            centroid = xy.mean(axis=0)
            l_major, l_minor = self._principal_extent(xy - centroid)
            elong = l_major / max(l_minor, 1e-3)

            if l_major > self.seg_len_min:
                walls.append((centroid, l_major))                    # стена
            elif self.robot_min <= l_major <= self.robot_max and elong <= self.elong_max:
                candidates.append((centroid, l_major))               # робот-кандидат
            # остальное (мелкий шум / странная форма) -- игнор

        return walls, candidates

    @staticmethod
    def _principal_extent(centered):
        """Габариты кластера вдоль его главных осей (PCA в XY), м."""
        cov = np.cov(centered.T)
        if cov.shape != (2, 2):
            return 0.0, 0.0
        _, vecs = np.linalg.eigh(cov)          # столбцы -- собственные векторы
        proj = centered @ vecs                 # проекция на главные оси
        spans = proj.max(axis=0) - proj.min(axis=0)
        l_major, l_minor = float(spans.max()), float(spans.min())
        return l_major, l_minor

    # ------------------------------------------------------------------
    def _update_tracks(self, candidates):
        """NN-ассоциация кандидатов с треками + подтверждение по устойчивости."""
        used = set()
        for tr in self.tracks:
            best_i, best_d = -1, self.gate
            for i, (c, _sz) in enumerate(candidates):
                if i in used:
                    continue
                d = float(np.hypot(*(c - tr['pos'])))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                c, sz = candidates[best_i]
                used.add(best_i)
                tr['pos'], tr['size'] = c, sz
                tr['hits'] += 1
                tr['miss'] = 0
            else:
                tr['miss'] += 1

        # новые треки из несвязанных кандидатов
        for i, (c, sz) in enumerate(candidates):
            if i not in used:
                self.tracks.append({'pos': c, 'size': sz, 'hits': 1,
                                    'miss': 0, 'tid': self._next_tid})
                self._next_tid += 1

        # закрыть протухшие
        self.tracks = [t for t in self.tracks if t['miss'] <= self.max_miss]

        # подтверждённые = прожили достаточно кадров
        confirmed = [t for t in self.tracks if t['hits'] >= self.min_hits]
        confirmed.sort(key=lambda t: -t['hits'])
        return confirmed

    # ------------------------------------------------------------------
    def _publish(self, confirmed, walls, frame, stamp):
        # первичный робот (самый устойчивый трек)
        if confirmed:
            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.header.stamp = stamp
            ps.pose.position.x = float(confirmed[0]['pos'][0])
            ps.pose.position.y = float(confirmed[0]['pos'][1])
            ps.pose.orientation.w = 1.0
            self.pub_robot.publish(ps)

        # все подтверждённые роботы
        pa = PoseArray()
        pa.header.frame_id = frame
        pa.header.stamp = stamp
        for t in confirmed:
            p = Pose()
            p.position.x = float(t['pos'][0])
            p.position.y = float(t['pos'][1])
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.pub_robots.publish(pa)

        self._publish_markers(confirmed, walls, frame, stamp)

        if confirmed:
            names = ", ".join(f"#{t['tid']}({t['size']:.2f}м)" for t in confirmed)
            self.get_logger().info(f"роботов: {len(confirmed)} [{names}]  стен: {len(walls)}")

    def _publish_markers(self, confirmed, walls, frame, stamp):
        arr = MarkerArray()
        clr = Marker()
        clr.header.frame_id = frame
        clr.action = Marker.DELETEALL
        arr.markers.append(clr)

        mid = 0
        for cx, size in ((w[0], w[1]) for w in walls):     # стены -- серые кубы
            m = self._mk(frame, stamp, mid, cx, size, (0.6, 0.6, 0.6, 0.5),
                         Marker.CUBE)
            arr.markers.append(m)
            mid += 1
        for t in confirmed:                                # роботы -- зелёные сферы
            m = self._mk(frame, stamp, mid, t['pos'], max(t['size'], 0.25),
                         (0.1, 0.9, 0.2, 0.9), Marker.SPHERE)
            arr.markers.append(m)
            mid += 1
            arr.markers.append(self._label(frame, stamp, mid, t))
            mid += 1

        self.pub_markers.publish(arr)

    @staticmethod
    def _mk(frame, stamp, mid, pos, size, rgba, shape):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns = 'detection'
        m.id = mid
        m.type = shape
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = float(size)
        m.scale.z = 0.3
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    @staticmethod
    def _label(frame, stamp, mid, track):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns = 'labels'
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(track['pos'][0])
        m.pose.position.y = float(track['pos'][1])
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.z = 0.2
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 1.0, 1.0, 1.0)
        m.text = f"robot #{track['tid']}"
        return m


def main(args=None):
    rclpy.init(args=args)
    node = RobotDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
