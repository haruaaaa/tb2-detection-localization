import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs

from . import core

class RobotDetector(Node):
    def __init__(self):
        super().__init__('robot_detector')

        # Parameters
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('detected_robot_frame', 'opponent_robot')
        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cloud_topic', '/livox/lidar/clust')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('target_topic', '/detection/opponent_robot')
        self.declare_parameter('use_scan', True)

        self.declare_parameter('robot_min_size', core.ROBOT_MIN_SIZE)
        self.declare_parameter('robot_max_size', core.ROBOT_MAX_SIZE)
        self.declare_parameter('seg_len_min', core.SEG_LEN_MIN)
        self.declare_parameter('elong_max', core.ELONG_MAX)
        self.declare_parameter('gate', core.GATE)
        self.declare_parameter('min_hits', core.MIN_HITS)
        self.declare_parameter('max_miss', core.MAX_MISS)
        self.declare_parameter('span_min', core.SPAN_MIN)

        self.declare_parameter('use_circle_fit', True)
        self.declare_parameter('circle_r_min', 0.14)
        self.declare_parameter('circle_r_max', 0.23)
        self.declare_parameter('max_circle_rms', 0.035)

        self.target_frame = self.get_parameter('target_frame').value
        self.detected_robot_frame = self.get_parameter('detected_robot_frame').value
        self.input_topic = self.get_parameter('input_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.cloud_topic = self.get_parameter('cloud_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        self.target_topic = self.get_parameter('target_topic').value
        self.use_scan = self.get_parameter('use_scan').value

        self.robot_min = self.get_parameter('robot_min_size').value
        self.robot_max = self.get_parameter('robot_max_size').value
        self.seg_len_min = self.get_parameter('seg_len_min').value
        self.elong_max = self.get_parameter('elong_max').value
        self.use_circle_fit = self.get_parameter('use_circle_fit').value
        self.circle_r_min = self.get_parameter('circle_r_min').value
        self.circle_r_max = self.get_parameter('circle_r_max').value
        self.max_circle_rms = self.get_parameter('max_circle_rms').value

        gate = self.get_parameter('gate').value
        min_hits = self.get_parameter('min_hits').value
        max_miss = self.get_parameter('max_miss').value
        span_min = self.get_parameter('span_min').value
        self.tracker = core.Tracker(gate, min_hits, max_miss, span_min)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # SLAM Map storage
        self.map_data = None
        self.map_info = None
        if self.map_topic:
            self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, 5)

        # Input subscriptions: either LaserScan or PointCloud2
        if self.use_scan:
            self.create_subscription(LaserScan, self.scan_topic, self.on_scan, 10)
        else:
            self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, 10)

        # Publishers
        self.pub_opponent = self.create_publisher(PoseStamped, self.target_topic, 10)
        self.pub_robot = self.create_publisher(PoseStamped, '/detection/robot', 10)
        self.pub_robots = self.create_publisher(PoseArray, '/detection/robots', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/detection/markers', 10)

        self.get_logger().info(
            f"Universal Robot Detector initialized! Target topic: {self.target_topic}, "
            f"frame: {self.target_frame}, use_scan: {self.use_scan}"
        )

    def on_map(self, msg: OccupancyGrid):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

    def filter_known_map_points(self, pts_map):
        if self.map_data is None or self.map_info is None or len(pts_map) == 0:
            return pts_map

        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h, w = self.map_data.shape

        gx = ((pts_map[:, 0] - ox) / res).astype(int)
        gy = ((pts_map[:, 1] - oy) / res).astype(int)

        valid_bounds = (gx >= 1) & (gx < w - 1) & (gy >= 1) & (gy < h - 1)
        free_mask = np.ones(len(pts_map), dtype=bool)

        for i in np.where(valid_bounds)[0]:
            cx, cy = gx[i], gy[i]
            # If current cell or 3x3 neighborhood has static wall occupancy (> 50), filter it out
            neighborhood = self.map_data[cy - 1:cy + 2, cx - 1:cx + 2]
            if np.any(neighborhood > 50):
                free_mask[i] = False

        return pts_map[free_mask]

    def on_scan(self, msg: LaserScan):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        valid = (ranges >= msg.range_min) & (ranges <= msg.range_max) & np.isfinite(ranges)
        if not np.any(valid):
            return

        r = ranges[valid]
        a = angles[valid]
        xs = r * np.cos(a)
        ys = r * np.sin(a)
        pts_local = np.column_stack((xs, ys, np.zeros(len(xs))))

        self.process_points(pts_local, msg.header.frame_id, msg.header.stamp)

    def on_cloud(self, msg: PointCloud2):
        has_intensity = any(f.name == "intensity" for f in msg.fields)
        if has_intensity:
            gen = point_cloud2.read_points(msg, field_names=("x", "y", "intensity"), skip_nans=True)
            pts = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
            if pts.shape[0] == 0:
                return
            # Pre-clustered cloud from cluster_node
            xy = pts[:, :2]
            ids = pts[:, 2]
            self.process_clustered_xy(xy, ids, msg.header.frame_id, msg.header.stamp)
            return

        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        if pts.shape[0] == 0:
            return
        z_mask = (pts[:, 2] >= -0.10) & (pts[:, 2] <= 0.30)
        pts_filtered = pts[z_mask]
        self.process_points(pts_filtered, msg.header.frame_id, msg.header.stamp)

    def process_points(self, pts_local, src_frame, stamp):
        if len(pts_local) < 5:
            return

        pts_xy = pts_local[:, :2]

        # 1. Transform points to global map frame if map available to subtract known walls
        pts_to_cluster = pts_xy
        is_global = False

        if self.map_data is not None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.target_frame, src_frame, stamp,
                    timeout=rclpy.duration.Duration(seconds=0.0)
                )
                tx = trans.transform.translation.x
                ty = trans.transform.translation.y
                q = trans.transform.rotation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)

                rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
                pts_map = (rot @ pts_xy.T).T + np.array([tx, ty])

                # Subtract known map walls
                pts_free = self.filter_known_map_points(pts_map)
                if len(pts_free) >= 5:
                    pts_to_cluster = pts_free
                    is_global = True
            except Exception:
                pass

        # 2. Cluster points with DBSCAN
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.18, min_samples=6).fit(pts_to_cluster)
        labels = db.labels_
        valid = labels != -1
        if not np.any(valid):
            return

        pts_valid = pts_to_cluster[valid]
        labels_valid = labels[valid]

        # 3. Classify clusters
        walls, candidates = core.classify_clusters(
            np.column_stack([pts_valid, np.zeros(len(pts_valid))]), labels_valid,
            self.robot_min, self.robot_max, self.seg_len_min, self.elong_max,
            self.circle_r_min, self.circle_r_max, self.max_circle_rms, self.use_circle_fit
        )

        # 4. Convert candidates and walls to global frame if not already
        if not is_global:
            candidates_global = []
            for c in candidates:
                pos_g = self.transform_point(c['pos'], src_frame, stamp)
                if pos_g is not None:
                    candidates_global.append({'pos': pos_g, 'size': c['size']})

            walls_global = []
            for w in walls:
                pos_g = self.transform_point(w['pos'], src_frame, stamp)
                if pos_g is not None:
                    walls_global.append({'pos': pos_g, 'size': w['size']})
        else:
            candidates_global = candidates
            walls_global = walls

        # 5. Tracker update
        confirmed = self.tracker.update(candidates_global)

        # 6. Publish outputs
        self.publish(confirmed, walls_global, self.target_frame, stamp)

    def process_clustered_xy(self, xy, ids, src_frame, stamp):
        walls_local, candidates_local = core.classify_clusters(
            np.column_stack([xy, np.zeros(len(xy))]), ids,
            self.robot_min, self.robot_max, self.seg_len_min, self.elong_max,
            self.circle_r_min, self.circle_r_max, self.max_circle_rms, self.use_circle_fit
        )

        candidates_global = []
        for c in candidates_local:
            pos_gtf = self.transform_point(c['pos'], src_frame, stamp)
            if pos_gtf is not None:
                candidates_global.append({'pos': pos_gtf, 'size': c['size']})

        walls_global = []
        for w in walls_local:
            pos_gtf = self.transform_point(w['pos'], src_frame, stamp)
            if pos_gtf is not None:
                walls_global.append({'pos': pos_gtf, 'size': w['size']})

        confirmed = self.tracker.update(candidates_global)
        self.publish(confirmed, walls_global, self.target_frame, stamp)

    def transform_point(self, pos_2d, src_frame, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                src_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.0)
            )
            ps_msg = Pose()
            ps_msg.position.x = float(pos_2d[0])
            ps_msg.position.y = float(pos_2d[1])
            ps_msg.position.z = 0.0
            ps_msg.orientation.w = 1.0

            tf_pose = tf2_geometry_msgs.do_transform_pose(ps_msg, transform)
            return np.array([tf_pose.position.x, tf_pose.position.y])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def publish_robot_tf(self, confirmed_robot, frame, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame
        t.child_frame_id = self.detected_robot_frame
        t.transform.translation.x = float(confirmed_robot['pos'][0])
        t.transform.translation.y = float(confirmed_robot['pos'][1])
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def publish(self, confirmed, walls, frame, stamp):
        if confirmed:
            best_robot = confirmed[0]

            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.header.stamp = stamp
            ps.pose.position.x = float(best_robot['pos'][0])
            ps.pose.position.y = float(best_robot['pos'][1])
            ps.pose.orientation.w = 1.0

            # Publish to target topic (opponent_robot) and backward-compatible robot topic
            self.pub_opponent.publish(ps)
            self.pub_robot.publish(ps)
            self.publish_robot_tf(best_robot, frame, stamp)

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

        self.publish_markers(confirmed, walls, frame, stamp)

    def publish_markers(self, confirmed, walls, frame, stamp):
        arr = MarkerArray()

        clr = Marker()
        clr.header.frame_id = frame
        clr.header.stamp = stamp
        clr.action = Marker.DELETEALL
        arr.markers.append(clr)

        mid = 0
        for w in walls:
            m = self.make_marker(frame, stamp, mid, w['pos'], w['size'],
                                 (0.6, 0.6, 0.6, 0.5), Marker.CUBE)
            arr.markers.append(m)
            mid += 1

        for i, t in enumerate(confirmed):
            is_best = (i == 0)
            if not is_best:
                continue

            color = (0.1, 1.0, 0.2, 1.0) if is_best else (0.5, 0.7, 0.5, 0.6)
            m = self.make_marker(frame, stamp, mid, t['pos'], max(t['size'], 0.25),
                                 color, Marker.SPHERE)
            arr.markers.append(m)
            mid += 1

            lbl = Marker()
            lbl.header.frame_id = frame
            lbl.header.stamp = stamp
            lbl.ns = 'labels'
            lbl.id = mid
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = float(t['pos'][0])
            lbl.pose.position.y = float(t['pos'][1])
            lbl.pose.position.z = 0.5
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.2
            lbl.color.r, lbl.color.g, lbl.color.b, lbl.color.a = 1.0, 1.0, 1.0, 1.0
            span = t.get('span', 0)
            lbl.text = f"#{t['id']} span={span:.2f}m sz={t['size']:.2f}m"
            arr.markers.append(lbl)
            mid += 1

        self.pub_markers.publish(arr)

    def make_marker(self, frame, stamp, mid, pos, size, rgba, shape):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns = 'detection'
        m.id = mid
        m.type = shape
        m.action = Marker.ADD

        if hasattr(pos, 'pose'):
            m.pose.position.x = float(pos.pose.position.x)
            m.pose.position.y = float(pos.pose.position.y)
        elif hasattr(pos, 'position'):
            m.pose.position.x = float(pos.position.x)
            m.pose.position.y = float(pos.position.y)
        else:
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])

        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = float(size)
        m.scale.z = 0.3
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()