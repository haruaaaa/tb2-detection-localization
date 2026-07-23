import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray

from . import core


class RobotDetector(Node):

    def __init__(self):
        super().__init__('robot_detector')

        self.declare_parameter('input_topic', '/livox/lidar/clust')
        self.declare_parameter('robot_min_size', core.ROBOT_MIN_SIZE)
        self.declare_parameter('robot_max_size', core.ROBOT_MAX_SIZE)
        self.declare_parameter('seg_len_min', core.SEG_LEN_MIN)
        self.declare_parameter('elong_max', core.ELONG_MAX)
        self.declare_parameter('gate', core.GATE)
        self.declare_parameter('min_hits', core.MIN_HITS)
        self.declare_parameter('max_miss', core.MAX_MISS)
        self.declare_parameter('span_min', core.SPAN_MIN)

        self.robot_min = self.get_parameter('robot_min_size').value
        self.robot_max = self.get_parameter('robot_max_size').value
        self.seg_len_min = self.get_parameter('seg_len_min').value
        self.elong_max = self.get_parameter('elong_max').value
        topic = self.get_parameter('input_topic').value

        gate = self.get_parameter('gate').value
        min_hits = self.get_parameter('min_hits').value
        max_miss = self.get_parameter('max_miss').value
        span_min = self.get_parameter('span_min').value
        self.tracker = core.Tracker(gate, min_hits, max_miss, span_min)

        self.sub = self.create_subscription(PointCloud2, topic, self.on_cloud, 10)
        self.pub_robot = self.create_publisher(PoseStamped, '/detection/robot', 10)
        self.pub_robots = self.create_publisher(PoseArray, '/detection/robots', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/detection/markers', 10)

        self.get_logger().info(f"listening {topic}, robot size {self.robot_min:.2f}..{self.robot_max:.2f}m")

    def on_cloud(self, msg):
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "intensity"), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        if pts.shape[0] == 0:
            return

        xy = pts[:, :2]
        ids = pts[:, 2]

        walls, candidates = core.classify_clusters(
            np.column_stack([xy, np.zeros(len(xy))]), ids,
            self.robot_min, self.robot_max, self.seg_len_min, self.elong_max
        )

        confirmed = self.tracker.update(candidates)
        self.publish(confirmed, walls, msg.header.frame_id, msg.header.stamp)

    def publish(self, confirmed, walls, frame, stamp):
        if confirmed:
            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.header.stamp = stamp
            ps.pose.position.x = float(confirmed[0]['pos'][0])
            ps.pose.position.y = float(confirmed[0]['pos'][1])
            ps.pose.orientation.w = 1.0
            self.pub_robot.publish(ps)

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

        if confirmed:
            self.get_logger().info(f"robots: {len(confirmed)}, walls: {len(walls)}")

    def publish_markers(self, confirmed, walls, frame, stamp):
        arr = MarkerArray()

        clr = Marker()
        clr.header.frame_id = frame
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
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
