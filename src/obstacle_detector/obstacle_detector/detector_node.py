import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs

from . import core

class RobotDetector(Node):

    def __init__(self):
        super().__init__('robot_detector')

        self.declare_parameter('target_frame', 'livox')
        self.declare_parameter('detected_robot_frame', 'turtlebot')
        self.declare_parameter('input_topic', '/livox/lidar/clust')
        self.declare_parameter('robot_min_size', core.ROBOT_MIN_SIZE)
        self.declare_parameter('robot_max_size', core.ROBOT_MAX_SIZE)
        self.declare_parameter('seg_len_min', core.SEG_LEN_MIN)
        self.declare_parameter('elong_max', core.ELONG_MAX)
        self.declare_parameter('gate', core.GATE)
        self.declare_parameter('min_hits', core.MIN_HITS)
        self.declare_parameter('max_miss', core.MAX_MISS)
        self.declare_parameter('span_min', core.SPAN_MIN)

        self.target_frame = self.get_parameter('target_frame').value
        self.detected_robot_frame = self.get_parameter('detected_robot_frame').value
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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.sub = self.create_subscription(PointCloud2, topic, self.on_cloud, 10)
        self.pub_robot = self.create_publisher(PoseStamped, '/detection/robot', 10)
        self.pub_robots = self.create_publisher(PoseArray, '/detection/robots', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/detection/markers', 10)

        self.get_logger().info(
            f"listening {topic}, target_frame: {self.target_frame}, "
            f"detected_frame: {self.detected_robot_frame}, "
            f"robot size {self.robot_min:.2f}..{self.robot_max:.2f}m"
        )

    def transform_point(self, pos_2d, src_frame, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                src_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            
            ps_msg = Pose()
            ps_msg.position.x = float(pos_2d[0])
            ps_msg.position.y = float(pos_2d[1])
            ps_msg.position.z = 0.0
            ps_msg.orientation.w = 1.0
            
            tf_pose = tf2_geometry_msgs.do_transform_pose(ps_msg, transform)
            return np.array([tf_pose.position.x, tf_pose.position.y])
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF transform error ({src_frame} -> {self.target_frame}): {e}",
                throttle_duration_sec=2.0
            )
            return None

    def on_cloud(self, msg):
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "intensity"), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        if pts.shape[0] == 0:
            return

        xy = pts[:, :2]
        ids = pts[:, 2]

        walls_local, candidates_local = core.classify_clusters(
            np.column_stack([xy, np.zeros(len(xy))]), ids,
            self.robot_min, self.robot_max, self.seg_len_min, self.elong_max
        )

        src_frame = msg.header.frame_id
        stamp = msg.header.stamp

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

            if confirmed:
                self.get_logger().info(f"robots: {len(confirmed)}, walls: {len(walls)}") 

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