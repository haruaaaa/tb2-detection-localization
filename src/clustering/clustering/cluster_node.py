import rclpy
from rclpy.node import Node
import numpy as np
from sklearn.cluster import DBSCAN

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class ClusterNode(Node):
    def __init__(self):
        super().__init__('point_cloud_cluster')

        self.declare_parameter('lidar_topic', '/livox/lidar')
        self.declare_parameter('clust_topic', '/livox/lidar/clust')
        self.declare_parameter('eps', 0.14)
        self.declare_parameter('min_samples', 20)
        self.declare_parameter('z_min', -0.35)
        self.declare_parameter('z_max', 0.325)
        self.declare_parameter('min_dist_xy', 0.35)
        self.declare_parameter('max_dist_xy', 6.0)

        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        clust_topic = self.get_parameter('clust_topic').get_parameter_value().string_value

        self.eps = self.get_parameter('eps').get_parameter_value().double_value
        self.min_samples = self.get_parameter('min_samples').get_parameter_value().integer_value
        self.z_min = self.get_parameter('z_min').get_parameter_value().double_value
        self.z_max = self.get_parameter('z_max').get_parameter_value().double_value
        self.min_dist_xy = self.get_parameter('min_dist_xy').get_parameter_value().double_value
        self.max_dist_xy = self.get_parameter('max_dist_xy').get_parameter_value().double_value

        self.subscription = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_callback,
            10
        )

        self.clust_pub = self.create_publisher(
            PointCloud2,
            clust_topic,
            10
        )

        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info(
            f"ClusterNode started: subscribing to {lidar_topic}, publishing to {clust_topic}"
        )

    def lidar_callback(self, msg: PointCloud2):
        # 1. Fast binary point extraction
        try:
            raw_bytes = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            points = raw_bytes[:, :12].copy().view(dtype=np.float32).reshape(-1, 3)
        except Exception:
            gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)

        if points.shape[0] == 0:
            return

        # 2. Filter out floor, high ceiling, and out-of-range points
        valid_z = (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        dists_xy = np.linalg.norm(points[:, :2], axis=1)
        valid_xy = (dists_xy >= self.min_dist_xy) & (dists_xy <= self.max_dist_xy)
        valid_mask = valid_z & valid_xy & np.isfinite(points).all(axis=1)

        points = points[valid_mask]
        if points.shape[0] < self.min_samples:
            return

        # 3. DBSCAN clustering (fast real-time sklearn implementation)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_

        cluster_mask = labels != -1
        clustered_points = points[cluster_mask]
        clustered_labels = labels[cluster_mask]

        if len(clustered_points) == 0:
            return

        # 4. Create PointCloud2 with intensity storing the cluster ID
        N = len(clustered_points)
        cloud_arr = np.empty(N, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32)
        ])
        cloud_arr['x'] = clustered_points[:, 0]
        cloud_arr['y'] = clustered_points[:, 1]
        cloud_arr['z'] = clustered_points[:, 2]
        cloud_arr['intensity'] = clustered_labels.astype(np.float32)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        clust_msg = point_cloud2.create_cloud(header, self.fields, cloud_arr)
        self.clust_pub.publish(clust_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterNode()
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
