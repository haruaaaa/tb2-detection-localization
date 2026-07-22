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

        self.subscription = self.create_subscription(
            PointCloud2, '/livox/lidar', self.lidar_callback, 10)

        self.clust_pub = self.create_publisher(
            PointCloud2, '/livox/lidar/clust', 10)

        self.eps = 0.1
        self.min_samples = 15
        self.z_min = -0.10
        self.z_max = 0.20

    def lidar_callback(self, msg):
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)

        if points.shape[0] == 0:
            return

        mask = (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        points = points[mask]

        if points.shape[0] == 0:
            return

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(points)

        valid = labels != -1
        clustered_points = points[valid]
        clustered_labels = labels[valid]

        if len(clustered_points) == 0:
            return

        output_data = []
        for i in range(len(clustered_points)):
            x, y, z = clustered_points[i]
            output_data.append((x, y, z, float(clustered_labels[i])))

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        clust_msg = point_cloud2.create_cloud(header, fields, output_data)
        self.clust_pub.publish(clust_msg)

        num_clusters = len(set(clustered_labels))
        self.get_logger().info(f"clusters: {num_clusters}, points: {len(clustered_points)}")


def main(args=None):
    rclpy.init(args=args)
    node = ClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
