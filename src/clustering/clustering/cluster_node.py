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

        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('output_topic', '/livox/lidar/clust')
        self.declare_parameter('z_min', -0.15)
        self.declare_parameter('z_max', 0.325)
        self.declare_parameter('min_dist_xy', 0.35)
        self.declare_parameter('max_dist_xy', 4.5)
        self.declare_parameter('eps', 0.18)
        self.declare_parameter('min_samples', 25)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.min_dist_xy = float(self.get_parameter('min_dist_xy').value)
        self.max_dist_xy = float(self.get_parameter('max_dist_xy').value)
        self.eps = float(self.get_parameter('eps').value)
        self.min_samples = int(self.get_parameter('min_samples').value)
        
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.lidar_callback,
            10
        )
        
        self.clust_pub = self.create_publisher(
            PointCloud2,
            output_topic,
            10
        )

    def lidar_callback(self, msg):
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        
        if points.shape[0] == 0:
            return

        z_mask = (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)

        dist_xy = np.linalg.norm(points[:, :2], axis=1)
        dist_mask = (dist_xy >= self.min_dist_xy) & (dist_xy <= self.max_dist_xy) 
        points = points[z_mask & dist_mask]


        if points.shape[0] == 0:
            return
       
        MAX_DIST = 3.0 
        MAX_EPS_INCREASE = 0.22 

        r = np.linalg.norm(points[:, :2], axis=1)
        
        scale = 1.0 / (1.0 + MAX_EPS_INCREASE * np.clip(r / MAX_DIST, 0.0, 1.0))

        fake_points = points.copy()
        fake_points[:, 0] *= scale
        fake_points[:, 1] *= scale
        fake_points[:, 2] *= scale

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(fake_points)
        labels = clustering.labels_
        
        mask = labels != -1
        
        clustered_points = points[mask] 
        clustered_labels = labels[mask]
        
        if len(clustered_points) == 0:
            return

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