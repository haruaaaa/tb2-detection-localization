#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class RobotClusterDetector(Node):
    def __init__(self):
        super().__init__('robot_cluster_detector')

        self.target_topic = '/livox/lidar/clust'

        self.min_dx, self.max_dx = 0.2, 0.354
        self.min_dy, self.max_dy = 0.2, 0.354
        self.min_dz, self.max_dz = 0.3, 0.42
        self.min_volume, self.max_volume = 0.01,  0.053
        self.min_points = 10
        
        self.subscription = self.create_subscription(
            PointCloud2,
            self.target_topic,
            self.listener_callback,
            10
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/detected_robots_markers', 10)
        self.get_logger().info("Детектор робота запущен!")

    def listener_callback(self, msg):

        points_gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        raw_data = list(points_gen)
            
        if not raw_data:
            return
            
        points_arr = np.array([[p[0], p[1], p[2], p[3]] for p in raw_data], dtype=np.float32)

        cluster_ids = np.unique(points_arr[:, 3])

        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        detected_count = 0

        for c_id in cluster_ids:
            cluster_mask = (points_arr[:, 3] == c_id)
            cluster_pts = points_arr[cluster_mask, :3]
            
            num_pts = len(cluster_pts)
            if num_pts < self.min_points:
                continue

            min_coords = np.min(cluster_pts, axis=0)
            max_coords = np.max(cluster_pts, axis=0)
            
            extent = max_coords - min_coords
            center = (min_coords + max_coords) / 2.0
            
            dx, dy, dz = float(extent[0]), float(extent[1]), float(extent[2])
            volume = dx * dy * dz
            dist_to_lidar = float(np.linalg.norm(center[:2]))

            if dist_to_lidar < 0.1:
                continue

            is_robot = (
                (self.min_dx <= dx <= self.max_dx) and
                (self.min_dy <= dy <= self.max_dy) and
                (self.min_dz <= dz <= self.max_dz) and
                (self.min_volume <= volume <= self.max_volume)
            )

            if is_robot:
                detected_count += 1
                self.get_logger().info(
                    f"[НАЙДЕН РОБОТ!] ID={int(c_id)} | Точек: {num_pts} | "
                    f"Размеры: {dx:.2f}x{dy:.2f}x{dz:.2f}м | "
                    f"Координаты: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}"
                )

                marker = Marker()
                marker.header = msg.header
                marker.ns = "detected_robots"
                marker.id = int(c_id)
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(center[0])
                marker.pose.position.y = float(center[1])
                marker.pose.position.z = float(center[2])
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = dx
                marker.scale.y = dy
                marker.scale.z = dz

                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                
                marker.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
                marker_array.markers.append(marker)

        if detected_count > 0:
            self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = RobotClusterDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()