import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class RobotDetector(Node):
    
    def __init__(self):
        super().__init__('robot_cluster_detector')

        self.min_dx, self.max_dx = 0.1, 0.354
        self.min_dy, self.max_dy = 0.1, 0.354
        self.min_dz, self.max_dz = 0.2, 0.42
        self.min_vol, self.max_vol = 0.01, 0.053
        self.min_pts = 100

        self.min_sph = 0.5
        self.max_sph = 0.85

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/livox/lidar/clust',
            self.listener_callback,
            10
        )

        self.marker_publisher = self.create_publisher(MarkerArray, '/detected_robot', 10)
        self.get_logger().info("Детектор робота запущен!")

    def listener_callback(self, msg):

        raw = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            
        if not raw:
            return
            
        points = np.array([[p[0], p[1], p[2], p[3]] for p in raw], dtype=np.float32)
        idx = np.unique(points[:, 3])

        markers = MarkerArray()
        
        clear_m = Marker()
        clear_m.action = Marker.DELETEALL
        markers.markers.append(clear_m)
        
        found = 0

        for c_idx in idx:

            c_mask = (points[:, 3] == c_idx)
            c_pts = points[c_mask, :3]
            
            pts_count = len(c_pts)

            if pts_count < self.min_pts:
                continue

            point_mn = np.min(c_pts, axis=0)
            point_mx = np.max(c_pts, axis=0)
            
            ext = point_mx - point_mn
            center = (point_mn + point_mx) / 2.0
            
            dx, dy, dz = float(ext[0]), float(ext[1]), float(ext[2])
            volume = dx * dy * dz
            dist = float(np.linalg.norm(center[:2]))

            if dist < 0.1:
                continue

            r_dists = np.linalg.norm(c_pts - center, axis=1)
            r = np.max(r_dists)
            
            if r > 1e-5:
                v_sph = (4.0 / 3.0) * np.pi * (r ** 3)
                sph = volume / v_sph
            else:
                sph = 0.0

            robot = (
                (self.min_dx <= dx <= self.max_dx) and
                (self.min_dy <= dy <= self.max_dy) and
                (self.min_dz <= dz <= self.max_dz) and
                (self.min_vol <= volume <= self.max_vol) and
                (self.min_sph <= sph <= self.max_sph)
            )

            if robot:
                found += 1
                self.get_logger().info(
                    f"[НАЙДЕН РОБОТ!] ID={int(c_idx)} | Точек: {pts_count} | "
                    f"Размеры: {dx:.2f}x{dy:.2f}x{dz:.2f}м | Сферичность: {sph:.2f} | "
                    f"Координаты: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}"
                )

                marker = Marker()
                marker.header = msg.header
                marker.ns = "detected_robots"
                marker.id = int(c_idx)
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
                markers.markers.append(marker)

        if found > 0:
            self.marker_publisher.publish(markers)

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