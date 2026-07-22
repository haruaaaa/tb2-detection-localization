import os
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import trimesh
import xacro
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import ament_index_python


def build_model_pcd_from_xacro(xacro_path: str, logger, num_samples: int = 3000) -> o3d.geometry.PointCloud:
    if not os.path.exists(xacro_path):
        raise FileNotFoundError(f"Xacro файл не найден: {xacro_path}")

    doc = xacro.process_file(xacro_path)
    urdf_string = doc.toxml()

    root = ET.fromstring(urdf_string)
    loaded_meshes = []

    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if not filename:
            continue

        resolved_path = None
        if filename.startswith("package://"):
            parts = filename.replace("package://", "").split("/", 1)
            pkg_name, rel_path = parts[0], parts[1]
            try:
                pkg_share = ament_index_python.get_package_share_directory(pkg_name)
                resolved_path = os.path.join(pkg_share, rel_path)
            except Exception:
                continue
        else:
            resolved_path = filename

        if resolved_path and os.path.exists(resolved_path):
            try:
                m = trimesh.load(resolved_path)
                if isinstance(m, trimesh.Scene):
                    m = m.dump(concatenate=True)
                loaded_meshes.append(m)
            except Exception:
                pass

    if not loaded_meshes:
        raise RuntimeError("Не удалось загрузить ни один меш из xacro файла.")

    combined_mesh = trimesh.util.concatenate(loaded_meshes)
    pts, _ = trimesh.sample.sample_surface(combined_mesh, num_samples)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    min_b = pcd.get_min_bound()
    max_b = pcd.get_max_bound()
    extents = max_b - min_b
    logger.info(f"Исходные габариты модели (X, Y, Z): {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f}")

    if np.max(extents) > 5.0:
        logger.info("Обнаружен масштаб в миллиметрах. Выполняется масштабирование в метры (x0.001)...")
        pcd.scale(0.001, center=pcd.get_center())
        extents = pcd.get_max_bound() - pcd.get_min_bound()
        logger.info(f"Новые габариты модели: {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} м")

    pcd.translate(-pcd.get_center())
    return pcd


class RobotDetectorICP(Node):

    def __init__(self):
        super().__init__('robot_icp_detector')

        self.declare_parameter('xacro_path', '/path/to/your/robot.xacro')
        xacro_path = self.get_parameter('xacro_path').get_parameter_value().string_value

        # Пороги грубой первичной фильтрации
        self.min_dx, self.max_dx = 0.10, 0.354
        self.min_dy, self.max_dy = 0.10, 0.354
        self.min_dz, self.max_dz = 0.15, 0.42
        self.min_vol, self.max_vol = 0.01, 0.053
        self.min_sph, self.max_sph = 0.60, 0.85
        self.min_pts = 50

        # Параметры ICP
        self.max_icp_dist = 0.05
        self.fitness_thresh = 0.86
        self.rmse_thresh = 0.045

        self.get_logger().info(f"Загрузка модели из Xacro: {xacro_path}")
        try:
            self.pcd_model = build_model_pcd_from_xacro(xacro_path, self.get_logger(), num_samples=3000)
            self.get_logger().info("Модель успешно загружена!")
        except Exception as e:
            self.get_logger().error(f"Ошибка загрузки модели: {e}")
            self.pcd_model = None

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/livox/lidar/clust',
            self.listener_callback,
            10
        )

        self.marker_publisher = self.create_publisher(MarkerArray, '/detected_robot_pose', 10)
        self.get_logger().info("ICP Детектор робота готов!")

    def listener_callback(self, msg):
        if self.pcd_model is None:
            return

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

            if len(c_pts) < self.min_pts:
                continue

            # 1. Первичная геометрическая фильтрация
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
            sph = (volume / ((4.0 / 3.0) * np.pi * (r ** 3))) if r > 1e-5 else 0.0

            candidate = (
                (self.min_dx <= dx <= self.max_dx) and
                (self.min_dy <= dy <= self.max_dy) and
                (self.min_dz <= dz <= self.max_dz) and
                (self.min_vol <= volume <= self.max_vol) and
                (self.min_sph <= sph <= self.max_sph)
            )

            if not candidate:
                self.get_logger().debug(
                    f"ID={int(c_idx)} не прошел фильтр: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, "
                    f"vol={volume:.3f}, sph={sph:.2f}"
                )
                continue

            fitness, rmse, transform = self.match_icp(c_pts, center)

            if fitness >= self.fitness_thresh and rmse <= self.rmse_thresh:
                found += 1

                pos = transform[:3, 3]
                rot_matrix = transform[:3, :3].copy()
                quat = R.from_matrix(rot_matrix).as_quat()

                self.get_logger().info(
                    f"[РОБОТ ЛОКАЛИЗОВАН] ID={int(c_idx)} | "
                    f"Fitness={fitness:.2f} | RMSE={rmse * 1000:.1f}мм | "
                    f"Pos: X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}"
                )

                marker = Marker()
                marker.header = msg.header
                marker.ns = "icp_detected_robots"
                marker.id = int(c_idx)
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = float(pos[2])

                marker.pose.orientation.x = float(quat[0])
                marker.pose.orientation.y = float(quat[1])
                marker.pose.orientation.z = float(quat[2])
                marker.pose.orientation.w = float(quat[3])

                marker.scale.x = dx
                marker.scale.y = dy
                marker.scale.z = dz

                marker.color.r = 0.0
                marker.color.g = 0.8
                marker.color.b = 0.2
                marker.color.a = 0.85

                marker.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
                markers.markers.append(marker)
            else:
                self.get_logger().info(
                    f"ID={int(c_idx)} не прошел ICP: Fitness={fitness:.2f} (порог {self.fitness_thresh}), "
                    f"RMSE={rmse*1000:.1f}мм"
                )

        if found > 0:
            self.marker_publisher.publish(markers)

    def match_icp(self, cluster_pts: np.ndarray, cluster_center: np.ndarray):
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_pts)

        init_trans = np.identity(4)
        init_trans[:3, 3] = cluster_center

        reg = o3d.pipelines.registration.registration_icp(
            self.pcd_model,
            pcd_cluster,
            self.max_icp_dist,
            init_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
        )

        return reg.fitness, reg.inlier_rmse, reg.transformation


def main(args=None):
    rclpy.init(args=args)
    node = RobotDetectorICP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()