import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2


def fit_circle_kasa(xy: np.ndarray):
    """
    Algebraic circle fit using the Kasa method.
    Returns: (center_x, center_y, radius, rmse)
    """
    if len(xy) < 4:
        return None, None, None, 999.0
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc = sol[0] / 2.0
        yc = sol[1] / 2.0
        r_sq = sol[2] + xc**2 + yc**2
        if r_sq <= 0:
            return None, None, None, 999.0
        r = float(np.sqrt(r_sq))
        dists = np.sqrt((x - xc)**2 + (y - yc)**2)
        rmse = float(np.sqrt(np.mean((dists - r)**2)))
        return float(xc), float(yc), r, rmse
    except Exception:
        return None, None, None, 999.0


class RobotTracker:
    """
    Smooths detections and handles brief occlusions (coasting).
    Prevents false jumps to distant walls or background clutter.
    """
    def __init__(self, max_coasting_frames=6, max_jump_dist=0.8, smoothing_alpha=0.65):
        self.pos = None
        self.center_z = 0.15
        self.missed = 999
        self.hits = 0
        self.max_coasting_frames = max_coasting_frames
        self.max_jump_dist = max_jump_dist
        self.smoothing_alpha = smoothing_alpha

    def update(self, candidates):
        if not candidates:
            self.missed += 1
            if self.missed <= self.max_coasting_frames and self.pos is not None:
                return self.pos, self.center_z, True
            return None, None, False

        # If no active track, initialize with best-scored candidate
        if self.pos is None or self.missed > self.max_coasting_frames:
            best = min(candidates, key=lambda c: c['score'])
            self.pos = best['pos'].copy()
            self.center_z = float(best['center_z'])
            self.missed = 0
            self.hits = 1
            return self.pos, self.center_z, False

        # Match closest candidate to existing track
        dists = [np.linalg.norm(c['pos'][:2] - self.pos[:2]) for c in candidates]
        min_idx = int(np.argmin(dists))

        if dists[min_idx] <= self.max_jump_dist:
            best = candidates[min_idx]
            self.pos = self.smoothing_alpha * best['pos'] + (1.0 - self.smoothing_alpha) * self.pos
            self.center_z = 0.5 * best['center_z'] + 0.5 * self.center_z
            self.missed = 0
            self.hits += 1
            return self.pos, self.center_z, False
        else:
            # Candidate is far away: coast on current track to prevent false jump to walls
            self.missed += 1
            if self.missed <= self.max_coasting_frames and self.pos is not None:
                return self.pos, self.center_z, True

            # If coasting expired, re-lock onto new valid candidate
            best = min(candidates, key=lambda c: c['score'])
            self.pos = best['pos'].copy()
            self.center_z = float(best['center_z'])
            self.missed = 0
            self.hits = 1
            return self.pos, self.center_z, False


class RobotDetector(Node):
    def __init__(self):
        super().__init__('robot_detector')

        self.declare_parameter('clust_topic', '/livox/lidar/clust')
        self.declare_parameter('marker_topic', '/detected_robot')
        self.declare_parameter('marker_pose_topic', '/detected_robot_pose')

        clust_topic = self.get_parameter('clust_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        marker_pose_topic = self.get_parameter('marker_pose_topic').get_parameter_value().string_value

        # Robot physical bounds (TurtleBot 2: diameter ~ 0.354m, height ~ 0.35m)
        self.tb2_diameter = 0.354
        self.tb2_radius = 0.177
        self.min_pts = 14

        # Strict Wall & Post Rejection Thresholds:
        self.min_height_dz = 0.22         # TB2 is a 3D cylindrical body; flat wall scan stripes have dz < 0.20m
        self.max_height_dz = 0.42         # Structural walls and columns have dz > 0.50m
        self.ceiling_z_threshold = -0.16  # TB2 never extends above z = -0.15m in inverted Livox frame
        self.min_wall_length = 0.16       # Eliminates tiny wall patches
        self.max_wall_length = 0.44       # Eliminates long walls
        self.min_depth = 0.08             # Flat walls have depth < 0.06m; cylinder visible arc has depth >= 0.08m
        self.max_aspect_ratio = 2.6       # TB2 is circular (ratio 1.2 - 2.2); walls are elongated (ratio > 2.8)
        self.min_radius = 0.10
        self.max_radius = 0.23
        self.max_circle_rmse = 0.048

        self.tracker = RobotTracker()

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            clust_topic,
            self.clust_callback,
            10
        )

        # Publish to both standard topics so all RViz configs / downstream nodes receive markers
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.marker_pose_pub = self.create_publisher(MarkerArray, marker_pose_topic, 10)

        self.get_logger().info(
            f"RobotDetector started: subscribing to {clust_topic}, "
            f"publishing markers to {marker_topic} and {marker_pose_topic}"
        )

    def clust_callback(self, msg: PointCloud2):
        # 1. Fast binary point extraction
        try:
            raw_bytes = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            points = raw_bytes[:, :16].copy().view(dtype=np.float32).reshape(-1, 4)
        except Exception:
            raw = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            if not raw:
                return
            points = np.array([[p[0], p[1], p[2], p[3]] for p in raw], dtype=np.float32)

        if len(points) == 0:
            return

        unique_labels = np.unique(points[:, 3])
        candidates = []

        for c_idx in unique_labels:
            if c_idx < 0:
                continue

            c_pts = points[points[:, 3] == c_idx, :3]
            n_pts = len(c_pts)

            if n_pts < self.min_pts:
                continue

            p_min = np.min(c_pts, axis=0)
            p_max = np.max(c_pts, axis=0)
            dz = float(p_max[2] - p_min[2])

            # Filter 1: Vertical height span & ceiling cutoff
            # TB2 sits on the floor and stands ~35cm tall.
            # Ceiling columns and walls reach z < -0.16m. Flat wall scan stripes have dz < 0.22m.
            if p_min[2] < self.ceiling_z_threshold or dz < self.min_height_dz or dz > self.max_height_dz:
                continue

            center = (p_min + p_max) / 2.0
            dist_to_sensor = float(np.linalg.norm(center[:2]))
            if dist_to_sensor < 0.35 or dist_to_sensor > 5.5:
                continue

            xy = c_pts[:, :2]

            # Filter 2: PCA in XY plane to detect flatness and elongation
            cov = np.cov(xy - np.mean(xy, axis=0), rowvar=False)
            evals, _ = np.linalg.eigh(cov)
            l_maj = 2.0 * float(np.sqrt(max(evals[1], 1e-6)))
            l_min = 2.0 * float(np.sqrt(max(evals[0], 1e-6)))

            # Reject flat wall stripes (l_min < 0.08m) and long wall segments (l_maj > 0.44m)
            if l_maj < self.min_wall_length or l_maj > self.max_wall_length or l_min < self.min_depth:
                continue

            # Aspect ratio check: TB2 is rounded (ratio <= 2.6); walls are elongated planes
            ratio = l_maj / max(l_min, 1e-3)
            if ratio > self.max_aspect_ratio:
                continue

            # Filter 3: Kasa Circle / Arc Fit
            xc, yc, r, rmse = fit_circle_kasa(xy)
            if r is None or r < self.min_radius or r > self.max_radius or rmse > self.max_circle_rmse:
                continue

            # Direction check: circle center must be behind the visible arc
            d_center = float(np.linalg.norm([xc, yc]))
            d_arc = float(np.linalg.norm(np.mean(xy, axis=0)))
            if d_center < d_arc - 0.04:
                continue

            # Score candidate based on closeness to TB2 geometric dimensions
            score = abs(r - self.tb2_radius) * 3.0 + rmse * 4.0 + abs(dz - 0.35) * 1.0
            candidates.append({
                'pos': np.array([xc, yc]),
                'center_z': center[2],
                'score': score,
                'r': r,
                'pts': n_pts
            })

        # 4. Multi-frame tracker update
        robot_pos, center_z, is_coasting = self.tracker.update(candidates)

        # 5. Publish detection markers
        markers = MarkerArray()
        clear_m = Marker()
        clear_m.header.frame_id = msg.header.frame_id
        clear_m.header.stamp = msg.header.stamp
        clear_m.action = Marker.DELETEALL
        markers.markers.append(clear_m)

        if robot_pos is not None:
            dist = float(np.linalg.norm(robot_pos[:2]))

            # Cylinder Model Marker
            m_robot = Marker()
            m_robot.header.frame_id = msg.header.frame_id
            m_robot.header.stamp = msg.header.stamp
            m_robot.ns = "opponent_robot"
            m_robot.id = 1
            m_robot.type = Marker.CYLINDER
            m_robot.action = Marker.ADD
            m_robot.pose.position.x = float(robot_pos[0])
            m_robot.pose.position.y = float(robot_pos[1])
            m_robot.pose.position.z = float(center_z)
            m_robot.pose.orientation.w = 1.0
            m_robot.scale.x = self.tb2_diameter
            m_robot.scale.y = self.tb2_diameter
            m_robot.scale.z = 0.35

            # Vibrant Green
            m_robot.color.r = 0.0
            m_robot.color.g = 1.0
            m_robot.color.b = 0.2
            m_robot.color.a = 0.85 if not is_coasting else 0.50
            m_robot.lifetime = Duration(seconds=0.35).to_msg()
            markers.markers.append(m_robot)

            # Text Label Marker
            m_text = Marker()
            m_text.header.frame_id = msg.header.frame_id
            m_text.header.stamp = msg.header.stamp
            m_text.ns = "opponent_robot_text"
            m_text.id = 2
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(robot_pos[0])
            m_text.pose.position.y = float(robot_pos[1])
            m_text.pose.position.z = float(center_z) + 0.30
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.15
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = f"TB2 ROBOT | dist={dist:.2f}m"
            m_text.lifetime = Duration(seconds=0.35).to_msg()
            markers.markers.append(m_text)

            self.get_logger().info(
                f"[OPPONENT ROBOT DETECTED] Pos: X={robot_pos[0]:.2f}, Y={robot_pos[1]:.2f}, "
                f"Z={center_z:.2f} | dist={dist:.2f}m"
            )

        self.marker_pub.publish(markers)
        self.marker_pose_pub.publish(markers)


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
