import os
import re
import pytest
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'obstacle_detector'))
from obstacle_detector import core


BAGS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'rosbags')
TOPIC = '/livox/lidar'


def read_bag_frames(bag_path, max_frames=200):
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter = rosbag2_py.ConverterOptions('', '')
    reader.open(storage, converter)

    frames = []
    while reader.has_next() and len(frames) < max_frames:
        topic, data, _ = reader.read_next()
        if topic != TOPIC:
            continue
        msg = deserialize_message(data, PointCloud2)
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = np.array([[p[0], p[1], p[2]] for p in gen], dtype=np.float32)
        if len(pts) > 0:
            frames.append(pts)

    return frames


def run_detection(frames, min_hits=5):
    tracker = core.Tracker(min_hits=min_hits)
    all_confirmed = []

    for pts in frames:
        clustered_pts, labels = core.cluster_points(pts)
        if len(clustered_pts) == 0:
            tracker.update([])
            continue

        _, candidates = core.classify_clusters(clustered_pts, labels)
        confirmed = tracker.update(candidates)

        for t in confirmed:
            all_confirmed.append({
                'pos': t['pos'].copy(),
                'size': t['size'],
                'id': t['id'],
                'history': [h.copy() for h in t['history']]
            })

    final_tracks = {}
    for t in tracker.tracks:
        if t['hits'] >= min_hits:
            final_tracks[t['id']] = {
                'pos': t['pos'],
                'size': t['size'],
                'span': core.compute_span(t['history']),
                'hits': t['hits']
            }

    return final_tracks


def get_bag_label(bag_name):
    m = re.search(r'(\d+\.?\d*)m', bag_name)
    if m:
        return float(m.group(1))
    return None


def get_bags():
    if not os.path.isdir(BAGS_DIR):
        return []
    bags = []
    for name in os.listdir(BAGS_DIR):
        path = os.path.join(BAGS_DIR, name)
        if os.path.isdir(path) and name.startswith('rosbag2_'):
            bags.append((name, path))
    return sorted(bags)


class TestShapeDetection:

    @pytest.fixture(scope='class')
    def bags(self):
        return get_bags()

    def test_static_1m_shape(self, bags):
        bag = next((b for b in bags if 'static_1m' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1])
        tracks = run_detection(frames)

        assert len(tracks) > 0, "no robot detected by shape"

        best = max(tracks.values(), key=lambda t: t['hits'])
        dist = np.hypot(*best['pos'])
        assert 0.8 < dist < 1.3, f"robot distance {dist:.2f}m not near 1m"
        assert best['size'] < 0.6, f"robot size {best['size']:.2f}m too large"

    def test_rot_05m_shape(self, bags):
        bag = next((b for b in bags if 'rot_0.5m' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1])
        tracks = run_detection(frames)

        assert len(tracks) > 0, "no robot detected"
        best = max(tracks.values(), key=lambda t: t['hits'])
        dist = np.hypot(*best['pos'])
        assert 0.3 < dist < 0.8, f"expected ~0.5m, got {dist:.2f}m"

    def test_rot_1m_shape(self, bags):
        bag = next((b for b in bags if 'rot_1m' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1])
        tracks = run_detection(frames)

        assert len(tracks) > 0, "no robot detected"
        best = max(tracks.values(), key=lambda t: t['hits'])
        dist = np.hypot(*best['pos'])
        assert 0.7 < dist < 1.3, f"expected ~1m, got {dist:.2f}m"

    def test_rot_2m_shape(self, bags):
        bag = next((b for b in bags if 'rot_2m' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1])
        tracks = run_detection(frames)

        assert len(tracks) > 0, "no robot detected"
        best = max(tracks.values(), key=lambda t: t['hits'])
        dist = np.hypot(*best['pos'])
        assert 1.5 < dist < 2.5, f"expected ~2m, got {dist:.2f}m"

    def test_rot_3m_shape(self, bags):
        bag = next((b for b in bags if 'rot_3m' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1])
        tracks = run_detection(frames)

        assert len(tracks) > 0, "no robot detected"
        best = max(tracks.values(), key=lambda t: t['hits'])
        dist = np.hypot(*best['pos'])
        assert 2.5 < dist < 3.5, f"expected ~3m, got {dist:.2f}m"


class TestMotionDetection:

    @pytest.fixture(scope='class')
    def bags(self):
        return get_bags()

    def test_mov_01_motion(self, bags):
        bag = next((b for b in bags if 'mov_01' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1], max_frames=300)
        tracks = run_detection(frames)

        assert len(tracks) > 0, "no robot detected"

        moving = [t for t in tracks.values() if t['span'] > 1.0]
        assert len(moving) > 0, "no moving robot found (span > 1m)"

        best = max(moving, key=lambda t: t['span'])
        assert best['span'] > 2.0, f"robot span {best['span']:.2f}m too small"
        assert best['size'] < 0.6, f"robot size {best['size']:.2f}m too large"

    def test_mov_02_motion(self, bags):
        bag = next((b for b in bags if 'mov_02' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1], max_frames=300)
        tracks = run_detection(frames)

        assert len(tracks) > 0, "no robot detected"

        moving = [t for t in tracks.values() if t['span'] > 1.0]
        assert len(moving) > 0, "no moving robot found"

        best = max(moving, key=lambda t: t['span'])
        assert best['span'] > 2.0, f"robot span {best['span']:.2f}m too small"


class TestWallRejection:

    @pytest.fixture(scope='class')
    def bags(self):
        return get_bags()

    def test_walls_not_detected_as_robots(self, bags):
        bag = next((b for b in bags if 'static_1m' in b[0]), None)
        if bag is None:
            pytest.skip("bag not found")

        frames = read_bag_frames(bag[1], max_frames=50)

        for pts in frames[:10]:
            clustered_pts, labels = core.cluster_points(pts)
            if len(clustered_pts) == 0:
                continue

            walls, candidates = core.classify_clusters(clustered_pts, labels)

            assert len(walls) > 0, "should detect walls"

            for w in walls:
                assert w['size'] > 0.6, "wall should be large"

            for c in candidates:
                assert c['size'] < 0.6, "candidate should be compact"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
