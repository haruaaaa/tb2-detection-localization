import numpy as np
from sklearn.cluster import DBSCAN


ROBOT_MIN_SIZE = 0.12
ROBOT_MAX_SIZE = 0.55
SEG_LEN_MIN = 0.60
ELONG_MAX = 2.5
GATE = 0.40
MIN_HITS = 5
MAX_MISS = 6


def cluster_points(points, eps=0.1, min_samples=15, z_min=-0.10, z_max=0.20):
    if len(points) == 0:
        return np.array([]), np.array([])

    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    pts = points[mask]

    if len(pts) < min_samples:
        return np.array([]), np.array([])

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    valid = labels != -1
    return pts[valid], labels[valid]


def principal_extent(xy):
    if len(xy) < 3:
        return 0.0, 0.0
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return 0.0, 0.0
    _, vecs = np.linalg.eigh(cov)
    proj = centered @ vecs
    spans = proj.max(axis=0) - proj.min(axis=0)
    return float(spans.max()), float(spans.min())


def classify_clusters(points, labels, robot_min=ROBOT_MIN_SIZE, robot_max=ROBOT_MAX_SIZE,
                      seg_len=SEG_LEN_MIN, elong_max=ELONG_MAX):
    walls = []
    candidates = []

    for cid in np.unique(labels):
        xy = points[labels == cid, :2]
        if len(xy) < 3:
            continue

        centroid = xy.mean(axis=0)
        l_major, l_minor = principal_extent(xy)
        elong = l_major / max(l_minor, 1e-3)

        if l_major > seg_len:
            walls.append({'pos': centroid, 'size': l_major})
        elif robot_min <= l_major <= robot_max and elong <= elong_max:
            candidates.append({'pos': centroid, 'size': l_major})

    return walls, candidates


class Tracker:
    def __init__(self, gate=GATE, min_hits=MIN_HITS, max_miss=MAX_MISS):
        self.gate = gate
        self.min_hits = min_hits
        self.max_miss = max_miss
        self.tracks = []
        self._next_id = 0

    def update(self, candidates):
        used = set()

        for tr in self.tracks:
            best_i, best_d = -1, self.gate
            for i, c in enumerate(candidates):
                if i in used:
                    continue
                d = np.hypot(*(c['pos'] - tr['pos']))
                if d < best_d:
                    best_d, best_i = d, i

            if best_i >= 0:
                used.add(best_i)
                tr['pos'] = candidates[best_i]['pos']
                tr['size'] = candidates[best_i]['size']
                tr['hits'] += 1
                tr['miss'] = 0
                tr['history'].append(tr['pos'].copy())
            else:
                tr['miss'] += 1

        for i, c in enumerate(candidates):
            if i not in used:
                self.tracks.append({
                    'pos': c['pos'],
                    'size': c['size'],
                    'hits': 1,
                    'miss': 0,
                    'id': self._next_id,
                    'history': [c['pos'].copy()]
                })
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t['miss'] <= self.max_miss]

        confirmed = [t for t in self.tracks if t['hits'] >= self.min_hits]
        confirmed.sort(key=lambda t: -t['hits'])
        return confirmed

    def reset(self):
        self.tracks = []
        self._next_id = 0


def compute_span(history):
    if len(history) < 2:
        return 0.0
    pts = np.array(history)
    return float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
