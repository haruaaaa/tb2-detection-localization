import numpy as np
from sklearn.cluster import DBSCAN


ROBOT_MIN_SIZE = 0.24
ROBOT_MAX_SIZE = 0.44
SEG_LEN_MIN = 0.60
ELONG_MAX = 1.5
GATE = 0.40
MIN_HITS = 1
MAX_MISS = 6
SPAN_MIN = 0.027
SPAN_MOVING = 0.1


def cluster_points(points, eps=0.1, min_samples=15, z_min=-0.15, z_max=0.325):
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


def fit_circle_kasa(xy):
    """Algebraic circle fitting (Kasa method). Returns (xc, yc, r, rms)."""
    if len(xy) < 5:
        return None, None, None, 999.0
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x**2 + y**2
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        r_sq = c + xc**2 + yc**2
        if r_sq <= 0:
            return None, None, None, 999.0
        r = np.sqrt(r_sq)
        dists = np.sqrt((x - xc)**2 + (y - yc)**2)
        rms = np.sqrt(np.mean((dists - r)**2))
        return float(xc), float(yc), float(r), float(rms)
    except Exception:
        return None, None, None, 999.0


def classify_clusters(points, labels, robot_min=ROBOT_MIN_SIZE, robot_max=ROBOT_MAX_SIZE,
                      seg_len=SEG_LEN_MIN, elong_max=ELONG_MAX,
                      circle_r_min=0.14, circle_r_max=0.23, max_circle_rms=0.035, use_circle_fit=True):
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
        elif (robot_min <= l_major <= robot_max and
              robot_min <= l_minor <= robot_max and
              elong <= elong_max):
            
            cand_pos = centroid
            cand_size = l_major
            is_robot = True

            if use_circle_fit and len(xy) >= 5:
                xc, yc, r, rms = fit_circle_kasa(xy)
                if r is not None and circle_r_min <= r <= circle_r_max and rms <= max_circle_rms:
                    # Kasa circle center is more accurate for cylindrical TB2 chassis
                    cand_pos = np.array([xc, yc])
                    cand_size = r * 2.0
                elif rms > max_circle_rms * 1.8:
                    # Poor circle fit on dense clusters indicates irregular clutter
                    is_robot = False

            if is_robot:
                candidates.append({'pos': cand_pos, 'size': cand_size})

    return walls, candidates


class Tracker:
    def __init__(self, gate=GATE, min_hits=MIN_HITS, max_miss=MAX_MISS, span_min=SPAN_MIN):
        self.gate = gate
        self.min_hits = min_hits
        self.max_miss = max_miss
        self.span_min = span_min
        self.tracks = []
        self._next_id = 0
        self._best_id = None

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

        candidates = []
        for t in self.tracks:
            if t['hits'] >= self.min_hits:
                t['span'] = compute_span(t['history'])
                if t['span'] >= self.span_min:
                    candidates.append(t)

        if not candidates:
            self._best_id = None
            return []

        moving = [t for t in candidates if t['span'] >= SPAN_MOVING]

        if moving:
            current_best = next((t for t in moving if t['id'] == self._best_id), None)
            top = max(moving, key=lambda t: t['span'])

            if current_best is None:
                self._best_id = top['id']
            elif top['span'] > current_best['span'] * 1.5:
                self._best_id = top['id']

            moving.sort(key=lambda t: (t['id'] != self._best_id, -t['span']))
            return moving
        else:
            current_best = next((t for t in candidates if t['id'] == self._best_id), None)
            top = max(candidates, key=lambda t: t['hits'])

            if current_best is None:
                self._best_id = top['id']
            elif top['hits'] > current_best['hits'] * 1.5:
                self._best_id = top['id']

            candidates.sort(key=lambda t: (t['id'] != self._best_id, -t['hits']))
            return candidates[:1]

    def reset(self):
        self.tracks = []
        self._next_id = 0
        self._best_id = None


def compute_span(history):
    if len(history) < 2:
        return 0.0
    pts = np.array(history)
    return float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
