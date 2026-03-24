import numpy as np


def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _principal_direction(points):
    """
    Principal direction of an ordered local point set, using PCA/SVD.
    The sign is oriented to agree with the local parameterization.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to compute a direction.")

    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    v = vh[0]  # principal direction

    # Fix sign so that it follows the ordering of the points
    ref = pts[-1] - pts[0]
    if np.dot(v, ref) < 0:
        v = -v

    return _normalize(v)


def _closest_point_on_closed_polyline(loop_xyz, p):
    """
    Closest point X on the closed polyline loop_xyz to point p.
    Returns:
        X         : closest point on the loop
        seg_idx   : segment index i such that X lies on segment i -> i+1
        t         : parameter in [0, 1] on that segment
    """
    loop_xyz = np.asarray(loop_xyz, dtype=float)
    p = np.asarray(p, dtype=float)

    n = len(loop_xyz)
    best_dist2 = np.inf
    best_X = None
    best_i = None
    best_t = None

    for i in range(n):
        a = loop_xyz[i]
        b = loop_xyz[(i + 1) % n]
        ab = b - a
        denom = np.dot(ab, ab)

        if denom < 1e-12:
            t = 0.0
            X = a
        else:
            t = np.dot(p - a, ab) / denom
            t = np.clip(t, 0.0, 1.0)
            X = a + t * ab

        dist2 = np.dot(p - X, p - X)
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_X = X
            best_i = i
            best_t = t

    return best_X, best_i, best_t


def _loop_neighborhood_indices(n, seg_idx, t, k_loop):
    """
    Choose a center vertex near the closest point X and return
    wrapped neighborhood indices of size 2*k_loop+1.
    """
    # Center on the nearer endpoint of the closest segment
    center = seg_idx if t < 0.5 else (seg_idx + 1) % n
    return [((center + j) % n) for j in range(-k_loop, k_loop + 1)]


def _tail_neighborhood_indices(m, tail_index, k_tail):
    """
    Open-chain neighborhood around tail_index, clipped to valid range.
    """
    a = max(0, tail_index - k_tail)
    b = min(m, tail_index + k_tail + 1)
    return list(range(a, b))


def compute_angle(loop_xyz, tail_xyz, tail_index, k_loop=2, k_tail=2):
    """
    Compute the signed angle at which the tail intersects the local loop surface.

    Parameters
    ----------
    loop_xyz : array-like, shape (n, 3)
        Ordered coordinates of the closed loop.
    tail_xyz : array-like, shape (m, 3)
        Ordered coordinates of the tail.
    tail_index : int
        Index of the tail point P = tail_xyz[tail_index].
    k_loop : int
        Uses 2*k_loop + 1 loop points around the closest loop location.
    k_tail : int
        Uses 2*k_tail + 1 tail points around tail_index (clipped at ends).

    Returns
    -------
    angle : float
        Signed angle in [-pi/2, pi/2].

    Notes
    -----
    The sign depends on the chosen orientation of the local loop normal.
    Here the normal is oriented consistently with:
        - local loop ordering, and
        - the side on which P lies.
    """
    loop_xyz = np.asarray(loop_xyz, dtype=float)
    tail_xyz = np.asarray(tail_xyz, dtype=float)

    if len(loop_xyz) < 3:
        raise ValueError("loop_xyz must contain at least 3 points.")
    if len(tail_xyz) < 2:
        raise ValueError("tail_xyz must contain at least 2 points.")
    if not (0 <= tail_index < len(tail_xyz)):
        raise IndexError("tail_index out of range.")

    P = tail_xyz[tail_index]

    # 1) Closest point X on the loop to P
    X, seg_idx, t = _closest_point_on_closed_polyline(loop_xyz, P)

    # 2) Local loop neighborhood around X
    loop_idx = _loop_neighborhood_indices(len(loop_xyz), seg_idx, t, k_loop)
    loop_pts = loop_xyz[loop_idx]

    # Local tail neighborhood around P
    tail_idx = _tail_neighborhood_indices(len(tail_xyz), tail_index, k_tail)
    tail_pts = tail_xyz[tail_idx]

    if len(loop_pts) < 3:
        raise ValueError("Need at least 3 loop points in the local neighborhood.")
    if len(tail_pts) < 2:
        raise ValueError("Need at least 2 tail points in the local neighborhood.")

    # 3) Principal direction of the tail
    v_tail = _principal_direction(tail_pts)

    # 4) Local loop tangent from ordered points
    v_loop = _principal_direction(loop_pts)

    # 5) Local loop normal from PCA plane
    centered = loop_pts - loop_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n_loop = _normalize(vh[-1])  # least-variance direction = plane normal

    # 6) Fix the normal sign consistently
    # Use the vector from X to P, projected orthogonally to the loop tangent
    r = P - X
    r = r - np.dot(r, v_loop) * v_loop
    r_norm = np.linalg.norm(r)

    if r_norm > 1e-12:
        r = r / r_norm
        ref_normal = np.cross(v_loop, r)
        if np.dot(n_loop, ref_normal) < 0:
            n_loop = -n_loop

    # 7) Signed angle in [-pi/2, pi/2]
    s = np.clip(np.dot(v_tail, n_loop), -1.0, 1.0)
    angle = float(np.arcsin(s))

    return angle