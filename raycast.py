import numpy as np


def raycast(
    vertices: np.ndarray, origin: np.ndarray, angle: float, *, closed: bool = True
):
    """
    Cast a half-ray from `origin` in direction `angle` (radians) and
    find its first intersection with the line segments defined by
    `vertices`.

    Parameters
    ----------
    vertices : (N, 2) float array
        Vertices of one or more polygonal chains.  Consecutive vertices
        are treated as an edge; if `closed` is True the last vertex is
        also connected back to the first.
    origin   : (2,) float array
        Ray start position.
    angle    : float
        Direction of the ray in **radians** (0 along +x, π/2 along +y).
    closed   : bool, default True
        Whether to connect the last vertex back to the first.

    Returns
    -------
    hit_point : (2,) float array or None
        Coordinates of the closest hit, or ``None`` when the ray
        misses the geometry.
    distance  : float or None
        Ray-parameter *t* (‖hit_point – origin‖). ``None`` if no hit.
    hit_index : int or None
        Index *i* such that the hit lies on segment (i, i+1).  ``None``
        if no hit.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("vertices has to be an (N,2) array")

    # Build the segment list --------------------------------------------------
    if closed:
        seg_starts = v
        seg_ends = np.roll(v, -1, axis=0)
    else:
        seg_starts = v[:-1]
        seg_ends = v[1:]

    # Pre-compute constants ----------------------------------------------------
    d = np.array([np.cos(angle), np.sin(angle)])  # ray direction
    r0 = origin.astype(float)

    best_t = np.inf
    best_hit = None
    best_idx = None

    # tiny helper for 2-D perp product  a×b = ax*by − ay*bx
    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for i, (p, q) in enumerate(zip(seg_starts, seg_ends)):
        e = q - p  # segment vector
        denom = cross2(d, e)
        if np.isclose(denom, 0.0):  # parallel (or collinear)
            continue

        # Solve r0 + t d = p + u e
        diff = p - r0
        t = cross2(diff, e) / denom
        u = cross2(diff, d) / denom

        if t >= 0.0 and 0.0 <= u <= 1.0:  # ray hits this segment
            if t < best_t:  # keep closest
                best_t = t
                best_hit = r0 + t * d
                best_idx = i

    if best_hit is None:
        return None, None, None
    return best_hit, best_t, best_idx


def raycast_squares(squares: np.ndarray, origin: np.ndarray, angle: float):
    """
    Cast a ray against multiple squares and find the closest hit.

    This is a specialized version optimized for square geometry.

    Parameters
    ----------
    squares : list of (4, 2) float arrays
        List of squares, where each square is defined by 4 vertices
        in order (typically bottom-left, bottom-right, top-right, top-left).
    origin : (2,) float array
        Ray start position.
    angle : float
        Direction of the ray in **radians** (0 along +x, π/2 along +y).

    Returns
    -------
    hit_point : (2,) float array or None
        Coordinates of the closest hit, or ``None`` when the ray
        misses all squares.
    distance : float or None
        Distance to the closest hit. ``None`` if no hit.
    square_index : int or None
        Index of the square that was hit. ``None`` if no hit.
    edge_index : int or None
        Index of the edge within the hit square (0=bottom, 1=right, 2=top, 3=left).
        ``None`` if no hit.
    """
    d = np.array([np.cos(angle), np.sin(angle)])  # ray direction
    r0 = origin.astype(float)

    closest_hit = None
    closest_dist = np.inf
    closest_square_idx = None
    closest_edge_idx = None

    # tiny helper for 2-D perp product  a×b = ax*by − ay*bx
    def cross2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for square_idx, square in enumerate(squares):
        square = np.asarray(square, dtype=float)
        if square.shape != (4, 2):
            raise ValueError(f"Square {square_idx} must have exactly 4 vertices")

        # Check each edge of the square
        for edge_idx in range(4):
            p = square[edge_idx]  # start of edge
            q = square[(edge_idx + 1) % 4]  # end of edge

            e = q - p  # edge vector
            denom = cross2(d, e)

            if np.isclose(denom, 0.0):  # parallel (or collinear)
                continue

            # Solve r0 + t d = p + u e
            diff = p - r0
            t = cross2(diff, e) / denom
            u = cross2(diff, d) / denom

            if t >= 0.0 and 0.0 <= u <= 1.0:  # ray hits this edge
                if t < closest_dist:  # keep closest
                    closest_dist = t
                    closest_hit = r0 + t * d
                    closest_square_idx = square_idx
                    closest_edge_idx = edge_idx

    if closest_hit is None:
        return None, None, None, None
    return closest_hit, closest_dist, closest_square_idx, closest_edge_idx


def raycast_axis_aligned_squares_batch_ultra(
    squares: np.ndarray, origin: np.ndarray, angles: np.ndarray
) -> np.ndarray:
    """
    Ultra-optimized batch raycast using full vectorization over both rays and squares.

    This function casts multiple rays from a single origin against multiple axis-aligned
    squares simultaneously, using full vectorization for maximum performance. All ray-square
    combinations are processed in parallel using NumPy broadcasting.

    Parameters
    ----------
    squares : np.ndarray
        Array of squares in one of two formats:
        - Shape (M, 4, 2): M squares with 4 vertices each (x, y coordinates)
        - List of tuples: [(min_x, min_y, max_x, max_y), ...] bounding box format
    origin : np.ndarray
        Shape (2,) array containing the ray origin coordinates [x, y]
    angles : np.ndarray
        Shape (N,) array of ray angles in radians (0 = +x direction, π/2 = +y direction)

    Returns
    -------
    distances : np.ndarray
        Array of N distances from origin to hit points. -1 for rays that miss all squares.

    Notes
    -----
    This function is optimized for scenarios with many rays and squares by:
    - Using full vectorization over both rays and squares simultaneously
    - Broadcasting ray-square computations to leverage SIMD operations
    - Pre-computing inverse ray directions to avoid division in inner loops
    - Using 32-bit floats for better cache performance

    For rays originating inside a square, the function returns the exit point
    rather than the entry point to ensure consistent behavior.

    Examples
    --------
    >>> squares = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]])
    >>> origin = np.array([5.0, -5.0])
    >>> angles = np.array([np.pi/2, 0.0])  # Up and right
    >>> distances = raycast_axis_aligned_squares_batch_ultra(squares, origin, angles)
    """
    if squares is None or len(squares) == 0:
        n_rays = len(angles)
        return np.full(n_rays, -1.0, dtype=np.float32)

    angles = np.asarray(angles, dtype=np.float32)
    n_rays = len(angles)
    origin = np.asarray(origin, dtype=np.float32)

    # Pre-compute ray directions for all angles (N, 2)
    ray_dirs = np.column_stack([np.cos(angles), np.sin(angles)])

    # Handle degenerate ray directions
    eps = 1e-7
    ray_dirs_safe = np.where(np.abs(ray_dirs) < eps, eps, ray_dirs)
    inv_ray_dirs = 1.0 / ray_dirs_safe  # (N, 2)

    # Pre-compute all bounding boxes (M, 4)
    n_squares = len(squares)
    bounds = np.zeros((n_squares, 4), dtype=np.float32)

    for i, square in enumerate(squares):
        square_arr = np.asarray(square, dtype=np.float32)
        bounds[i, :2] = np.min(square_arr, axis=0)
        bounds[i, 2:] = np.max(square_arr, axis=0)

    # Vectorized computation for all ray-square pairs
    # Reshape for broadcasting: rays (N, 1, 2), bounds (1, M, 4)
    # ray_dirs_bc = ray_dirs[:, np.newaxis, :]      # (N, 1, 2)
    inv_ray_dirs_bc = inv_ray_dirs[:, np.newaxis, :]  # (N, 1, 2)
    bounds_bc = bounds[np.newaxis, :, :]  # (1, M, 4)
    origin_bc = origin[np.newaxis, np.newaxis, :]  # (1, 1, 2)

    # Calculate t values for all ray-square combinations (N, M, 2)
    t_mins = (bounds_bc[:, :, :2] - origin_bc) * inv_ray_dirs_bc
    t_maxs = (bounds_bc[:, :, 2:] - origin_bc) * inv_ray_dirs_bc

    # Ensure t_min <= t_max for each axis
    t_near = np.minimum(t_mins, t_maxs)  # (N, M, 2)
    t_far = np.maximum(t_mins, t_maxs)  # (N, M, 2)

    # Calculate entry and exit times (N, M)
    t_entry = np.maximum(t_near[:, :, 0], t_near[:, :, 1])
    t_exit = np.minimum(t_far[:, :, 0], t_far[:, :, 1])

    # Check if ray origin is inside boxes (1, M) -> broadcast to (N, M)
    inside_mask = (
        (origin[0] >= bounds[:, 0])
        & (origin[0] <= bounds[:, 2])
        & (origin[1] >= bounds[:, 1])
        & (origin[1] <= bounds[:, 3])
    )
    inside_mask_bc = inside_mask[np.newaxis, :]  # (1, M) -> broadcasts to (N, M)

    # Choose hit times
    t_hit = np.where(inside_mask_bc, t_exit, t_entry)  # (N, M)

    # Valid hits mask (N, M)
    valid_hits = (t_entry <= t_exit) & (t_exit >= eps) & (t_hit >= eps)

    # For each ray, find the closest valid hit
    distances = np.full(n_rays, -1.0, dtype=np.float32)

    for ray_idx in range(n_rays):
        ray_valid = valid_hits[ray_idx]  # (M,) boolean mask for this ray

        if np.any(ray_valid):
            ray_t_hits = t_hit[ray_idx]  # (M,) t values for this ray
            ray_t_valid = ray_t_hits[ray_valid]  # Only valid t values

            closest_t = np.min(ray_t_valid)
            distances[ray_idx] = closest_t

    return distances
