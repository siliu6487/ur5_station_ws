#!/usr/bin/env python3
"""
In-repo marker-blob detector for GelSight Mini tactile images.

Written here instead of depending on the vendored gsrobotics.utilities.marker_tracker
so detection parameters — especially binary_threshold — can be tuned for our own
hardware setup without touching third-party code. Built because a mounted tool can
locally reduce marker contrast (e.g. compressing the gel near one edge), which a
fixed, untunable threshold sometimes misses entirely for a whole row of markers.

Algorithm (same core idea as gsrobotics, minus its grid-fitting step — neither
caller in this repo needs grid-snapped positions, only raw blob centroids):
  1. highpass = grayscale - large-sigma Gaussian blur of grayscale.
     Subtracting a broad blur removes slow/broad lighting gradients (vignette,
     tilt) while keeping the sharp local dip each marker creates.
  2. Threshold the highpass to a binary mask, morphologically close small gaps,
     label connected components.
  3. Keep components whose pixel area falls within [min_blob_size, max_blob_size].
  4. Report each surviving component's centroid.

Usage:
    tracker = MarkerTracker(frame_float01)   # BGR or RGB, float, any range/scale
    tracker.centers      # (N, 2) as (row, col) == (y, x)
    tracker.marker_mask  # (H, W) uint8, nonzero where a kept blob is

For frames where a tool or a scratch locally distorts the gel enough that fresh
blob detection drops some markers (see detect_markers_with_fallback below), a
known-good reference capture can fill the gaps via optical-flow tracking instead
of a blob-detection threshold — LK matches local image gradients directly, so
it isn't fooled by the same contrast/merging issues.

Known limitation (2026-07-22): per-frame blob detection here is inherently less
reliable than downstream LK flow-tracking, since a fixed binary_threshold + area
filter can be defeated outright by a scratch/merge (see dev_notes/logs/07_22_log.md).
Not worth perfecting further — real sensor-tracking reliability comes from flow
tracking, which only needs this module once, to seed. TODO if precise per-frame
marker count/identity is needed later: adaptive/local thresholding instead of one
fixed global value, and a shape (circularity) filter alongside the area filter.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.morphology import closing, disk

_DEFAULT_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def inner_region_mask(
    centers: np.ndarray, height: int, width: int, border_fraction: float
) -> np.ndarray:
    """Boolean mask: True where a (row, col) center is more than border_fraction
    of the frame away from every edge.

    Used to exclude near-edge detections (housing occlusion, vignette false
    positives) from being *seeded* for tracking. This is meant to be applied
    once at seed time, not every frame — a marker that starts in the inner
    region keeps being tracked normally even if it later moves toward or into
    the border during a large motion (see track_markers_lk / compute_marker_flow,
    neither of which re-applies this filter after seeding).
    """
    if len(centers) == 0:
        return np.zeros(0, dtype=bool)
    r_min, r_max = height * border_fraction, height * (1 - border_fraction)
    c_min, c_max = width * border_fraction, width * (1 - border_fraction)
    return (
        (centers[:, 0] >= r_min) & (centers[:, 0] <= r_max) &
        (centers[:, 1] >= c_min) & (centers[:, 1] <= c_max)
    )


class MarkerTracker:
    """Detects dark marker blobs in a GelSight-style tactile image."""

    def __init__(
        self,
        image: np.ndarray,
        filter_sigma: float = 20.0,
        binary_threshold: float = -0.05,
        min_blob_size: int = 20,
        max_blob_size: int = 500,
        closing_radius: int = 3,
    ):
        """
        Args:
            image: Float image. Channel order (BGR/RGB) doesn't matter — only
                the per-pixel max across channels is used. A grayscale (H, W)
                array is also accepted.
            filter_sigma: Gaussian-blur sigma (px) used to estimate the slow
                background that gets subtracted out before thresholding.
            binary_threshold: highpass cutoff below which a pixel counts as
                part of a marker blob. More negative = stricter (needs a
                darker dip to count). Loosen (move toward 0) if markers with
                reduced contrast — e.g. from a mounted tool locally
                compressing the gel — are being missed.
            min_blob_size / max_blob_size: accepted connected-component area
                in pixels. Components outside this range (too small = noise,
                too large = merged markers) are dropped.
            closing_radius: morphological-closing disk radius (px); fills
                small gaps inside a blob before labeling.
        """
        self.filter_sigma     = filter_sigma
        self.binary_threshold = binary_threshold
        self.min_blob_size    = min_blob_size
        self.max_blob_size    = max_blob_size
        self.closing_radius   = closing_radius

        arr = np.asarray(image, dtype=np.float32)
        gray = np.max(arr, axis=-1) if arr.ndim == 3 else arr

        smoothed = gaussian_filter(gray, sigma=filter_sigma)
        highpass = gray - smoothed

        binary_mask = highpass < binary_threshold
        closed_mask = closing(binary_mask, footprint=disk(closing_radius))
        labels = measure.label(closed_mask)

        centers = []
        keep = np.zeros_like(labels, dtype=bool)
        for i in range(1, labels.max() + 1):
            blob = labels == i
            size = int(blob.sum())
            if min_blob_size <= size <= max_blob_size:
                ys, xs = np.nonzero(blob)
                centers.append((float(ys.mean()), float(xs.mean())))
                keep |= blob

        self.centers = (
            np.array(centers, dtype=np.float32) if centers else np.zeros((0, 2), dtype=np.float32)
        )
        self.marker_mask = np.where(keep, 255, 0).astype(np.uint8)

        # Alias matching the attribute name the vendored gsrobotics.MarkerTracker
        # used, so this is a drop-in replacement for existing callers.
        self.initial_marker_center = self.centers


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    """Grayscale uint8, from either a BGR uint8 frame or a float [0,1] frame."""
    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr


def track_markers_lk(
    reference_image: np.ndarray,
    reference_centers: np.ndarray,
    current_image: np.ndarray,
    lk_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track marker positions from a reference frame into current_image via
    Lucas-Kanade optical flow.

    More robust than re-running blob detection to small local changes — a tool
    locally compressing the gel, a scratch merging a marker's blob with a
    neighboring artifact — since LK matches local image gradients directly
    instead of thresholding a highpass image.

    Args:
        reference_image: BGR uint8 or float [0,1] image reference_centers were
            detected on.
        reference_centers: (N, 2) array of (row, col) positions to track.
        current_image: image to track into — same size as reference_image,
            same format rules.
        lk_params: cv2.calcOpticalFlowPyrLK params. Defaults to a 15x15 window,
            2 pyramid levels (same defaults used elsewhere in this repo).

    Returns:
        tracked_centers: (N, 2) float32 (row, col) positions in current_image.
            Entries where tracking failed keep the reference position.
        found: (N,) bool — True where LK successfully tracked that point.
    """
    if lk_params is None:
        lk_params = _DEFAULT_LK_PARAMS

    ref_gray = _to_gray_u8(reference_image)
    cur_gray = _to_gray_u8(current_image)

    reference_centers = np.asarray(reference_centers, dtype=np.float32)
    if len(reference_centers) == 0:
        return reference_centers.copy(), np.zeros(0, dtype=bool)

    # LK uses (x, y) = (col, row); our convention is (row, col).
    p0 = reference_centers[:, ::-1].reshape(-1, 1, 2).copy()

    p1, st, _ = cv2.calcOpticalFlowPyrLK(ref_gray, cur_gray, p0, None, **lk_params)

    tracked = reference_centers.copy()
    found = np.zeros(len(reference_centers), dtype=bool)
    if p1 is not None and st is not None:
        good = st.ravel() == 1
        tracked[good] = p1.reshape(-1, 2)[good][:, ::-1]  # (x, y) -> (row, col)
        found = good

    return tracked, found


def _cluster_1d(values: np.ndarray, split_at: float) -> np.ndarray:
    """Label sorted 1-D values into bands, breaking wherever a gap exceeds split_at."""
    order = np.argsort(values)
    s = values[order]
    labels = np.zeros(len(s), dtype=int)
    for i in range(1, len(s)):
        labels[i] = labels[i - 1] + (1 if s[i] - s[i - 1] > split_at else 0)
    out = np.empty(len(values), dtype=int)
    out[order] = labels
    return out


def fit_marker_grid(
    centers: np.ndarray,
    grid_shape=None,
    pitch_px: Optional[float] = None,
    min_band_frac: float = 0.5,
):
    """
    Keep only markers that belong to the regular lattice; drop strays.

    GelSight markers are printed on a fixed grid, so a detection that does not
    share a row AND a column with the rest is not a marker — it is vignette,
    a scratch, or a housing edge. The border filter cannot catch these: a stray
    just inside the border cutoff survives it, while a real outer-grid marker
    just outside gets removed. This tests lattice membership instead of distance
    from the edge, so it removes the right points regardless of where they sit.

    Method: markers in the same row share a row coordinate, so sorting each axis
    and splitting wherever the gap exceeds half a pitch recovers the rows and
    columns directly. Real bands hold a full row/column of markers; a stray sits
    alone in a band of its own.

    Args:
        centers: (N, 2) as (row, col).
        grid_shape: (n_rows, n_cols) to keep exactly that many bands per axis,
            largest first — use it when the grid is known (a GelSight Mini is
            7x9 = 63). "auto" keeps every band holding at least min_band_frac of
            the median band size. None disables filtering entirely.
        pitch_px: lattice spacing. Measured from the data when omitted.
        min_band_frac: "auto" mode only — how small a band may be, relative to
            the median band, before it is treated as stray.

    Returns:
        keep: (N,) bool — True for markers on the lattice.
        row_idx, col_idx: (N,) int band indices (only meaningful where keep).
        info: dict with the measured pitch and the band sizes per axis.
    """
    n = len(centers)
    if grid_shape is None or n == 0:
        return (np.ones(n, dtype=bool), np.zeros(n, dtype=int),
                np.zeros(n, dtype=int), {})

    if pitch_px is None:
        pitch_px = _median_nn_distance(centers[:, ::-1])
    if not pitch_px or not np.isfinite(pitch_px):
        return (np.ones(n, dtype=bool), np.zeros(n, dtype=int),
                np.zeros(n, dtype=int), {})

    split = 0.5 * pitch_px
    row_idx = _cluster_1d(centers[:, 0], split)
    col_idx = _cluster_1d(centers[:, 1], split)

    def _keep_bands(idx, want):
        labels, counts = np.unique(idx, return_counts=True)
        if want is not None:
            # Largest `want` bands; ties broken by band position for determinism.
            order = np.lexsort((labels, -counts))
            return set(labels[order[:want]].tolist()), counts.tolist()
        floor = max(2.0, min_band_frac * float(np.median(counts)))
        return set(labels[counts >= floor].tolist()), counts.tolist()

    want_r, want_c = (grid_shape if isinstance(grid_shape, (tuple, list))
                      else (None, None))
    keep_r, row_counts = _keep_bands(row_idx, want_r)
    keep_c, col_counts = _keep_bands(col_idx, want_c)

    keep = np.array(
        [(r in keep_r) and (c in keep_c) for r, c in zip(row_idx, col_idx)],
        dtype=bool,
    )

    # One marker per lattice cell — but only where the candidates are genuinely
    # the same marker detected twice (a blob that split), not two real markers
    # whose bands merged.
    #
    # Banding is 1-D and axis-aligned, so it assumes an undeformed lattice. When
    # the gel is already pressed at seed time a whole column can shift into its
    # neighbour and merge into one band; every cell there then looks occupied
    # twice. Deleting those would discard real markers — measured: 8 lost on one
    # trial. Requiring duplicates to be much closer than the pitch separates the
    # two cases, since a split blob is a few px apart while merged neighbours sit
    # a full pitch apart.
    dup_max_dist = 0.5 * pitch_px
    cells = {}
    for i in np.flatnonzero(keep):
        cells.setdefault((row_idx[i], col_idx[i]), []).append(i)
    n_duplicates = 0
    for (r, c), members in cells.items():
        if len(members) < 2:
            continue
        in_row = [i for i in np.flatnonzero(keep) if row_idx[i] == r]
        in_col = [i for i in np.flatnonzero(keep) if col_idx[i] == c]
        target = np.array([np.median(centers[in_row, 0]),
                           np.median(centers[in_col, 1])])
        best = min(members, key=lambda i: np.linalg.norm(centers[i] - target))
        for i in members:
            if i == best:
                continue
            if np.linalg.norm(centers[i] - centers[best]) <= dup_max_dist:
                keep[i] = False
                n_duplicates += 1

    info = {
        "pitch_px": float(pitch_px),
        "row_band_sizes": row_counts,
        "col_band_sizes": col_counts,
        "n_rows_kept": len(keep_r),
        "n_cols_kept": len(keep_c),
        "n_duplicates_dropped": n_duplicates,
        "n_dropped": int((~keep).sum()),
    }
    return keep, row_idx, col_idx, info


def _median_nn_distance(points: np.ndarray) -> float:
    """Median nearest-neighbour distance — the marker lattice pitch in pixels.

    Returns 0.0 when there are too few points to define a spacing.
    """
    if len(points) < 2:
        return 0.0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def reference_marker_set(
    reference_centers: np.ndarray,
    frame_shape,
    border_fraction: float = 0.15,
    grid_shape=None,
):
    """
    Reduce a reference capture's detections to the canonical marker set.

    Filtering here rather than on a working frame is what makes every consumer
    agree: the reference is captured with the gel undistorted, so its lattice is
    clean and the row/column bands separate properly. The same filter applied to
    a frame where the tool is already pressed can merge two deformed columns into
    one band (measured on real data), which is both less accurate and — since the
    mask and the flow analysis run on different frames — a source of disagreement
    between them.

    Args:
        reference_centers: (N, 2) (row, col) detections on the reference frame.
        frame_shape: (H, W) or an image whose shape gives it.
        border_fraction / grid_shape: as elsewhere.

    Returns:
        centers: (M, 2) (row, col) canonical markers, in a fixed order.
        info: dict from fit_marker_grid, plus n_border_excluded.
    """
    shape = getattr(frame_shape, "shape", frame_shape)
    H, W = shape[0], shape[1]

    keep, _, _, info = fit_marker_grid(reference_centers, grid_shape)
    centers = np.asarray(reference_centers, dtype=np.float32)[keep]
    n_after_grid = len(centers)

    inner = inner_region_mask(centers, H, W, border_fraction)
    centers = centers[inner]

    info = dict(info)
    info["n_off_grid"] = int((~keep).sum())
    info["n_border_excluded"] = n_after_grid - len(centers)
    return centers, info


def locate_reference_markers(
    frame_bgr: np.ndarray,
    reference_frame_bgr: np.ndarray,
    reference_centers: np.ndarray,
    max_match_dist_px: float = 5.0,
    **tracker_kwargs,
):
    """
    Find where a fixed set of reference markers sits in frame_bgr.

    Unlike detect_markers_with_fallback, this NEVER invents markers: the output
    has exactly one entry per reference marker, in the same order. Fresh blob
    detections that do not correspond to a reference marker are ignored rather
    than appended — they are the strays (vignette, housing edge, split blobs)
    that made the mask and the flow analysis disagree about how many markers
    there are.

    Each marker is placed by LK tracking from the reference, then refined to a
    fresh detection when one lies within max_match_dist_px (fresh centroids are
    more precise than a tracked estimate).

    Returns:
        positions: (M, 2) float32 (row, col) in frame_bgr.
        source:    (M,) "detected" | "tracked" — how each position was obtained.
        found:     (M,) bool — False where LK lost the marker entirely; its
                   position falls back to the reference position.
        marker_mask: the fresh blob mask, for visualisation.
    """
    reference_centers = np.asarray(reference_centers, dtype=np.float32)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detected = MarkerTracker(np.float32(frame_rgb) / 255.0, **tracker_kwargs)

    tracked, found = track_markers_lk(
        reference_frame_bgr, reference_centers, frame_bgr
    )

    positions = tracked.copy()
    source = np.full(len(reference_centers), "tracked", dtype=object)
    if len(detected.centers):
        for i in range(len(positions)):
            d = np.linalg.norm(detected.centers - positions[i], axis=1)
            j = int(np.argmin(d))
            if d[j] <= max_match_dist_px:
                positions[i] = detected.centers[j]
                source[i] = "detected"

    return positions, source, found, detected.marker_mask


class MarkerFlowTracker:
    """
    Incremental Lucas-Kanade tracking of gel markers, with cumulative displacement
    reported against the seed frame.

    Single implementation shared by compute_marker_flow (saved trials),
    TactileFlowGuard (live safety guard) and tactile_displacement_live (live
    preview). Those three previously each had their own copy and had drifted
    apart, each getting a different part right; the combination below is what
    none of them had individually:

      1. Fallback seeding — a known-good reference capture recovers markers that
         fresh blob detection drops on the seed frame. With a tool pressed into
         the gel this is worth ~25 of 63 markers, i.e. ~40% of the sensor area
         that would otherwise go unmonitored. Falls back to plain detection when
         no reference is supplied.
      2. Incremental LK — tracks frame-to-frame, never from an ageing reference.
         LK is a small-displacement algorithm: matching a stale reference against
         a heavily sheared frame silently locks onto the wrong marker.
      3. Correct carry-forward — a marker LK drops keeps its LAST TRACKED
         position as the next search point, not its seed position. Resetting to
         the seed makes the next frame search a deformed image from a stale
         origin, which is how the wrong-marker lock happens in the first place.
      4. Forward-backward validation — LK reports status=1 for a confident match
         onto the WRONG marker, so status alone cannot detect a mis-lock.
         Tracking back and requiring the point to return within fb_max_error_px
         catches mis-locks caused by a marker being occluded or losing contrast.
      5. Step-plausibility bound — the safeguard that actually covers a marker
         GRID. Markers sit on a regular lattice, so a frame-to-frame step of
         roughly one marker pitch is indistinguishable from a correct match:
         forward and backward tracking are then BOTH self-consistent and the
         forward-backward check passes a wrong answer (measured: 15.7 px mean
         error across all markers, only 2 of 14 frames flagged). A real marker
         cannot cross half the distance to its neighbour between frames, so any
         match that does is rejected. max_step_px defaults to half the measured
         marker pitch.

      6. Neighbour-guided recovery — without it, a rejected marker's search point
         freezes while the real marker keeps moving, so it can never re-acquire
         and under-reports for the rest of the run (measured on real data: two
         rejected frames left one marker reading 3.6 px where the truth was
         ~10.8 px). Under-reporting is the dangerous direction for a safety
         guard. The gel deforms continuously, so a lost marker's search point is
         advanced by the median motion of its nearest successfully-tracked
         neighbours, keeping it near the real marker until LK can pick it up
         again. Its REPORTED displacement still stays frozen — the prediction
         moves the search window, it is never published as a measurement.

    Together these bound the error rather than eliminate it. If markers are
    genuinely moving faster than ~half a pitch per frame, no per-frame matcher
    can disambiguate them — the tracker reports those markers as lost instead of
    guessing, which is the signal that the capture rate is too low for the motion.

    Known limit: a perfectly regular lattice displaced by exactly one full pitch
    in a single frame is genuinely ambiguous. Forward and backward tracking are
    both self-consistent and every check above passes, reporting zero motion.
    Only sampling fast enough relative to the motion avoids this.

    Displacement convention: (dx, dy) = (delta column, delta row), matching all
    existing call sites. Positions are exposed in both orders because callers
    disagree: initial_positions_xy is (col, row) as OpenCV wants, and
    initial_positions_rc is (row, col) as MarkerTracker returns.

    Usage:
        tr = MarkerFlowTracker(frame0, reference_frame_bgr=ref, reference_centers=rc)
        for frame in frames[1:]:
            disp, found = tr.update(frame)   # disp (N,2) cumulative, found (N,) bool
    """

    def __init__(
        self,
        seed_frame_bgr: np.ndarray,
        reference_frame_bgr: Optional[np.ndarray] = None,
        reference_centers: Optional[np.ndarray] = None,
        border_fraction: float = 0.15,
        grid_shape=None,
        marker_binary_threshold: Optional[float] = None,
        fb_max_error_px: Optional[float] = 1.0,
        max_step_px: Optional[float] = None,
        neighbour_recovery_k: int = 4,
        anchor_to_reference: bool = True,
        lk_params: Optional[dict] = None,
        max_match_dist_px: float = 5.0,
    ):
        """
        Args:
            seed_frame_bgr: frame whose marker positions become the zero-displacement
                origin. Everything is measured relative to this.
            reference_frame_bgr / reference_centers: known-good capture (e.g. taken
                with the gel undistorted) used to recover markers fresh detection
                misses on the seed frame. Both must be given for fallback to apply.
            border_fraction: exclude markers seeded within this fraction of an edge.
                Applied ONCE at seed time — a marker that starts inside keeps being
                tracked even if it later moves into the border.
            grid_shape: (n_rows, n_cols) of the printed marker lattice — (7, 9) for
                a GelSight Mini — or "auto". Drops detections that do not share a
                row and a column with the rest: vignette blobs, scratches, housing
                edges. Complements border_fraction rather than duplicating it, since
                a stray just inside the border cutoff survives the border filter
                while a real outer-grid marker just outside it does not.
                None disables lattice filtering.
            marker_binary_threshold: override MarkerTracker's blob threshold.
            fb_max_error_px: forward-backward tolerance. A point must track back to
                within this many pixels of where it started or it is treated as
                lost. None disables the check (roughly halves the per-frame cost).
            max_step_px: reject any match that moves further than this in one
                frame — see note 5 above. None (default) derives it as half the
                measured marker pitch, which is the largest step that cannot be
                confused with the neighbouring marker. Pass a float to override,
                or math.inf to disable.
            neighbour_recovery_k: how many nearest tracked neighbours vote on
                where a lost marker has drifted to — see note 6 above. 0 disables
                recovery, leaving lost markers frozen in place.
            anchor_to_reference: when a reference is supplied, take the marker set
                from it (filtered once, while the gel is undistorted) instead of
                from whatever blob detection finds on the seed frame. This is what
                makes the marker mask and the flow analysis agree — both then
                describe the same markers in the same order, even though they run
                on different frames. Ignored when no reference is given.
            lk_params: cv2.calcOpticalFlowPyrLK params.
            max_match_dist_px: fresh detection within this distance of an LK-tracked
                reference point counts as the same marker.
        """
        self.lk_params = lk_params if lk_params is not None else _DEFAULT_LK_PARAMS
        self.fb_max_error_px = fb_max_error_px
        self.border_fraction = border_fraction
        self.neighbour_recovery_k = neighbour_recovery_k
        self._max_step_px_arg = max_step_px

        tracker_kwargs = (
            {} if marker_binary_threshold is None
            else {"binary_threshold": marker_binary_threshold}
        )

        has_reference = (
            reference_frame_bgr is not None
            and reference_centers is not None
            and len(reference_centers) > 0
        )
        self.anchored = bool(has_reference and anchor_to_reference)

        if self.anchored:
            # Canonical marker set, defined once on the undistorted reference.
            ref_set, self.grid_info = reference_marker_set(
                reference_centers, seed_frame_bgr, border_fraction, grid_shape,
            )
            centers, source, _found, mask = locate_reference_markers(
                seed_frame_bgr, reference_frame_bgr, ref_set,
                max_match_dist_px=max_match_dist_px, **tracker_kwargs,
            )
            self.reference_set = ref_set
            self.n_recovered = int(np.sum(source == "tracked"))
            self.n_detected_total = len(reference_centers)
            self.n_off_grid = self.grid_info.get("n_off_grid", 0)
            self.n_border_excluded = self.grid_info.get("n_border_excluded", 0)
            self.marker_mask = mask
            self._finish_seed(centers, seed_frame_bgr)
            return

        if has_reference:
            centers, source, mask = detect_markers_with_fallback(
                seed_frame_bgr, reference_frame_bgr, reference_centers,
                max_match_dist_px=max_match_dist_px, **tracker_kwargs,
            )
            self.n_recovered = int(np.sum(source == "tracked"))
        else:
            frame_rgb = cv2.cvtColor(seed_frame_bgr, cv2.COLOR_BGR2RGB)
            t = MarkerTracker(np.float32(frame_rgb) / 255.0, **tracker_kwargs)
            centers, mask = t.centers, t.marker_mask
            self.n_recovered = 0

        H, W = seed_frame_bgr.shape[:2]
        self.n_detected_total = len(centers)

        # Lattice filter first, while whole rows and columns are still present —
        # the border filter would thin them out and blur the band structure.
        on_grid, _, _, self.grid_info = fit_marker_grid(centers, grid_shape)
        centers = centers[on_grid]
        self.n_off_grid = int((~on_grid).sum())

        inner = inner_region_mask(centers, H, W, border_fraction)
        centers = centers[inner]
        self.n_border_excluded = len(on_grid[on_grid]) - len(centers)
        self.marker_mask = mask
        self.reference_set = None
        self._finish_seed(centers, seed_frame_bgr)

    def _finish_seed(self, centers, seed_frame_bgr):
        """Common state setup once the seed marker set has been decided."""
        if len(centers) == 0:
            raise ValueError(
                "No markers seeded inside the inner region. Check border_fraction, "
                "sensor contact, or marker_binary_threshold."
            )

        # OpenCV LK works in (x, y) = (col, row); MarkerTracker gives (row, col).
        self._p0_init = np.array(
            [[c[1], c[0]] for c in centers], dtype=np.float32
        ).reshape(-1, 1, 2)
        self._p0 = self._p0_init.copy()
        self._prev_gray = _to_gray_u8(seed_frame_bgr)

        self.n_markers = len(centers)
        self.initial_positions_xy = self._p0_init.reshape(-1, 2).copy()
        self.initial_positions_rc = centers.astype(np.float32).copy()

        # Marker pitch = median nearest-neighbour distance on the seeded lattice.
        # Half of it is the largest single-frame step that cannot be mistaken for
        # the adjacent marker.
        self.marker_pitch_px = _median_nn_distance(self.initial_positions_xy)
        if self._max_step_px_arg is None:
            self.max_step_px = (
                0.5 * self.marker_pitch_px if self.marker_pitch_px > 0 else float("inf")
            )
        else:
            self.max_step_px = self._max_step_px_arg
        # Cumulative (dx, dy) per marker; lost markers hold their last known value.
        self.displacement = np.zeros((self.n_markers, 2), dtype=np.float32)
        self.found = np.ones(self.n_markers, dtype=bool)
        # Consecutive frames each marker has gone untracked — a marker with a long
        # streak is reporting a stale displacement, however plausible it looks.
        self.lost_streak = np.zeros(self.n_markers, dtype=int)

        # Neighbour order by seed geometry, for recovery. Row i lists the other
        # markers nearest to marker i.
        if self.neighbour_recovery_k > 0 and self.n_markers > 1:
            dist = np.linalg.norm(
                self.initial_positions_xy[:, None, :]
                - self.initial_positions_xy[None, :, :], axis=2
            )
            np.fill_diagonal(dist, np.inf)
            self._nn_order = np.argsort(dist, axis=1)
        else:
            self._nn_order = None

    def _lk(self, gray_a, gray_b, pts):
        return cv2.calcOpticalFlowPyrLK(gray_a, gray_b, pts, None, **self.lk_params)

    def update(self, frame_bgr: np.ndarray):
        """
        Track one more frame.

        Returns:
            displacement: (N, 2) float32 cumulative (dx, dy) from the seed frame.
                Markers not tracked this frame retain their previous value rather
                than reporting zero or a bogus jump.
            found: (N,) bool — True where this frame's tracking was accepted.
        """
        cur_gray = _to_gray_u8(frame_bgr)
        p1, st, _ = self._lk(self._prev_gray, cur_gray, self._p0)

        if p1 is None or st is None:
            self.found = np.zeros(self.n_markers, dtype=bool)
            self._prev_gray = cur_gray
            return self.displacement.copy(), self.found.copy()

        good = st.ravel() == 1

        # Step-plausibility: a marker cannot travel half way to its neighbour in
        # one frame. Rejecting such matches is what stops a lattice of markers
        # aliasing onto each other, which the forward-backward check below cannot
        # see (both directions agree when the step is a whole pitch).
        if np.isfinite(self.max_step_px) and np.any(good):
            step = np.linalg.norm(
                p1.reshape(-1, 2) - self._p0.reshape(-1, 2), axis=1
            )
            good &= step <= self.max_step_px

        # Forward-backward check: a correct match tracks back to where it came
        # from. Catches mis-locks from occlusion or lost contrast.
        if self.fb_max_error_px is not None and np.any(good):
            p0r, st_r, _ = self._lk(cur_gray, self._prev_gray, p1)
            if p0r is not None and st_r is not None:
                fb_err = np.linalg.norm(
                    p0r.reshape(-1, 2) - self._p0.reshape(-1, 2), axis=1
                )
                good &= (st_r.ravel() == 1) & (fb_err <= self.fb_max_error_px)
            else:
                good[:] = False

        if np.any(good):
            all_xy = p1.reshape(-1, 2)
            prev_xy = self._p0.reshape(-1, 2)
            self.displacement[good] = (
                all_xy[good] - self._p0_init.reshape(-1, 2)[good]
            )

            # Carry the last tracked position forward for everything else.
            p0_new = prev_xy.copy()
            p0_new[good] = all_xy[good]

            # Advance lost markers by what their neighbours just did, so their
            # search window follows the deforming gel instead of freezing. This
            # only moves where we LOOK next frame; displacement stays frozen.
            if self._nn_order is not None and not np.all(good):
                frame_delta = all_xy[good] - prev_xy[good]
                good_idx = np.flatnonzero(good)
                rank = np.full(self.n_markers, -1, dtype=int)
                rank[good_idx] = np.arange(len(good_idx))
                for i in np.flatnonzero(~good):
                    nbrs = [j for j in self._nn_order[i] if good[j]]
                    if not nbrs:
                        continue
                    take = nbrs[: self.neighbour_recovery_k]
                    p0_new[i] = prev_xy[i] + np.median(
                        frame_delta[rank[take]], axis=0
                    )

            self._p0 = p0_new.reshape(-1, 1, 2)

        self._prev_gray = cur_gray
        self.found = good
        self.lost_streak = np.where(good, 0, self.lost_streak + 1)
        return self.displacement.copy(), good.copy()

    @property
    def positions_xy(self) -> np.ndarray:
        """Current (col, row) position of every marker, (N, 2)."""
        return self._p0.reshape(-1, 2).copy()

    def max_displacement(self):
        """(max norm, max |dx|, max |dy|) over markers tracked at least once."""
        d = self.displacement
        if len(d) == 0:
            return 0.0, 0.0, 0.0
        return (
            float(np.linalg.norm(d, axis=1).max()),
            float(np.abs(d[:, 0]).max()),
            float(np.abs(d[:, 1]).max()),
        )


def detect_markers_with_fallback(
    frame_bgr: np.ndarray,
    reference_frame_bgr: Optional[np.ndarray] = None,
    reference_centers: Optional[np.ndarray] = None,
    max_match_dist_px: float = 5.0,
    **tracker_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect markers in frame_bgr, then fill in any reference marker that fresh
    blob detection missed using LK optical flow from reference_frame_bgr /
    reference_centers.

    Use this instead of a bare MarkerTracker() call whenever a known-good
    reference capture is available (e.g. before a tool was mounted, or from an
    earlier clean health check where every marker was detected) — it recovers
    markers that a fixed intensity threshold or the blob-merging area filter
    would otherwise drop outright, without loosening those filters globally
    (which trades missed markers for false positives elsewhere, see
    dev_notes/logs — loosening binary_threshold made real cases worse).

    Args:
        frame_bgr: current frame (BGR uint8) to detect markers in.
        reference_frame_bgr: a frame (BGR uint8) where reference_centers is
            known-good. If None (or reference_centers is empty), this is
            equivalent to a plain MarkerTracker(...) call — no fallback.
        reference_centers: (N, 2) (row, col) positions on reference_frame_bgr.
        max_match_dist_px: a fresh detection within this distance of an
            LK-tracked reference point counts as the same marker (the fresh
            detection wins — more precise); otherwise the LK-tracked position
            is used as a fallback.
        **tracker_kwargs: forwarded to MarkerTracker (e.g. binary_threshold).

    Returns:
        centers: (M, 2) float32 (row, col) — fresh detections plus any
            reference markers recovered via fallback tracking.
        source: (M,) array of "detected" | "tracked", one per row of centers.
        marker_mask: the fresh MarkerTracker's binary mask (blob-detected
            markers only — fallback markers aren't blobs, so they aren't part
            of this mask).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_f32 = np.float32(frame_rgb) / 255.0
    tracker = MarkerTracker(img_f32, **tracker_kwargs)
    detected = tracker.centers

    if (
        reference_frame_bgr is None
        or reference_centers is None
        or len(reference_centers) == 0
    ):
        source = np.full(len(detected), "detected", dtype=object)
        return detected, source, tracker.marker_mask

    tracked, found = track_markers_lk(reference_frame_bgr, reference_centers, frame_bgr)

    fallback = []
    for i in range(len(tracked)):
        if not found[i]:
            continue
        if len(detected):
            if np.linalg.norm(detected - tracked[i], axis=1).min() <= max_match_dist_px:
                continue  # a fresh detection already covers this marker
        fallback.append(tracked[i])

    if fallback:
        fallback_arr = np.array(fallback, dtype=np.float32)
        centers = np.vstack([detected, fallback_arr]) if len(detected) else fallback_arr
        source = np.array(
            ["detected"] * len(detected) + ["tracked"] * len(fallback_arr), dtype=object
        )
    else:
        centers = detected
        source = np.full(len(detected), "detected", dtype=object)

    return centers, source, tracker.marker_mask
