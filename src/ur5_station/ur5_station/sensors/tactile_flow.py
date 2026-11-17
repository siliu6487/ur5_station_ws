#!/usr/bin/env python3
"""
Marker-flow analysis for saved GelSight Mini image sequences.

Algorithm — all of it lives in MarkerFlowTracker (sensor_utils.marker_tracker),
shared with TactileFlowGuard and tactile_displacement_live so the three cannot
drift apart again:
  1. Seed markers on frame 0, recovering any that blob detection misses via LK
     from a known-good reference capture (pass reference_frame_bgr/centers).
  2. Track frame-to-frame with Lucas-Kanade, validating each match and carrying
     lost markers along with their neighbours.
  3. Displacement = current_position − initial_position  (cumulative, not frame-to-frame).

Usage:
    from ur5_station.sensors.tactile_flow import compute_marker_flow

    result = compute_marker_flow(
        image_dir  = trial_dir / 'tactile_images',
        output_dir = trial_dir / 'tactile_flow',   # optional: write annotated frames
    )
    # result['displacements']  shape (N_frames, N_markers, 2) — (dx, dy) in pixels
    # result['initial_positions']  shape (N_markers, 2) — (x, y) pixel coords
"""
import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ur5_station.sensors.marker_tracker import MarkerFlowTracker

def compute_marker_flow(
    image_dir: Path,
    output_dir: Optional[Path] = None,
    arrow_scale: float = 1.0,  # for debugging, set value > 1.0 to exaggerate small displacements
    arrow_color: tuple = (255, 255, 255),
    border_fraction: float = 0.15,
    grid_shape=None,
    expected_markers: Optional[int] = 63,
    marker_binary_threshold: Optional[float] = None,
    reference_frame_bgr: Optional[np.ndarray] = None,
    reference_centers: Optional[np.ndarray] = None,
    fb_max_error_px: Optional[float] = 1.0,
    max_step_px: Optional[float] = None,
    neighbour_recovery_k: int = 4,
) -> dict:
    """
    Compute cumulative marker displacement for a saved tactile image sequence.

    Markers are seeded only from the inner (non-border) region of frame 0 so that
    border-region noise is not tracked. LK then follows each marker freely across
    the full frame — including into the border if the marker physically moves there.

    Args:
        image_dir:       Path to tactile_images/ (contains tactile_metadata.csv + frame_*.jpg).
        output_dir:      If set, write flow-annotated frames and a summary quiver plot here.
        arrow_scale:     Multiply displacement by this factor before drawing arrows.
        arrow_color:     BGR colour for drawn arrows.
        border_fraction: Fraction of image width/height to exclude when seeding markers
                         (default 0.15 matches the GelSight Mini border).
        marker_binary_threshold: override MarkerTracker's blob-detection threshold (default
                         -0.05). Loosen (move toward 0) if a mounted tool locally reduces
                         marker contrast enough that some markers go undetected on frame 0.
        grid_shape: (n_rows, n_cols) of the printed marker lattice, e.g. (7, 9) for a
                         GelSight Mini, or "auto". Drops detections that are not on the
                         lattice (vignette blobs, housing edges) — which border_fraction
                         alone cannot do, since a stray just inside the cutoff survives it.
        reference_frame_bgr / reference_centers: known-good capture (gel undistorted).
                         STRONGLY recommended — by frame 0 the tool is already pressed
                         into the gel, and fresh blob detection alone can miss ~40% of
                         markers, which then go untracked for the whole trial. With the
                         reference, LK recovers them at seed time.
        fb_max_error_px / max_step_px / neighbour_recovery_k: match-validation and
                         recovery settings, forwarded to MarkerFlowTracker (see there).

    Returns:
        {
          'timestamps':        list[float]          — unix timestamps (or frame indices if no CSV)
          'initial_positions': np.ndarray (N, 2)    — (x, y) pixel positions of each marker
          'displacements':     np.ndarray (F, N, 2) — cumulative (dx, dy) per frame per marker
          'n_markers':         int
          'output_dir':        Path | None
        }
    """
    image_dir = Path(image_dir)

    # ── Load frame list ───────────────────────────────────────────────────────
    csv_path = image_dir / "tactile_metadata.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        frame_paths = [image_dir / r["filename"] for r in rows]
        timestamps = [float(r["timestamp"]) for r in rows]
    else:
        frame_paths = sorted(image_dir.glob("frame_*.jpg"))
        timestamps = list(range(len(frame_paths)))

    if len(frame_paths) < 2:
        raise ValueError(
            f"Need ≥2 frames, found {len(frame_paths)} in {image_dir}"
        )

    # ── Seed markers on frame 0 ──────────────────────────────────────────────
    first_bgr = cv2.imread(str(frame_paths[0]))
    if first_bgr is None:
        raise FileNotFoundError(f"Could not read {frame_paths[0]}")

    # Shared tracker: fallback seeding + incremental LK + validated matching.
    # See MarkerFlowTracker for why each safeguard is there.
    tracker = MarkerFlowTracker(
        first_bgr,
        reference_frame_bgr=reference_frame_bgr,
        reference_centers=reference_centers,
        border_fraction=border_fraction,
        grid_shape=grid_shape,
        marker_binary_threshold=marker_binary_threshold,
        fb_max_error_px=fb_max_error_px,
        max_step_px=max_step_px,
        neighbour_recovery_k=neighbour_recovery_k,
    )
    n_markers = tracker.n_markers
    warnings = []

    # ── Marker count sanity checks ────────────────────────────────────────────
    n_all = tracker.n_detected_total
    n_border = tracker.n_border_excluded
    print(f"[tactile_flow] seeded {n_all} markers total "
          f"({tracker.n_recovered} recovered from reference), "
          f"{n_border} in border (excluded), {n_markers} tracked")
    if reference_centers is None:
        print("[tactile_flow] NOTE: no reference frame supplied — markers that fresh "
              "detection misses on frame 0 (e.g. under a mounted tool) are not tracked")

    if expected_markers is not None and n_markers > expected_markers:
        msg = (f"seeded {n_markers} markers but expected ≤{expected_markers}; "
               f"consider increasing border_fraction (currently {border_fraction})")
        warnings.append(msg)
        print(f"[tactile_flow] WARNING: {msg}")

    # initial_positions stored as (N, 2) in (x, y) order for downstream use
    initial_positions = tracker.initial_positions_xy

    # ── Track through sequence ────────────────────────────────────────────────
    n_frames = len(frame_paths)
    displacements = np.zeros((n_frames, n_markers, 2), dtype=np.float32)
    # frame 0 → all zeros (by definition)
    lost_per_frame = np.zeros(n_frames, dtype=int)   # how many markers lost each frame

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Save annotated frame 0 (no arrows yet — just marker dots)
        vis0 = first_bgr.copy()
        for cx, cy in initial_positions:
            cv2.circle(vis0, (int(cx), int(cy)), 3, arrow_color, -1)
        cv2.imwrite(str(Path(output_dir) / "flow_000000.jpg"), vis0)

    for i, fpath in enumerate(frame_paths[1:], start=1):
        frame_bgr = cv2.imread(str(fpath))
        if frame_bgr is None:
            displacements[i] = displacements[i - 1]
            continue

        # Lost markers keep their previous displacement inside update(), so the
        # returned array is always full length and never reports a bogus zero.
        disp, good_mask = tracker.update(frame_bgr)
        displacements[i] = disp

        n_lost = int((~good_mask).sum())
        lost_per_frame[i] = n_lost
        if n_lost > 0:
            lost_indices = np.where(~good_mask)[0].tolist()
            warnings.append(f"frame {i}: lost {n_lost} marker(s) {lost_indices}")

        if output_dir is not None:
            vis = frame_bgr.copy()
            for (cx, cy), (dx, dy), ok in zip(
                initial_positions, disp, good_mask
            ):
                x0, y0 = int(cx), int(cy)
                if not ok:
                    # Stale value — mark it rather than drawing an arrow that
                    # looks as trustworthy as a tracked one.
                    cv2.circle(vis, (x0, y0), 4, (0, 0, 255), 1)
                    continue
                x1 = int(cx + arrow_scale * dx)
                y1 = int(cy + arrow_scale * dy)
                cv2.arrowedLine(
                    vis, (x0, y0), (x1, y1), arrow_color,
                    thickness=1, line_type=cv2.LINE_AA, tipLength=0.25,
                )
            cv2.imwrite(str(Path(output_dir) / f"flow_{i:06d}.jpg"), vis)

    max_simultaneous_lost = int(lost_per_frame.max())
    if max_simultaneous_lost > 0:
        print(f"[tactile_flow] {(lost_per_frame > 0).sum()} frame(s) had lost markers "
              f"(max simultaneous: {max_simultaneous_lost})")

    result = {
        "timestamps":            timestamps,
        "initial_positions":     initial_positions,
        "displacements":         displacements,
        "n_markers":             n_markers,
        "lost_per_frame":        lost_per_frame,   # np.ndarray (F,) — lost count per frame
        "max_simultaneous_lost": max_simultaneous_lost,
        "warnings":              warnings,
        "output_dir":            Path(output_dir) if output_dir else None,
        "image_dir":             image_dir,
        "frame_paths":           frame_paths,
    }

    if output_dir is not None:
        np.save(str(Path(output_dir) / "displacements.npy"), displacements)
        _save_flow_summary(result, Path(output_dir))
        _save_displacement_plots(result, Path(output_dir))

    return result


def check_flow_anomalies(
    result: dict,
    min_alarm_markers: int = 1,
    deviation_threshold_px: float = 5.0,
    n_neighbors: int = 6,
    min_displacement_px: float = 2.0,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Flag frames where markers have displacements inconsistent with their spatial
    neighbours.  An alarm fires when >= min_alarm_markers anomalous markers
    appear in the same frame.

    Frames where overall motion is below min_displacement_px are treated as
    at-rest and skipped — tiny deviations there are noise, not anomalies.

    A marker is anomalous if its displacement deviates from a local linear
    deformation trend fitted through its n_neighbors nearest neighbours.
    Using a linear fit (rather than the neighbour median) removes directional
    bias for edge/corner markers: the gradient is extrapolated correctly, so a
    top-row marker that legitimately moves more than its below-only neighbours
    will not be flagged.  Only genuine tracking jumps — which break the smooth
    local trend — produce large residuals.

    Args:
        result:                 Output dict from compute_marker_flow.
        min_alarm_markers:      Number of concurrent anomalies needed to trigger an alarm.
        deviation_threshold_px: Residual from the local linear fit (px) that flags a marker.
        n_neighbors:            Neighbours used to fit the local linear trend (min 3).
        min_displacement_px:    Frames where max marker displacement is below this value
                                are skipped (sensor at rest / noise regime).
        output_dir:             If set, save anomaly_metadata.json and annotated images here.
                                Suggested: flow_output_dir / 'anomalies'.
        verbose:                Print alarm summary to stdout.

    Returns:
        {
          'alarm_frames':      list[int]          — frame indices with >= min_alarm_markers anomalies
          'n_anomalous':       np.ndarray (F,)    — anomalous marker count per frame
          'anomalous_markers': np.ndarray (F, N) bool
          'deviation':         np.ndarray (F, N)  — residual from local linear fit (px)
          'skipped_frames':    np.ndarray (F,) bool — frames below min_displacement_px
        }
    """
    import json

    initial_positions = result["initial_positions"]  # (N, 2) x,y
    displacements     = result["displacements"]       # (F, N, 2)
    frame_paths       = result.get("frame_paths", [])
    N = len(initial_positions)

    # ── Spatial neighbour graph (fixed, based on initial marker positions) ────
    pairwise = np.linalg.norm(
        initial_positions[:, None, :] - initial_positions[None, :, :], axis=2
    )  # (N, N)
    np.fill_diagonal(pairwise, np.inf)
    k = max(min(n_neighbors, N - 1), 3)  # need at least 3 to fit a linear plane
    neighbor_idx = np.argsort(pairwise, axis=1)[:, :k]  # (N, k)

    # ── Pass 1: rough outlier mask via neighbour median ──────────────────────
    # The median is resistant to up to k/2 bad values, so one bad neighbour
    # cannot corrupt it.  We use 2× the final threshold so only gross tracking
    # failures are marked here — edge-gradient bias stays well below 2×.
    nbr_disp_all  = displacements[:, neighbor_idx, :]          # (F, N, k, 2)
    median_disp   = np.median(nbr_disp_all, axis=2)            # (F, N, 2)
    rough_dev     = np.linalg.norm(displacements - median_disp, axis=2)  # (F, N)
    rough_outlier = rough_dev > 2.0 * deviation_threshold_px   # (F, N) bool

    # ── Pass 2: local linear fit with outlier neighbours excluded ─────────────
    # For each marker j and frame f, remove any neighbour flagged in pass 1
    # before fitting  disp = a + b*x + c*y.  This breaks the contamination
    # chain: a bad marker cannot bias its neighbours' residuals.
    # Fast vectorised path for frames where all neighbours are clean; slow
    # per-frame path only when some neighbours are outliers (expected to be rare).
    F = len(displacements)
    deviation = np.zeros((F, N), dtype=np.float32)
    for j in range(N):
        nbr     = neighbor_idx[j]                            # (k,)
        pos_nbr = initial_positions[nbr]                     # (k, 2)
        A_full  = np.column_stack([np.ones(k), pos_nbr])    # (k, 3)
        pvec    = np.array([1., initial_positions[j, 0], initial_positions[j, 1]])

        has_bad_nbr = rough_outlier[:, nbr].any(axis=1)     # (F,) bool

        # Fast path — no outlier neighbours: vectorise over all such frames at once
        clean_frames = np.where(~has_bad_nbr)[0]
        if len(clean_frames):
            Bx = displacements[clean_frames][:, nbr, 0].T   # (k, |clean_frames|)
            By = displacements[clean_frames][:, nbr, 1].T
            Cx, _, _, _ = np.linalg.lstsq(A_full, Bx, rcond=None)  # (3, |clean_frames|)
            Cy, _, _, _ = np.linalg.lstsq(A_full, By, rcond=None)
            deviation[clean_frames, j] = np.sqrt(
                (displacements[clean_frames, j, 0] - pvec @ Cx) ** 2 +
                (displacements[clean_frames, j, 1] - pvec @ Cy) ** 2
            )

        # Slow path — exclude outlier neighbours per frame
        for f in np.where(has_bad_nbr)[0]:
            clean   = ~rough_outlier[f, nbr]                # (k,) which nbrs are ok
            n_clean = int(clean.sum())
            if n_clean < 3:                                  # too few to fit — fall back
                deviation[f, j] = rough_dev[f, j]
                continue
            A = np.column_stack([np.ones(n_clean), pos_nbr[clean]])
            Cx, _, _, _ = np.linalg.lstsq(A, displacements[f, nbr[clean], 0], rcond=None)
            Cy, _, _, _ = np.linalg.lstsq(A, displacements[f, nbr[clean], 1], rcond=None)
            pred = np.array([pvec @ Cx, pvec @ Cy])
            deviation[f, j] = np.linalg.norm(displacements[f, j] - pred)

    # ── Skip frames where the sensor is essentially at rest ───────────────────
    frame_max_disp = np.linalg.norm(displacements, axis=2).max(axis=1)  # (F,)
    skipped_frames = frame_max_disp < min_displacement_px                # (F,) bool

    anomalous   = (deviation > deviation_threshold_px) & ~skipped_frames[:, None]
    n_anomalous = anomalous.sum(axis=1)               # (F,)
    alarm_frames = sorted(np.where(n_anomalous >= min_alarm_markers)[0].tolist())

    if verbose:
        n_skipped = int(skipped_frames.sum())
        if n_skipped:
            print(f"[tactile_flow] {n_skipped} frame(s) skipped "
                  f"(max displacement < {min_displacement_px:.1f} px)")
        if alarm_frames:
            print(
                f"[tactile_flow] ANOMALY ALARM: {len(alarm_frames)} frame(s) with "
                f">={min_alarm_markers} anomalous markers "
                f"(threshold={deviation_threshold_px:.1f} px, k={k} neighbours, local-linear fit)"
            )
            for f in alarm_frames:
                bad = np.where(anomalous[f])[0].tolist()
                print(
                    f"  frame {f:5d}: {len(bad)} anomalous marker(s) "
                    f"indices={bad}  max_dev={deviation[f, bad].max():.1f} px"
                )
        else:
            print(
                f"[tactile_flow] no anomalies detected "
                f"(threshold={deviation_threshold_px:.1f} px, "
                f"min_alarm_markers={min_alarm_markers})"
            )

    # ── Optional: save metadata + annotated images ────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        meta = {
            "settings": {
                "min_alarm_markers":      min_alarm_markers,
                "deviation_threshold_px": deviation_threshold_px,
                "n_neighbors":            k,
                "min_displacement_px":    min_displacement_px,
            },
            "n_alarm_frames":  len(alarm_frames),
            "n_skipped_frames": int(skipped_frames.sum()),
            "alarm_frames": {
                str(f): {
                    "n_anomalous":    int(n_anomalous[f]),
                    "marker_indices": np.where(anomalous[f])[0].tolist(),
                    "deviations_px":  [round(float(v), 2)
                                       for v in deviation[f, anomalous[f]]],
                    "frame_max_disp_px": round(float(frame_max_disp[f]), 2),
                }
                for f in alarm_frames
            },
        }
        with open(out / "anomaly_metadata.json", "w") as fp:
            json.dump(meta, fp, indent=2)

        # Annotated images: flow arrow (white) + red ring on anomalous markers
        for f in alarm_frames:
            if f >= len(frame_paths):
                continue
            frame_bgr = cv2.imread(str(frame_paths[f]))
            if frame_bgr is None:
                continue

            curr_positions = initial_positions + displacements[f]  # (N, 2)
            for j in range(N):
                x0, y0 = int(initial_positions[j, 0]), int(initial_positions[j, 1])
                x1, y1 = int(curr_positions[j, 0]),    int(curr_positions[j, 1])
                cv2.arrowedLine(frame_bgr, (x0, y0), (x1, y1),
                                (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.25)
                if anomalous[f, j]:
                    cv2.circle(frame_bgr, (x1, y1), 9, (0, 0, 255), 2)

            cv2.imwrite(str(out / f"anomaly_{f:06d}.jpg"), frame_bgr)

        print(f"[tactile_flow] anomaly output saved to {out}  "
              f"({len(alarm_frames)} annotated frame(s))")

    return {
        "alarm_frames":      alarm_frames,
        "n_anomalous":       n_anomalous,
        "anomalous_markers": anomalous,
        "deviation":         deviation,
        "skipped_frames":    skipped_frames,
    }


def _save_flow_summary(result: dict, output_dir: Path) -> None:
    """Quiver plot of final-frame marker displacement + magnitude heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    displacements      = result["displacements"]      # (F, N, 2)
    initial_positions  = result["initial_positions"]  # (N, 2) as (x, y)
    timestamps         = result["timestamps"]
    t0                 = timestamps[0] if timestamps else 0.0

    x = initial_positions[:, 0]
    y = initial_positions[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: quiver of final displacement ───────────────────────────────────
    ax = axes[0]
    final = displacements[-1]   # (N, 2)
    u, v  = final[:, 0], final[:, 1]
    mag   = np.linalg.norm(final, axis=1)
    q = ax.quiver(x, y, u, -v, mag, cmap="hot", scale=None,
                  scale_units="xy", angles="xy", width=0.004)
    plt.colorbar(q, ax=ax, label="displacement (px)")
    ax.scatter(x, y, s=8, c="white", zorder=5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_facecolor("#111")
    ax.set_title(f"Final displacement  (frame {len(displacements)-1})")

    # ── right: max magnitude over time ────────────────────────────────────────
    ax2 = axes[1]
    mag_over_time = np.linalg.norm(displacements, axis=2).max(axis=1)  # (F,)
    t_rel = [ts - t0 for ts in timestamps]
    ax2.plot(t_rel, mag_over_time, linewidth=1.5, color="tab:orange")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("max marker displacement (px)")
    ax2.set_title("Peak displacement over time")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Marker flow — {result['n_markers']} markers  "
        f"{len(displacements)} frames",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(str(output_dir / "flow_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _symmetric_ylim(data: np.ndarray, floor: float, margin: float = 0.02) -> tuple:
    """Zero-centered y-limits: +/-(max(|data.min()|, |data.max()|, floor) + margin).

    Keeps 0.0 in the middle of the panel always. `floor` sets the minimum half
    -range so a trial with only sub-pixel jitter doesn't auto-zoom the axis in
    on the noise; a trial with real motion still shows its full (symmetric)
    range. `margin` pads both edges a bit further so the plotted min/max value
    never touches the axis border."""
    lim = max(abs(float(data.min())), abs(float(data.max())), floor) + margin
    return -lim, lim


def _save_displacement_plots(
    result: dict, output_dir: Path, y_axis_floor: float = 0.3
) -> None:
    """Per-marker dx, dy, and cumulative x/y drift over time, with behavior events
    as vertical lines. Data is plotted raw (unscaled); each panel's y-axis is
    zero-centered via _symmetric_ylim() (see there).

    dx, dy       — frame-to-frame optical flow: delta vs. the *previous* frame.
    drift_x/y    — cumulative drift vs. *frame 0*: direction + distance from the
                   start of the recording. Both are signed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    import csv as _csv

    displacements = result["displacements"]   # (F, N, 2)
    timestamps    = result["timestamps"]
    t0    = timestamps[0] if timestamps else 0.0
    t_rel = [ts - t0 for ts in timestamps]
    N     = result["n_markers"]

    # ── Load events.csv from the trial dir (parent of tactile_images/) ────────
    events: list[dict] = []
    image_dir = result.get("image_dir")
    if image_dir is not None:
        events_path = Path(image_dir).parent / "events.csv"
        if events_path.exists():
            with open(events_path) as f:
                for row in _csv.DictReader(f):
                    events.append({
                        "t_rel": float(row["timestamp"]) - t0,
                        "name":  row["event"],
                    })

    # Only show movement-start events — drop _done / _stop to avoid clutter
    events = [ev for ev in events if "_begin" in ev["name"] or ev["name"] == "trial_start"]

    def _event_style(name: str):
        if "poke_begin" in name:     return "#ff4444", "-"
        if "approach_begin" in name: return "#4488ff", "--"
        if "retract_begin" in name:  return "#44cc44", "--"
        return "#aaaaaa", ":"

    # ── Build figure ──────────────────────────────────────────────────────────
    cmap = plt.get_cmap("tab20", N)
    bg   = "#1e1e1e"
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.patch.set_facecolor(bg)

    t_start, t_end = t_rel[0], t_rel[-1]

    def _draw_panel(ax, data, ylabel, title, y_min, y_max):
        ax.set_facecolor("#111")
        for j in range(N):
            ax.plot(t_rel, data[:, j], linewidth=0.8, alpha=0.7, color=cmap(j))
        ax.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylim(y_min, y_max)

        # Event vertical lines (only those within the tactile time window ±1 s)
        for ev in events:
            if t_start - 1.0 <= ev["t_rel"] <= t_end + 1.0:
                color, ls = _event_style(ev["name"])
                ax.axvline(ev["t_rel"], color=color, linewidth=0.9,
                           linestyle=ls, alpha=0.75)

        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title, color="white", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.grid(True, alpha=0.2)

    # Cumulative drift vs. frame 0 — displacements is already exactly this.
    drift_x = displacements[:, :, 0]   # (F, N)
    drift_y = displacements[:, :, 1]

    # Frame-to-frame optical flow — delta vs. the previous frame, not frame 0.
    # Frame 0 has no previous frame, so its flow is 0 (prepend duplicates frame 0
    # so the diff there comes out to exactly zero, keeping the array length F).
    flow = np.diff(displacements, axis=0, prepend=displacements[:1])   # (F, N, 2)
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    _draw_panel(
        axes[0], flow_x, "dx (px)",
        f"Marker x-displacement — {N} markers (frame-to-frame)",
        *_symmetric_ylim(flow_x, y_axis_floor),
    )
    _draw_panel(
        axes[1], flow_y, "dy (px)", "Marker y-displacement (frame-to-frame)",
        *_symmetric_ylim(flow_y, y_axis_floor),
    )
    _draw_panel(
        axes[2], drift_x, "drift x (px)", "Cumulative x-drift from frame 0",
        *_symmetric_ylim(drift_x, y_axis_floor),
    )
    _draw_panel(
        axes[3], drift_y, "drift y (px)", "Cumulative y-drift from frame 0",
        *_symmetric_ylim(drift_y, y_axis_floor),
    )

    # Event name labels along the top of the upper axis only
    ax0 = axes[0]
    y_label = ax0.get_ylim()[1]
    for ev in events:
        if t_start - 1.0 <= ev["t_rel"] <= t_end + 1.0:
            color, _ = _event_style(ev["name"])
            ax0.text(ev["t_rel"], y_label, ev["name"],
                     rotation=90, fontsize=8, color=color,
                     va="top", ha="right", alpha=0.9)

    axes[3].set_xlabel("time (s)", color="white")
    axes[3].tick_params(axis="x", colors="white")

    fig.tight_layout()
    fig.savefig(
        str(output_dir / "displacement_xy.png"),
        dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
