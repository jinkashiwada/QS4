"""Step 3: Estimate optical flow within ROIs and visualise surface velocities."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from matplotlib import cm
import math
import re

COLORMAP = cm.get_cmap("jet_r")


@dataclass
class ROI:
    """Axis-aligned region of interest in pixel coordinates."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    def slice(self) -> Tuple[slice, slice]:
        return slice(self.top, self.bottom), slice(self.left, self.right)


@dataclass
class VideoConfig:
    """Configuration entry loaded from the settings CSV."""

    video_path: Path
    roi: Optional[ROI]
    index: int


def discover_candidate_videos() -> List[Path]:
    output_root = Path("output")
    candidates = sorted(output_root.glob("*.mp4"))
    return candidates


def prompt_video_selection(candidates: List[Path]) -> List[Path]:
    if candidates:
        print("Available videos:")
        for idx, path in enumerate(candidates):
            print(f"  [{idx}] {path}")
    else:
        print("No MP4 files found under output/. Please enter paths manually.")

    while True:
        selection = input("Select indices (comma-separated) or enter paths: ").strip()
        if not selection:
            print("Please provide at least one selection.")
            continue

        tokens = [token.strip() for token in re.split(r",|\s+", selection) if token.strip()]
        chosen: List[Path] = []
        invalid = False
        for token in tokens:
            if token.isdigit() and candidates:
                idx = int(token)
                if 0 <= idx < len(candidates):
                    chosen.append(candidates[idx])
                else:
                    print(f"Index {idx} is out of range.")
                    invalid = True
            else:
                path = Path(token)
                if path.exists():
                    chosen.append(path)
                else:
                    print(f"Path not found: {token}")
                    invalid = True
        if invalid or not chosen:
            continue
        return chosen


def initialise_settings(csv_path: Path) -> pd.DataFrame:
    candidates = discover_candidate_videos()
    selected = prompt_video_selection(candidates)
    rows = []
    for path in selected:
        rows.append(
            {
                "video_path": str(path),
                "roi_top": pd.NA,
                "roi_left": pd.NA,
                "roi_bottom": pd.NA,
                "roi_right": pd.NA,
            }
        )
    df = pd.DataFrame(rows, columns=["video_path", "roi_top", "roi_left", "roi_bottom", "roi_right"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Created settings file at {csv_path}")
    return df


def load_video_settings(csv_path: Path) -> Tuple[pd.DataFrame, List[VideoConfig]]:
    if not csv_path.exists():
        df = initialise_settings(csv_path)
    else:
        df = pd.read_csv(csv_path)

    required = ["video_path", "roi_top", "roi_left", "roi_bottom", "roi_right"]
    for column in required:
        if column not in df.columns:
            df[column] = pd.NA

    df["video_path"] = df["video_path"].astype(str)

    configs: List[VideoConfig] = []
    for idx, row in df.iterrows():
        video_path = Path(str(row.get("video_path", "")).strip())
        if not video_path:
            continue
        roi = None
        roi_values = [row.get("roi_top"), row.get("roi_left"), row.get("roi_bottom"), row.get("roi_right")]
        if not pd.isna(roi_values).any():
            roi = ROI(
                top=int(float(row["roi_top"])),
                left=int(float(row["roi_left"])),
                bottom=int(float(row["roi_bottom"])),
                right=int(float(row["roi_right"])),
            )
        configs.append(VideoConfig(video_path=video_path, roi=roi, index=idx))

    return df, configs


def create_valid_pixel_mask(frame: np.ndarray, roi: ROI, threshold: int, neighbor_size: int) -> np.ndarray:
    """Return a boolean mask where True indicates pixels suitable for flow estimation."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, threshold, 255)
    mask = cv2.bitwise_not(mask)
    kernel = np.ones((neighbor_size, neighbor_size), np.uint8)
    dilated = cv2.dilate(mask, kernel)
    valid = cv2.bitwise_not(dilated)
    roi_mask = valid[roi.slice()]
    return roi_mask == 255


def prompt_positive_float(message: str) -> float:
    while True:
        value = input(message).strip()
        try:
            number = float(value)
            if number > 0:
                return number
        except ValueError:
            pass
        print("Please enter a positive number.")


def resolve_pixels_per_meter(args) -> float:
    if args.pixels_per_meter is not None and args.pixels_per_meter > 0:
        return args.pixels_per_meter

    config_path = Path("output/02_region_config.csv")
    if config_path.exists():
        try:
            config_df = pd.read_csv(config_path)
            if not config_df.empty:
                value = float(config_df.iloc[0].get("pix_per_meter", float("nan")))
                if value > 0:
                    print(f"Using pix-per-meter {value} from {config_path}")
                    return value
        except Exception as exc:
            print(f"Warning: failed to read pix-per-meter from {config_path}: {exc}")

    return prompt_positive_float("Enter pix-per-meter for optical flow: ")


def ensure_roi(video_path: Path, existing_roi: Optional[ROI]) -> ROI:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for ROI selection: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame for ROI selection: {video_path}")

    if existing_roi is not None:
        preview = frame.copy()
        cv2.rectangle(
            preview,
            (existing_roi.left, existing_roi.top),
            (existing_roi.right, existing_roi.bottom),
            (0, 255, 0),
            2,
        )
        window_name = f"ROI Preview - {video_path.name}"
        cv2.imshow(window_name, preview)
        cv2.waitKey(1)
        response = input(
            f"Use existing ROI (top={existing_roi.top}, left={existing_roi.left}, "
            f"bottom={existing_roi.bottom}, right={existing_roi.right})? [Y/n]: "
        ).strip().lower()
        cv2.destroyWindow(window_name)
        if response in {"", "y", "yes"}:
            return existing_roi

    select_window = f"Select ROI - {video_path.name}"
    print("Drag to select ROI, then press ENTER or SPACE to confirm.")
    bbox = cv2.selectROI(select_window, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(select_window)
    if bbox[2] == 0 or bbox[3] == 0:
        raise SystemExit("ROI selection cancelled.")

    left = int(bbox[0])
    top = int(bbox[1])
    right = left + int(bbox[2])
    bottom = top + int(bbox[3])
    return ROI(top=top, left=left, bottom=bottom, right=right)


def prepare_rois(
    configs: List[VideoConfig],
    settings_df: pd.DataFrame,
    settings_path: Path,
) -> List[VideoConfig]:
    updated = False

    for config in configs:
        if not config.video_path.exists():
            print(f"Warning: video not found, skipping ROI selection for {config.video_path}")
            continue

        roi = ensure_roi(config.video_path, config.roi)
        if config.roi is None or roi != config.roi:
            updated = True
        config.roi = roi
        settings_df.loc[config.index, ["roi_top", "roi_left", "roi_bottom", "roi_right"]] = [
            roi.top,
            roi.left,
            roi.bottom,
            roi.right,
        ]

    if updated:
        settings_df.to_csv(settings_path, index=False)
        print(f"Updated ROI saved to {settings_path}")

    return configs


def magnitude_to_color(magnitude: float, reference_value: float) -> Tuple[int, int, int]:
    """Map a magnitude to a BGR tuple using a discrete jet colormap."""

    if reference_value <= 0:
        return 255, 255, 255

    normalized = np.clip(magnitude / reference_value, 0.0, 1.0)
    levels = np.linspace(0, 1, 10)
    discrete = levels[np.argmin(np.abs(levels - normalized))]
    r, g, b, _ = COLORMAP(discrete)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_arrow(frame: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
    cv2.line(frame, start, end, color, thickness, lineType=cv2.LINE_AA)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    angle = math.atan2(dy, dx)
    head_length = max(5, int(length * 0.2))
    left_angle = angle + math.radians(135)
    right_angle = angle - math.radians(135)
    left = (
        int(end[0] + head_length * math.cos(left_angle)),
        int(end[1] + head_length * math.sin(left_angle)),
    )
    right = (
        int(end[0] + head_length * math.cos(right_angle)),
        int(end[1] + head_length * math.sin(right_angle)),
    )
    cv2.line(frame, end, left, color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, end, right, color, thickness, lineType=cv2.LINE_AA)


def draw_flow_field(
    frame: np.ndarray,
    flow: np.ndarray,
    step: int,
    scale: float,
    reference_value: float,
    offset: Tuple[int, int] = (0, 0),
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw flow vectors on ``frame`` and return the annotated image."""

    output = frame.copy()
    h, w = flow.shape[:2]
    step = max(1, step)
    grid = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
    y_coords, x_coords = grid
    vectors = flow[y_coords, x_coords] * scale
    magnitudes = np.linalg.norm(vectors, axis=1)

    x_offset, y_offset = offset
    for (x0, y0), (dx, dy), mag in zip(zip(x_coords, y_coords), vectors, magnitudes):
        start = (int(x0 + x_offset), int(y0 + y_offset))
        end = (int(x0 + x_offset + dx), int(y0 + y_offset + dy))
        color = magnitude_to_color(mag, reference_value)
        draw_arrow(output, start, end, color, line_thickness)

    return output


def draw_scale_bar(
    frame: np.ndarray,
    reference_value: float,
    scale: float,
    origin: Tuple[int, int] = (20, 20),
    thickness: int = 2,
) -> None:
    """Draw a simple scale bar referencing ``reference_value``."""

    length = max(1, int(reference_value))
    start = (origin[0], origin[1] + 220)
    end = (start[0] + length, start[1])
    color = (255, 255, 255)
    cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.1)
    cv2.putText(frame, "0", (start[0], start[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    effective_scale = scale if scale else 1.0
    label = f"{reference_value / effective_scale:.1f}"
    cv2.putText(frame, label, (end[0] - 30, end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_color_legend(
    frame: np.ndarray,
    reference_value: float,
    scale: float,
    origin: Tuple[int, int] = (20, 20),
    height: int = 200,
    width: int = 40,
) -> None:
    """Draw a vertical color legend that matches the flow colouring."""

    x0, y0 = origin
    for i in range(height):
        fraction = 1 - i / (height - 1)
        value = fraction * reference_value
        color = magnitude_to_color(value, reference_value)
        cv2.line(frame, (x0, y0 + i), (x0 + width, y0 + i), color, 1)

    effective_scale = scale if scale else 1.0
    top_label = f"{reference_value / effective_scale:.1f}"
    cv2.putText(frame, top_label, (x0 + width + 5, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "0", (x0 + width + 5, y0 + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_heatmap_legend(
    frame: np.ndarray,
    vmin: float,
    vmax: float,
    origin: Tuple[int, int] = (20, 20),
    height: int = 200,
    width: int = 40,
) -> None:
    x0, y0 = origin
    gradient = np.linspace(0, 255, height).astype(np.uint8)
    gradient = cv2.applyColorMap(gradient.reshape(-1, 1), cv2.COLORMAP_JET)
    for i in range(height):
        color = tuple(int(c) for c in gradient[height - 1 - i, 0])
        cv2.line(frame, (x0, y0 + i), (x0 + width, y0 + i), color, 1)

    cv2.putText(frame, f"{vmax:.1f}", (x0 + width + 5, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{vmin:.1f}", (x0 + width + 5, y0 + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def apply_sketch_effect(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    sketch = cv2.convertScaleAbs(edges)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def adjust_background(frame: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)


def compute_heatmap_limits(magnitude: np.ndarray, user_min: Optional[float], user_max: Optional[float]) -> Tuple[float, float]:
    valid = magnitude[np.isfinite(magnitude)]
    if valid.size == 0:
        return 0.0, 1.0

    vmin = user_min if user_min is not None else 0.0
    auto_max = np.percentile(valid, 95) if valid.size else valid.max()
    if not np.isfinite(auto_max):
        auto_max = valid.max()
    auto_max = max(auto_max, 1e-6)
    vmax = user_max if user_max is not None else math.ceil(auto_max)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def draw_vectors_on_frame(
    frame: np.ndarray,
    flow: np.ndarray,
    roi_left: int,
    roi_top: int,
    step: int,
    scale: float,
    thickness: int,
) -> None:
    h, w = flow.shape[:2]
    step = max(1, step)
    y_indices, x_indices = np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
    vectors = flow[y_indices, x_indices] * scale
    for (x0, y0), (dx, dy) in zip(zip(x_indices, y_indices), vectors):
        start = (int(roi_left + x0), int(roi_top + y0))
        end = (int(start[0] + dx), int(start[1] + dy))
        draw_arrow(frame, start, end, (255, 255, 255), thickness)


def render_heatmap_with_vectors(
    background: np.ndarray,
    flow_mps: np.ndarray,
    roi: ROI,
    vector_step: int,
    display_scale: float,
    vmin: float,
    vmax: float,
    line_thickness: int,
) -> np.ndarray:
    magnitude = np.linalg.norm(flow_mps, axis=2)
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((magnitude - vmin) / (vmax - vmin), 0.0, 1.0)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    result = background.copy()
    roi_slice = (slice(roi.top, roi.bottom), slice(roi.left, roi.right))
    blended = cv2.addWeighted(heatmap, 0.7, result[roi_slice], 0.3, 0)
    result[roi_slice] = blended

    draw_vectors_on_frame(result, flow_mps, roi.left, roi.top, vector_step, display_scale, line_thickness)
    return result


def process_video(
    video_path: Path,
    roi: ROI,
    reference_value: float,
    display_scale: float,
    pixels_per_meter: float,
    vector_step: int,
    arrow_thickness: int,
    black_threshold: int,
    neighbor_size: int,
    save_masks: bool,
    output_root: Path,
) -> Path:
    """Run optical flow analysis and store results alongside diagnostic media."""

    output_dir = output_root / "03_OpticalFlow" / video_path.stem
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    vector_step = max(1, vector_step)
    arrow_thickness = max(1, arrow_thickness)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read the first frame of {video_path}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_height, frame_width = prev_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_dir / "optical_flow_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    roi_flow = np.zeros((roi.height, roi.width, 2), dtype=np.float32)
    valid_counts = np.zeros((roi.height, roi.width), dtype=np.int32)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_roi = prev_gray[roi.slice()]
        curr_roi = gray[roi.slice()]

        if prev_roi.size == 0 or curr_roi.size == 0:
            prev_gray = gray
            frame_index += 1
            continue

        valid_mask = create_valid_pixel_mask(frame, roi, black_threshold, neighbor_size)
        flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow[~valid_mask] = 0

        roi_flow[..., 0] += flow[..., 0]
        roi_flow[..., 1] += flow[..., 1]
        valid_counts += valid_mask.astype(np.int32)

        overlay = draw_flow_field(
            cv2.cvtColor(curr_roi, cv2.COLOR_GRAY2BGR),
            flow,
            step=vector_step,
            scale=display_scale,
            reference_value=reference_value,
            line_thickness=arrow_thickness,
        )
        frame_copy = frame.copy()
        frame_copy[roi.slice()] = overlay
        writer.write(frame_copy)

        if save_masks:
            mask_path = masks_dir / f"mask_{frame_index:04d}.png"
            cv2.imwrite(str(mask_path), valid_mask.astype(np.uint8) * 255)

        prev_gray = gray
        frame_index += 1

    cap.release()
    writer.release()

    valid_counts[valid_counts == 0] = 1
    average_flow = roi_flow / valid_counts[..., None]

    csv_path = output_dir / "average_optical_flow_vectors.csv"
    df = pd.DataFrame(
        {
            "flow_x": average_flow[..., 0].flatten(),
            "flow_y": average_flow[..., 1].flatten(),
            "roi_top": roi.top,
            "roi_left": roi.left,
            "roi_bottom": roi.bottom,
            "roi_right": roi.right,
            "fps": fps,
            "pixels_per_meter": pixels_per_meter,
        }
    )
    df.to_csv(csv_path, index=False)
    return output_dir


def visualise_flow(
    video_path: Path,
    roi: ROI,
    reference_value: float,
    display_scale: float,
    pixels_per_meter: float,
    heatmap_min: Optional[float],
    heatmap_max: Optional[float],
    vector_step: int,
    line_thickness: int,
    output_root: Path,
    background_alpha: float,
    background_beta: float,
) -> None:
    """Create static visualisations from previously saved optical flow data."""

    output_dir = output_root / "03_OpticalFlow" / video_path.stem
    csv_path = output_dir / "average_optical_flow_vectors.csv"
    if not csv_path.exists():
        print(f"Warning: CSV not found for {video_path}, expected {csv_path}")
        return

    df = pd.read_csv(csv_path)
    flow_x = df["flow_x"].to_numpy().reshape((roi.height, roi.width))
    flow_y = df["flow_y"].to_numpy().reshape((roi.height, roi.width))
    fps = df.get("fps", pd.Series([30.0])).iloc[0]
    pixels_per_meter = df.get("pixels_per_meter", pd.Series([pixels_per_meter])).iloc[0]

    flow = np.dstack([flow_x, flow_y])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: unable to reopen {video_path} for visualisation")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, background_frame = cap.read()
    cap.release()
    if not ret:
        print(f"Warning: failed to capture background frame from {video_path}")
        return

    grayscale_background = adjust_background(background_frame, alpha=background_alpha, beta=background_beta)
    sketch_background = apply_sketch_effect(background_frame)

    effective_ppm = pixels_per_meter if pixels_per_meter else 1.0
    meters_per_second = flow * fps / effective_ppm
    magnitude = np.linalg.norm(meters_per_second, axis=2)
    vmin, vmax = compute_heatmap_limits(magnitude, heatmap_min, heatmap_max)

    annotated_gray = render_heatmap_with_vectors(
        grayscale_background,
        meters_per_second,
        roi,
        vector_step,
        display_scale,
        vmin,
        vmax,
        line_thickness,
    )
    annotated_sketch = render_heatmap_with_vectors(
        sketch_background,
        meters_per_second,
        roi,
        vector_step,
        display_scale,
        vmin,
        vmax,
        line_thickness,
    )

    draw_scale_bar(annotated_gray, reference_value, display_scale, thickness=line_thickness)
    draw_scale_bar(annotated_sketch, reference_value, display_scale, thickness=line_thickness)
    draw_heatmap_legend(annotated_gray, vmin, vmax)
    draw_heatmap_legend(annotated_sketch, vmin, vmax)

    gray_path = csv_path.with_name("flow_heatmap.png")
    sketch_path = csv_path.with_name("flow_heatmap_sketch.png")
    cv2.imwrite(str(gray_path), annotated_gray)
    cv2.imwrite(str(sketch_path), annotated_sketch)

    mean_speed = magnitude.mean()
    print(f"Saved visualisations to {gray_path} and {sketch_path} (mean speed ≈ {mean_speed:.2f} m/s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optical flow based surface velocity toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--settings",
        type=Path,
        default=Path("OpticalFlow_setting.csv"),
        help="CSV with video_path, roi_top, roi_left, roi_bottom, roi_right columns.",
    )
    common.add_argument(
        "--reference-value",
        type=float,
        default=5.0,
        help="Reference speed in metres per second used for colouring and the scale bar.",
    )
    common.add_argument(
        "--display-scale",
        type=float,
        default=10.0,
        help="Multiplier applied to the flow vectors when rendering arrows.",
    )
    common.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="Pixel-to-meter conversion (defaults to output/02_region_config.csv).",
    )
    common.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Base folder for analysis outputs.",
    )

    analyze = subparsers.add_parser("analyze", parents=[common], help="Run optical flow analysis and export CSV data.")
    analyze.add_argument("--black-threshold", type=int, default=10, help="Minimum grayscale value treated as valid.")
    analyze.add_argument("--neighbor-size", type=int, default=50, help="Kernel size used to dilate invalid pixels.")
    analyze.add_argument("--save-masks", action="store_true", help="Persist per-frame validity masks for inspection.")
    analyze.add_argument("--render-only", action="store_true", help="Skip optical flow computation and only regenerate visualisations from stored results.")
    analyze.add_argument("--heatmap-min", type=float, help="Minimum value for heatmap (default: 0).")
    analyze.add_argument("--heatmap-max", type=float, help="Maximum value for heatmap (default: auto from data).")
    analyze.add_argument("--vector-step", type=int, default=10, help="Pixel spacing between vectors in visualisations.")

    visualize = subparsers.add_parser("visualize", parents=[common], help="Generate static visualisations from CSV data.")
    visualize.add_argument("--line-thickness", type=int, default=1, help="Arrow line thickness in pixels.")
    visualize.add_argument("--background-alpha", type=float, default=1.0, help="Contrast multiplier applied to the grayscale background.")
    visualize.add_argument("--background-beta", type=float, default=0.0, help="Brightness offset applied to the grayscale background.")
    visualize.add_argument("--heatmap-min", type=float, help="Minimum value for heatmap (default: 0).")
    visualize.add_argument("--heatmap-max", type=float, help="Maximum value for heatmap (default: auto from data).")
    visualize.add_argument("--vector-step", type=int, default=10, help="Pixel spacing between vectors in visualisations.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings_df, configs = load_video_settings(args.settings)
    pixels_per_meter = resolve_pixels_per_meter(args)
    configs = prepare_rois(configs, settings_df, args.settings)

    reference_scaled = args.reference_value * pixels_per_meter
    heatmap_min = getattr(args, "heatmap_min", None)
    heatmap_max = getattr(args, "heatmap_max", None)
    vector_step = getattr(args, "vector_step", 10)
    line_thickness_visual = getattr(args, "line_thickness", 1)
    background_alpha = getattr(args, "background_alpha", 1.0)
    background_beta = getattr(args, "background_beta", 0.0)

    if args.command == "analyze":
        if args.render_only:
            for config in configs:
                if not config.video_path.exists():
                    print(f"Warning: video not found, skipping {config.video_path}")
                    continue
                if config.roi is None:
                    print(f"Skipping {config.video_path} because no ROI was selected.")
                    continue
                visualise_flow(
                    video_path=config.video_path,
                    roi=config.roi,
                    reference_value=reference_scaled,
                    display_scale=args.display_scale,
                    pixels_per_meter=pixels_per_meter,
                    heatmap_min=heatmap_min,
                    heatmap_max=heatmap_max,
                    vector_step=vector_step,
                    line_thickness=line_thickness_visual,
                    output_root=args.output_root,
                    background_alpha=background_alpha,
                    background_beta=background_beta,
                )
            return

        for config in configs:
            if not config.video_path.exists():
                print(f"Warning: video not found, skipping {config.video_path}")
                continue
            if config.roi is None:
                print(f"Skipping {config.video_path} because no ROI was selected.")
                continue
            try:
                output_dir = process_video(
                    config.video_path,
                    config.roi,
                    reference_value=reference_scaled,
                    display_scale=args.display_scale,
                    pixels_per_meter=pixels_per_meter,
                    vector_step=vector_step,
                    arrow_thickness=line_thickness_visual,
                    black_threshold=args.black_threshold,
                    neighbor_size=args.neighbor_size,
                    save_masks=args.save_masks,
                    output_root=args.output_root,
                )
            except Exception as exc:  # pragma: no cover - user feedback
                print(f"Error processing {config.video_path}: {exc}")
                continue
            print(f"Analysis complete: {output_dir}")
            visualise_flow(
                video_path=config.video_path,
                roi=config.roi,
                reference_value=reference_scaled,
                display_scale=args.display_scale,
                pixels_per_meter=pixels_per_meter,
                heatmap_min=heatmap_min,
                heatmap_max=heatmap_max,
                vector_step=vector_step,
                line_thickness=line_thickness_visual,
                output_root=args.output_root,
                background_alpha=background_alpha,
                background_beta=background_beta,
            )

    elif args.command == "visualize":
        for config in configs:
            if not config.video_path.exists():
                print(f"Warning: video not found, skipping {config.video_path}")
                continue
            if config.roi is None:
                print(f"Skipping visualisation for {config.video_path} because no ROI was selected.")
                continue
            visualise_flow(
                video_path=config.video_path,
                roi=config.roi,
                reference_value=reference_scaled,
                display_scale=args.display_scale,
                pixels_per_meter=pixels_per_meter,
                heatmap_min=heatmap_min,
                heatmap_max=heatmap_max,
                vector_step=vector_step,
                line_thickness=args.line_thickness,
                output_root=args.output_root,
                background_alpha=args.background_alpha,
                background_beta=args.background_beta,
            )


if __name__ == "__main__":
    main()
