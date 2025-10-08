"""Step 2-3: Apply the homography to stabilized frames for orthorectification."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
import math


def load_gcp_table(gcp_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gcp_csv)
    required = {"X Coordinate", "Y Coordinate", "valid"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing required columns in {gcp_csv}: {', '.join(sorted(missing))}")
    return df


def enrich_coordinates(df: pd.DataFrame, destination_crs: str) -> pd.DataFrame:
    if {"Xreal", "Yreal"}.issubset(df.columns):
        enriched = df.copy()
    elif {"lat", "lon"}.issubset(df.columns):
        transformer = Transformer.from_crs("EPSG:4326", destination_crs, always_xy=True)
        enriched = df.copy()
        enriched[["Xreal", "Yreal"]] = enriched.apply(
            lambda row: transformer.transform(row["lon"], row["lat"]), axis=1, result_type="expand"
        )
    else:
        raise ValueError("GCP table must include either Xreal/Yreal or lat/lon columns.")

    enriched[["Xreal", "Yreal"]] = enriched[["Yreal", "Xreal"]]
    enriched["Xreal"] = -enriched["Xreal"]
    enriched["Xreal_image"] = enriched["Yreal"]
    enriched["Yreal_image"] = enriched["Xreal"]
    return enriched


def compute_homography(valid_df: pd.DataFrame, pix_per_meter: float) -> np.ndarray:
    src_pts = valid_df[["X Coordinate", "Y Coordinate"]].to_numpy(dtype=np.float32)
    dst_pts = (valid_df[["Xreal_image", "Yreal_image"]] * pix_per_meter).to_numpy(dtype=np.float32)
    matrix, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    if matrix is None:
        raise RuntimeError("Homography estimation failed; ensure valid points are available.")
    return matrix


def determine_bounds(
    matrix: np.ndarray,
    image_shape: Tuple[int, int],
    bounds: Optional[Sequence[float]],
    pix_per_meter: float,
    margin: int,
) -> Tuple[int, int, int, int]:
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        return (
            int(np.floor(x_min * pix_per_meter)),
            int(np.ceil(x_max * pix_per_meter)),
            int(np.floor(y_min * pix_per_meter)),
            int(np.ceil(y_max * pix_per_meter)),
        )

    h, w = image_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, matrix)
    x_vals = warped[:, 0, 0]
    y_vals = warped[:, 0, 1]
    return (
        int(np.floor(x_vals.min())) - margin,
        int(np.ceil(x_vals.max())) + margin,
        int(np.floor(y_vals.min())) - margin,
        int(np.ceil(y_vals.max())) + margin,
    )


def orthorectify_frame(
    frame: np.ndarray,
    matrix: np.ndarray,
    bounds: Tuple[int, int, int, int],
    alpha: float,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    output_width = x_max - x_min
    output_height = y_max - y_min

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    adjusted_matrix = translation @ matrix
    warped = cv2.warpPerspective(frame, adjusted_matrix, (output_width, output_height))

    background = np.full_like(warped, 255)
    return cv2.addWeighted(warped, alpha, background, 1 - alpha, 0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2-3: Apply the Step 2 homography to the stabilized frames."
    )
    parser.add_argument("--gcp", type=Path, default=Path("GCPdata.csv"), help="Ground control point CSV file.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("output/01_QSIs"),
        help="Directory containing stabilised frames (defaults to output/01_QSIs).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("output/02_region_config.csv"),
        help="CSV file storing output bounds (meters) and pix_per_meter.",
    )
    parser.add_argument(
        "--pix-per-meter",
        type=float,
        help="Scale factor between meters and pixels (overrides config).",
    )
    parser.add_argument(
        "--destination-crs",
        type=str,
        default="EPSG:6675",
        help="Projected CRS used when converting lat/lon coordinates.",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Optional output bounds in meters (overrides stored configuration).",
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha blending factor for the warped frame.")
    return parser.parse_args()


def calculate_default_bounds(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    working_df = df[df["valid"] == 1]
    if working_df.empty:
        working_df = df

    x_min = math.floor(working_df["Xreal_image"].min())
    x_max = math.ceil(working_df["Xreal_image"].max())
    y_min = math.floor(working_df["Yreal_image"].min())
    y_max = math.ceil(working_df["Yreal_image"].max())

    span_x = max(1, x_max - x_min)
    span_y = max(1, y_max - y_min)
    margin_x = math.ceil(span_x * 0.2)
    margin_y = math.ceil(span_y * 0.2)

    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y
    return x_min, x_max, y_min, y_max


def prompt_pix_per_meter() -> float:
    while True:
        user_input = input("Enter pix-per-meter (positive float): ").strip()
        try:
            value = float(user_input)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Invalid input. Please enter a positive number.")


def load_or_create_region_config(
    df: pd.DataFrame,
    config_path: Path,
    pix_per_meter_override: Optional[float],
    bounds_override: Optional[Tuple[float, float, float, float]],
) -> Tuple[Tuple[int, int, int, int], float]:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    stored_bounds = None
    stored_ppm = None
    if config_path.exists():
        try:
            data = pd.read_csv(config_path)
            if not data.empty:
                row = data.iloc[0]
                stored_bounds = (
                    float(row.get("x_min", math.nan)),
                    float(row.get("x_max", math.nan)),
                    float(row.get("y_min", math.nan)),
                    float(row.get("y_max", math.nan)),
                )
                if any(math.isnan(val) for val in stored_bounds):
                    stored_bounds = None
                stored_ppm = row.get("pix_per_meter")
                stored_ppm = float(stored_ppm) if pd.notna(stored_ppm) else None
        except Exception:
            print(f"Warning: failed to read config {config_path}, regenerating.")

    if bounds_override is not None:
        stored_bounds = bounds_override
    if stored_bounds is None:
        stored_bounds = calculate_default_bounds(df)

    if pix_per_meter_override is not None:
        stored_ppm = pix_per_meter_override
    if stored_ppm is None or stored_ppm <= 0:
        stored_ppm = prompt_pix_per_meter()

    config_df = pd.DataFrame(
        [
            {
                "x_min": stored_bounds[0],
                "x_max": stored_bounds[1],
                "y_min": stored_bounds[2],
                "y_max": stored_bounds[3],
                "pix_per_meter": stored_ppm,
            }
        ]
    )
    config_df.to_csv(config_path, index=False)
    print(f"Region configuration saved to {config_path}")
    return stored_bounds, stored_ppm


def bounds_to_pixels(bounds_meters: Tuple[float, float, float, float], pix_per_meter: float) -> Tuple[int, int, int, int]:
    x_min, x_max, y_min, y_max = bounds_meters
    x_min_pix = int(math.floor(x_min * pix_per_meter))
    x_max_pix = int(math.ceil(x_max * pix_per_meter))
    y_min_pix = int(math.floor(y_min * pix_per_meter))
    y_max_pix = int(math.ceil(y_max * pix_per_meter))
    return x_min_pix, x_max_pix, y_min_pix, y_max_pix



def resolve_frames_directory(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise SystemExit(f"Frame directory not found: {base_dir}")
    if base_dir.is_file():
        raise SystemExit(f"Expected a directory for frames: {base_dir}")

    direct_frames = [p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if direct_frames:
        return base_dir

    candidate_dirs = [p for p in sorted(base_dir.iterdir()) if p.is_dir()]
    valid_dirs = [
        d
        for d in candidate_dirs
        if any(child.is_file() and child.suffix.lower() == ".png" for child in d.iterdir())
    ]

    if len(valid_dirs) == 1:
        return valid_dirs[0]
    if not valid_dirs:
        raise SystemExit(f"No PNG frames found in {base_dir} or its immediate subdirectories.")

    options = "\n".join(f"  - {d}" for d in valid_dirs)
    raise SystemExit(
        "Multiple frame directories detected under "
        f"{base_dir}.\nPlease re-run with --frames-dir pointing to the desired frames set.\n"
        f"Candidates:\n{options}"
    )


def main() -> None:
    args = parse_arguments()

    frames_dir = resolve_frames_directory(args.frames_dir)
    frames = sorted(
        path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not frames:
        raise SystemExit(f"No PNG frames found in {frames_dir}")

    df = load_gcp_table(args.gcp)
    df = enrich_coordinates(df, args.destination_crs)
    valid_df = df[df["valid"] == 1]
    if valid_df.empty:
        raise SystemExit("No valid control points available (valid == 1).")

    bounds_m, pix_per_meter = load_or_create_region_config(
        df,
        args.config,
        args.pix_per_meter,
        tuple(args.bounds) if args.bounds else None,
    )

    homography = compute_homography(valid_df, pix_per_meter)

    bounds = bounds_to_pixels(bounds_m, pix_per_meter)

    base_frames_root = Path("output/01_QSIs")
    output_root = Path("output/02-3_Ortho-QSIs")
    try:
        relative_section = frames_dir.relative_to(base_frames_root)
        output_dir = output_root / relative_section
    except ValueError:
        output_dir = output_root / frames_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: failed to load {frame_path}, skipping")
            continue

        transformed = orthorectify_frame(frame, homography, bounds, args.alpha)
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), transformed)
        print(f"Saved {output_path}")

    print(f"Homography applied to {len(frames)} frames. Results stored in {output_dir}")


if __name__ == "__main__":
    main()
