"""Step 2-2: Compute the homography from GCPs and produce diagnostic outputs."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Transformer = None

def load_gcp_table(gcp_csv: Path) -> pd.DataFrame:
    """Read the ground control point table and validate the expected columns."""

    df = pd.read_csv(gcp_csv)
    missing = {"Image Name", "X Coordinate", "Y Coordinate", "valid"} - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {gcp_csv}: {missing_cols}")

    numeric_columns = [
        "X Coordinate",
        "Y Coordinate",
        "lat",
        "lon",
        "x",
        "y",
        "valid",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["valid"] = df["valid"].fillna(0).astype(int)
    return df


def resolve_image_path(image_name: str) -> Optional[Path]:
    candidate = Path(image_name)
    if candidate.exists():
        return candidate

    annotated_root = Path("output") / "02-1_GCPs"
    base_name = Path(image_name).stem
    suffix = Path(image_name).suffix or ".png"

    annotated_path = annotated_root / base_name / f"{base_name}_click{suffix}"
    if annotated_path.exists():
        return annotated_path

    frames_root = Path("output")
    matches = list(frames_root.rglob(image_name))
    if matches:
        return matches[0]

    return None


def enrich_coordinates(
    df: pd.DataFrame,
    destination_crs: str,
) -> pd.DataFrame:
    """Add metric coordinates based on either existing fields or lat/lon."""

    if {"x", "y"}.issubset(df.columns) and not df[["x", "y"]].isna().all().all():
        df = df.copy()
        df["Xreal"] = df["x"]
        df["Yreal"] = df["y"]
    elif {"Xreal", "Yreal"}.issubset(df.columns):
        df = df.copy()
    elif {"lat", "lon"}.issubset(df.columns):
        df = df.copy()
        if Transformer is not None:
            transformer = Transformer.from_crs("EPSG:4326", destination_crs, always_xy=True)
            df[["Xreal", "Yreal"]] = df.apply(
                lambda row: transformer.transform(row["lon"], row["lat"]), axis=1, result_type="expand"
            )
        else:  # pragma: no cover - fallback when pyproj is unavailable
            mean_lat = df["lat"].mean()
            mean_lon = df["lon"].mean()
            earth_radius = 6378137.0
            lat_rad = np.radians(df["lat"])
            lon_rad = np.radians(df["lon"])
            mean_lat_rad = np.radians(mean_lat)
            mean_lon_rad = np.radians(mean_lon)
            df["Xreal"], df["Yreal"] = (
                (lon_rad - mean_lon_rad) * np.cos(mean_lat_rad) * earth_radius,
                (lat_rad - mean_lat_rad) * earth_radius,
            )
    else:
        raise ValueError("The GCP table must include either Xreal/Yreal or lat/lon columns.")

    # Align with the image coordinate convention used downstream
    df[["Xreal", "Yreal"]] = df[["Yreal", "Xreal"]]
    df["Xreal"] = -df["Xreal"]
    df["Xreal_image"] = df["Yreal"]
    df["Yreal_image"] = df["Xreal"]
    return df


def compute_homography_matrix(valid_df: pd.DataFrame, pix_per_meter: float) -> np.ndarray:
    """Estimate the homography that maps image coordinates to projected space."""

    src_pts = valid_df[["X Coordinate", "Y Coordinate"]].to_numpy(dtype=np.float32)
    dst_pts = (valid_df[["Xreal_image", "Yreal_image"]] * pix_per_meter).to_numpy(dtype=np.float32)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    if matrix is None:
        matrix, mask = cv2.findHomography(src_pts, dst_pts, method=0)
    if matrix is None:
        raise RuntimeError("Homography estimation failed; check the input points and validity flags.")
    return matrix


def project_all_points(df: pd.DataFrame, matrix: np.ndarray, pix_per_meter: float) -> pd.DataFrame:
    """Apply the homography to every row and compute residuals."""

    photo_coords = df[["X Coordinate", "Y Coordinate"]].to_numpy(dtype=np.float32)
    ones = np.ones((photo_coords.shape[0], 1), dtype=np.float32)
    homogeneous = np.hstack([photo_coords, ones])
    projected = (matrix @ homogeneous.T).T
    cartesian = projected[:, :2] / projected[:, 2:3]

    df = df.copy()
    df["Xcal"] = cartesian[:, 0] / pix_per_meter
    df["Ycal"] = cartesian[:, 1] / pix_per_meter
    df["ErrX"] = df["Xreal_image"] - df["Xcal"]
    df["ErrY"] = df["Yreal_image"] - df["Ycal"]
    return df


def plot_residuals(df: pd.DataFrame, output_dir: Path, base_filename: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        print("matplotlib is not installed; skipping scatter plot generation.")
        return
    except Exception as exc:  # pragma: no cover - environment issues
        print(f"Unable to initialise matplotlib ({exc}); skipping scatter plot.")
        return

    plot_path = output_dir / f"{base_filename}.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["Xreal_image"], df["Yreal_image"], c="blue", label="Input")
    ax.scatter(df["Xcal"], df["Ycal"], c="red", marker="x", label="Calculated")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("GCP vs Calculated Coordinates")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Scatter plot saved to {plot_path}")


def determine_bounds(
    matrix: np.ndarray,
    image_shape: Tuple[int, int],
    bounds: Optional[Sequence[float]],
    pix_per_meter: float,
    margin: int,
) -> Tuple[int, int, int, int]:
    """Return pixel bounds for the warped output (x_min, x_max, y_min, y_max)."""

    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        x_min_pix = int(np.floor(x_min * pix_per_meter))
        x_max_pix = int(np.ceil(x_max * pix_per_meter))
        y_min_pix = int(np.floor(y_min * pix_per_meter))
        y_max_pix = int(np.ceil(y_max * pix_per_meter))
        return x_min_pix, x_max_pix, y_min_pix, y_max_pix

    h, w = image_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, matrix)

    x_min_pix = int(np.floor(warped[:, 0, 0].min())) - margin
    x_max_pix = int(np.ceil(warped[:, 0, 0].max())) + margin
    y_min_pix = int(np.floor(warped[:, 0, 1].min())) - margin
    y_max_pix = int(np.ceil(warped[:, 0, 1].max())) + margin
    return x_min_pix, x_max_pix, y_min_pix, y_max_pix


def orthorectify_image(
    image: np.ndarray,
    matrix: np.ndarray,
    bounds: Tuple[int, int, int, int],
    overlay_alpha: float,
) -> np.ndarray:
    """Warp the input image using the homography and return the transformed result."""

    x_min, x_max, y_min, y_max = bounds
    output_width = x_max - x_min
    output_height = y_max - y_min

    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]],
        dtype=np.float64,
    )
    adjusted_matrix = translation @ matrix
    warped = cv2.warpPerspective(image, adjusted_matrix, (output_width, output_height))

    background = np.full_like(warped, 255)
    blended = cv2.addWeighted(warped, overlay_alpha, background, 1 - overlay_alpha, 0)
    return blended


def annotate_points(
    image: np.ndarray,
    df: pd.DataFrame,
    pix_per_meter: float,
    bounds: Tuple[int, int, int, int],
) -> np.ndarray:
    """Overlay measured and calculated points on the warped image."""

    annotated = image.copy()
    x_min, _, y_min, _ = bounds
    for _, row in df.iterrows():
        number = str(row["Click Number"])
        x_cal = int(row["Xcal"] * pix_per_meter - x_min)
        y_cal = int(row["Ycal"] * pix_per_meter - y_min)
        x_real = int(row["Xreal_image"] * pix_per_meter - x_min)
        y_real = int(row["Yreal_image"] * pix_per_meter - y_min)

        cv2.putText(annotated, f"{number} cal", (x_cal, y_cal), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(annotated, (x_cal, y_cal), 4, (0, 255, 0), -1)
        cv2.putText(annotated, f"{number} real", (x_real, y_real), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(annotated, (x_real, y_real), 4, (0, 0, 255), -1)

    return annotated


def create_overall_map(df: pd.DataFrame, pix_per_meter: float, output_path: Path, margin: int = 50) -> None:
    """Render a plan-view map summarising all calculated and reference points."""

    x_vals = df["Xreal_image"] * pix_per_meter
    y_vals = df["Yreal_image"] * pix_per_meter

    x_min = int(np.floor(x_vals.min())) - margin
    x_max = int(np.ceil(x_vals.max())) + margin
    y_min = int(np.floor(y_vals.min())) - margin
    y_max = int(np.ceil(y_vals.max())) + margin

    width = x_max - x_min
    height = y_max - y_min
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for _, row in df.iterrows():
        x_cal = int(row["Xcal"] * pix_per_meter - x_min)
        y_cal = int(row["Ycal"] * pix_per_meter - y_min)
        x_real = int(row["Xreal_image"] * pix_per_meter - x_min)
        y_real = int(row["Yreal_image"] * pix_per_meter - y_min)
        label = str(row["Click Number"])

        cv2.putText(canvas, f"{label} cal", (x_cal, y_cal), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(canvas, (x_cal, y_cal), 4, (0, 255, 0), -1)
        cv2.putText(canvas, f"{label} real", (x_real, y_real), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(canvas, (x_real, y_real), 4, (0, 0, 255), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate a homography from GCPs and orthorectify imagery.")
    parser.add_argument("--gcp", type=Path, default=Path("GCPdata.csv"), help="Ground control point CSV file.")
    parser.add_argument(
        "--pix-per-meter",
        type=float,
        default=10.0,
        help="Scale factor between meters and pixels (higher values increase output size).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/02-2_transformed"),
        help="Directory for outputs (imagery, CSV, plots).",
    )
    parser.add_argument(
        "--matrix-name",
        type=str,
        default="homography_matrix.csv",
        help="Filename for the homography matrix CSV within the output directory.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="homography_transformation_output.csv",
        help="Filename for the enriched GCP table within the output directory.",
    )
    parser.add_argument(
        "--overall-map-name",
        type=str,
        default="overall_map.png",
        help="Filename for the overview map within the output directory.",
    )
    parser.add_argument(
        "--enable-plot",
        action="store_true",
        help="Generate residual scatter plots (disable to avoid matplotlib overhead).",
    )
    parser.add_argument(
        "--destination-crs",
        type=str,
        default="EPSG:6675",
        help="Projected CRS used to convert lat/lon into planar coordinates.",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Optional output bounds in meters (overrides automatic detection).",
    )
    parser.add_argument("--margin", type=int, default=300, help="Padding (in pixels) added around automatic bounds.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha blending factor for the warped image.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    df = load_gcp_table(args.gcp)
    df = enrich_coordinates(df, args.destination_crs)
    valid_df = df[df["valid"] == 1]
    if valid_df.empty:
        raise SystemExit("No valid control points found (valid == 1).")

    homography = compute_homography_matrix(valid_df, args.pix_per_meter)
    df = project_all_points(df, homography, args.pix_per_meter)

    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = output_dir / args.matrix_name
    summary_csv = output_dir / args.summary_name
    overall_map = output_dir / args.overall_map_name

    np.savetxt(matrix_csv, homography, delimiter=",")
    df.to_csv(summary_csv, index=False)

    imagery_dir = output_dir / "frames"
    imagery_dir.mkdir(parents=True, exist_ok=True)
    for image_name, image_df in df.groupby("Image Name"):
        image_path = resolve_image_path(image_name)
        if image_path is None:
            print(f"Warning: image not found, skipping {image_name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: failed to load {image_path}, skipping")
            continue

        bounds = determine_bounds(homography, image.shape[:2], args.bounds, args.pix_per_meter, args.margin)
        warped = orthorectify_image(image, homography, bounds, args.alpha)
        annotated = annotate_points(warped, image_df, args.pix_per_meter, bounds)

        output_path = imagery_dir / f"transformed_{image_path.name}"
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved {output_path}")

    create_overall_map(df, args.pix_per_meter, overall_map)

    if args.enable_plot:
        plot_residuals(df, output_dir, "residual_scatter")
    else:
        print("Residual scatter plotting disabled (use --enable-plot to generate).")

    print(f"Homography matrix saved to {matrix_csv}")
    print(f"Summary CSV saved to {summary_csv}")
    print(f"Overall map saved to {overall_map}")


if __name__ == "__main__":
    main()
