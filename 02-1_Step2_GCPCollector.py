"""Step 2-1: Collect ground control points from stabilised Step 1 frames."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DEFAULT_CSV_HEADER = [
    "Image Name",
    "Click Number",
    "X Coordinate",
    "Y Coordinate",
    "Timestamp",
    "lat",
    "lon",
    "x",
    "y",
    "valid",
]


class InteractiveViewer:
    """Window helper that scales the image to fit within the requested window."""

    def __init__(
        self,
        window_name: str,
        base_image: np.ndarray,
        window_width: int,
        window_height: int,
    ) -> None:
        self.window = window_name
        self.window_width = window_width
        self.window_height = window_height
        self.base_image = base_image
        self.display_image = base_image
        self.scale = 1.0
        self.scaled_shape = base_image.shape[:2]

        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        self._render()

    def _compute_scaled_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(self.window_width / width, self.window_height / height, 1.0)
        if scale < 1.0:
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            self.scale = scale
            self.scaled_shape = (new_size[1], new_size[0])
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        self.scale = 1.0
        self.scaled_shape = image.shape[:2]
        return image

    def _render(self) -> None:
        scaled = self._compute_scaled_image(self.display_image)
        cv2.imshow(self.window, scaled)

    def refresh(self, image: np.ndarray) -> None:
        self.display_image = image
        self._render()

    def to_original_coordinates(self, x: int, y: int) -> tuple[int, int]:
        original_x = int(round(x / self.scale)) if self.scale > 0 else x
        original_y = int(round(y / self.scale)) if self.scale > 0 else y

        height, width = self.base_image.shape[:2]
        original_x = int(np.clip(original_x, 0, width - 1))
        original_y = int(np.clip(original_y, 0, height - 1))
        return original_x, original_y


def ensure_csv(csv_path: Path) -> None:
    """Ensure that ``csv_path`` exists and conforms to the expected header."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="") as file:
            csv.writer(file).writerow(DEFAULT_CSV_HEADER)
        return

    with csv_path.open("r", newline="") as file:
        reader = csv.reader(file)
        rows = [row for row in reader]

    if not rows:
        with csv_path.open("w", newline="") as file:
            csv.writer(file).writerow(DEFAULT_CSV_HEADER)
        return

    existing_header = [column.strip() for column in rows[0]]
    if existing_header == DEFAULT_CSV_HEADER:
        return

    data_rows = rows[1:]
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(DEFAULT_CSV_HEADER)
        for row in data_rows:
            row_dict = {key: value for key, value in zip(existing_header, row)}
            writer.writerow([row_dict.get(column, "") for column in DEFAULT_CSV_HEADER])


def load_existing_points(csv_path: Path, image_name: str) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return []
        return [row for row in reader if row.get("Image Name") == image_name]


def discover_frames(frames_dir: Path) -> list[Path]:
    pngs = sorted(frames_dir.glob("*.png"))
    if pngs:
        return pngs
    return sorted(frames_dir.rglob("*.png"))


def select_image(args: argparse.Namespace) -> Path:
    if args.image_file is not None:
        return args.image_file

    frames_dir = args.frames_dir or Path("output/01_QSIs")
    if not frames_dir.exists():
        raise SystemExit(f"Frame directory not found: {frames_dir}")

    candidates = discover_frames(frames_dir)
    if not candidates:
        raise SystemExit(f"No PNG frames located under {frames_dir}")

    if args.frame_index is not None:
        if not (0 <= args.frame_index < len(candidates)):
            raise SystemExit(f"frame-index must be within [0, {len(candidates) - 1}]")
        return candidates[args.frame_index]

    print("Available frames:")
    preview_count = min(10, len(candidates))
    for idx in range(preview_count):
        print(f"  [{idx}] {candidates[idx]}")
    if len(candidates) > preview_count:
        print(f"  ... ({len(candidates) - preview_count} more)")

    selection = input("Select frame index or enter a file path: ").strip()
    if selection.isdigit():
        chosen = int(selection)
        if not (0 <= chosen < len(candidates)):
            raise SystemExit(f"Selected index must be within [0, {len(candidates) - 1}]")
    return candidates[chosen]

    candidate_path = Path(selection)
    if candidate_path.is_file():
        return candidate_path

    raise SystemExit("Invalid selection. Provide a valid frame index or file path.")


def annotate_points(
    image: np.ndarray,
    csv_path: Path,
    image_path: Path,
    marker_dir: Path,
    window_width: int,
    window_height: int,
) -> None:
    ensure_csv(csv_path)

    marker_dir.mkdir(parents=True, exist_ok=True)

    overlay = image.copy()
    existing = load_existing_points(csv_path, image_path.name)
    max_existing_index = 0
    for row in existing:
        try:
            x = int(float(row["X Coordinate"]))
            y = int(float(row["Y Coordinate"]))
        except (TypeError, ValueError, KeyError):
            continue

        cv2.circle(overlay, (x, y), radius=8, color=(0, 255, 0), thickness=2)
        label = row.get("Click Number") or ""
        cv2.putText(overlay, str(label), (x + 12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        try:
            max_existing_index = max(max_existing_index, int(row.get("Click Number", 0)))
        except ValueError:
            pass

    click_state = {"count": max_existing_index + 1, "image": overlay}
    viewer = InteractiveViewer(image_path.name, click_state["image"], window_width, window_height)

    def handle_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x_orig, y_orig = viewer.to_original_coordinates(x, y)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with csv_path.open("a", newline="") as file:
            csv.writer(file).writerow(
                [
                    image_path.name,
                    click_state["count"],
                    x_orig,
                    y_orig,
                    timestamp,
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        overlay = click_state["image"]
        cv2.circle(overlay, (x_orig, y_orig), radius=8, color=(0, 255, 0), thickness=2)
        cv2.putText(
            overlay,
            str(click_state["count"]),
            (x_orig + 12, y_orig + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        output_name = image_path.with_name(image_path.stem + "_click" + image_path.suffix)
        annotated_path = marker_dir / output_name.name
        cv2.imwrite(str(annotated_path), overlay)

        click_state["count"] += 1
        viewer.refresh(overlay)
        print(f"Logged point #{click_state['count'] - 1} at ({x_orig}, {y_orig})")

    cv2.setMouseCallback(image_path.name, handle_mouse)

    while True:
        key = cv2.waitKey(50)
        if key in {27, ord("q"), ord("Q")}:
            break

    cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture control points from stabilized frames.")
    parser.add_argument("image_file", nargs="?", type=Path, help="Target image (PNG/JPG). Optional when using --frames-dir.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory containing frames exported from Step 1 (defaults to output/01_QSIs).",
    )
    parser.add_argument("--frame-index", type=int, help="Index of the frame within the discovered list to load.")
    parser.add_argument("--csv", type=Path, default=Path("GCPdata.csv"), help="CSV file used to store collected coordinates.")
    parser.add_argument(
        "--marker-dir",
        type=Path,
        help="Directory for annotated preview images (defaults to output/02-1_GCPs/<image_stem>).",
    )
    parser.add_argument("--window-width", type=int, default=1280, help="Viewer window width in pixels.")
    parser.add_argument("--window-height", type=int, default=720, help="Viewer window height in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.image_file is not None and args.image_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise SystemExit("Please provide an image with .png or .jpg extension.")

    image_path = select_image(args)
    print(f"Using image: {image_path}")
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise SystemExit(f"Unsupported image format: {image_path.suffix}")

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to load image: {image_path}")

    if args.marker_dir is not None:
        marker_dir = args.marker_dir
    else:
        marker_root = Path("output") / "02-1_GCPs"
        marker_dir = marker_root / image_path.stem

    annotate_points(
        image=image,
        csv_path=args.csv,
        image_path=image_path,
        marker_dir=marker_dir,
        window_width=args.window_width,
        window_height=args.window_height,
    )

    print("\nAnnotation complete.")
    print(f"Annotated preview saved under: {marker_dir}")
    print(f"Please review {args.csv} and populate either lat/lon (degrees) or x/y (meters) for each control point,\n"
          "then set the valid column to 1 for points to include in the homography calculation.")
    print("When supplying latitude/longitude, ensure that step 02 is configured with the correct target CRS.")


if __name__ == "__main__":
    main()
