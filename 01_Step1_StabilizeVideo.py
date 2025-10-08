"""Step 1: Stabilize raw footage and export per-frame homographies/frames."""

import argparse
import csv
import traceback
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def estimate_motion(
    kp_reference,
    kp_target,
    matches,
    previous_matrix: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate the frame-to-frame homography for the current pair of frames."""

    if len(matches) < 4:
        return previous_matrix, None

    src_pts = np.float32([kp_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_reference[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if matrix is None:
        return previous_matrix, None
    return matrix, mask


def apply_transformation(
    frame: np.ndarray,
    matrix: Optional[np.ndarray],
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Apply the perspective transformation and center-align the output."""

    if matrix is None:
        return frame

    height, width = frame.shape[:2]
    output_width, output_height = output_size
    translation = np.array(
        [
            [1, 0, (output_width - width) / 2],
            [0, 1, (output_height - height) / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    adjusted_matrix = translation @ matrix
    return cv2.warpPerspective(frame, adjusted_matrix, output_size)


def draw_matches(image_reference, kp_reference, image_target, kp_target, matches):
    """Return a debug visualization of AKAZE matches."""

    return cv2.drawMatches(
        image_reference,
        kp_reference,
        image_target,
        kp_target,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def create_match_visualization(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    kp_reference,
    kp_target,
    matches,
    inlier_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Create a side-by-side visualization of inlier matches."""

    ref_vis = reference_image.copy()
    tgt_vis = target_image.copy()

    if ref_vis.ndim == 2:
        ref_vis = cv2.cvtColor(ref_vis, cv2.COLOR_GRAY2BGR)
    if tgt_vis.ndim == 2:
        tgt_vis = cv2.cvtColor(tgt_vis, cv2.COLOR_GRAY2BGR)

    ref_h, ref_w = ref_vis.shape[:2]
    tgt_h, tgt_w = tgt_vis.shape[:2]
    canvas_h = max(ref_h, tgt_h)
    canvas_w = ref_w + tgt_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:ref_h, :ref_w] = ref_vis
    canvas[:tgt_h, ref_w : ref_w + tgt_w] = tgt_vis

    if kp_reference is None or kp_target is None or not matches:
        return canvas

    if inlier_mask is not None:
        mask_iterable = inlier_mask.ravel().astype(bool).tolist()
    else:
        mask_iterable = [True] * len(matches)

    for match, keep in zip(matches, mask_iterable):
        if not keep:
            continue

        ref_pt = kp_reference[match.queryIdx].pt
        tgt_pt = kp_target[match.trainIdx].pt

        ref_point = (int(round(ref_pt[0])), int(round(ref_pt[1])))
        tgt_point = (int(round(tgt_pt[0])) + ref_w, int(round(tgt_pt[1])))

        cv2.circle(canvas, ref_point, 4, (0, 255, 0), thickness=2)
        cv2.circle(canvas, tgt_point, 4, (0, 255, 0), thickness=2)
        cv2.line(canvas, ref_point, tgt_point, (255, 0, 0), thickness=1)

    return canvas


def write_homography_row(
    writer: Optional[csv.writer],
    matrix: Optional[np.ndarray],
    raw_match_count: int,
    inlier_count: int,
) -> None:
    if writer is None:
        return

    if matrix is not None:
        row = matrix.reshape(-1).astype(float).tolist()
    else:
        row = [float("nan")] * 9
    row.extend([raw_match_count, inlier_count])
    writer.writerow(row)


def export_frames(video_path: Path, frames_dir: Path) -> int:
    """Save every frame of ``video_path`` to ``frames_dir`` as PNG files."""

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for frame export: {video_path}")

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = frames_dir / f"frame_{index:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        index += 1

    cap.release()
    return index


def process_video(
    video_path: Path,
    reference_frame_index: int,
    roi: Optional[Tuple[int, int, int, int]],
    output_dir: Path,
    csv_path: Optional[Path],
    output_scale: float,
    display_matches: bool = False,
) -> Path:
    """Process a video to produce a stabilized version and return the output path."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if reference_frame_index >= total_frames:
        raise ValueError("Reference frame index exceeds video length.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_index)
    ret, reference_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the reference frame.")

    frame_height, frame_width = reference_frame.shape[:2]
    if roi is None:
        roi = (0, 0, frame_width, frame_height)
    x, y, w, h = roi
    reference_roi = reference_frame[y : y + h, x : x + w]

    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    scale = max(1.0, output_scale)
    output_size = (
        max(1, int(round(frame_width * scale))),
        max(1, int(round(frame_height * scale))),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = None
    csv_writer = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "h00",
                "h01",
                "h02",
                "h10",
                "h11",
                "h12",
                "h20",
                "h21",
                "h22",
                "matches_total",
                "matches_inliers",
            ]
        )

    raw_output_path = output_dir / f"{video_path.stem}_homography_preview.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    preview_writer = cv2.VideoWriter(str(raw_output_path), fourcc, fps, output_size)

    match_output_dir = output_dir / "01_match_viz"
    match_output_dir.mkdir(parents=True, exist_ok=True)
    match_output_path = match_output_dir / f"{video_path.stem}_matches.mp4"
    match_width = int(w * 2)
    match_height = int(h)
    match_writer = cv2.VideoWriter(str(match_output_path), fourcc, fps, (match_width, match_height))

    akaze = cv2.AKAZE_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    kp_reference, des_reference = akaze.detectAndCompute(reference_roi, None)

    previous_matrix = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_roi = frame[y : y + h, x : x + w]
            kp_target, des_target = akaze.detectAndCompute(frame_roi, None)

            raw_match_count = 0
            inlier_count = 0
            matches_for_visual = []
            inlier_mask = None

            if des_target is not None and des_reference is not None:
                matches = matcher.knnMatch(des_reference, des_target, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
                raw_match_count = len(good_matches)
                matches_for_visual = good_matches
                previous_matrix, inlier_mask = estimate_motion(
                    kp_reference,
                    kp_target,
                    good_matches,
                    previous_matrix,
                )
                if inlier_mask is not None:
                    inlier_mask = inlier_mask.ravel().astype(bool)
                    inlier_count = int(inlier_mask.sum())
            else:
                previous_matrix, _ = estimate_motion(
                    kp_reference,
                    kp_target,
                    [],
                    previous_matrix,
                )

            write_homography_row(csv_writer, previous_matrix, raw_match_count, inlier_count)

            transformed = apply_transformation(frame, previous_matrix, output_size)
            preview_writer.write(transformed)

            match_frame = create_match_visualization(
                reference_roi,
                frame_roi,
                kp_reference,
                kp_target,
                matches_for_visual,
                inlier_mask,
            )
            if match_writer.isOpened():
                match_writer.write(match_frame)

            if display_matches and matches_for_visual:
                if inlier_mask is not None:
                    matches_to_display = [m for m, keep in zip(matches_for_visual, inlier_mask) if keep]
                else:
                    matches_to_display = matches_for_visual
                matched_frame = draw_matches(reference_frame, kp_reference, frame, kp_target, matches_to_display)
                cv2.imshow("AKAZE Matches", matched_frame)
                cv2.imshow("Transformed Frame", transformed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    display_matches = False
                    cv2.destroyAllWindows()
    finally:
        cap.release()
        preview_writer.release()
        if match_writer.isOpened():
            match_writer.release()
        if display_matches:
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    print(f"Match visualization video saved to {match_output_path}")
    return raw_output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate frame-wise homographies and create a stabilized video.",
    )
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Index of the reference frame used as baseline for matching.",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Region of interest (x, y, width, height) used for feature matching.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to store intermediate and stabilized outputs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV file that captures the per-frame homography matrices.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory to store exported stabilized frames (defaults to output_dir/01_QSIs/<video_stem>).",
    )
    parser.add_argument(
        "--skip-frame-export",
        action="store_true",
        help="Skip exporting the stabilized video as individual PNG frames.",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the output canvas (default 1.0).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display AKAZE matches while processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    csv_path = args.csv or (args.output_dir / f"{args.video.stem}_homography.csv")
    frames_dir = args.frames_dir or (args.output_dir / "01_QSIs" / args.video.stem)

    try:
        stabilized_path = process_video(
            video_path=args.video,
            reference_frame_index=args.reference_frame,
            roi=tuple(args.roi) if args.roi else None,
            output_dir=args.output_dir,
            csv_path=csv_path,
            output_scale=args.output_scale,
            display_matches=args.show,
        )
    except Exception as exc:  # pragma: no cover - user-facing error message
        traceback.print_exc()
        raise SystemExit(1) from exc

    print(f"Stabilized preview video saved to {stabilized_path}")
    print(f"Homography coefficients logged to {csv_path}")

    if not args.skip_frame_export:
        frame_count = export_frames(stabilized_path, frames_dir)
        print(f"Exported {frame_count} stabilized frames to {frames_dir}")


if __name__ == "__main__":
    main()
