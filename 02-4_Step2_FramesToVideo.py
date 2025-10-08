"""Step 2-4: Assemble a range of orthorectified PNG frames into an MP4 video."""

from pathlib import Path

import cv2


def list_frames(frame_dir: Path) -> list[Path]:
    return sorted(path for path in frame_dir.iterdir() if path.suffix.lower() == ".png")


def prompt_int(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        value = input(prompt).strip()
        if value.isdigit():
            number = int(value)
            if minimum <= number <= maximum:
                return number
        print(f"Please enter an integer between {minimum} and {maximum}.")


def prompt_float(prompt: str) -> float:
    while True:
        value = input(prompt).strip()
        try:
            number = float(value)
            if number > 0:
                return number
        except ValueError:
            pass
        print("Please enter a positive number.")



def select_frame_directory(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"Frame root not found: {root}")

    candidates = sorted({path.parent for path in root.rglob("*.png")})
    if not candidates:
        raise SystemExit(f"No PNG frames found under {root}")

    if len(candidates) == 1:
        return candidates[0]

    print("Multiple orthorectified frame sets detected:")
    for idx, candidate in enumerate(candidates):
        try:
            label = candidate.relative_to(root)
        except ValueError:
            label = candidate
        print(f"  [{idx}] {label}")

    selection = prompt_int(
        f"Select frame set index (0 to {len(candidates) - 1}): ",
        0,
        len(candidates) - 1,
    )
    return candidates[selection]

def main() -> None:
    frame_root = Path("output/02-3_Ortho-QSIs")
    frame_dir = select_frame_directory(frame_root)
    print(f"Using frames from: {frame_dir}")

    frames = list_frames(frame_dir)
    if not frames:
        raise SystemExit(f"No PNG frames found in {frame_dir}")

    print("Available frames (first 10 shown):")
    for idx, path in enumerate(frames[:10]):
        print(f"  [{idx}] {path.name}")

    start = prompt_int(
        f"Enter start frame index (0 to {len(frames) - 1}): ",
        0,
        len(frames) - 1,
    )
    end = prompt_int(
        f"Enter end frame index ({start} to {len(frames) - 1}): ",
        start,
        len(frames) - 1,
    )
    fps = prompt_float("Enter frames per second (fps): ")

    subset = frames[start : end + 1]
    first_frame = cv2.imread(str(subset[0]))
    if first_frame is None:
        raise SystemExit(f"Failed to load the first frame: {subset[0]}")
    height, width = first_frame.shape[:2]

    output_path = Path(f"output/Ortho-QSI_{start:04d}_{end:04d}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_path in subset:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: could not read {frame_path}, skipping")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        print(f"Added {frame_path.name}")

    writer.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
