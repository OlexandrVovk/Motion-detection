"""
Script to create a video from frames in the multiscale_ema_frames folder.
Output resolution: 848x464
"""

import cv2
import os
import re
from pathlib import Path


def natural_sort_key(path: Path) -> tuple:
    """Extract numeric parts for natural sorting."""
    name = path.stem
    parts = re.split(r'(\d+)', name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def create_video_from_frames(
    frames_dir: str,
    output_path: str,
    resolution: tuple = (848, 464),
    fps: int = 30,
    frame_pattern: str = "frame_"
):
    """
    Create a video from a sequence of frames.

    Args:
        frames_dir: Directory containing the frame images
        output_path: Path for the output video file
        resolution: Output video resolution (width, height)
        fps: Frames per second for the output video
        frame_pattern: Pattern to filter frame files
    """
    frames_path = Path(frames_dir)

    # Get all frame files and sort them naturally
    frame_files = [f for f in frames_path.iterdir()
                   if f.is_file() and f.suffix.lower() == '.png'
                   and frame_pattern in f.name]

    frame_files = sorted(frame_files, key=natural_sort_key)

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Found {len(frame_files)} frames")
    print(f"Output resolution: {resolution[0]}x{resolution[1]}")
    print(f"FPS: {fps}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    if not out.isOpened():
        print("Error: Could not open video writer")
        return

    # Process each frame
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))

        if frame is None:
            print(f"Warning: Could not read {frame_file.name}, skipping...")
            continue

        # Resize frame to target resolution
        frame_resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_LANCZOS4)

        # Write frame to video
        out.write(frame_resized)

        # Progress update every 100 frames
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames...")

    # Release video writer
    out.release()
    print(f"\nVideo saved to: {output_path}")
    print(f"Total frames: {len(frame_files)}")
    print(f"Duration: {len(frame_files) / fps:.2f} seconds")


if __name__ == "__main__":
    # Configuration
    FRAMES_DIR = "../data/multiscale_ema_frames"
    OUTPUT_PATH = "../data/multiscale_ema_video.mp4"
    RESOLUTION = (848, 464)
    FPS = 30

    create_video_from_frames(
        frames_dir=FRAMES_DIR,
        output_path=OUTPUT_PATH,
        resolution=RESOLUTION,
        fps=FPS
    )
