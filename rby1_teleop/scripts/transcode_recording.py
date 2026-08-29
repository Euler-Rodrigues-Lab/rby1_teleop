# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Transcode a recorded OpenXR CSV into a device-neutral geo_kin_core frame stream.

The CSV format needs the monolith's device stack (pandas + xr_robot_teleop_server
bone schemas); a frame stream is plain numpy and replays anywhere. Run this once
to produce sample/regression data, then everything downstream — the offline
replay demo, tests, public CI — reads the .npz and needs no device deps at all.

Example::

    python -m rby1_teleop.scripts.transcode_recording \\
        --csv_file $GEO_TELEOP_MONOLITH/References/recordings/ipman_roll.csv \\
        --out rby1_teleop/assets/sample_motion/ipman_roll.npz \\
        --fps 60 --duration 8
"""

import argparse

from geo_kin_core.frames import save_frames

from rby1_teleop.input import OfflineCSVAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Recorded CSV -> geo_kin_core frame stream")
    parser.add_argument("--csv_file", required=True, help="Recorded OpenXR body-pose CSV")
    parser.add_argument("--out", required=True, help="Output .npz frame stream")
    parser.add_argument("--fps", type=float, default=60.0, help="Sampling rate (Hz)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (s)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Seconds to transcode (default: to the end)")
    parser.add_argument("--monolith_path", default=None,
                        help="SEW-Geometric-Teleop checkout (else GEO_TELEOP_MONOLITH)")
    parser.add_argument("--notes", default="", help="Free-text note stored in the stream")
    return parser.parse_args()


def main():
    args = parse_args()
    source = OfflineCSVAdapter(args.csv_file, loop=False, monolith_path=args.monolith_path)
    duration = source.duration if args.duration is None else min(args.duration,
                                                                 source.duration - args.start)
    n = int(duration * args.fps)
    frames = []
    for i in range(n):
        frame = source.frame_at_time(args.start + i / args.fps)
        if frame is None:
            break
        frames.append(frame)
    if not frames:
        raise SystemExit("No frames transcoded — check --start/--duration against the recording.")
    path = save_frames(args.out, frames, fps=args.fps,
                       source=str(args.csv_file), notes=args.notes)
    print(f"Wrote {len(frames)} frames ({len(frames) / args.fps:.2f}s @ {args.fps:g}Hz) to {path}")


if __name__ == "__main__":
    main()
