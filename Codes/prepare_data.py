"""
Video-to-frame extraction tool.

Splits video files into JPEG frames with a directory structure compatible
with the anomaly prediction pipeline:

    Data/<dataset>/
    ├── training/frames/
    │   ├── 01/
    │   │   ├── 000.jpg
    │   │   ├── 001.jpg
    │   │   └── ...
    │   ├── 02/
    │   └── ...
    └── testing/frames/
        ├── 01/
        └── ...

Usage:
    # Single video file
    python prepare_data.py --video_dir /path/to/video.mp4 --dataset my_dataset --split training
    # Directory of videos
    python prepare_data.py --video_dir /path/to/videos/ --dataset my_dataset --split training
    # Auto split train/test
    python prepare_data.py --video_dir /path/to/videos/ --dataset my_dataset --split_ratio 0.8
"""

import os
import sys
import argparse
import cv2


VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpg', '.mpeg'}


def get_video_files(path):
    """Get video files from a path (single file or directory)."""
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            return [path]
        print(f'[ERROR] {path} is not a supported video file')
        sys.exit(1)

    files = []
    for f in sorted(os.listdir(path)):
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
            files.append(os.path.join(path, f))
    if not files:
        print(f'[ERROR] No video files found in {path}')
        sys.exit(1)
    return files


def extract_frames(video_path, output_dir, resize=None):
    """Extract all frames from a video file to output_dir as numbered JPEGs.

    Args:
        video_path: path to the video file.
        output_dir: directory to write frames into.
        resize: optional (width, height) tuple.

    Returns:
        Number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[WARN] Cannot open {video_path}, skipping.')
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Determine zero-padding width so string sort == numeric sort
    pad = max(3, len(str(total - 1))) if total > 0 else 3

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize)
        cv2.imwrite(os.path.join(output_dir, f'{count:0{pad}d}.jpg'), frame)
        count += 1

    cap.release()
    return count


def main():
    parser = argparse.ArgumentParser(description='Extract video frames for anomaly prediction pipeline.')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Path to a single video file or a directory containing video files.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (output under Data/<dataset>/).')
    parser.add_argument('--data_root', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'Data'),
                        help='Root data directory (default: ../Data relative to this script).')
    parser.add_argument('--split', type=str, default=None, choices=['training', 'testing'],
                        help='Put all videos into this split. Mutually exclusive with --split_ratio.')
    parser.add_argument('--split_ratio', type=float, default=None,
                        help='Fraction of videos for training (rest goes to testing). '
                             'E.g. 0.8 means 80%% train, 20%% test.')
    parser.add_argument('--resize', type=str, default=None,
                        help='Resize frames to WxH, e.g. "256x256". Default: keep original size.')

    args = parser.parse_args()

    # Parse resize
    resize = None
    if args.resize:
        w, h = args.resize.lower().split('x')
        resize = (int(w), int(h))

    # Validate split args
    if args.split and args.split_ratio is not None:
        parser.error('--split and --split_ratio are mutually exclusive.')
    if args.split is None and args.split_ratio is None:
        parser.error('Must specify either --split or --split_ratio.')

    video_files = get_video_files(args.video_dir)
    print(f'Found {len(video_files)} video(s) in {args.video_dir}')

    # Assign videos to splits
    if args.split:
        split_map = {args.split: video_files}
    else:
        n_train = max(1, int(len(video_files) * args.split_ratio))
        split_map = {
            'training': video_files[:n_train],
            'testing': video_files[n_train:]
        }
        if not split_map['testing']:
            print('[WARN] No videos assigned to testing split. Consider lowering --split_ratio.')

    dataset_dir = os.path.join(args.data_root, args.dataset)

    for split_name, videos in split_map.items():
        frames_root = os.path.join(dataset_dir, split_name, 'frames')
        print(f'\n--- {split_name}: {len(videos)} video(s) ---')

        for idx, vpath in enumerate(videos):
            video_subdir = os.path.join(frames_root, f'{idx + 1:02d}')
            n_frames = extract_frames(vpath, video_subdir, resize=resize)
            print(f'  [{idx + 1:02d}] {os.path.basename(vpath)} -> {n_frames} frames')

    print(f'\nDone. Dataset directory: {os.path.abspath(dataset_dir)}')
    print(f'Structure:')
    for split_name in split_map:
        frames_root = os.path.join(dataset_dir, split_name, 'frames')
        if os.path.isdir(frames_root):
            subdirs = sorted(os.listdir(frames_root))
            print(f'  {split_name}/frames/ -> {len(subdirs)} video(s)')


if __name__ == '__main__':
    main()
