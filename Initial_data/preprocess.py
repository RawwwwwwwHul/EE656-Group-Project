"""
Preprocessing script for FF++ dataset.

Runs MTCNN face detection on all videos once, saves cropped 6-frame clips
as .pt files to an output folder. After this, training never needs to run
MTCNN again -- just loads the .pt files directly.

Output folder structure:
  clips/
    real/
      000_clip0.pt
      000_clip1.pt
      ...
    fake/
      DeepFakeDetection_001_clip0.pt
      ...

Each .pt file contains a dict:
  {
    "clip"  : FloatTensor of shape (T, C, H, W),
    "label" : int (0=real, 1=fake),
    "video" : str (source video filename),
    "method": str (e.g. "Face2Face" or "real")
  }

Usage:
  python preprocess.py --src D:\Summer\ISTVT\mini_ffpp --dst D:\Summer\ISTVT\clips

You can then send the entire clips/ folder to your friend's device and
train directly from it without needing the original videos or MTCNN.
"""

import os
import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN


# --- settings ---
REAL_DIR  = "original"
FAKE_DIRS = ["DeepFakeDetection", "Face2Face", "FaceSwap", "NeuralTextures"]
SEQ_LEN   = 6      # frames per clip
IMG_SIZE  = 128    # resize face crops to this
MAX_FRAMES = 270   # max frames to read per video (paper's cap)


def get_nose_centered_box(landmarks, img_w, img_h, scale=1.25):
    """
    Bounding box centered on nose tip, scale x face extent.
    landmarks: (5, 2) -- [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    nose      = landmarks[2]
    left_eye  = landmarks[0]
    right_eye = landmarks[1]
    left_mouth = landmarks[3]

    cx, cy = nose
    face_w = abs(right_eye[0] - left_eye[0]) * 2
    face_h = abs(left_mouth[1] - left_eye[1]) * 2
    size   = max(face_w, face_h) * scale

    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = int(cx + size / 2)
    y2 = int(cy + size / 2)

    # clamp to frame
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    return x1, y1, x2, y2


def process_video(video_path, mtcnn, img_size, seq_len, max_frames):
    """
    Reads a video, detects faces, returns list of (T, C, H, W) clip tensors.
    Returns empty list if not enough faces detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    crops = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # MTCNN detection
        pil_img = Image.fromarray(rgb)
        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)

        if boxes is None or landmarks is None:
            continue

        best = int(np.argmax(probs))
        lm   = landmarks[best]

        x1, y1, x2, y2 = get_nose_centered_box(lm, w, h, scale=1.25)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (img_size, img_size))
        crop_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        crops.append(crop_t)

    cap.release()

    if len(crops) < seq_len:
        return []

    # assemble non-overlapping clips
    clips = []
    for i in range(0, len(crops) - seq_len + 1, seq_len):
        clip = torch.stack(crops[i:i + seq_len], dim=0)  # (T, C, H, W)
        clips.append(clip)

    return clips


def collect_videos(src_root):
    """Returns list of (video_path, label, method) tuples."""
    videos = []

    # real
    real_path = os.path.join(src_root, REAL_DIR)
    for f in sorted(os.listdir(real_path)):
        if f.endswith(".mp4"):
            videos.append((os.path.join(real_path, f), 0, "real"))

    # fake
    for method in FAKE_DIRS:
        method_path = os.path.join(src_root, method)
        if not os.path.exists(method_path):
            print(f"  Skipping missing folder: {method_path}")
            continue
        for f in sorted(os.listdir(method_path)):
            if f.endswith(".mp4"):
                videos.append((os.path.join(method_path, f), 1, method))

    return videos


def main(src_root, dst_root):
    os.makedirs(os.path.join(dst_root, "real"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "fake"), exist_ok=True)

    mtcnn = MTCNN(keep_all=False, device="cpu", post_process=False)

    videos = collect_videos(src_root)
    print(f"\nFound {len(videos)} videos to process")
    print(f"Saving clips to: {dst_root}\n")

    total_clips = 0
    skipped     = 0

    # tqdm gives us the progress bar with ETA
    for video_path, label, method in tqdm(videos, desc="Processing videos",
                                           unit="video", ncols=80):
        clips = process_video(video_path, mtcnn, IMG_SIZE, SEQ_LEN, MAX_FRAMES)

        if not clips:
            skipped += 1
            tqdm.write(f"  Skipped (no face / too short): {os.path.basename(video_path)}")
            continue

        # save each clip as a separate .pt file
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        subfolder  = "real" if label == 0 else "fake"

        for i, clip in enumerate(clips):
            filename = f"{method}_{video_name}_clip{i}.pt"
            save_path = os.path.join(dst_root, subfolder, filename)
            torch.save({
                "clip"  : clip,
                "label" : label,
                "video" : os.path.basename(video_path),
                "method": method,
            }, save_path)

        total_clips += len(clips)
        tqdm.write(f"  {os.path.basename(video_path)} -> {len(clips)} clips")

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Videos processed : {len(videos) - skipped}/{len(videos)}")
    print(f"  Videos skipped   : {skipped}")
    print(f"  Total clips saved: {total_clips}")
    print(f"  Output folder    : {dst_root}")
    print(f"\nReal clips : {len(os.listdir(os.path.join(dst_root, 'real')))}")
    print(f"Fake clips : {len(os.listdir(os.path.join(dst_root, 'fake')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Path to mini_ffpp folder")
    parser.add_argument("--dst", required=True,
                        help="Where to save the processed clips")
    args = parser.parse_args()
    main(args.src, args.dst)
