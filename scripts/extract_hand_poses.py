import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

def get_model(model_path: Path) -> Path:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
    return model_path

def landmarks_to_pose(lm_xyzw: np.ndarray) -> dict:
    # Wrist
    wrist_pos = lm_xyzw[0]

    # Calculate orientation matrix from index finger and middle finger vectors to find normal
    v1 = lm_xyzw[9] - lm_xyzw[0]
    v2 = lm_xyzw[17] - lm_xyzw[5]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-8 or n2 < 1e-8:
        # If the vectors are too small, return the identity matrix as a fallback.
        basis = np.eye(3)
    else:
        # Find the normal vector to the plane defined by the two vectors
        v1 = v1 / n1
        v2 = v2 / n2
        z = np.cross(v1, v2)
        nz = np.linalg.norm(z)
        if nz < 1e-8:
            basis = np.eye(3)
        else:
            z = z / nz
            x = v1
            y = np.cross(z, x)
            basis = np.stack([x, y, z])

    thumb_index_dist = float(np.linalg.norm(lm_xyzw[4] - lm_xyzw[8]))

    return {
        "wrist_pos": wrist_pos.tolist(),
        "orientation_matrix": basis.tolist(),
        "thumb_index_dist": thumb_index_dist,
        "landmarks": lm_xyzw.tolist(),
    }

def draw_overlay(frame: np.ndarray, lm_xyzw: np.ndarray | None,
                 thumb_index_dist: float | None, frame_idx: int, n_frames: int):
    vis = frame.copy()
    h, w = vis.shape[:2]

    if lm_xyzw is not None:
        pts = [(int(x * w), int(y * h)) for x, y, _ in lm_xyzw]
        for s, e in HAND_CONNECTIONS:
            cv2.line(vis, pts[s], pts[e], (0, 255, 0), 2)
        for cx, cy in pts:
            cv2.circle(vis, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        # Thumb - index line
        cv2.line(vis, pts[4], pts[8], (255, 0, 255), 2)

        # thumb_index_dist text
        if thumb_index_dist is not None:
            color = (0, 255, 0) if thumb_index_dist > 0.05 else (0, 128, 255)
            cv2.putText(vis, f"thumb index distance={thumb_index_dist:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(vis, "no hand", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(vis, f"frame {frame_idx}/{n_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis

def extract_one(video_path: Path, out_path: Path, debug_path: Path | None, model_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"{video_path} does not exist.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer if debug_path is provided
    writer = None
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (w, h))

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    frames = []
    n_detected = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(idx * (1000.0 / fps))
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            lm_xyzw = np.array([[l.x, l.y, l.z] for l in lm])
            pose = landmarks_to_pose(lm_xyzw)
            pose["frame"] = idx
            pose["time"] = idx / fps
            n_detected += 1
            
            if writer:
                vis_frame = draw_overlay(frame, lm_xyzw, pose["thumb_index_dist"], idx, n_total)
                writer.write(vis_frame)
        else:
            pose = {
                "frame": idx,
                "time": idx / fps,
                "wrist_pos": None,
                "orientation_matrix": None,
                "thumb_index_dist": None,
                "landmarks": None,
            }
            if writer:
                vis_frame = draw_overlay(frame, None, None, idx, n_total)
                writer.write(vis_frame)

        frames.append(pose)

        idx += 1
        if idx % 60 == 0:
            pct = 100.0 * idx / max(n_total, 1)
            print(f"  [{video_path.name}] {idx}/{n_total} ({pct:.0f}%)", end="\r", flush=True)

    cap.release()
    if writer:
        writer.release()
    landmarker.close()
    
    summary = {
        "video": video_path.name,
        "fps": fps,
        "width": w,
        "height": h,
        "n_frames": len(frames),
        "n_detected": n_detected,
        "detection_rate": n_detected / max(len(frames), 1),
        "frames": frames,
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary))

    print(f"{video_path.name}: {n_detected}/{len(frames)} "
          f"({100 * summary['detection_rate']:.1f}%) detected".ljust(80))
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/poses")
    ap.add_argument("--debug_dir", default="data/poses/debug")
    ap.add_argument("--model", default="hand_landmarker.task")
    ap.add_argument("--glob", default="*")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    debug_dir = Path(args.debug_dir).expanduser() if args.debug_dir else None

    model_path = get_model(Path(args.model))

    videos = []
    for ext in ["*.mp4"]:
        videos.extend(raw_dir.glob(f"{args.glob}{ext[1:]}"))
    videos = sorted(set(videos))

    print(f"Processing {len(videos)} video(s) from {raw_dir}\n")

    for v in videos:
        out_path = out_dir / f"{v.stem}_poses.json"
        dbg_path = (debug_dir / f"{v.stem}_debug.mp4") if debug_dir else None
        extract_one(v, out_path, dbg_path, model_path)

if __name__ == "__main__":
    main()