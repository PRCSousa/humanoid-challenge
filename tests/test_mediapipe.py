import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd

# setup model
import urllib.request
import os
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                               # Palm base
]

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

vid_path = 'data/raw/clap_1.mp4'
out_path = 'outputs/clap_1.mp4'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

vid = cv2.VideoCapture(vid_path)
fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# write output video to mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

frame_idx = 0
n_detected = 0

landmark_data = pd.DataFrame(columns=['frame_idx', 'landmark_data', 'thumb_index_distance', 'landmark_zero_pos'])

while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    # cv2 default is BGR, convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # create mediapipe image and detect hands
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int(frame_idx * (1000 / fps))
    result = detector.detect_for_video(mp_image, ts_ms)

    # if hand detected
    if result.hand_landmarks:

        n_detected += 1

        # get landmarks to draw
        for hand_lms in result.hand_landmarks:

            h, w, _ = frame.shape

            for s, e in HAND_CONNECTIONS:

                # for each pair of HAND_CONNECTIONS, draw a line between the two landmarks
                cv2.line(frame,
                         (int(hand_lms[s].x * w), int(hand_lms[s].y * h)),
                         (int(hand_lms[e].x * w), int(hand_lms[e].y * h)),
                         (0, 0, 0), 2)
            
            # for each landmark, draw a little point
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

            # compute distance between thumb tip and index tip
            thumb = hand_lms[4]
            index = hand_lms[8]

            thumb_index_dist = float(np.linalg.norm(
                np.array([thumb.x, thumb.y]) -
                np.array([index.x, index.y])
            )) # calculate euclidean distance
            
            # draw line between thumb and index
            cv2.line(frame,
                     (int(thumb.x * w), int(thumb.y * h)),
                     (int(index.x * w), int(index.y * h)),
                     (0, 0, 255), 2) # color in BGR
            
            cv2.putText(frame, f"thumb_index_distance = {thumb_index_dist:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            row = pd.DataFrame({
                'frame_idx': [frame_idx],
                'landmark_data': [[(lm.x, lm.y, lm.z) for lm in hand_lms]],
                'thumb_index_distance': [thumb_index_dist],
                'landmark_zero_pos': [(hand_lms[0].x, hand_lms[0].y, hand_lms[0].z)]
            })
            landmark_data = pd.concat([landmark_data, row], ignore_index=True)
    else:
        cv2.putText(frame, "no hand detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        row = pd.DataFrame({
            'frame_idx': [frame_idx],
            'landmark_data': [None],
            'thumb_index_distance': [None],
            'landmark_zero_pos': [None]
        })
        landmark_data = pd.concat([landmark_data, row], ignore_index=True)

    cv2.putText(frame, f"frame {frame_idx}/{frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # write frame to output video
    writer.write(frame)
    frame_idx += 1
    if frame_idx % 60 == 0:
        print(f"  {frame_idx}/{frame_count} frames", end="\r")

vid.release()
writer.release()
detector.close()

print(f"\nResult in: {out_path}")
landmark_data.to_csv('data/poses/debug_landmark_data.csv', index=False)