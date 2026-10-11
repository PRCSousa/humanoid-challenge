import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PALM_3D_MODEL = np.array([
    [ 0.000,  0.000,  0.000],   # Wrist
    [-0.030,  0.080,  0.005],   # Index MCP
    [ 0.000,  0.085,  0.010],   # Middle MCP
    [ 0.015,  0.080,  0.006],   # Ring MCP
    [ 0.030,  0.070, -0.002],   # Pinky MCP (added small Z offsets to break coplanarity and help with PnP disambiguation)
], dtype=np.float32)

PALM_LANDMARK_INDICES = [0, 5, 9, 13, 17]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
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

def draw_skeleton(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 255, 0), 1)
    for cx, cy in pts:
        cv2.circle(frame, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

def draw_bar(img, label, value, min_val, max_val, x, y, w, h, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (160, 160, 160), 1)
    val_clamped = max(min_val, min(value, max_val))
    
    if min_val < 0:
        zero_px = x + int(w * (0 - min_val) / (max_val - min_val))
        val_px = x + int(w * (val_clamped - min_val) / (max_val - min_val))
        if val_px > zero_px:
            cv2.rectangle(img, (zero_px, y), (val_px, y + h), color, -1)
        else:
            cv2.rectangle(img, (val_px, y), (zero_px, y + h), color, -1)
        cv2.line(img, (zero_px, y - 2), (zero_px, y + h + 2), (255, 255, 255), 1)
    else:
        val_px = x + int(w * (val_clamped - min_val) / (max_val - min_val))
        cv2.rectangle(img, (x, y), (val_px, y + h), color, -1)
    cv2.putText(img, f"{label}: {value:+.2f}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

def solve_both_branches(model, img_pts, cam_matrix, dist_coeffs):
    """Return both valid PnP solutions for a planar/semi-planar palm."""
    ok_a, rvec_a, tvec_a = cv2.solvePnP(
        model, img_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    if not ok_a:
        return []

    model_b = model.copy()
    model_b[:, 2] *= -1.0
    ok_b, rvec_b, tvec_b = cv2.solvePnP(
        model_b, img_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    if not ok_b:
        return [(rvec_a, tvec_a)]

    return [(rvec_a, tvec_a), (rvec_b, tvec_b)]

def refine_branch(model, img_pts, cam_matrix, dist_coeffs, rvec, tvec):
    ok, rvec, tvec = cv2.solvePnP(
        model, img_pts, cam_matrix, dist_coeffs,
        rvec=rvec.copy(), tvec=tvec.copy(),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    return rvec, tvec

def auto_disambiguate_by_depth(lm, candidate_branches, active_model):
    
    # Extract mediapipe's relative depth for the 5 palm landmarks
    # Smaller MediaPipe Z = closer to camera

    mp_z = np.array([lm[idx].z for idx in PALM_LANDMARK_INDICES])
    mp_z = mp_z - np.mean(mp_z)  # Center to get relative depth slopes
    
    best_branch = None
    best_corr = -float('inf')
    
    for rvec, tvec in candidate_branches:

        # Project the active 3D model into camera space
        R, _ = cv2.Rodrigues(rvec)
        
        # pts_cam: The 3D coordinates of the hand model in OpenCV camera space
        pts_cam = (R @ active_model.T).T + tvec.T
        
        # Extract the camera-space Z for the 5 points
        pnp_z = pts_cam[:, 2].flatten()
        pnp_z = pnp_z - np.mean(pnp_z) 
        
        # Correlate PnP depth with MediaPipe depth
        # The solution with the highest positive correlation is likely the correct orrientation
        corr = np.dot(mp_z, pnp_z)
        
        if corr > best_corr:
            best_corr = corr
            best_branch = (rvec, tvec)
            
    return best_branch

def main():

    model_path = get_model(Path("hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    f = w * 0.8
    cam_matrix = np.array([
        [f, 0.0, w / 2.0],
        [0.0, f, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    ACTIVE_PALM_MODEL = PALM_3D_MODEL.copy()

    mode = "waiting"
    R_0 = tvec_0 = prev_rvec = prev_tvec = None

    EMA_ALPHA = 0.3
    ema_dx = ema_dy = ema_dz = ema_pinch = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int((time.time() - start_time) * 1000))

        # HUD
        cv2.rectangle(frame, (10, 10), (220, 160), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (220, 160), (120, 120, 120), 1)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_skeleton(frame, lm, w, h)
            
            img_pts = np.array(
                [[lm[idx].x * w, lm[idx].y * h] for idx in PALM_LANDMARK_INDICES],
                dtype=np.float32)

            if mode == "locked":
                rvec, tvec = refine_branch(
                    ACTIVE_PALM_MODEL, img_pts, cam_matrix, dist_coeffs,
                    prev_rvec, prev_tvec)

                if rvec is None:
                    mode = "waiting"
                    continue

                prev_rvec, prev_tvec = rvec.copy(), tvec.copy()
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.08, 2)

                delta_local = R_0.T @ (tvec - tvec_0)
                raw_dx, raw_dy, raw_dz = (delta_local.flatten() * 100)
                raw_pinch = float(np.linalg.norm(
                    np.array([lm[4].x * w, lm[4].y * h]) - np.array([lm[8].x * w, lm[8].y * h])))

                ema_dx = (1 - EMA_ALPHA) * ema_dx + EMA_ALPHA * raw_dx
                ema_dy = (1 - EMA_ALPHA) * ema_dy + EMA_ALPHA * raw_dy
                ema_dz = (1 - EMA_ALPHA) * ema_dz + EMA_ALPHA * raw_dz
                ema_pinch = (1 - EMA_ALPHA) * ema_pinch + EMA_ALPHA * raw_pinch

                draw_bar(frame, "Local X (Right)",   ema_dx, -15, 15, 20, 40, 190, 10, (0, 0, 255))
                draw_bar(frame, "Local Y (Forward)", ema_dy, -15, 15, 20, 80, 190, 10, (0, 255, 0))
                draw_bar(frame, "Local Z (Down)",      ema_dz, -15, 15, 20, 120, 190, 10, (255, 0, 0))

        if mode == "waiting":
            prev_rvec = prev_tvec = None
            text = "PRESS SPACE" if result.hand_landmarks else "NO HAND"
            color = (255, 255, 0) if result.hand_landmarks else (0, 0, 255)
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        cv2.imshow("Camera-Agnostic PnP", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if result.hand_landmarks:
                w_pt = np.array([lm[0].x, lm[0].y, lm[0].z])
                m_pt = np.array([lm[9].x, lm[9].y, lm[9].z])
                i_pt = np.array([lm[5].x, lm[5].y, lm[5].z])
                p_pt = np.array([lm[17].x, lm[17].y, lm[17].z])
                
                true_length = np.linalg.norm(m_pt - w_pt)
                true_width = np.linalg.norm(p_pt - i_pt)
                
                if true_length > 1e-6:
                    target_width = 0.085 * (true_width / true_length)
                    scale_x = target_width / 0.060
                    ACTIVE_PALM_MODEL = PALM_3D_MODEL.copy()
                    ACTIVE_PALM_MODEL[:, 0] *= scale_x

                candidate_branches = solve_both_branches(
                    ACTIVE_PALM_MODEL, img_pts, cam_matrix, dist_coeffs)
                
                if candidate_branches:
                    rvec, tvec = auto_disambiguate_by_depth(lm, candidate_branches, ACTIVE_PALM_MODEL)
                    if rvec is not None:
                        tvec_0 = tvec.copy()
                        R_0, _ = cv2.Rodrigues(rvec)
                        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()
                        ema_dx = ema_dy = ema_dz = ema_pinch = 0.0
                        mode = "locked"
            else:
                R_0 = tvec_0 = prev_rvec = prev_tvec = None
                mode = "waiting"

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()