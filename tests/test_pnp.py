import cv2
import json
import numpy as np

cam_matrix = np.array([
    [1280.0, 0.0, 640.0],
    [0.0, 1280.0, 360.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

PALM_3D_MODEL = np.array([
    [ 0.000,  0.000,  0.0],
    [-0.030, -0.080,  0.0],
    [-0.010, -0.085,  0.0],
    [ 0.010, -0.080,  0.0],
    [ 0.030, -0.070,  0.0],
], dtype=np.float32)

theta = np.deg2rad(40.0)
R_pitch = np.array([
    [1.0, 0.0,            0.0],
    [0.0, np.cos(theta), -np.sin(theta)],
    [0.0, np.sin(theta),  np.cos(theta)],
])

cap = cv2.VideoCapture("data/raw/move_forward_1.mp4")
with open("data/poses/move_forward_1_poses.json") as f:
    pose_data = json.load(f)

writer = cv2.VideoWriter("pnp_visualization.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))

for frame_idx, pose in enumerate(pose_data["frames"]):
    ret, frame = cap.read()
    if not ret: break

    landmarks = pose.get("landmarks")
    if landmarks:
        img_pts = np.array([
            [landmarks[i][0] * 1280, landmarks[i][1] * 720]
            for i in [0, 5, 9, 13, 17]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            PALM_3D_MODEL, img_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # such a cool function
            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1, 4)            
            # map the translation vector to world coordinates
            world_pos = R_pitch @ tvec.flatten() # x y z
            
            depth = world_pos[1]
            height = -world_pos[2]
            
            cv2.putText(frame, f"Depth (Y): {depth:+.3f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Height (Z): {height:+.3f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    writer.write(frame)

cap.release()
writer.release()