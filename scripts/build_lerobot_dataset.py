import argparse
import sys
import shutil
from pathlib import Path
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMAGE_SHAPE = (256, 256, 3)

FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channels"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": IMAGE_SHAPE,
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": [
            "eef_x", "eef_y", "eef_z",
            "eef_ax", "eef_ay", "eef_az",
            "gripper_qpos1", "gripper_qpos2",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
}

def load_episodes(ep_dir: Path) -> list[dict]:
    files = sorted(ep_dir.glob("*.npz"))
    if not files:
        sys.exit(f"No files found in {ep_dir}")

    episodes = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        episodes.append({
            "name": f.stem,
            "agentview": d["agentview"],
            "wrist": d["wrist"],
            "state": d["state"],
            "action": d["action"],
            "task": str(d.get("task", "pick up the object")),
            "fps": int(d.get("fps", 20)),
            "success": bool(d.get("success", False)),
        })
    return episodes

def build_dataset(episodes: list[dict], repo_id: str, root: Path, fps: int = 20):
    if root.exists():
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=FEATURES,
        root=root,
        robot_type="panda",
        use_videos=True
    )

    for i, ep in enumerate(episodes):
        T = len(ep["action"])
        print(f"[{i+1}/{len(episodes)}] {ep['name']}: {T} frames")

        for t in range(T):
            dataset.add_frame({
                "observation.images.image": np.asarray(ep["agentview"][t], dtype=np.uint8),
                "observation.images.image2": np.asarray(ep["wrist"][t], dtype=np.uint8),
                "observation.state": np.asarray(ep["state"][t], dtype=np.float32),
                "action": np.asarray(ep["action"][t], dtype=np.float32),
                "task": ep["task"],
            })
        dataset.save_episode()

    return dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/episodes")
    ap.add_argument("--root", default="data/lerobot_dataset")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--push_to_hub", action="store_true")
    args = ap.parse_args()

    ep_dir = Path(args.episodes).expanduser()
    root = Path(args.root).expanduser()

    episodes = load_episodes(ep_dir)
    dataset = build_dataset(episodes, args.repo_id, root, fps=args.fps)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "smolvla", "humanoid-challenge"],
            private=False,
        )

if __name__ == "__main__":
    main()