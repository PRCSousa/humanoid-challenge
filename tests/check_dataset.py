"""Verify the built LeRobot dataset loads back correctly."""
import sys
from pathlib import Path

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


root = Path("data/lerobot_dataset").expanduser()
repo_id = "ReAscalon/humanoid_move_thing"

dataset = LeRobotDataset(repo_id=repo_id, root=root)

print(f"Loaded dataset: {len(dataset)} frames")
print(f"Episodes: {dataset.num_episodes}")
print(f"FPS: {dataset.fps}")
print()
print("Features:")
for k, v in dataset.features.items():
    print(f"  {k}: shape={v['shape']}, dtype={v['dtype']}")
print()

# Sample frame
s = dataset[0]
print("Frame 0 keys:", list(s.keys()))
for k, v in s.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)}, {v.dtype}")
    else:
        print(f"  {k}: {v}")

# Sample action from mid-episode
mid = len(dataset) // 2
s2 = dataset[mid]
print()
print(f"Frame {mid}:")
print(f"  action: {s2['action'].numpy() if hasattr(s2['action'], 'numpy') else s2['action']}")
print(f"  state:  {s2['observation.state'].numpy() if hasattr(s2['observation.state'], 'numpy') else s2['observation.state']}")