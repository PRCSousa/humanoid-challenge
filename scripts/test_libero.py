import os
os.environ["MUJOCO_GL"] = "osmesa"  # egl doesnt work
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_object"]()
print(f"Number of tasks: {task_suite.n_tasks}")

task = task_suite.get_task(0)
print(f"Task: {task.language}")

init_states = task_suite.get_task_init_states(0)
print(f"Number of init states: {len(init_states)}")

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)
print(f"BDDL file: {task_bddl_file}")
assert os.path.exists(task_bddl_file), f"BDDL not found at {task_bddl_file}"

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=256,
    camera_widths=256,
)
obs = env.reset()
print(f"Observation keys: {obs.keys()}")

for i in range(5):
    action = np.random.uniform(-0.1, 0.1, size=7)
    obs, reward, done, info = env.step(action)
    print(f"Step {i}: reward={reward:.3f}, done={done}")

print("LIBERO OK")
env.close()
