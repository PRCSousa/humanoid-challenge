import os
os.environ["MUJOCO_GL"] = "osmesa"  # egl doesnt work
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import cv2

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_object"]()
task = task_suite.get_task(0)
init_states = task_suite.get_task_init_states(0)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=1024,
    camera_widths=1024,
)
obs = env.reset()

# view
for i in range(200):
    action = np.random.uniform(-0.1, 0.1, size=7)
    obs, reward, done, info = env.step(action)
    
    # main camera view (Libero usually provides "agentview_image")
    img = obs["agentview_image"] 
    
    # osmesa renders upside down by default (???)
    img = cv2.flip(img, 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Libero", img_bgr)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

env.close()
cv2.destroyAllWindows()