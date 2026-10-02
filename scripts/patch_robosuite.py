import robosuite
import os

# Force OSMesa because the default wasn't working
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
