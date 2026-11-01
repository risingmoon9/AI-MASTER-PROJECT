"""Portable project/environment initialization; no local Windows paths."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MINARI_DATASETS_PATH', str(ROOT / 'minari_datasets'))
os.environ.setdefault('MUJOCO_GL', 'glfw' if sys.platform == 'win32' else 'egl')


def initialize():
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    os.chdir(ROOT)
    return ROOT
