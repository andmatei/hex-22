import gymnasium as gym

from envs.hex import HexEnv

gym.register("hex", entry_point="envs.hex:HexEnv")