import gymnasium as gym
import numpy as np
from gymnasium import spaces
import environment
from environment import MDPEnv
import torch
import torch.nn as nn
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# class RewardLoggingCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.total_rewards = []

#     def _on_step(self) -> bool:
#         return True

#     def _on_rollout_end(self) -> None:
#         """
#         This event is triggered before updating the policy.
#         """
#         print('episode reward', self.training_env.info)
#         pass

MimicEnv = MDPEnv(environment.config)
MimicEnv = Monitor(MimicEnv, filename='ppo_rewards.csv', allow_early_resets=False)
model = PPO("MlpPolicy", MimicEnv, verbose=1, tensorboard_log="")
model.learn(total_timesteps=1000000)

# obs = env.reset()
# n_steps = 20
# for step in range(n_steps):
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   env.render(mode='console')
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     breaks