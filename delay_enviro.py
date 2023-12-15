# import gym 
# from collections import deque
# import numpy as np
# import torch 
# from torch import distributions as pyd
# from gymnasium.utils import seeding

# mdp_params = np.load("mdp/mdp_25.npz")

# reward = mdp_params['reward']
# mu = mdp_params['mu']
# p_transition = mdp_params['p_transition']
# states = mdp_params['states']
# actions = mdp_params['n_actions']
# H = 1000
# K = 10000
# D = 50

# config = {
#         'action_dim': int(actions),
#         'state_dim': int(states),
#         'transition_function': torch.from_numpy(p_transition),
#         'initial_state_distribution': torch.from_numpy(mu),
#         'reward_function': torch.from_numpy(reward),
#         'seed': 0,
#         'training horizon H': H
# }

# def sample_from_logits(logits):
#     # pytorch sample from categorical distribution
#     sample = pyd.Categorical(logits=logits).sample()
#     return sample.item()

# class DelayedRewardEnv(gym.Env):
#     def __init__(self, env, delay):
#         self.env = env
#         self.delay = delay
#         self.reward_buffer = deque(maxlen=delay)

#         self.config = config
#         self.seed(config["seed"])
#         self.action_space = gym.spaces.Discrete(config['action_dim'])
#         self.observation_space = gym.spaces.Discrete(config['state_dim'])
#         # self.state_space = gym.spaces.Discrete(config['state_dim'])
#         self.initial_state_distribution = config["initial_state_distribution"]
#         self.transition_function = config["transition_function"]
#         self.reward_function = config["reward_function"]


#     def step(self, action):
#         observation = sample_from_logits(self.transition_function[self.state][action])
#         reward = self.reward_function[self.state, action].item()
#         terminated = False
#         truncated = False

#         observation, reward, done, info = self.env.step(action)
#         self.reward_buffer.append(reward)

#         info = {}
#         self.state = observation

#         delayed_reward = 0
#         if len(self.reward_buffer) == self.delay:
#             delayed_reward = self.reward_buffer.popleft()

#         return observation, delayed_reward, terminated, truncated, info

#     def reset(self):
#         self.reward_buffer.clear()
#         return self.env.reset()

#     def seed(self, seed=None):
#         """
#         Seed the environment.

#         Parameters
#         ----------
#         seed : int
#             Seed to use
#         """
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
    
#     def __str__(self):
#         return "DelayedMimicEnv"
    

# from stable_baselines3.common.env_checker import check_env
# MimicEnv = DelayedRewardEnv(config)

# # It will check your custom environment and output additional warnings if needed
# check_env(MimicEnv)
 
# # from gym.envs.registration import register

# # register(
# #     id='gym_examples/MimicEnv-v0',
# #     entry_point='MimicEnv',
# #     max_episode_steps=200,
# #     kwargs = {'config': 
# #     config}
# # )


# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3 import PPO

# MimicEnv = DelayedRewardEnv(config)
# MimicEnv = Monitor(MimicEnv, filename='ppo_rewards.csv', allow_early_resets=False)
# model = PPO("MlpPolicy", MimicEnv, verbose=1, tensorboard_log="", batch_size = 64)
# model.learn(total_timesteps=1000000)

#? #! PPO 8 is the only one 
#! PPO 8

import collections
from gymnasium import RewardWrapper
import numpy as np 

class DelayedRewardWrapper(RewardWrapper):
    def __init__(self, env, delay):
        super().__init__(env)
        self.delay = delay
        self.reward_buffer = collections.deque([0] * delay, maxlen=delay)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.reward_buffer.append(reward)

        # Get the delayed reward
        delayed_reward = self.reward_buffer[0]

        return observation, delayed_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset the buffer when the environment is reset
        self.reward_buffer.clear()
        self.reward_buffer.extend([0] * self.delay)
        return super().reset(**kwargs)

def random_delay(current_step, max_delay, total_timesteps):
    return np.random.randint(0, min(max_delay, total_timesteps - current_step))

class StochasticDelayedRewardWrapper(RewardWrapper):
    def __init__(self, env, max_delay, total_timesteps, delay_function):
        super().__init__(env)
        self.max_delay = max_delay
        self.total_timesteps = total_timesteps
        self.delay_function = delay_function
        self.reward_buffer = collections.deque([], maxlen=max_delay)
        self.current_step = 0

    def step(self, action):
        # Determine the delay for the current step
        current_delay = self.delay_function(self.current_step, self.max_delay, self.total_timesteps)
        current_delay = min(max(0, current_delay), len(self.reward_buffer))

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.reward_buffer.append(reward)

        # Get the delayed reward
        delayed_reward = self.reward_buffer[-current_delay - 1] if current_delay < len(self.reward_buffer) else 0

        self.current_step += 1
        return observation, delayed_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.reward_buffer.clear()
        self.current_step = 0
        return super().reset(**kwargs)

from environment import MDPEnv
import environment
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

total_timesteps = 1000000
MimicEnv = MDPEnv(environment.config)
MimicEnv = Monitor(MimicEnv, filename='ppo_rewards.csv', allow_early_resets=False)
stochastic_delayed_env = StochasticDelayedRewardWrapper(MimicEnv, max_delay=10000, total_timesteps = total_timesteps, delay_function=random_delay)

#! THIS WORKS
# delayed_env = DelayedRewardWrapper(MimicEnv, delay=5)  # Set the delay as needed
# model = PPO("MlpPolicy", delayed_env, verbose=1, tensorboard_log="")

model = PPO("MlpPolicy", stochastic_delayed_env, verbose=1, tensorboard_log="")
model.learn(total_timesteps=total_timesteps)

# env = gym.make('YourGymEnvironment')
# delayed_env = DelayedRewardWrapper(env, delay=5)  # Set the delay as needed

# Now use delayed_env for your training loop
