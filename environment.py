import random

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from torch import distributions as pyd
import torch

# state space tensor
# action space tensor
# reward function
# transition function (transition probability array)
# initial state distribution

def reward (state, action):
    pass

# def transition (state, action, next_state):
#     transition_probs = np.array([[0.5, 0.5], [0.5, 0.5]])
#     return transition_probs[state][next_state][action]
#     pass

config = {
            # action space is list of actions, dimensions A * 1
          'action_space':['one', 'two', 'three', 'four', 'five', 'six'],
            # state space is list of states, dimensions S * 1
          'state_space': ['one', 'two', 'three', 'four', 'five', 'six'],
            # initial state distribution is a list of probabilities, dimensions S * 1
          'initial_state_distribution': np.array([0.5, 0.5]),
            # transition probabilities are a tensor of probabilities, dimensions S * S * A
          'transition_function': torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
            # reward function is a function S * A -> R
          'reward_function': reward}


def sample_from_logits(logits):
    # pytorch sample from categorical distribution
    sample = pyd.Categorical(logits=logits).sample()
    return sample.item()

class MDPEnv(gym.Env):
    def __init__(self, config):
        """
        Initialize environment.

        Parameters
        ----------
        config : dict
            Environment configuration
            If to seed the action space as well
        """
        self.config = config
        self.seed(config["seed"])
        self.action_space = gym.spaces.Discrete(len(config["action_space"]))
        self.state_space = gym.spaces.Discrete(len(config["state_space"]))
        self.state = sample_from_logits(config["initial_state_distribution"])
        self.transition_function = config["transition_function"]
        self.reward_function = config["reward_function"]

    def step(self, action):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            Action to take

        Returns
        -------
        observation : int
            Observation after taking action
        reward : float
            Reward after taking action
        done : bool
            Whether the episode is over
        info : dict
            Additional information
        """
        observation = sample_from_logits(self.transition_function[self.state][action])
        reward = self.reward_function(self.state, action)
        done = False
        info = {}
        self.state = observation
        return observation, reward, done, info
    
    def reset(self):
        """
        Reset the environment.

        Returns
        -------
        observation : int
            Initial observation
        """
        self.state = sample_from_logits(self.config["initial_state_distribution"])
        return self.state
    
    def seed(self, seed=None):
        """
        Seed the environment.

        Parameters
        ----------
        seed : int
            Seed to use
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode="human"):
        """
        Render the environment.

        Parameters
        ----------
        mode : str
            Mode to render with
        """
        pass

    def close(self):
        """
        Close the environment.
        """
        pass

    def __str__(self):
        return "MimicEnv"
