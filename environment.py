import random

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from torch import distributions as pyd
import torch


mdp_params = np.load("mdp/mdp_25.npz")

reward = mdp_params['reward']
mu = mdp_params['mu']
p_transition = mdp_params['p_transition']
states = mdp_params['states']
actions = mdp_params['n_actions']
H = 1000
K = 10000
D = 50

# def transition (state, action, next_state):
#     transition_probs = np.array([[0.5, 0.5], [0.5, 0.5]])
#     return transition_probs[state][next_state][action]
#     pass

config = {
        # this is for both the environment and the agent
        # action space is list of actions, dimensions A * 1
        'action_dim': int(actions),
        # state space is list of states, dimensions S * 1
        'state_dim': int(states),
        # transition probabilities are a tensor of probabilities, dimensions S * S * A
        # 'transition_function': torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        'transition_function': torch.from_numpy(p_transition),
        # initial state distribution is logits
        'initial_state_distribution': torch.from_numpy(mu),
        # reward function is S * A
        'reward_function': torch.from_numpy(reward),
        'seed': 0,
        'training horizon H': H
}

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
        self.action_space = gym.spaces.Discrete(config['action_dim'])
        self.observation_space = gym.spaces.Discrete(config['state_dim'])
        # self.state_space = gym.spaces.Discrete(config['state_dim'])
        self.initial_state_distribution = config["initial_state_distribution"]
        self.transition_function = config["transition_function"]
        self.reward_function = config["reward_function"]

        # self.H = config["training horizon H"]
        # self.info = {}
        # self.episode = 0
        # self.total_reward = 0.



    def reset(self, seed=None):
        """
        Reset the environment.

        Returns
        -------
        observation : int
            Initial observation
        """
        super().reset(seed=seed)
        self.state = sample_from_logits(self.initial_state_distribution)
        info = {}
        
        # self.info[self.episode] = {'r': self.total_reward,
        #                           'l': self.H,
        #                           't': 0.}
        # self.total_reward = 0.
        # info = {'total reward': self.total_reward}
        # # print('total reward', self.total_reward)
        # self.info = info
        return self.state, info


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
        reward = self.reward_function[self.state, action].item()
        terminated = False
        truncated = False
        # if self.H == 0:
        #     terminated = True
        #     self.episode += 1
        #     self.info[self.episode] = {'r': self.total_reward,
        #                                'l': self.H,
        #                                't': 0.}
        #     self.total_reward = 0.
        # else:
        #     self.H =- 1
        
        info = {}
        self.state = observation
        # self.total_reward += reward

        # if terminated:
        #     info['total reward'] = self.total_reward
        #     self.total_reward = 0.

        return observation, reward, terminated, truncated, info
    
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



# check environment
from stable_baselines3.common.env_checker import check_env
MimicEnv = MDPEnv(config)

# It will check your custom environment and output additional warnings if needed
check_env(MimicEnv)


from gym.envs.registration import register

register(
    id='gym_examples/MimicEnv-v0',
    entry_point='MimicEnv',
    max_episode_steps=200,
    kwargs = {'config': config}
)
