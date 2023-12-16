# import gym
# import dapo
# import environment as env_setup
# from  environment import MDPEnv

import torch
import numpy as np
from dapo import DAPO

config = {
        # this is for both the environment and the agent
        # action space is list of actions, dimensions A * 1
        'action_space':['one', 'two', 'three', 'four', 'five', 'six'],
        # state space is list of states, dimensions S * 1
        'state_space': ['one', 'two', 'three', 'four', 'five', 'six'],
        # transition probabilities are a tensor of probabilities, dimensions S * S * A
        # 'transition_function': torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        'transition_function': torch.ones((6, 6, 6)),

        # initial state distribution is logits
        'initial_state_distribution': torch.tensor([0.5, 0.5]),
        # reward function is S * A
        'reward_function': torch.ones((6, 6)),

        'training horizon H': 5,
        'episodes K': 100,
        'eta': 0.01,
        'gamma': 0.1        
}

MIMIC_DAPO = DAPO(config)
MIMIC_DAPO.run()