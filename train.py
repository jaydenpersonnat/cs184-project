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
# def main():
    
#     MIMIC_DAPO.run()
    


    # Initialize the environment
    

    # Initialize the agent with the environment
    # agent = DAPO(config)
    
#     for episode in range(num_episodes): 
#         agent.run(episode)

#     num_episodes = 1000  # Define the number of episodes for training

#     for episode in range(num_episodes):
#         state = env.reset()
#         done = False
#         total_reward = 0

#         while not done:
#             # The agent selects an action
#             action = agent.select_action(state)

#             # Environment executes the action and returns next state, reward, and done status
#             next_state, reward, done, _ = env.step(action)

#             # The agent learns from the experience
#             agent.learn(state, action, reward, next_state, done)

#             state = next_state
#             total_reward += reward

#         print(f"Episode: {episode}, Total Reward: {total_reward}")

#     # Save the trained model, if needed
#     agent.save_model("path_to_save_model")

# if __name__ == "__main__":
#     main()
