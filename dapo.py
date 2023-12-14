import torch
# import torch.optim as optim
from torch import distributions as pyd
import numpy as np
# import environment as env_setup

# MimicEnv = env_setup.MDPEnv(env_setup.config_env)

# write a 6x6 tensor with half zeros and half ones randomly


thing = torch.zeros((6, 6))
thing[2][4] = 1
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
        'reward_function': thing,

        'training horizon H': 100,
        'episodes K': 1000,
        'eta': 0.01,
        'gamma': 0.1        
}


class DAPO:
    def __init__(self, config): 
        # Initialize the environment
        # self.env = env
        self.action_space = config['action_space']
        self.state_space = config['state_space']
        self.transition_function = config['transition_function']
        self.initial_state_distribution = config['initial_state_distribution']
        self.H = config["training horizon H"]
        self.K = config["episodes K"]
        self.reward_function = config['reward_function']
        
        # learning rate, exploration parameter
        self.eta = config["eta"]
        self.gamma = config["gamma"]

        # initialize state and time step
        # self.state = sample_from_logits(self.initial_state_distribution)

        # initialize space dimensions
        self.A = 6 # self.action_space.shape[0]
        self.S = 6 # self.state_space.shape[0]

        # initialize policy, occupancy measures, and visit counters
        # self.policy_history = torch.zeros(self.S, self.A, self.H, self.K)
        self.policy_history = (1.0 / self.A) * torch.ones(self.S, self.A, self.H, self.K)
        # print('policy_history shape', self.policy_history.shape)

        self.delay_dict = {}
        for i in range(self.K): self.delay_dict[i] = []

    def sample_from_logits(self, logits):
        # pytorch sample from categorical distribution
        sample = pyd.Categorical(logits=logits).sample()
        return sample.item()

    def play_one_step(self, current_state, policy, h):
        # Play one step in the environment and return the next state, reward, and done flag
        # print('policy at k', policy)
        # print('action distribution at k', policy[current_state, :, h])


        action = self.sample_from_logits(policy[current_state, :, h])
        next_state = self.sample_from_logits(self.transition_function[current_state][action])
        reward = self.reward_function[current_state, action]
        return action, next_state, reward

    def play_episode(self, policy):
        traj = []
        # sample initial state
        current_state = self.sample_from_logits(self.initial_state_distribution)
        for h in range(self.H):
            action, next_state, reward = self.play_one_step(current_state, policy, h)
            traj.append((current_state, action, reward))
            current_state = next_state
        return traj
    
    def observe_feedback(self, traj): # observe \{c^j_h(s^j_h,a^j_h)\}^{H}_{h=1}
        rewards = torch.zeros(self.H)
        for h in range(self.H - 1):
            s, a = traj[h][0], traj[h][1]
            rewards[h] = self.reward_function[s, a]
        return rewards
    
    def get_n_step_transition (self, h, k):
        # returns a tensor of shape S * S which gives the one step transition probabilities
        # from each state to each state

        # initialize transition probabilities
        p = torch.zeros(self.S, self.S)
        for i in range(self.S):
            for j in range(self.S):
                for a in range(self.A):
                    p[i, j] += self.policy_history[i, a, h, k] * self.transition_function[i, j, a]
        return p
    
    def get_occupancy(self, h, k):
        # get the n-step transition probabilities
        p = self.get_n_step_transition(h, k)
        p_h = torch.linalg.matrix_power(p, h)

        # adjust for the fact that initial state is stochastic
        p_adj = torch.zeros(self.S)
        for i in range(self.S):
            for j in range(self.S):
                p_adj[i] += p_h[j] * self.initial_state_distribution[j]

        # p_adj represents the unconditional n-step transition probabilities
        # p_adj[i] is the probability of being in state i after h steps

        occ_measure = torch.zeros(self.S, self.A)
        for i in range(self.S):
            for a in range(self.A):
                occ_measure[i, a] = self.policy_history[i, a, h, k] * p_adj[i]

        # we want the sum over actions
        occ_measure = torch.sum(occ_measure, dim = 1)

        return occ_measure

    def run(self):
        for k in range(self.K):

            # DELAYS! #! HOW ARE WE TRACKING THE DELAYED TRAJECTORIES? 
            # put k in list in dictionary at some value >= k, < K
            #! RANDOMMMMMMMMMM
            rdm_num = np.random.choice(np.arange(k, self.K))
            self.delay_dict[rdm_num].append(k)
            delayed = self.delay_dict[k]

            # Play episode k with policy $\pi_k$ and observe trajectory
            # print('policy at k', self.policy_history[:,:,:,k])
            k_trajectory = self.play_episode(self.policy_history[:,:,:,k])
            k_rewards = self.observe_feedback(k_trajectory)

            print(f"Episode: {k}, Total Reward: {torch.sum(k_rewards)}")
        
            Q = torch.zeros(self.S, self.A, self.H, len(delayed))
            B = torch.zeros(self.S, self.A, self.H + 1, len(delayed))

            # numerator_sum = 0

            #! A DICTIONARY 
            for j in delayed: 
                #? lol idk 
                # get trajectory corresponding to delayed trajectory
                trajectory = self.play_episode(self.policy_history[:,:,:,j])
                rewards = self.observe_feedback(trajectory)

                for h in range(self.H, 0): 
                    j_policy = self.policy_history[:, :, h, j]
                    k_policy = self.policy_history[:, :, h, k]

                    # calculate r - this is an S * A
                    r = j_policy / torch.max(j_policy, k_policy) # s by a matrix for time step h

                    # calculate L - this is also a scalar (not stored)
                    L = np.sum(rewards[h:]) 

                    # Q-hat has to be stored, so it's S * A * H * J
                    # replace [s][a] entry of Q
                    sh, ah = trajectory[h][0], trajectory[h][1]
                    Q[sh][ah][h][j] = r[sh, ah] * L / \
                            (self.get_occupancy(h, j)[sh] * j_policy[sh][ah] + self.gamma)

                    # B also has to be stored, so it's S * A * H * J
                    # calculate b, don't need to store so it's just S * 1 tensor
                    b = torch.zeros(self.S)
                    for s in range(self.S):
                        b_sum = 0.
                        for a in range(self.A):
                            b_sum += k_policy[s][a] * r[s][a] / \
                                     (self.get_occupancy(h, j)[s] * j_policy[s][a] + self.gamma)
                        b[s] = 3 * self.gamma * self.H * b_sum

                    # b = (3 * self.gamma * self.H) * torch.sum(k_policy[s]) * torch.sum(r[s]) / \
                    #         (self.get_occupancy(h, k)[s] * torch.sum(j_policy[s]) + self.gamma)

                    # Calculate a slice of B for this h and j
                    for s in range(self.S):
                        for a in range(self.A):
                            summation = 0
                            for s_prime in range(self.S):
                                for a_prime in range(self.A): 
                                    term = self.transition_function[s][a][s_prime] \
                                        * self.policy_history[s_prime, a_prime, h+1, j][s_prime][a_prime] \
                                        * B[s_prime][a_prime][h+1][j]
                                    summation += term 
                            B[s][a][h][j] = b[s] + summation

            # POLICY IMPROVEMENT given we have (S x A x H x J) Q and B matrices
            for h in range(self.H): 
                for s in range(self.S): 
                    for a in range(self.A): 

                        summation = 0.
                        for j in range(len(delayed)): 
                            summation += Q[s][a][h][j] - B[s][a][h][j]

                        numerator = self.policy_history[s][a][h][k] * np.exp(-1 * self.eta * summation)
                        
                        denominator = 0.
                        
                        for a_prime in range(self.A):
                            inner_sum = 0.
                            for d in range(len(delayed)): 
                                inner_sum += Q[s][a_prime][h][d] - B[s][a_prime][h][d]

                            denom_sum = np.exp(-1 * self.eta * inner_sum)
                            denominator += self.policy_history[s, a_prime, h, k] * denom_sum

                        if k != (self.K - 1): # do not update policy at last episode
                            self.policy_history[s][a][h][k + 1] = numerator / (denominator if denominator != 0 else 1)            
        
        print(self.delay_dict)

DAPO(config).run()