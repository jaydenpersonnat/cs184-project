import torch
from torch import distributions as pyd
import numpy as np

mdp_params = np.load("mdp/mdp_25.npz")

reward = np.max(mdp_params['reward']) - mdp_params['reward']
mu = mdp_params['mu']
p_transition = mdp_params['p_transition'] # S * S' * A
states = mdp_params['states']
actions = mdp_params['n_actions']
H = 100
K = 10
D = 50

eta = (H^2 * int(states) * int(actions) * K + H^4 * (K + D)) ** (-(1/2))
# eta = 0.5
# print('eta', eta)

config = {
        'action_dim': int(actions),
        'state_dim': int(states),
        'transition_function': torch.from_numpy(p_transition),
        'initial_state_distribution': torch.from_numpy(mu),
        'reward_function': torch.from_numpy(reward),

        'training horizon H': 1000,
        'episodes K': 1000,
        'eta': 0.01,
        'gamma': 0.1        
}


class DAPO:
    def __init__(self, config): 
        self.transition_function = config['transition_function']
        self.initial_state_distribution = config['initial_state_distribution']
        self.H = config["training horizon H"]
        self.K = config["episodes K"]
        self.reward_function = config['reward_function']
        
        # learning rate, exploration parameter
        self.eta = config["eta"]
        self.gamma = config["gamma"]

        # initialize space dimensions
        self.A = config['action_dim']
        self.S = config['state_dim']

        # self.policy_history = torch.zeros(self.S, self.A, self.H, self.K)
        self.policy_history = (1.0 / self.A) * torch.ones(self.S, self.A, self.H, self.K)

        self.delay_dict = {}
        for i in range(self.K): self.delay_dict[i] = []

    def sample_from_logits(self, logits):
        sample = pyd.Categorical(logits=logits).sample()
        return sample.item()

    def play_one_step(self, current_state, policy, h):  
        # Play one step in the environment
        
        # sample from policy at timestep h
        action = self.sample_from_logits(policy[current_state, :, h])
        next_state = self.sample_from_logits(self.transition_function[current_state, :, action])
        reward = self.reward_function[current_state, action]
        return action, next_state, reward

    def play_episode(self, policy):
        traj = []
        # sample initial state
        current_state = self.sample_from_logits(self.initial_state_distribution)
        
        # for 0, ..., H-1 play a step and set the current state to the next state
        for h in range(self.H):
            action, next_state, reward = self.play_one_step(current_state, policy, h)
            traj.append((current_state, action, reward))
            current_state = next_state
        return traj
    
    def observe_feedback(self, traj):
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

        total_rewards = [] 

        for k in range(self.K):

            rdm_num = np.random.choice(np.arange(k, k + 10))
            self.delay_dict[rdm_num].append(k)
            delayed = self.delay_dict[k]

            # Play episode k with policy $\pi_k$ and observe trajectory
            k_trajectory = self.play_episode(self.policy_history[:,:,:,k])
            k_rewards = self.observe_feedback(k_trajectory)

            reward_sum = torch.sum(k_rewards)

            total_rewards.append(reward_sum)

            print(f"Episode: {k}, Total Reward: {reward_sum}")
        
            Q = torch.zeros(self.S, self.A, self.H, len(delayed))
            B = torch.zeros(self.S, self.A, self.H + 1, len(delayed))

            for j in delayed: 
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
                                    term = self.transition_function[s][s_prime][a] \
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