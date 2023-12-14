import torch
import torch.optim as optim
import numpy as np
import environment as env_setup

config_agent = {
            # action space is list of actions, dimensions A * 1
        #   'action_space':['one', 'two', 'three', 'four', 'five', 'six'],
            # state space is list of states, dimensions S * 1
        #   'state_space': ['one', 'two', 'three', 'four', 'five', 'six'],
          'training_horizon': 20,
          'episodes': 100,
          'eta': 0.01,
            #   'delta': 0.1,
          'gamma': 0.1,
}

MimicEnv = env_setup.MDPEnv(env_setup.config_env)

class DAPO:        
    def __init__(self, config, env): 
        # Initialize the environment
        self.env = env
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space
        self.transition = self.env.transition_function
        self.H = config["training_horizon"]
        self.K = config["episodes"]
        self.reward_function = self.env.reward_function
        
        # learning rate, exploration parameter, confidence parameter
        self.eta = config_agent["eta"]
        self.gamma = config_agent["gamma"]

        # initialize state and time step
        self.h = 0
        self.k = 0
        self.state = self.env.state

        # initialize space dimensions
        self.A = self.action_space.shape[0]
        self.S = self.state_space.shape[0]

        # initialize policy, occupancy measures, and visit counters
        self.policy_history = torch.zeros(self.S, self.A, self.H, self.K)
        self.policy_history = (1.0 / self.A) * torch.ones(self.S, self.A, self.H, self.K)
#         self.policy_history[:, :, :, self.K]


    def play_one_step(self, policy):
        # Play one step in the environment and return the next state, reward, and done flag
        action = env_setup.sample_from_logits(logits = policy[self.state, :, self.h])
        next_state, reward, done, _ = self.env.step(action)
        
        return action, next_state, reward, done

    def play_episode(self, policy):
        traj = []
        for i in range(self.H):
            action, next_state, reward, _ = self.play_one_step(policy)
            traj.append((self.state, action, reward))
            self.state = next_state
            # self.h += 1
        return traj
    
    def observe_feedback(self, traj): # objerve \{c^j_h(s^j_h,a^j_h)\}^{H}_{h=1}
        rewards = torch.zeros(self.H)
        for h in range(self.H - 1):
            s, a = traj[h][0], traj[h][1]
            rewards[h] = self.reward_function(s, a)
            
        return rewards
    
    def get_n_step_transition (self, h, k):
        # returns a tensor of shape S * S which gives the one step transition probabilities
        # from each state to each state

        # initialize transition probabilities
        p = torch.zeros(self.S, self.S)
        for i in range(self.S):
            for j in range(self.S):
                for a in range(self.A):
                    p[i, j] += self.policy_history[i, a, h, k] * self.env.transition_function[i, j, a]
        return p
    
    def get_occupancy(self, h, k):
        # get the n-step transition probabilities
        p = self.get_n_step_transition(h, k)
        p_h = torch.linalg.matrix_power(p, h)

        # adjust for the fact that initial state is stochastic
        p_adj = torch.zeros(self.S)
        for i in range(self.S):
            for j in range(self.S):
                p_adj[i] += p_h[j] * self.env.initial_state_distribution[j]

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
        delayed = []

        for k in range(self.K):
            # Play episode k with policy $\pi_k$ and observe trajectory
            # trajectory = self.play_episode(self.current_policy_history[:,:,:,k])
            
            B = torch.zeros(self.S, self.A, len(delayed))
            Q = torch.zeros(self.S, self.A, len(delayed))

            for j in delayed: #! HOW ARE WE TRACKING THE DELAYED TRAJECTORIES? 
                #? lol idk 
                # get trajectory corresponding to delayed trajectory
                trajectory = self.play_episode(self.policy_history[:,:,:,j])
                rewards = self.observe_feedback(trajectory)

                for h in range(self.self.H): 
                    j_policy = self.policy_history[:, :, h, j]
                    k_policy = self.policy_history[:, :, h, k]

                    r = j_policy / torch.max(j_policy, k_policy)
                    L = np.sum(rewards[h:])
            
                    # replace [s][a] entry of Q
                    sh, ah = trajectory[h][0], trajectory[h][1]
                    Q[sh][ah][j] = r[sh, ah] * L / \
                            (self.get_occupancy(h, k)[sh] * j_policy[sh][ah] + self.gamma)

                    b = (3 * self.gamma * self.H) * torch.sum(k_policy[sh]) * torch.sum(r[sh]) / \
                            (self.get_occupancy(h, k)[sh] * torch.sum(j_policy[sh]) + self.gamma)
        
                    # replace [s][a] entry of B
                    for s, a in range(self.S, self.A):
                        # self.transition[sh][a] * 
                        j_policy[s][a] * B[sh][ah][j]

                    r = self.policy_history[self.S, self.A, h, k] / np.max()