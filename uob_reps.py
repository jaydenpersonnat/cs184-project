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
          'delta': 0.1,
          'gamma': 0.1,
}

MimicEnv = env_setup.MDPEnv(env_setup.config_env)

class DelayedUOBREPS:
    def __init__(self, config, env): 
        # Initialize the environment
        self.env = env
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space
        self.H = config["training_horizon"]
        self.K = config["episodes"]
        # learning rate, exploration parameter, confidence parameter
        self.eta, self.gamma, self.delta = config["eta"], config["gamma"], config["delta"] 

        # initialize state and time step
        self.h = 0
        self.state = self.env.state

        # initialize space dimensions
        self.A = self.action_space.shape[0]
        self.S = self.state_space.shape[0]

        # initialize policy, occupancy measures, and visit counters
        self.policy = (1.0 / self.A) * torch.ones(self.S, self.A, self.H, dtype = torch.float32)
        self.q = (1.0 / (self.S * self.S * self.A)) * torch.ones(self.S, self.A, self.S, self.H, dtype = torch.float32)
        self.m_sa = torch.zeros(self.S, self.A, self.H, dtype = torch.float32)
        self.m_sas = torch.zeros(self.S, self.A, self.S, self.H, dtype = torch.float32)

        self.P = torch.zeros(self.S, self.A, self.S, self.H, dtype = torch.float32)

        # initialize 

    # def initialize_policy_and_variables(self):
    #     # Initialize policy \pi as a uniform distribution over actions
    #     # \pi_h^1(a | s) = \frac{1}{A}
    #     policy = (1.0 / self.A) * \
    #              torch.ones(self.S, self.A, self.H, dtype = torch.float32)
        
    #     # Next, initialize occupancy measures => 
    #     # Initialize q as uniform over state-action pairs and next state
    #     # q_h^1(s, a, s') = \frac{1}{S^2A}
    #     q = (1.0 / (self.S * self.S * self.A)) * \
    #         torch.ones(self.S, self.A, self.S, self.H, dtype = torch.float32)
        
    #     # Initialize m as zeros for all s, a, s', h ------------- m_h^1(s,a) = 0 
    #     # m_sa is S * A * H, m_sas is S * A * S' * H
    #     m_sa = torch.zeros(self.S, self.A, self.H, dtype = torch.float32)
    #     m_sas = torch.zeros(self.S, self.A, self.S, self.H, dtype = torch.float32)
        
    #     return policy, q, m_sa, m_sas

    def update_confidence_set(self, traj):
        # Update the confidence set P_{k+1} by algorithm 9 
        # Extract states (s), actions (a), and rewards (r) from the trajectory
        # Note: You'll need to adjust this according to how trajectory is represented
        states, actions, rewards = zip(*traj)

        p = torch.zeros(self.S, self.A, self.S, self.H, dtype = torch.float32)

        for h in range(self.H - 1):

            # update visit counters
            s, a, s_prime = traj[h][0], traj[h][1], traj[h+1][0]
            self.m_sa[s, a, h] += 1
            self.m_sas[s, a, s_prime, h] += 1

            # update empirical transition function
            p[:, :, :, h] = self.m_sas[:, :, :, h] / (self.m_sa[:, :, h] if self.m_sa[:, :, h] != 0 else 1)
    
        # define confidence sets
        num_1 = 16 * p * torch.log(10 * self.H * self.S * self.A / self.delta)
        denom_1 = self.m_sas if self.m_sas != 0 else 1
        num_2 = 10 * torch.log(10 * self.H * self.S * self.A / self.delta)
        denom_2 = self.m_sa if self.m_sa != 0 else 1
        bound = torch.sqrt(num_1 / denom_1) + num_2 / denom_2

        # return upper and lower bounds of confidence sets
        # note that p' is in the confidence bound if p - bound <= p' <= p + bound
        return p + bound, p - bound

        # update empirical transitions
        # for h in range(self.H - 1):
        #     s, a, s_prime = traj[h][0], traj[h][1], traj[h+1][0]
        #     self.q[s, a, s_prime, h] += 1 / self.m_sa[s, a, h]

        # # Update visit counts m_k
        # for h in range(self.H - 1):
        #     s, a, s_prime = states[h], actions[h], states[h+1] 
        #     self.m_k[s, a, h] += 1
        
        # Update empirical transitions p_k
        for h in range(self.H - 1):
            s, a, s_prime = states[h], actions[h], states[h+1]
            self.p_k[s, a, s_prime, h] += 1 / self.m_k[s, a, h]

        # Define confidence sets P_{k+1}
        # Note: This is a simplified version; the actual implementation should follow the bounds provided in the algorithm
        # P_k_plus_1 = self.p_k.clone()
        # confidence_bounds = torch.sqrt(torch.log(1 / self.delta) / (self.m_k + 1))
        # P_k_plus_1 += confidence_bounds.unsqueeze(2).expand_as(self.p_k)
        
        # return P_k_plus_1

    def compute_upper_occupancy_bound(self, P_k, s, a):
        # Compute the upper occupancy bound u_k(s, a)
        pass

    def compute_loss_estimator(self, c_h, s, a):
        # Compute the delay-adapted cost estimator c_hat using equation (5)
        pass

    def update_occupancy_measure(self, q_k, P_k):
        # Update and return the occupancy measure q_{k+1}
        # This will require solving the optimization problem in Eq. (31), which involves KL divergence
        pass

    def update_policy(self, q_k_plus_1):
        # Update the policy based on the new occupancy measure q_{k+1}
        pass

    def run(self):
        for k in range(1, self.K + 1):
            trajectory = self.play_episode(self.policy)
            self.P = self.update_confidence_set(trajectory)

            
            
            for h in range(1, self.H + 1):
                u_k = self.compute_upper_occupancy_bound(P_k, s, a)
                c_hat = self.compute_loss_estimator(c_h, s, a)
                
                q_k_plus_1 = self.update_occupancy_measure(q_k, P_k_plus_1)
                self.update_policy(q_k_plus_1)
        return self.policy
    
    def play_one_step(self, policy):
        # Play one step in the environment and return the next state, reward, and done flag
        action = env_setup.sample_from_logits(logits=policy[self.state, :, self.h])
        next_state, reward, done, _ = self.env.step(action)
        return action, next_state, reward, done

    def play_episode(self, policy):
        trajectory = []
        for i in range(self.H):
            action, next_state, reward, _ = self.play_one_step(policy)
            trajectory.append((self.state, action, reward))
            self.state = next_state
            self.h += 1
        return trajectory


# Example usage
state_space_size = 10  # replace with actual size of state space
action_space_size = 5  # replace with actual size of action space
horizon = 20  # replace with actual horizon
episodes = 100  # replace with the desired number of episodes
eta = 0.01  # replace with the desired learning rate
gamma = 0.99  # replace with the desired exploration parameter
delta = 0.1  # replace with the desired confidence parameter

algorithm = DelayedUOBREPS(state_space_size, action_space_size, 
                           horizon, episodes, eta, gamma, delta)
print(algorithm.initialize_policy_and_variables())
