# Exploring and Evaluating Delayed Reward Algorithm Performance on the MIMIC-IV Healthcare Datase

Contributors: Victoria Li, Jayden Personnat, Johnathan Sun

This repository contains implementations of Maximum Entropy Inverse Reinforcement Learning,
Value Iteration, Proximal Policy Optimization (PPO), and Delayed Proximal Policy Optimization (DPPO)
using the MIMIC-IV v2.2 dataset to construct an Markov Decision Process (MDP) environment.

`process.py` contains data processing code to construct the MDP.

`maxent.py` contains an implementation of Maximum Entropy IRL. 

``environment.py`` contains the Gym environment that we created to represent our MDP

`solver.py` contains code for value iteration.

`stable_baseline_ppo.py` has an implementation of PPO while we use `ppo.py` for implementing DPPO.

``dapo.py``contains an implmentation of Delay-Adapted Policy Optimization (DAPO)

``uob_reps.py``contains an unfinished implementation of the UOB-REPS algorithm for delayed feeback.
