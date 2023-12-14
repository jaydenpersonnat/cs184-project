from sklearn.cluster import KMeans
from utils import * 
from maxent import irl, irl_causal, feature_expectation_from_trajectories
import optimizer as O 
import solver as S                  
from sklearn.preprocessing import OneHotEncoder
        # MDP solver (value-iteration)


def construct_trajectories(p_events, p_vitals): 
    """
    p_events: events for each patient 
    p_vitals: vital readings for each patient 
    """
    trajs = {} 
    
    for patient in p_events: 
        tau = trajs_from_patient(p_events[patient], p_vitals[patient])
        # drop trajectories with length = 0
        if (len(tau) > 1):
            trajs[int(patient)] = trajs_from_patient(p_events[patient], p_vitals[patient])

    return trajs 


def calc_start_dist(taus, S): 
    X = np.zeros(S)
    n = len(taus)

    for tau in taus: 
        X[tau[0]] += 1 

    return X / n 


def calc_tran_model(taus, states, actions, smoothing_value=1): 
    p_transition = np.zeros((states, states, actions)) + smoothing_value

    for traj in taus:
        for tran in traj:

            p_transition[tran[0], tran[2], tran[1]] +=1

        p_transition = p_transition/ p_transition.sum(axis = 1)[:, np.newaxis, :]

    return p_transition

def calc_terminal_states(taus): 
    terminal_states = set() 

    for patient in taus: 
        terminal_states.add(taus[patient][-1])
    
    return list(terminal_states)

def convert_traj(trajectories):
    lst = []
    for patient in trajectories:
        traj = trajectories[patient]
        row = []
        n = len(traj)
        for i in range(0, n-2, 2):
            row.append((traj[i], traj[i+1], traj[i+2]))
        
        lst.append(row)
    
    return lst



def train_single_intent(T, p_transition, states, terminal_states, discount=0.9): 
    # set up features: we use one feature vector per state (1 hot encoding for each cluster/state)
    # choose our parameter initialization strategy:
    #   initialize parameters with constant
        init = O.Constant(1.0)

        # choose our optimization strategy:
        #   we select exponentiated stochastic gradient descent with linear learning-rate decay
        optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

        state_encoder = OneHotEncoder(sparse=False, categories= [np.arange(states)])
        features = state_encoder.fit_transform(np.arange(states).reshape(-1, 1))

        # actually do some inverse reinforcement learning
        # reward_maxent = maxent_irl(p_transition, features, terminal_states, trajectories, optim, init, eps= 1e-3)

        reward_maxent_causal, theta_causal = irl_causal(p_transition, features, terminal_states, T, optim, init, discount,
                    eps=1e-3, eps_svf=1e-4, eps_lap=1e-4)
        
        
        return reward_maxent_causal, theta_causal



def discretize_S(M, n_clusters=100, random_state=42): 
    """
    M: matrix representation of data 

    returns a model that takes in a feature set D -> state s from 0 to K - 1
    where K is the number of clusters 
    """

    # instantiate model and fit data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit(M)

    # return model 
    return kmeans


if __name__ == "__main__": 

    try: 
        states = int(input("Number of states: ")) 
    except: 
        print("Error: provide valid number for states")

    # read from CSV files 
    patients_df = read_csv_to_dataframe("data/patients.csv")
    inputevents_df = read_csv_to_dataframe("data/inputevents.csv")
    vitalsign_df = read_csv_to_dataframe("data/vitalsign.csv")

    print("---Read from dataframe---")

    # merge patients with vitals dataframe 
    data_pv = pd.merge(patients_df, vitalsign_df, on='subject_id', how='inner')

    # fill in missing values 
    fill_NANS(data_pv)

    # map rhythms  
    unique_rhythms = data_pv['rhythm'].unique()
    n_rhythms = len(unique_rhythms)
    rhythms_mapping = {k:v for k,v in zip(unique_rhythms, range(n_rhythms))}

    features = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "rhythm"]

    M = construct_M(data_pv, features, rhythms_mapping) 

    print("---Created Matrix M---")

    state_model = discretize_S(M, n_clusters=states)

    print("---discretized state space---")


    inputevents_sample = inputevents_df.sample(n=6000000, random_state = 42)

    n_actions = len(inputevents_df['ordercategorydescription'].unique()) + 1 
    actions = ["No Treatment"] + list(inputevents_df['ordercategorydescription'].unique())
    action_mapping = {k:v for k, v in zip(actions, range(n_actions))}

    patient_data = {} 

    patient_events = find_patient_events(inputevents_sample, action_mapping)
    patient_vitals = find_patient_vitals(data_pv, state_model, features, rhythms_mapping)

    print("---constructing trajectories---")

    p_events, p_vitals = intersect_vitals_events(patient_events, patient_vitals)
    trajectories = construct_trajectories(p_events, p_vitals)

    save_json(trajectories, f"data/trajectories_{states}.json")

    T = convert_traj(trajectories)

    p_transition = calc_tran_model(T, states, n_actions)
    terminal_states = calc_terminal_states(trajectories)

    reward, _ = train_single_intent(T, p_transition, states, terminal_states)

    print("--Trained inverse RL model--")

    reward = np.tile(reward.reshape(-1, 1), n_actions)

    mu = calc_start_dist(list(trajectories.values()), states)

    np.savez(f"mdp/mdp_{states}", states=states, n_actions=n_actions, p_transition=p_transition, reward=reward, terminal_states=terminal_states, mu=mu)
