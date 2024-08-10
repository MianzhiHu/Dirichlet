from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import ComputationalModels
import pandas as pd
import numpy as np
import time
import gc
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def proportion_chosen(x):
    return (x == 'C').sum() / len(x)


def value_generator(epsilon=0.01):
    while True:
        # Randomly generate A such that 0.5 <= A <= 1
        A = np.random.uniform(0.5, 1)

        # Calculate B such that A + B = 1
        B = 1 - A

        # Ensure B is between 0 and 0.5
        if not (0 <= B <= 0.5):
            continue

        # Randomly generate C such that C > A + 0.01 and 0.5 <= C <= 1
        C = np.random.uniform(max(0.5, A + epsilon), 1)

        # Calculate D as 1 - C to ensure C + D = 1
        D = 1 - C

        # Ensure D is between 0 and 0.5 and D < B - 0.01
        if not (0 <= D <= 0.5 and D < B - epsilon):
            continue

        # If all conditions are met, break the loop and return the values
        if C > A + epsilon and D < B - epsilon:
            break

    return A, B, C, D


def simulation_unpacker(dict):
    all_sims = []

    for res in dict:
        sim_num = res['simulation_num']
        a_val = res['a']
        t_val = res['t']
        tau_val = res['tau']
        for trial_idx, trial_detail in zip(res['trial_indices'], res['trial_details']):
            data_row = {
                'simulation_num': sim_num,
                'trial_index': trial_idx,
                'a': a_val,
                't': t_val,
                'tau': tau_val,
                'pair': trial_detail['pair'],
                'choice': trial_detail['choice'],
                'reward': trial_detail['reward'],
            }
            all_sims.append(data_row)

    return pd.DataFrame(all_sims)


# ======================================================================================================================
#                                           Initialize models
# ======================================================================================================================
model = DualProcessModel()
decay = ComputationalModels("decay")
delta = ComputationalModels("delta")
actr = ComputationalModels("ACTR")

# ======================================================================================================================
#                              Simulation for randomly drawn reward values and variances
# ======================================================================================================================
# randomly draw reward values and variances
n = 5000
epsilon = 0.01


# Define the simulation function
def run_simulation(i):

    logging.info(f'Simulation {i + 1}/{n}')

    # Randomly draw the reward values
    a_val, b_val, c_val, d_val = value_generator(epsilon)

    # Randomly draw the variance
    var_val = np.random.uniform(0.11, 0.48)
    var = [var_val, var_val, var_val, var_val]

    # Simulate the data
    dual_simulation = model.simulate([a_val, b_val, c_val, d_val], var, model='Entropy_Dis_ID',
                                     AB_freq=100, CD_freq=50, num_iterations=1000, weight_Gau='softmax',
                                     weight_Dir='softmax', arbi_option='Entropy', Dir_fun='Linear_Recency',
                                     Gau_fun='Naive_Recency')

    decay_simulation = simulation_unpacker(decay.simulate([a_val, b_val, c_val, d_val], var,
                                                          AB_freq=100, CD_freq=50, num_iterations=1000))

    delta_simulation = simulation_unpacker(delta.simulate([a_val, b_val, c_val, d_val], var,
                                                          AB_freq=100, CD_freq=50, num_iterations=1000))

    actr_simulation = simulation_unpacker(actr.simulate([a_val, b_val, c_val, d_val], var,
                                                        AB_freq=100, CD_freq=50, num_iterations=1000))

    # Summarize the results
    dual_results = dual_simulation[dual_simulation['pair'] == ('C', 'A')].groupby('simulation_num').agg(
        choice=('choice', proportion_chosen),
        t=('t', 'mean'),
        a=('a', 'mean'),
        param_weight=('param_weight', 'mean'),
        obj_weight=('obj_weight', 'mean'),
        weight_dir=('weight_Dir', 'mean'),
    ).reset_index()

    decay_results = decay_simulation[decay_simulation['pair'] == ('C', 'A')].groupby('simulation_num').agg(
        choice=('choice', proportion_chosen),
        t=('t', 'mean'),
        a=('a', 'mean')
    ).reset_index()

    delta_results = delta_simulation[delta_simulation['pair'] == ('C', 'A')].groupby('simulation_num').agg(
        choice=('choice', proportion_chosen),
        t=('t', 'mean'),
        a=('a', 'mean')
    ).reset_index()

    actr_results = actr_simulation[actr_simulation['pair'] == ('C', 'A')].groupby('simulation_num').agg(
        choice=('choice', proportion_chosen),
        t=('t', 'mean'),
        a=('a', 'mean'),
        tau=('tau', 'mean')
    ).reset_index()

    # Add reward difference and variance to summaries
    for res in [dual_results, decay_results, delta_results, actr_results]:
        res['diff'] = c_val - a_val
        res['reward_ratio'] = c_val / (c_val + a_val)
        res['var'] = var_val

    return dual_results, decay_results, delta_results, actr_results


# Run the simulation
if __name__ == '__main__':

    start_time = time.time()

    dual_filepath = './data/Simulation/random_dual.csv'
    decay_filepath = './data/Simulation/random_decay.csv'
    delta_filepath = './data/Simulation/random_delta.csv'
    actr_filepath = './data/Simulation/random_actr.csv'

    # Define the headers
    dual_headers = ['simulation_num', 'choice', 't', 'a', 'param_weight', 'obj_weight', 'weight_dir', 'diff',
                    'reward_ratio', 'var']
    decay_headers = ['simulation_num', 'choice', 't', 'a', 'diff', 'reward_ratio', 'var']
    delta_headers = ['simulation_num', 'choice', 't', 'a', 'diff', 'reward_ratio', 'var']
    actr_headers = ['simulation_num', 'choice', 't', 'a', 'tau', 'diff', 'reward_ratio', 'var']

    # Initialize the file and write headers
    pd.DataFrame(columns=dual_headers).to_csv(dual_filepath, index=False)
    pd.DataFrame(columns=decay_headers).to_csv(decay_filepath, index=False)
    pd.DataFrame(columns=delta_headers).to_csv(delta_filepath, index=False)
    pd.DataFrame(columns=actr_headers).to_csv(actr_filepath, index=False)

    with ProcessPoolExecutor(max_workers=24) as executor:
        for i, (dual_results, decay_results, delta_results, actr_results) in enumerate(
                tqdm(executor.map(run_simulation, range(n)), total=n)):
            dual_results.to_csv(dual_filepath, mode='a', header=False, index=False)
            decay_results.to_csv(decay_filepath, mode='a', header=False, index=False)
            delta_results.to_csv(delta_filepath, mode='a', header=False, index=False)
            actr_results.to_csv(actr_filepath, mode='a', header=False, index=False)

            # Free up memory
            del dual_results, decay_results, delta_results, actr_results
            gc.collect()

    total_time = time.time() - start_time
    print(f'Total Time: {total_time}')


# ======================================================================================================================
#                                           Traditional Simulation
# ======================================================================================================================
# reward_means = [0.65, 0.35, 0.75, 0.25]
# reward_means_uncertainty = [0.70, 0.30, 0.70, 0.30]
# hv = [0.48, 0.48, 0.43, 0.43]
# mv = [0.24, 0.24, 0.22, 0.22]
# lv = [0.12, 0.12, 0.11, 0.11]
# uncertainty = [0.43, 0.43, 0.12, 0.12]

# # ========== Dirichlet Model ==========
# dir_hv = model.simulate(reward_means, hv, model='Dir', AB_freq=100, CD_freq=50, num_iterations=10000)
# dir_mv = model.simulate(reward_means, mv, model='Dir', AB_freq=100, CD_freq=50, num_iterations=10000)
# dir_lv = model.simulate(reward_means, lv, model='Dir', AB_freq=100, CD_freq=50, num_iterations=10000)
#
# dir_hv.to_csv('./data/Simulation/dir_hv.csv', index=False)
# dir_mv.to_csv('./data/Simulation/dir_mv.csv', index=False)
# dir_lv.to_csv('./data/Simulation/dir_lv.csv', index=False)
#
# # ========== Multivariate Gaussian Model ==========
# gau_hv = model.simulate(reward_means, hv, model='Gau', AB_freq=100, CD_freq=50, num_iterations=10000)
# gau_mv = model.simulate(reward_means, mv, model='Gau', AB_freq=100, CD_freq=50, num_iterations=10000)
# gau_lv = model.simulate(reward_means, lv, model='Gau', AB_freq=100, CD_freq=50, num_iterations=10000)
#
# gau_hv.to_csv('./data/Simulation/gau_hv.csv', index=False)
# gau_mv.to_csv('./data/Simulation/gau_mv.csv', index=False)
# gau_lv.to_csv('./data/Simulation/gau_lv.csv', index=False)

# # ========= Dual Process Model ==========
# dual_hv = model.simulate(reward_means, hv, model='Entropy_Dis_ID', AB_freq=100, CD_freq=50, num_iterations=2000,
#                          weight_Gau='softmax', weight_Dir='softmax', arbi_option='Entropy', Dir_fun='Linear_Recency',
#                          Gau_fun='Naive_Recency')
# dual_mv = model.simulate(reward_means, mv, model='Entropy_Dis_ID', AB_freq=100, CD_freq=50, num_iterations=2000,
#                          weight_Gau='softmax', weight_Dir='softmax', arbi_option='Entropy', Dir_fun='Linear_Recency',
#                          Gau_fun='Naive_Recency')
# dual_lv = model.simulate(reward_means, lv, model='Entropy_Dis_ID', AB_freq=100, CD_freq=50, num_iterations=2000,
#                          weight_Gau='softmax', weight_Dir='softmax', arbi_option='Entropy', Dir_fun='Linear_Recency',
#                          Gau_fun='Naive_Recency')

# dual_hv.to_csv('./data/Simulation/dual_hv.csv', index=False)
# dual_mv.to_csv('./data/Simulation/dual_mv.csv', index=False)
# dual_lv.to_csv('./data/Simulation/dual_lv.csv', index=False)

# ========== Parametric Model ==========
# mixed_hv = model.simulate(reward_means, hv, model='Param', AB_freq=100, CD_freq=50)
# mixed_mv = model.simulate(reward_means, mv, model='Param', AB_freq=100, CD_freq=50)
# mixed_lv = model.simulate(reward_means, lv, model='Param', AB_freq=100, CD_freq=50)

# mixed_hv.to_csv('./data/Simulation/mixed_hv.csv', index=False)
# mixed_mv.to_csv('./data/Simulation/mixed_mv.csv', index=False)
# mixed_lv.to_csv('./data/Simulation/mixed_lv.csv', index=False)

# # ========== Decay Model ==========
# decay_hv = decay.simulate(reward_means, hv, AB_freq=100, CD_freq=50, num_iterations=10000)
# decay_mv = decay.simulate(reward_means, mv, AB_freq=100, CD_freq=50, num_iterations=10000)
# decay_lv = decay.simulate(reward_means, lv, AB_freq=100, CD_freq=50, num_iterations=10000)
#
# # ========== Delta Model =========
# delta_hv = delta.simulate(reward_means, hv, AB_freq=100, CD_freq=50, num_iterations=10000)
# delta_mv = delta.simulate(reward_means, mv, AB_freq=100, CD_freq=50, num_iterations=10000)
# delta_lv = delta.simulate(reward_means, lv, AB_freq=100, CD_freq=50, num_iterations=10000)
#
# # ========= ACT-R Model ==========
# actr_hv = actr.simulate(reward_means, hv, AB_freq=100, CD_freq=50, num_iterations=10000)
# actr_mv = actr.simulate(reward_means, mv, AB_freq=100, CD_freq=50, num_iterations=10000)
# actr_lv = actr.simulate(reward_means, lv, AB_freq=100, CD_freq=50, num_iterations=10000)
#
# # unpacking the results
# for i, sim in enumerate([decay_hv, decay_mv, decay_lv, delta_hv, delta_mv, delta_lv, actr_hv, actr_mv, actr_lv]):
#
#     file_name = ['decay_hv', 'decay_mv', 'decay_lv', 'delta_hv', 'delta_mv', 'delta_lv', 'actr_hv', 'actr_mv',
#                  'actr_lv'][i]
#
#     df = simulation_unpacker(sim)
#     df.to_csv('./data/Simulation/' + file_name + '.csv', index=False)
