from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import ComputationalModels
import pandas as pd
import numpy as np
import ast
import time


def proportion_chosen(x):
    return (x == 'C').sum() / len(x)


def value_generator(min_value, max_value, ref_val, epsilon, option=None):
    value = np.random.uniform(min_value, max_value)
    if option == 'C':
        while value <= ref_val + epsilon:
            value = np.random.uniform(min_value, max_value)
    elif option == 'D':
        while value >= ref_val - epsilon:
            value = np.random.uniform(min_value, max_value)
    return value


def simulation_unpacker(dict):
    all_sims = []

    for res in dict:
        sim_num = res['simulation_num']
        a_val = res['a']
        t_val = res['t']
        tau_val = res['tau']
        for trial_idx, trial_detail, ev in zip(res['trial_indices'], res['trial_details'], res['EV_history']):
            data_row = {
                'simulation_num': sim_num,
                'trial_index': trial_idx,
                'a': a_val,
                't': t_val,
                'tau': tau_val,
                'pair': trial_detail['pair'],
                'choice': trial_detail['choice'],
                'reward': trial_detail['reward'],
                'EV_A': ev[0],
                'EV_B': ev[1],
                'EV_C': ev[2],
                'EV_D': ev[3]
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

# testing
reward_means = [0.65, 0.35, 0.75, 0.25]
reward_means_uncertainty = [0.70, 0.30, 0.70, 0.30]
hv = [0.48, 0.48, 0.43, 0.43]
mv = [0.24, 0.24, 0.22, 0.22]
lv = [0.12, 0.12, 0.11, 0.11]
uncertainty = [0.43, 0.43, 0.12, 0.12]
decay_hv = decay.simulate(reward_means, hv, AB_freq=100, CD_freq=50, num_iterations=10)

# unpack the results
df = simulation_unpacker(decay_hv)

# ======================================================================================================================
#                              Simulation for randomly drawn reward values and variances
# ======================================================================================================================
# randomly draw reward values and variances
n = 2000
epsilon = 0.01
start_time = time.time()

for i in range(n):

    print(f'====================================')
    print(f'Simulation {i + 1}')
    print(f'====================================')

    # randomly draw the reward values
    a_val = np.random.uniform(0.5, 1)
    b_val = np.random.uniform(0, 0.5)
    c_val = value_generator(0.5, 1, a_val, epsilon, option='C')
    d_val = value_generator(0, 0.5, b_val, epsilon, option='D')

    # randomly draw the variance
    var_val = np.random.uniform(0.11, 0.48)
    var = [var_val, var_val, var_val, var_val]

    # simulate the data
    # dual process model
    print(f'------------------------------------')
    print(f'Dual Process Model')
    print(f'------------------------------------')
    dual_simulation = model.simulate([a_val, b_val, c_val, d_val], var, model='Entropy_Dis_ID',
                                     AB_freq=100, CD_freq=50, num_iterations=1000, weight_Gau='softmax',
                                     weight_Dir='softmax', arbi_option='Entropy', Dir_fun='Linear_Recency',
                                     Gau_fun='Naive_Recency')

    # decay model
    print(f'------------------------------------')
    print(f'Decay Model')
    print(f'------------------------------------')
    decay_simulation = simulation_unpacker(decay.simulate([a_val, b_val, c_val, d_val], var,
                                                          AB_freq=100, CD_freq=50, num_iterations=1000))

    # delta model
    print(f'------------------------------------')
    print(f'Delta Model')
    print(f'------------------------------------')
    delta_simulation = simulation_unpacker(delta.simulate([a_val, b_val, c_val, d_val], var,
                                                          AB_freq=100, CD_freq=50, num_iterations=1000))

    # actr model
    print(f'------------------------------------')
    print(f'ACT-R Model')
    print(f'------------------------------------')
    actr_simulation = simulation_unpacker(actr.simulate([a_val, b_val, c_val, d_val], var,
                                                        AB_freq=100, CD_freq=50, num_iterations=1000))

    # summarize the results
    dual_results = dual_simulation[dual_simulation['pair'] == ('C', 'A')].groupby('simulation_num').agg(
        choice=('choice', proportion_chosen),
        t=('t', 'mean'),
        a=('a', 'mean'),
        param_weight=('param_weight', 'mean'),
        obj_weight=('obj_weight', 'mean'),
        weight_dir=('weight_Dir', 'mean'),
    ).reset_index()

    # for classic models
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
        a=('a', 'mean')
    ).reset_index()

    # add the reward difference and variance to the summaries
    for res in [dual_results, decay_results, delta_results, actr_results]:
        res.loc[:, 'diff'] = c_val - a_val
        res.loc[:, 'var'] = var_val

    if i == 0:
        dual_all = dual_results
        decay_all = decay_results
        delta_all = delta_results
        actr_all = actr_results
    else:
        dual_all = pd.concat([dual_all, dual_results], ignore_index=True)
        decay_all = pd.concat([decay_all, decay_results], ignore_index=True)
        delta_all = pd.concat([delta_all, delta_results], ignore_index=True)
        actr_all = pd.concat([actr_all, actr_results], ignore_index=True)

# save the results
dual_all.to_csv('./data/Simulation/random_dual.csv', index=False)
decay_all.to_csv('./data/Simulation/random_decay.csv', index=False)
delta_all.to_csv('./data/Simulation/random_delta.csv', index=False)
actr_all.to_csv('./data/Simulation/random_actr.csv', index=False)

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
