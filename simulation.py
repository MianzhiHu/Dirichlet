from utilities.utility_DualProcess import DualProcessModel
from utilities.utility_ComputationalModeling import ComputationalModels
import pandas as pd


model = DualProcessModel()
reward_means = [0.65, 0.35, 0.75, 0.25]
reward_means_uncertainty = [0.70, 0.30, 0.70, 0.30]
hv = [0.43, 0.43, 0.43, 0.43]
mv = [0.265, 0.265, 0.265, 0.265]
lv = [0.1, 0.1, 0.1, 0.1]
uncertainty = [0.43, 0.43, 0.12, 0.12]

# # model simulation
# dir_hv = model.simulate(reward_means, hv, model='Dir', AB_freq=100, CD_freq=50)
# dir_mv = model.simulate(reward_means, mv, model='Dir', AB_freq=100, CD_freq=50)
# dir_lv = model.simulate(reward_means, lv, model='Dir', AB_freq=100, CD_freq=50)
#
# gau_hv = model.simulate(reward_means, hv, model='Gau', AB_freq=100, CD_freq=50)
# gau_mv = model.simulate(reward_means, mv, model='Gau', AB_freq=100, CD_freq=50)
# gau_lv = model.simulate(reward_means, lv, model='Gau', AB_freq=100, CD_freq=50)
#
# dual_hv = model.simulate(reward_means, hv, model='Dual', AB_freq=100, CD_freq=50)
# dual_mv = model.simulate(reward_means, mv, model='Dual', AB_freq=100, CD_freq=50)
# dual_lv = model.simulate(reward_means, lv, model='Dual', AB_freq=100, CD_freq=50)
#
# mixed_hv = model.simulate(reward_means, hv, model='Param', AB_freq=100, CD_freq=50)
# mixed_mv = model.simulate(reward_means, mv, model='Param', AB_freq=100, CD_freq=50)
# mixed_lv = model.simulate(reward_means, lv, model='Param', AB_freq=100, CD_freq=50)
#
# uncertainty_dual = model.simulate(reward_means_uncertainty, uncertainty, model='Dual', AB_freq=75, CD_freq=75)

# uncertainty_dir = model.simulate(reward_means, uncertainty, model='Dir', AB_freq=100, CD_freq=50)
#
# uncertainty_gau = model.simulate(reward_means, uncertainty, model='Gau', AB_freq=100, CD_freq=50)
#
# uncertainty_mixed = model.simulate(reward_means, uncertainty, model='Param', AB_freq=100, CD_freq=50)

# # save the simulation results
# dir_hv.to_csv('./data/Simulation/dir_hv.csv', index=False)
# dir_mv.to_csv('./data/Simulation/dir_mv.csv', index=False)
# dir_lv.to_csv('./data/Simulation/dir_lv.csv', index=False)
#
# gau_hv.to_csv('./data/Simulation/gau_hv.csv', index=False)
# gau_mv.to_csv('./data/Simulation/gau_mv.csv', index=False)
# gau_lv.to_csv('./data/Simulation/gau_lv.csv', index=False)
#
# dual_hv.to_csv('./data/Simulation/dual_hv.csv', index=False)
# dual_mv.to_csv('./data/Simulation/dual_mv.csv', index=False)
# dual_lv.to_csv('./data/Simulation/dual_lv.csv', index=False)

# mixed_hv.to_csv('./data/Simulation/mixed_hv.csv', index=False)
# mixed_mv.to_csv('./data/Simulation/mixed_mv.csv', index=False)
# mixed_lv.to_csv('./data/Simulation/mixed_lv.csv', index=False)

# uncertainty_dual.to_csv('./data/Simulation/dual_uncertainty_UO.csv', index=False)
# uncertainty_dir.to_csv('./data/Simulation/dir_uncertainty.csv', index=False)
# uncertainty_gau.to_csv('./data/Simulation/gau_uncertainty.csv', index=False)
# uncertainty_mixed.to_csv('./data/Simulation/mixed_uncertainty.csv', index=False)

# # simulate the decay and delta models
# decay = ComputationalModels("decay")
# delta = ComputationalModels("delta")
#
# decay_uncertainty = decay.simulate(reward_means, uncertainty, AB_freq=100, CD_freq=50)
# delta_uncertainty = delta.simulate(reward_means, uncertainty, AB_freq=100, CD_freq=50)
#
# # unpacking the results
# all_data = []
#
# for res in delta_uncertainty:
#     sim_num = res['simulation_num']
#     a_val = res['a']
#     b_val = res['b']
#     t_val = res['t']
#     for trial_idx, trial_detail, ev in zip(res['trial_indices'], res['trial_details'], res['EV_history']):
#         data_row = {
#             'simulation_num': sim_num,
#             'trial_index': trial_idx,
#             'a': a_val,
#             't': t_val,
#             'pair': trial_detail['pair'],
#             'choice': trial_detail['choice'],
#             'reward': trial_detail['reward'],
#             'EV_A': ev[0],
#             'EV_B': ev[1],
#             'EV_C': ev[2],
#             'EV_D': ev[3]
#         }
#         all_data.append(data_row)
#
# df = pd.DataFrame(all_data)
#
# df.to_csv('./data/Simulation/delta_uncertainty.csv', index=False)

