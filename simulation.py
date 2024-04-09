import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utilities.utility_DualProcess import DualProcessModel

# # after the simulation has been completed, we can just load the simulated data from the folder
# folder_path = './data/Simulation'
# simulations = {}
#
# for file in os.listdir(folder_path):
#     if file.endswith('.csv'):  # Check if the file is a CSV
#         file_path = os.path.join(folder_path, file)  # Get the full path of the file
#         df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
#         simulations[df_name] = pd.read_csv(file_path)
#
# # unnest the dictionary into dfs
# for key in simulations:
#     globals()[key] = simulations[key]

model = DualProcessModel()
reward_means = [0.65, 0.35, 0.75, 0.25]
hv = [0.43, 0.43, 0.43, 0.43]
mv = [0.265, 0.265, 0.265, 0.265]
lv = [0.1, 0.1, 0.1, 0.1]
uncertainty = [0.43, 0.43, 0.12, 0.12]

# model simulation
dir_hv = model.simulate(reward_means, hv, model='Dir', AB_freq=100, CD_freq=50)
dir_mv = model.simulate(reward_means, mv, model='Dir', AB_freq=100, CD_freq=50)
dir_lv = model.simulate(reward_means, lv, model='Dir', AB_freq=100, CD_freq=50)

gau_hv = model.simulate(reward_means, hv, model='Gau', AB_freq=100, CD_freq=50)
gau_mv = model.simulate(reward_means, mv, model='Gau', AB_freq=100, CD_freq=50)
gau_lv = model.simulate(reward_means, lv, model='Gau', AB_freq=100, CD_freq=50)

dual_hv = model.simulate(reward_means, hv, model='Dual', AB_freq=100, CD_freq=50)
dual_mv = model.simulate(reward_means, mv, model='Dual', AB_freq=100, CD_freq=50)
dual_lv = model.simulate(reward_means, lv, model='Dual', AB_freq=100, CD_freq=50)

mixed_hv = model.simulate(reward_means, hv, model='Param', AB_freq=100, CD_freq=50)
mixed_mv = model.simulate(reward_means, mv, model='Param', AB_freq=100, CD_freq=50)
mixed_lv = model.simulate(reward_means, lv, model='Param', AB_freq=100, CD_freq=50)

uncertainty_dual = model.simulate(reward_means, uncertainty, model='Dual', AB_freq=100, CD_freq=50)

uncertainty_dir = model.simulate(reward_means, uncertainty, model='Dir', AB_freq=100, CD_freq=50)

uncertainty_gau = model.simulate(reward_means, uncertainty, model='Gau', AB_freq=100, CD_freq=50)

uncertainty_mixed = model.simulate(reward_means, uncertainty, model='Param', AB_freq=100, CD_freq=50)

# save the simulation results
dir_hv.to_csv('./data/Simulation/dir_hv.csv', index=False)
dir_mv.to_csv('./data/Simulation/dir_mv.csv', index=False)
dir_lv.to_csv('./data/Simulation/dir_lv.csv', index=False)

gau_hv.to_csv('./data/Simulation/gau_hv.csv', index=False)
gau_mv.to_csv('./data/Simulation/gau_mv.csv', index=False)
gau_lv.to_csv('./data/Simulation/gau_lv.csv', index=False)

dual_hv.to_csv('./data/Simulation/dual_hv.csv', index=False)
dual_mv.to_csv('./data/Simulation/dual_mv.csv', index=False)
dual_lv.to_csv('./data/Simulation/dual_lv.csv', index=False)

mixed_hv.to_csv('./data/Simulation/mixed_hv.csv', index=False)
mixed_mv.to_csv('./data/Simulation/mixed_mv.csv', index=False)
mixed_lv.to_csv('./data/Simulation/mixed_lv.csv', index=False)

uncertainty_dual.to_csv('./data/Simulation/dual_uncertainty.csv', index=False)
uncertainty_dir.to_csv('./data/Simulation/dir_uncertainty.csv', index=False)
uncertainty_gau.to_csv('./data/Simulation/gau_uncertainty.csv', index=False)
uncertainty_mixed.to_csv('./data/Simulation/mixed_uncertainty.csv', index=False)



# # visualize the simulation results
# cols_to_mean_dir = ['EV_A_Dir', 'EV_B_Dir', 'EV_C_Dir', 'EV_D_Dir']
# cols_to_mean_gau = ['EV_A_Gau', 'EV_B_Gau', 'EV_C_Gau', 'EV_D_Gau']
# cols_to_mean_mixed = ['EV_A', 'EV_B', 'EV_C', 'EV_D']
# df_avg = dual_lv.groupby('trial_index')[cols_to_mean_dir + cols_to_mean_gau].mean().reset_index()
# df_mixed = mixed_lv.groupby('trial_index')[cols_to_mean_mixed].mean().reset_index()
#
# fig, ax = plt.subplots(4, 2, figsize=(20, 20))
# for i, col in enumerate(cols_to_mean_dir):
#     ax[i, 0].plot(df_avg['trial_index'], df_avg[col])
#     ax[i, 0].set_title(col)
#
# for i, col in enumerate(cols_to_mean_gau):
#     ax[i, 1].plot(df_avg['trial_index'], df_avg[col])
#     ax[i, 1].set_title(col)
#
# plt.show()
#
# # plot together
# fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# for i, col in enumerate(cols_to_mean_dir):
#     ax[0].plot(df_avg['trial_index'], df_avg[col], label=col)
#     ax[0].set_title('Dirichlet Model')
#     ax[0].legend()
#
# for i, col in enumerate(cols_to_mean_gau):
#     ax[1].plot(df_avg['trial_index'], df_avg[col], label=col)
#     ax[1].set_title('Multivariate Gaussian Model')
#     ax[1].legend()
#
# plt.show()
#
# # calculate the percentage of choosing the best option
# # create a new column to indicate the best option
# dual_lv['pair'] = dual_lv['pair'].map(lambda x: ''.join(x)).astype(str)
# best_option_dict = {'AB': 'A', 'CA': 'C', 'AD': 'A',
#                     'CB': 'C', 'BD': 'B', 'CD': 'C'}
# dual_lv['BestOption'] = dual_lv['pair'].map(best_option_dict)
# dual_lv['BestOptionChosen'] = dual_lv['choice'] == dual_lv['BestOption']
# propoptimal = dual_lv.groupby('pair')['BestOptionChosen'].mean().reset_index()
# propoptimal_by_trial = dual_lv.groupby(['pair', 'trial_index'])['BestOptionChosen'].mean().reset_index()
#
#
# # plot the percentage of choosing the best option
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.bar(propoptimal['pair'], propoptimal['BestOptionChosen'])
# ax.set_title('Percentage of Choosing the Best Option')
# ax.set_ylabel('Percentage')
# ax.set_xlabel('Pair')
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
# plt.show()
#
# # plot the percentage of choosing the best option by trial
# fig, ax = plt.subplots(3, 2, figsize=(20, 20))
# for i, pair in enumerate(propoptimal_by_trial['pair'].unique()):
#     df = propoptimal_by_trial[propoptimal_by_trial['pair'] == pair]
#     ax[i // 2, i % 2].plot(df['trial_index'], df['BestOptionChosen'])
#     ax[i // 2, i % 2].set_title(f'Pair {pair}')
#     ax[i // 2, i % 2].set_ylabel('Percentage')
#     ax[i // 2, i % 2].set_xlabel('Trial Index')
#     ax[i // 2, i % 2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
#
# plt.show()
#
# # calculate the percentage of process chosen
# process_chosen = dual_lv.groupby('trial_index')['process'].value_counts(normalize=True).unstack().reset_index()
#
# # plot the percentage of process chosen
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# for col in process_chosen.columns[1:]:
#     ax.plot(process_chosen['trial_index'], process_chosen[col], label=col)
#     ax.set_title('Percentage of Process Chosen')
#     ax.set_ylabel('Percentage')
#     ax.set_xlabel('Trial Index')
#     ax.legend()
#     ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
#
# plt.show()
#
# # this is the same set of plotting functions except that there are only four options to be plotted
# fig, ax = plt.subplots(4, 1, figsize=(10, 20))
# for i, col in enumerate(cols_to_mean_mixed):
#     ax[i].plot(df_mixed['trial_index'], df_mixed[col])
#     ax[i].set_title(col)
#
# plt.show()
#
# # plot together
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# for i, col in enumerate(cols_to_mean_mixed):
#     ax.plot(df_mixed['trial_index'], df_mixed[col], label=col)
#     ax.set_title('Mixed Model - LV')
#     ax.legend()
#
# plt.show()
#
# #
# dual_lv['pair'] = dual_lv['pair'].map(lambda x: ''.join(x)).astype(str)
# best_option_dict = {'AB': 'A', 'CA': 'C', 'AD': 'A',
#                     'CB': 'C', 'BD': 'B', 'CD': 'C'}
# dual_lv['BestOption'] = dual_lv['pair'].map(best_option_dict)
# dual_lv['BestOptionChosen'] = dual_lv['choice'] == dual_lv['BestOption']
# propoptimal = dual_lv.groupby('pair')['BestOptionChosen'].mean().reset_index()
# propoptimal_by_trial = dual_lv.groupby(['pair', 'trial_index'])['BestOptionChosen'].mean().reset_index()