import numpy as np
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# after the simulation has been completed, we can just load the simulated data from the folder
folder_path = './data/Simulation'
simulations = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)  # Get the full path of the file
        df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
        simulations[df_name] = pd.read_csv(file_path)

# calculate the percentage of choosing the best option
for key, df in simulations.items():
    # create a new column to indicate the best option
    df['pair'] = df['pair'].map(lambda x: ''.join(ast.literal_eval(x)))
    best_option_dict = {'AB': 'A', 'CA': 'C', 'AD': 'A',
                        'CB': 'C', 'BD': 'B', 'CD': 'C'}
    df['BestOption'] = df['pair'].map(best_option_dict)
    df['BestOptionChosen'] = df['choice'] == df['BestOption']

# unnest the dictionary into dfs
for key in simulations:
    globals()[key] = simulations[key]


# visualize the simulation results
cols_to_mean_dir = ['EV_A_Dir', 'EV_B_Dir', 'EV_C_Dir', 'EV_D_Dir']
cols_to_mean_gau = ['EV_A_Gau', 'EV_B_Gau', 'EV_C_Gau', 'EV_D_Gau']
cols_to_mean_mixed = ['EV_A', 'EV_B', 'EV_C', 'EV_D']
df_avg = dual_uncertainty.groupby('trial_index')[cols_to_mean_dir + cols_to_mean_gau].mean().reset_index()
propoptimal = dual_uncertainty.groupby('pair')['BestOptionChosen'].mean().reset_index()
propoptimal_by_trial = dual_uncertainty.groupby(['pair', 'trial_index'])['BestOptionChosen'].mean().reset_index()

fig, ax = plt.subplots(4, 2, figsize=(20, 20))
for i, col in enumerate(cols_to_mean_dir):
    ax[i, 0].plot(df_avg['trial_index'], df_avg[col])
    ax[i, 0].set_title(col)

for i, col in enumerate(cols_to_mean_gau):
    ax[i, 1].plot(df_avg['trial_index'], df_avg[col])
    ax[i, 1].set_title(col)

plt.show()

# plot together
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
for i, col in enumerate(cols_to_mean_dir):
    ax[0].plot(df_avg['trial_index'], df_avg[col], label=col)
    ax[0].set_title('Dirichlet Model')
    ax[0].legend()

for i, col in enumerate(cols_to_mean_gau):
    ax[1].plot(df_avg['trial_index'], df_avg[col], label=col)
    ax[1].set_title('Multivariate Gaussian Model')
    ax[1].legend()

plt.show()

# plot the percentage of choosing the best option
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.bar(propoptimal['pair'], propoptimal['BestOptionChosen'])
ax.set_title('Percentage of Choosing the Best Option')
ax.set_ylabel('Percentage')
ax.set_xlabel('Pair')
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.show()

# plot the percentage of choosing the best option by trial
fig, ax = plt.subplots(3, 2, figsize=(20, 20))
for i, pair in enumerate(propoptimal_by_trial['pair'].unique()):
    df = propoptimal_by_trial[propoptimal_by_trial['pair'] == pair]
    ax[i // 2, i % 2].plot(df['trial_index'], df['BestOptionChosen'])
    ax[i // 2, i % 2].set_title(f'Pair {pair}')
    ax[i // 2, i % 2].set_ylabel('Percentage')
    ax[i // 2, i % 2].set_xlabel('Trial Index')
    ax[i // 2, i % 2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

plt.show()

# calculate the percentage of process chosen
process_chosen = dual_uncertainty.groupby('trial_index')['process'].value_counts(normalize=True).unstack().reset_index()
process_chosen_percentage = dual_uncertainty['process'].value_counts(normalize=True).reset_index()
# take only trial 151 to 250
transfer_trials = dual_uncertainty[dual_uncertainty['trial_index'] > 150]
transfer_process_chosen = transfer_trials.groupby('pair')['process'].value_counts(normalize=True).unstack().reset_index()


# plot the percentage of process chosen
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for col in process_chosen.columns[1:]:
    ax.plot(process_chosen['trial_index'], process_chosen[col], label=col)
    ax.set_title('Percentage of Process Chosen')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Trial Index')
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
bar_width = 0.35
processes = transfer_process_chosen.columns[1:]
for i, process in enumerate(processes):
    if i == 0:
        bars = ax.bar(transfer_process_chosen['pair'], transfer_process_chosen[process], bar_width, label=process)
    else:
        bars = ax.bar(transfer_process_chosen['pair'], transfer_process_chosen[process], bar_width, label=process, bottom=transfer_process_chosen[processes[i-1]])

ax.set_title('Percentage of Process Chosen in Transfer Trials')
ax.set_ylabel('Percentage')
ax.set_xlabel('Pair')
ax.legend()
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.show()

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
