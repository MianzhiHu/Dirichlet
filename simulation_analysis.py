import numpy as np
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import f_oneway, ttest_1samp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

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

# ================== Simulation Analysis ==================
# Observed data
CA_avg = [0.6154838709677419, 0.5504, 0.4324]  # LV, MV, HV

dual_list = [recency_lv, recency_mv, recency_hv]
decay_list = [decay_lv, decay_mv, decay_hv]
delta_list = [delta_lv, delta_mv, delta_hv]
actr_list = [actr_lv, actr_mv, actr_hv]

# calculate the RMSE for each model
rmse_list = []

for model in [dual_list, decay_list, delta_list, actr_list]:

    model_name = ['Dual', 'Decay', 'Delta', 'ACTR']

    for i in range(len(model)):
        CA_trials = model[i][model[i]['pair'] == 'CA']
        CA_pred = CA_trials.groupby('simulation_num')['BestOptionChosen'].mean()
        # one sample t-test
        t, p = ttest_1samp(CA_pred, 0.5)
        print(f"Model: {model_name[i]}, Condition: {['LV', 'MV', 'HV'][i]}, t-statistic: {t}, p-value: {p}")
        rmse_val = np.sqrt(np.mean((CA_pred - CA_avg[i]) ** 2))
        rmse_list.append(rmse_val)


# calculate if process is related to the best option chosen
sim_CA = dual_lv[dual_lv['pair'] == 'CA']
# convert the process column to numeric
sim_CA.loc[:, 'process'] = sim_CA['process'].map({'Dir': 0, 'Gau': 1})
sim_CA.loc[:, 'BestOptionChosen'] = sim_CA['BestOptionChosen'].astype(int)

# mixed effects model
model = smf.mixedlm("BestOptionChosen ~ process", sim_CA, groups=sim_CA['simulation_num'])
result = model.fit()
print(result.summary())


# ================== Visualization ==================
# visualize the simulation results
cols_to_mean_dir = ['EV_A_Dir', 'EV_B_Dir', 'EV_C_Dir', 'EV_D_Dir']
cols_to_mean_gau = ['EV_A_Gau', 'EV_B_Gau', 'EV_C_Gau', 'EV_D_Gau']
cols_to_mean_mixed = ['EV_A', 'EV_B', 'EV_C', 'EV_D']
df_avg = dual_hv.groupby('trial_index')[cols_to_mean_dir + cols_to_mean_gau].mean().reset_index()
propoptimal = dual_hv.groupby('pair')['BestOptionChosen'].mean().reset_index()
propoptimal_by_trial = dual_hv.groupby(['pair', 'trial_index'])['BestOptionChosen'].mean().reset_index()

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
# define the order of the pairs
pair_order = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']
propoptimal['pair'] = pd.Categorical(propoptimal['pair'], categories=pair_order, ordered=True)
propoptimal = propoptimal.sort_values('pair')

plt.bar(propoptimal['pair'], propoptimal['BestOptionChosen'])
plt.title('Percentage of Choosing the Best Option')
plt.ylabel('Percentage')
plt.xlabel('Pair')
plt.ylim(0, 0.9)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
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

# plot the percentage of choosing the best option only for CA pair
mean_CA = []
se_CA = []
upper_CI = []
lower_CI = []


var_condition = dual_list
for var in var_condition:
    propoptimal_CA = var[var['pair'] == 'CA'].groupby('simulation_num')['BestOptionChosen'].mean()
    mean_CA.append(propoptimal_CA.mean())
    # calculate the standard error
    propoptimal_CA_se = propoptimal_CA.std() / np.sqrt(len(propoptimal_CA))
    # find the .975 quantile
    upper_CI.append(propoptimal_CA.quantile(0.975))
    # find the .025 quantile
    lower_CI.append(propoptimal_CA.quantile(0.025))
    se_CA.append(propoptimal_CA_se)

# conduct a one-way ANOVA for all the var in the list
f_stat, p_val = f_oneway(*[var[var['pair'] == 'CA'].groupby('simulation_num')['BestOptionChosen'].mean()
                           for var in var_condition])
print(f'F-statistic: {f_stat}, p-value: {p_val}')

# pairwise comparison
# conduct the Tukey HSD test
df = pd.concat([var[var['pair'] == 'CA'].groupby('simulation_num')['BestOptionChosen'].mean() for var in var_condition])
df = pd.DataFrame(df)
df['condition'] = ['LV' for _ in range(10000)] + ['MV' for _ in range(10000)] + ['HV' for _ in range(10000)]
tukey = pairwise_tukeyhsd(endog=df['BestOptionChosen'], groups=df['condition'], alpha=0.05)


# Define colors for each bar
palette = sns.color_palette("pastel", 3)

# Plot the percentage of choosing the best option only for CA pair
plt.clf()
plt.bar(['LV', 'MV', 'HV'], mean_CA, yerr=se_CA, color=palette)
plt.ylim(0, .75)
# plt.ylabel('Percentage of Selecting C in CA Pair')
plt.axhline(y=0.5, color='black', linestyle='--')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
sns.despine()
plt.show()

# calculate the percentage of process chosen
process_chosen = recency_hv.groupby('trial_index')['process'].value_counts(normalize=True).unstack().reset_index()
process_chosen_percentage = recency_hv['process'].value_counts(normalize=True).reset_index()


# take only trial 151 to 250
def transfer_process_chosen(df):
    transfer_trials = df[df['trial_index'] > 150]
    if 'process' in df.columns:
        transfer_process_chosen_df = transfer_trials.groupby('pair')['process'].value_counts(normalize=True).unstack().reset_index()
    if 'weight_Dir' in df.columns:
        transfer_weight_df = transfer_trials.groupby('pair')['weight_Dir'].mean().reset_index()

    if ('process' in df.columns) and ('weight_Dir' in df.columns):
        return transfer_process_chosen_df, transfer_weight_df
    elif 'process' in df.columns:
        return transfer_process_chosen_df
    elif 'weight_Dir' in df.columns:
        return transfer_weight_df


var_df = [recency_lv, recency_mv, recency_hv]

transfer_process_chosen = pd.concat([transfer_process_chosen(df) for df in var_df])
# create a new column to indicate the condition
# the first 6 pairs are from the low variance condition
condition_list = ['LV' for _ in range(4)] + ['MV' for _ in range(4)] + ['HV' for _ in range(4)]
transfer_process_chosen['Condition'] = condition_list
transfer_process_chosen = transfer_process_chosen[transfer_process_chosen['pair'] == 'CA']
dir_se = transfer_process_chosen['Dir'].std() / np.sqrt(10000)

# calculate the weight of the process
transfer_weight = pd.concat([transfer_weight(df) for df in var_df])
transfer_weight['Condition'] = condition_list
transfer_weight = transfer_weight[transfer_weight['pair'] == 'CA']


# Plot the percentage of process chosen for each pair in the transfer phase
plt.figure()
plt.bar(transfer_process_chosen['Condition'], transfer_process_chosen['Dir'], color=palette, yerr=dir_se)
# plt.ylabel('Percentage of Dirichlet-Based Decisions')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.axhline(y=0.5, color='black', linestyle='--')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
sns.despine()
plt.show()


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
