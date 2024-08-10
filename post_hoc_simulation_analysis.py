import numpy as np
import pandas as pd
import os
import ast
from utilities.utility_DataAnalysis import RMSE_calculation
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# import original data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")

LV = data[data['Condition'] == 'LV']
MV = data[data['Condition'] == 'MV']
HV = data[data['Condition'] == 'HV']

LV = LV.reset_index(drop=True)
MV = MV.reset_index(drop=True)
HV = HV.reset_index(drop=True)

# import post-hoc simulation results
folder_path = './data/Post_hoc'
simulations = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)  # Get the full path of the file
        df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
        simulations[df_name] = pd.read_csv(file_path)

# unnest the dictionary into dfs
for key in simulations:
    if 'HV' in key:
        # combine the original trial sequence with the simulated data
        simulations[key] = pd.concat([HV[['bestOption', 'TrialType']], simulations[key]], axis=1)
        simulations[key]['Condition'] = 'HV'
    elif 'MV' in key:
        simulations[key] = pd.concat([MV[['bestOption', 'TrialType']], simulations[key]], axis=1)
        simulations[key]['Condition'] = 'MV'
    else:
        simulations[key] = pd.concat([LV[['bestOption', 'TrialType']], simulations[key]], axis=1)
        simulations[key]['Condition'] = 'LV'
    # now separate the key to find the model plot_type
    model_type = key.split('_')[0]
    # add the model plot_type to the dataframe
    simulations[key]['model'] = model_type

    # now, combine all the elements in the dictionary into a single dataframe
    if 'all_posthoc' not in locals():
        all_posthoc = simulations[key]
    else:
        all_posthoc = pd.concat([all_posthoc, simulations[key]], axis=0)

    globals()[key] = simulations[key]

# add summary columns to all_posthoc
all_posthoc['pred_choice'] = np.where(all_posthoc['choice'] > 0.5, 1, 0)
all_posthoc['AE'] = np.abs(all_posthoc['bestOption'] - all_posthoc['choice'])
all_posthoc['squared_error'] = all_posthoc['AE'] ** 2

# ========================================
# filter steps (comment out if not needed)
# ========================================
included_models = ['Dual', 'delta', 'decay', 'actr', 'Recency']
all_posthoc = all_posthoc[all_posthoc['model'].isin(included_models)]

# reset subject number
n_models = len(included_models)
n_trials = 250
# sort by condition
all_posthoc = all_posthoc.sort_values(by=['Condition', 'Subnum', 'model', 'trial_index'])
all_posthoc = all_posthoc.reset_index(drop=True)
all_posthoc['Subnum'] = all_posthoc.index // (n_models * n_trials) + 1

# # save the data
# all_posthoc.to_csv('./data/all_posthoc.csv', index=False)

for condition, model in all_posthoc.groupby(['Condition', 'model']):
    print(f'{condition[0]}: {condition[1]}')
    print(f'MSE: {model["squared_error"].mean()}')
    print(f'RMSE: {np.sqrt(model["squared_error"].mean())}')
    print(f'MAE: {model["AE"].mean()}')

MSE_by_participant = all_posthoc.groupby(['Condition', 'model', 'Subnum', 'TrialType'])[[
    'AE', 'squared_error']].mean().reset_index()
# MSE_by_participant.to_csv('./data/MSE_by_participant.csv', index=False)

# calculate the proportion of accurately predicting the best option
all_posthoc['correct'] = np.where(all_posthoc['bestOption'] == all_posthoc['pred_choice'], 1, 0)
proportion_correct = all_posthoc.groupby(['Condition', 'model', 'Subnum', 'TrialType'])['correct'].mean().reset_index()
proportion_correct_CA = proportion_correct[proportion_correct['TrialType'] == 'CA']

proportion_correct.to_csv('./data/proportion_correct.csv', index=False)

# calculate the proportion of best option chosen
proportion_best_option = all_posthoc.groupby(['Condition', 'model', 'Subnum', 'TrialType'])[
    'choice'].mean().reset_index()
proportion_best_option_by_trial = all_posthoc.groupby(['Condition', 'model', 'TrialType'])[
    'choice'].mean().reset_index()
proportion_best_option_by_trial_AD = proportion_best_option_by_trial[proportion_best_option_by_trial['TrialType'] == 'AD']

proportion_best_option_CA = proportion_best_option[proportion_best_option['TrialType'] == 'CA']
proportion_best_option_CA = proportion_best_option_CA[proportion_best_option_CA['model'].isin(included_models)]
proportion_best_option_CA['Condition'] = pd.Categorical(proportion_best_option_CA['Condition'],
                                                        categories=['LV', 'MV', 'HV'], ordered=True)


# plot the proportion of correctly predicting the best option
palette = sns.color_palette("pastel", 4)
sns.set_theme(style='white')
plt.figure(figsize=(10, 6))
sns.barplot(data=proportion_correct_CA, x='model', y='correct', hue='Condition', palette=palette)
plt.title('Proportion of correctly predicting the best opt  ion')
plt.ylabel('Proportion')
plt.xlabel('Model')
plt.show()

# MSE for the proportion of optimal choices for each trial plot_type
MAE_by_proportion = all_posthoc.groupby(['Condition', 'model', 'TrialType'])[
    ['choice', 'bestOption']].mean().reset_index()
MAE_by_proportion = MAE_by_proportion.groupby(['Condition', 'model', 'TrialType']).apply(
    lambda x: np.abs(x['choice'] - x['bestOption']).mean()).reset_index()
# divide the df by the condition
MSE_by_proportion_HV = MAE_by_proportion[MAE_by_proportion['Condition'] == 'HV']
MSE_by_proportion_MV = MAE_by_proportion[MAE_by_proportion['Condition'] == 'MV']
MSE_by_proportion_LV = MAE_by_proportion[MAE_by_proportion['Condition'] == 'LV']
MSE_by_proportion = MAE_by_proportion[MAE_by_proportion['TrialType'] == 'CA']

# MSE for processes chosen
process_data = pd.read_csv('./data/CombinedVarianceData.csv')
HV_process = process_data[process_data['Condition'] == 'HV']
MV_process = process_data[process_data['Condition'] == 'MV']
LV_process = process_data[process_data['Condition'] == 'LV']
