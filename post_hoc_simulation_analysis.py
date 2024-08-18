import numpy as np
import pandas as pd
import os
import ast
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# import original data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")

mapping = {(0, 1): 'AB',
           (2, 3): 'CD',
           (2, 0): 'CA',
           (2, 1): 'CB',
           (1, 3): 'BD',
           (0, 3): 'AD'
           }

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
        simulations[key]['Condition'] = 'HV'
    elif 'MV' in key:
        simulations[key]['Condition'] = 'MV'
    else:
        simulations[key]['Condition'] = 'LV'

    # now separate the key to find the model plot_type
    model_type = key.split('_')[0]
    # add the model plot_type to the dataframe
    simulations[key]['model'] = model_type

    # transform delta, decay, and actr simulations
    if model_type in ['delta', 'decay', 'actr']:
        simulations[key]['pair'] = simulations[key]['pair'].astype(str).apply(lambda x: mapping[ast.literal_eval(x)])

    simulations[key].rename(columns={'pair': 'TrialType'}, inplace=True)

    # now, combine all the elements in the dictionary into a single dataframe
    if 'all_posthoc' not in locals():
        all_posthoc = simulations[key]
    else:
        all_posthoc = pd.concat([all_posthoc, simulations[key]], axis=0)

    globals()[key] = simulations[key]


# empirical
data_summary = data.groupby(['subnum', 'Condition', 'TrialType'])['bestOption'].mean().reset_index()
data_summary.rename(columns={'subnum': 'Subnum'}, inplace=True)

# sort all posthoc by condition
condition_order = ['LV', 'MV', 'HV']
all_posthoc['Condition'] = pd.Categorical(all_posthoc['Condition'], categories=condition_order, ordered=True)
all_posthoc = all_posthoc.sort_values(by=['Condition', 'Subnum', 'model'])

# redefine the subnum so that it is consistent with the empirical data
n_models = len(all_posthoc['model'].unique())
n_trialtypes = len(all_posthoc['TrialType'].unique())
all_posthoc = all_posthoc.reset_index(drop=True)
all_posthoc['Subnum'] = all_posthoc.index // (n_models * n_trialtypes) + 1

# join the empirical data with the simulations
all_posthoc = all_posthoc.merge(data_summary, on=['Subnum', 'Condition', 'TrialType'], how='left')

# add summary columns to all_posthoc
all_posthoc['AE'] = np.abs(all_posthoc['bestOption'] - all_posthoc['choice'])
all_posthoc['squared_error'] = all_posthoc['AE'] ** 2

# get RMSE for each model
RMSE = all_posthoc.groupby(['model'])['squared_error'].mean().reset_index()
RMSE['RMSE'] = RMSE['squared_error'].apply(np.sqrt)

RMSE_CA = all_posthoc[all_posthoc['TrialType'] == 'CA'].groupby(['model'])['squared_error'].mean().reset_index()
RMSE_CA['RMSE'] = RMSE_CA['squared_error'].apply(np.sqrt)

RMSE_by_condition = all_posthoc.groupby(['Condition', 'model'])['squared_error'].mean().reset_index()
RMSE_by_condition['RMSE'] = RMSE_by_condition['squared_error'].apply(np.sqrt)

MAE = all_posthoc.groupby(['model'])['AE'].mean().reset_index()
MAE_CA = all_posthoc[all_posthoc['TrialType'] == 'CA'].groupby(['model'])['AE'].mean().reset_index()
MAE_by_condition = all_posthoc.groupby(['Condition', 'model'])['AE'].mean().reset_index()

# Plot the distribution of AE
sns.set_theme(style='white')
plt.figure(figsize=(10, 6))
sns.boxplot(data=all_posthoc, x='model', y='AE', hue='TrialType')
plt.title('Absolute Error')
plt.ylabel('Absolute Error')
plt.xlabel('Model')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
sns.despine()
plt.show()


# # add summary columns to all_posthoc
# all_posthoc['pred_choice'] = np.where(all_posthoc['choice'] > 0.5, 1, 0)
# all_posthoc['AE'] = np.abs(all_posthoc['bestOption'] - all_posthoc['choice'])
# all_posthoc['squared_error'] = all_posthoc['AE'] ** 2

# # ========================================
# # filter steps (comment out if not needed)
# # ========================================
# included_models = ['Dual', 'delta', 'decay', 'actr', 'Obj']
# all_posthoc = all_posthoc[all_posthoc['model'].isin(included_models)]
#
# # reset subject number
# n_models = len(included_models)
# n_trials = 250
# # sort by condition
# all_posthoc = all_posthoc.sort_values(by=['Condition', 'Subnum', 'model', 'trial_index'])
# all_posthoc = all_posthoc.reset_index(drop=True)
# all_posthoc['Subnum'] = all_posthoc.index // (n_models * n_trials) + 1
#
# # # save the data
# # all_posthoc.to_csv('./data/all_posthoc.csv', index=False)
#
# for condition, model in all_posthoc.groupby(['Condition', 'model']):
#     print(f'{condition[0]}: {condition[1]}')
#     print(f'MSE: {model["squared_error"].mean()}')
#     print(f'RMSE: {np.sqrt(model["squared_error"].mean())}')
#     print(f'MAE: {model["AE"].mean()}')
#
# MSE_by_participant = all_posthoc.groupby(['Condition', 'model', 'Subnum', 'TrialType'])[[
#     'AE', 'squared_error']].mean().reset_index()
# # MSE_by_participant.to_csv('./data/MSE_by_participant.csv', index=False)
#
# # calculate the proportion of accurately predicting the best option
# all_posthoc['correct'] = np.where(all_posthoc['bestOption'] == all_posthoc['pred_choice'], 1, 0)
# proportion_correct = all_posthoc.groupby(['Condition', 'model', 'Subnum', 'TrialType'])['correct'].mean().reset_index()
# proportion_correct_CA = proportion_correct[proportion_correct['TrialType'] == 'CA']
#
# proportion_correct.to_csv('./data/proportion_correct.csv', index=False)
#
# # calculate the proportion of best option chosen
# proportion_best_option = all_posthoc.groupby(['Condition', 'model', 'Subnum', 'TrialType'])[
#     'choice'].mean().reset_index()
# proportion_best_option_by_trial = all_posthoc.groupby(['Condition', 'model', 'TrialType'])[
#     'choice'].mean().reset_index()
# proportion_best_option_by_trial_AD = proportion_best_option_by_trial[
#     proportion_best_option_by_trial['TrialType'] == 'AD']
#
# proportion_best_option_CA = proportion_best_option[proportion_best_option['TrialType'] == 'CA']
# proportion_best_option_CA = proportion_best_option_CA[proportion_best_option_CA['model'].isin(included_models)]
# proportion_best_option_CA['Condition'] = pd.Categorical(proportion_best_option_CA['Condition'],
#                                                         categories=['LV', 'MV', 'HV'], ordered=True)
#
# # plot the proportion of correctly predicting the best option
# palette = sns.color_palette("pastel", 4)
# sns.set_theme(style='white')
# plt.figure(figsize=(10, 6))
# sns.barplot(data=proportion_correct_CA, x='model', y='correct', hue='Condition', palette=palette)
# plt.title('Proportion of correctly predicting the best option')
# plt.ylabel('Proportion')
# plt.xlabel('Model')
# plt.show()
#
# # MSE for the proportion of optimal choices for each trial plot_type
# MAE_by_proportion = all_posthoc.groupby(['Condition', 'model', 'TrialType'])[
#     ['choice', 'bestOption']].mean().reset_index()
# MAE_by_proportion = MAE_by_proportion.groupby(['Condition', 'model', 'TrialType']).apply(
#     lambda x: np.abs(x['choice'] - x['bestOption']).mean()).reset_index()
# # divide the df by the condition
# MSE_by_proportion_HV = MAE_by_proportion[MAE_by_proportion['Condition'] == 'HV']
# MSE_by_proportion_MV = MAE_by_proportion[MAE_by_proportion['Condition'] == 'MV']
# MSE_by_proportion_LV = MAE_by_proportion[MAE_by_proportion['Condition'] == 'LV']
# MSE_by_proportion = MAE_by_proportion[MAE_by_proportion['TrialType'] == 'CA']
#
# # MSE for processes chosen
# process_data = pd.read_csv('./data/CombinedVarianceData.csv')
# HV_process = process_data[process_data['Condition'] == 'HV']
# MV_process = process_data[process_data['Condition'] == 'MV']
# LV_process = process_data[process_data['Condition'] == 'LV']
