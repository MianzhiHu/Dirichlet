import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind, pearsonr
import statsmodels.formula.api as smf
from utilities.utility_DataAnalysis import mean_AIC_BIC, create_bayes_matrix, process_chosen_prop

# after the simulation has been completed, we can just load the simulated data from the folder
folder_path = './data/DataFitting/FittingResults'
fitting_results = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)  # Get the full path of the file
        df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
        fitting_results[df_name] = pd.read_csv(file_path)

# unnest the dictionary into dfs
for key in fitting_results:
    globals()[key] = fitting_results[key]

# calculate the mean AIC and BIC values
for key in fitting_results:
    print(f"Model: {key}")
    mean_AIC_BIC(fitting_results[key])

# import the data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")
uncertainty_data = pd.read_csv('./data/UncertaintyData.csv')
uncertaintyPropOptimal = pd.read_csv('./data/UncertaintyPropOptimal.csv')
condition_assignments = uncertainty_data[['Subnum', 'Condition']].drop_duplicates()

LV = data[data['Condition'] == 'LV']
MV = data[data['Condition'] == 'MV']
HV = data[data['Condition'] == 'HV']

uncertainty_uf = uncertainty_data[uncertainty_data['Condition'] == 'S2A1']
uncertainty_uo = uncertainty_data[uncertainty_data['Condition'] == 'S2A2']
uncertaintyPropOptimal = uncertaintyPropOptimal.merge(condition_assignments, on='Subnum')

dataframes = [LV, MV, HV]
for i in range(len(dataframes)):
    dataframes[i] = dataframes[i].reset_index(drop=True)
    dataframes[i].iloc[:, 1] = (dataframes[i].index // 250) + 1
    dataframes[i].rename(columns={'X': 'Subnum', 'setSeen': 'SetSeen.', 'choice': 'KeyResponse', 'points': 'Reward'}, inplace=True)
    dataframes[i]['KeyResponse'] = dataframes[i]['KeyResponse'] - 1
    dataframes[i]['SetSeen.'] = dataframes[i]['SetSeen.'] - 1
    dataframes[i]['trial_index'] = dataframes[i].groupby('Subnum').cumcount() + 1

LV_df, MV_df, HV_df = dataframes

uncertainty_condition_results = {}

for key, results in fitting_results.items():
    if 'uncertainty' in key:
        results.rename(columns={'participant_id': 'Subnum'}, inplace=True)
        # join the condition assignments to the uncertainty data
        results = results.merge(condition_assignments, on='Subnum')
        grouped_results = results.groupby('Condition')
        for condition, group in grouped_results:
            uncertainty_condition_results[f"{key}_{condition}"] = group
        # print(f"Model: {key}")
        # print(f"Mean AIC: {grouped_results['AIC'].mean()}")
        # print(f"Mean BIC: {grouped_results['BIC'].mean()}")

for key in uncertainty_condition_results:
    globals()[key] = uncertainty_condition_results[key]

# create bayes factor matrices
# Filter fitting_results for HV, MV, and LV
fitting_results_HV = {k: v for k, v in fitting_results.items() if 'HV' in k}
fitting_results_MV = {k: v for k, v in fitting_results.items() if 'MV' in k}
fitting_results_LV = {k: v for k, v in fitting_results.items() if 'LV' in k}
uncertainty_frequency_results = {k: v for k, v in uncertainty_condition_results.items() if 'S2A1' in k}
uncertainty_only_results = {k: v for k, v in uncertainty_condition_results.items() if 'S2A2' in k}

# # Create Bayes factor matrices for HV, MV, and LV
# bayes_matrix_HV = create_bayes_matrix(fitting_results_HV, 'HV Bayes Factor Matrix')
# bayes_matrix_MV = create_bayes_matrix(fitting_results_MV, 'MV Bayes Factor Matrix')
# bayes_matrix_LV = create_bayes_matrix(fitting_results_LV, 'LV Bayes Factor Matrix')
# bayes_matrix_uncertainty_frequency = create_bayes_matrix(uncertainty_frequency_results, 'UF Bayes Factor Matrix')
# bayes_matrix_uncertainty_only = create_bayes_matrix(uncertainty_only_results, 'UO Bayes Factor Matrix')

# explode ProcessChosen
HV_df, process_chosen_HV = process_chosen_prop(Dual_HV_results, HV_df, sub=True)
MV_df, process_chosen_MV = process_chosen_prop(Dual_MV_results, MV_df, sub=True)
LV_df, process_chosen_LV = process_chosen_prop(Dual_LV_results, LV_df, sub=True)
uncertainty_uf, process_chosen_uf = process_chosen_prop(Dual_uncertaintyOld_results_S2A1, uncertainty_uf, sub=True)
uncertainty_uo, process_chosen_uo = process_chosen_prop(Dual_uncertaintyOld_results_S2A2, uncertainty_uo, sub=True)

dfs = [HV_df, MV_df, LV_df]
process_chosen_df = [process_chosen_HV, process_chosen_MV, process_chosen_LV]

for i in range(len(dfs)):
    prop_optimal = dfs[i].groupby(['Subnum', 'TrialType'])['bestOption'].mean().reset_index()
    process_chosen_Dir = process_chosen_df[i][process_chosen_df[i]['best_process_chosen'] == 'Dir']
    df_corr = pd.merge(prop_optimal, process_chosen_Dir, on=['Subnum', 'TrialType'], how='outer')
    df_corr.fillna(0, inplace=True)
    df_corr = df_corr[df_corr['TrialType'] == 'CA']
    corr, p = pearsonr(df_corr['bestOption'], df_corr['proportion'])
    print(f"correlation: {corr}, p-value: {p}")


# test the uncertainty data
prop_uf = uncertaintyPropOptimal[uncertaintyPropOptimal['Condition'] == 'S2A1']
prop_uo = uncertaintyPropOptimal[uncertaintyPropOptimal['Condition'] == 'S2A2']

uncertainty_dfs = [uncertainty_uf, uncertainty_uo]
uncertainty_process_chosen = [process_chosen_uf, process_chosen_uo]
prop_optimal_dfs = [prop_uf, prop_uo]
rfile_names = ['uncertainty_uf', 'uncertainty_uo']

for i in range(len(uncertainty_dfs)):
    prop_optimal = prop_optimal_dfs[i]
    prop_optimal.rename(columns={'ChoiceSet': 'TrialType'}, inplace=True)
    process_chosen_Dir = uncertainty_process_chosen[i][uncertainty_process_chosen[i]['best_process_chosen'] == 'Dir']
    df_corr = pd.merge(prop_optimal, process_chosen_Dir, on=['Subnum', 'TrialType'], how='outer')
    df_corr.fillna(0, inplace=True)
    df_corr = df_corr[df_corr['TrialType'] == 'CA']
    corr, p = pearsonr(df_corr['PropOptimal'], df_corr['proportion'])
    print(f"correlation: {corr}, p-value: {p}")

# find out the significance of the model
HV_df['best_process_chosen'] = (HV_df['best_process_chosen'] == 'Dir').astype(int)
HV_df_CA = HV_df[HV_df['TrialType'] == 'CA']
# HV_df.to_csv('./data/HV_df_CA.csv', index=False)

model = smf.logit('bestOption ~ best_process_chosen', data=HV_df_CA)
result = model.fit()
print(result.summary())



# selected_data = uncertainty_data.iloc[:, 0:21].drop_duplicates()
# # selected_data = selected_data.merge(uncertaintyPropOptimal[uncertaintyPropOptimal['ChoiceSet'] == 'CA'], on='Subnum')
#
# un_df, process_chosen_un = process_chosen_prop(Dual_uncertaintyOld_results, uncertainty_data, sub=True)
# process_chosen_un_Dir = process_chosen_un[process_chosen_un['best_process_chosen'] == 'Dir']
# process_chosen_un_Dir.rename(columns={'TrialType': 'ChoiceSet'}, inplace=True)
# df_corr = pd.merge(uncertaintyPropOptimal, process_chosen_un_Dir, on=['Subnum', 'ChoiceSet'], how='outer')
# df_corr.fillna(0, inplace=True)
# df_corr = df_corr.merge(selected_data, on=['Subnum', 'Condition'])
# df_corr.to_csv('./data/UncertaintyDualProcess.csv', index=False)

# RT analysis
# process_rt = LV_df.groupby('best_process_chosen')['RT'].mean().reset_index()
# # do a t-test to compare the reaction times of the two processes
#
# ttest_ind(LV_df[LV_df['best_process_chosen'] == 'Gau']['RT'], LV_df[LV_df['best_process_chosen'] == 'Dir']['RT'])
#
# # Fit a linear model
# model = smf.mixedlm('RT ~ best_process_chosen', data=LV_df, groups=LV_df['Subnum'] + LV_df['SetSeen.'])
# result = model.fit()
# print(result.summary())

