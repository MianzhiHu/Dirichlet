import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind, pearsonr, norm
import statsmodels.formula.api as smf
from utilities.utility_DataAnalysis import (mean_AIC_BIC, create_bayes_matrix, process_chosen_prop,
                                            calculate_mean_squared_error, fitting_summary_generator,
                                            save_df_to_word, individual_param_generator, calculate_difference)
from utilities.utility_DataAnalysis import extract_all_parameters
import seaborn as sns
import matplotlib.pyplot as plt

# after the simulation has been completed, we can just load the simulated data from the folder
# folder_path = './data/DataFitting/FittingResults/AllCombinations/'
folder_path = './data/DataFitting/FittingResults/'
fitting_results = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, file)  # Get the full path of the file
        df_name = os.path.splitext(file)[0]  # Extract the file name without the extension
        fitting_results[df_name] = pd.read_csv(file_path)

# unnest the dictionary into dfs
for key in fitting_results:
    print(key)
    mean_AIC_BIC(fitting_results[key])
    globals()[key] = fitting_results[key]

# ======================================================================================================================
# Generate the fitting summary
# ======================================================================================================================
# select the models to be compared
included_models = ['decay', 'delta', 'actr', 'Dual']
indices_to_calculate = ['AIC', 'BIC']
fitting_summary = fitting_summary_generator(fitting_results, included_models, indices_to_calculate)
fitting_summary = fitting_summary.round(3)
# if the value is less than 0.001, replace it with <0.001
numeric_cols = fitting_summary.select_dtypes(include='number').columns
fitting_summary[numeric_cols] = fitting_summary[numeric_cols].map(
    lambda x: '<0.001' if x == 0 else x)
save_df_to_word(fitting_summary, 'FittingSummary.docx')

# calculate the mean AIC and BIC advantage of dual process model over the other models
fitting_summary['model'] = fitting_summary['index'].str.split('_').str[-2]
fitting_summary['condition'] = fitting_summary['index'].str.split('_').str[1]
fitting_summary.drop('index', axis=1, inplace=True)

# Specify the reference model to calculate the difference from
reference_model = 'Dual'



# Group by the condition and apply the difference calculation
fitting_summary_diff = fitting_summary.groupby('condition').apply(calculate_difference, reference_model,
                                                                  include_groups=False)
fitting_summary_diff = fitting_summary_diff[
    (fitting_summary_diff['model'] != reference_model) &
    (fitting_summary_diff['model'] != "Dir") &
    (fitting_summary_diff['model'] != "Gau")
]

# Calculate the average difference for each model
average_differences = fitting_summary_diff.groupby('condition')[['AIC_diff', 'BIC_diff']].mean().reset_index()

print(average_differences)


# ======================================================================================================================
# Extract the best fitting parameters and analyze the results
# ======================================================================================================================
param_cols = ['param_1', 'param_2', 'param_3']
individual_param_df = individual_param_generator(fitting_results, param_cols)

# filter for the conditions and models
individual_param_df = individual_param_df[individual_param_df['condition'].isin(['HV', 'MV', 'LV'])]
individual_param_df = individual_param_df[individual_param_df['model'].isin(included_models)]
individual_param_df = individual_param_df[individual_param_df['model'] != 'actr']

# plot the distribution of the parameters
sns.set_theme(style='white')
sns.displot(data=individual_param_df, x='param_2', hue='model', col='condition', kind='kde', fill=True)
# set the x-axis limit
plt.xlim(0, 1)
plt.show()

print(individual_param_df.groupby(['model', 'condition'])['param_2'].mean())
print(individual_param_df.groupby(['model', 'condition'])['param_2'].std())

# individual_param_df.to_csv('./data/IndividualParamResults.csv', index=False)

# ======================================================================================================================
# Extract the individual AIC and BIC values
# ======================================================================================================================
# Initialize a list to store individual AIC and BIC values along with the model key
individual_indices = []

for key in fitting_results:
    for i in range(len(fitting_results[key])):
        individual_indices.append({
            'index': key,
            'AIC': fitting_results[key]['AIC'].iloc[i],
            'BIC': fitting_results[key]['BIC'].iloc[i]
        })

# Convert the list to a DataFrame
individual_indices_df = pd.DataFrame(individual_indices)

# filter
individual_indices_df['condition'] = individual_indices_df['index'].str.split('_').str[1]
individual_indices_df['model'] = individual_indices_df['index'].str.split('_').str[0]
individual_indices_df = individual_indices_df[individual_indices_df['condition'].isin(['HV', 'MV', 'LV'])]
individual_indices_df = individual_indices_df[individual_indices_df['model'].isin(included_models)]
individual_indices_df.drop('index', axis=1, inplace=True)

# individual_indices_df.to_csv('./data/IndividualIndices.csv', index=False)


# import the data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")

LV = data[data['Condition'] == 'LV']
MV = data[data['Condition'] == 'MV']
HV = data[data['Condition'] == 'HV']

dataframes = [LV, MV, HV]
for i in range(len(dataframes)):
    dataframes[i] = dataframes[i].reset_index(drop=True)
    dataframes[i].iloc[:, 1] = (dataframes[i].index // 250) + 1
    dataframes[i].rename(columns={'X': 'Subnum', 'setSeen': 'SetSeen.', 'choice': 'KeyResponse', 'points': 'Reward'},
                         inplace=True)
    dataframes[i]['KeyResponse'] = dataframes[i]['KeyResponse'] - 1
    dataframes[i]['SetSeen.'] = dataframes[i]['SetSeen.'] - 1
    dataframes[i]['trial_index'] = dataframes[i].groupby('Subnum').cumcount() + 1

LV_df, MV_df, HV_df = dataframes

# # create bayes factor matrices
# # Filter fitting_results for HV, MV, and LV
fitting_results_HV = {k: v for k, v in fitting_results.items() if 'HV' in k}
fitting_results_MV = {k: v for k, v in fitting_results.items() if 'MV' in k}
fitting_results_LV = {k: v for k, v in fitting_results.items() if 'LV' in k}


# # Create Bayes factor matrices for HV, MV, and LV
bayes_matrix_HV = create_bayes_matrix(fitting_results_HV, 'HV Bayes Factor Matrix')
bayes_matrix_MV = create_bayes_matrix(fitting_results_MV, 'MV Bayes Factor Matrix')
bayes_matrix_LV = create_bayes_matrix(fitting_results_LV, 'LV Bayes Factor Matrix')

# explode ProcessChosen
HV_df, process_chosen_HV = process_chosen_prop(Recency_HV_results, HV_df, sub=True)
MV_df, process_chosen_MV = process_chosen_prop(Dual_MV_results, MV_df, sub=True)
LV_df, process_chosen_LV = process_chosen_prop(Dual_LV_results, LV_df, sub=True)

# the proportion of choosing the Dirichlet process
HV_df['best_process_chosen'] = (HV_df['best_process_chosen'] == 'Dir').astype(int)
print(HV_df.groupby('TrialType')['best_process_chosen'].mean())


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

# combine the AIC and BIC values
columns = ['AIC', 'BIC', 'Model']
hv_results = [delta_HV_results, decay_HV_results, Dual_HV_results]
mv_results = [delta_MV_results, decay_MV_results, Dual_MV_results]
lv_results = [delta_LV_results, decay_LV_results, Dual_LV_results]


def combine_results(results):
    combined_results = []
    model_names = ['Delta', 'Decay', 'Dual Process']
    for i in range(len(results)):
        model_results = results[i]
        model_results['Model'] = model_names[i]
        combined_results.append(model_results[columns])
    return pd.concat(combined_results)


combined_HV_results = combine_results(hv_results)
combined_MV_results = combine_results(mv_results)
combined_LV_results = combine_results(lv_results)

# # plot the AIC and BIC values
# sns.set_theme(style='white')
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# sns.barplot(x='Model', y='AIC', data=combined_LV_results, ax=ax[0])
# ax[0].set_title('Low Variance')
# sns.barplot(x='Model', y='AIC', data=combined_MV_results, ax=ax[1])
# ax[1].set_title('Moderate Variance')
# sns.barplot(x='Model', y='AIC', data=combined_HV_results, ax=ax[2])
# ax[2].set_title('High Variance')
# for i in range(3):
#     # remove the x label
#     ax[i].set_xlabel('')
#     # set lower y limit
#     ax[i].set_ylim(bottom=120)
# for i in (1, 2):
#     ax[i].set_ylabel('')
# sns.despine()
# plt.show()


# find out the significance of the model
for df in [HV_df, MV_df, LV_df]:
    df['best_process_chosen'] = (df['best_process_chosen'] == 'Dir').astype(int)

# combine the dataframes and add a column for the condition
HV_df['Condition'] = 'HV'
MV_df['Condition'] = 'MV'
LV_df['Condition'] = 'LV'
combined_df = pd.concat([HV_df, MV_df, LV_df]).reset_index()
combined_df['Subnum'] = combined_df.index // 250 + 1
# combined_df.to_csv('./data/CombinedVarianceData.csv', index=False)

combined_df_CA = combined_df[combined_df['TrialType'] == 'CA']

model = smf.logit('bestOption ~ best_process_chosen + TrialType + Condition', data=combined_df)
result = model.fit()
print(result.summary())

# model = smf.logit('bestOption ~ best_process_chosen + Condition', data=combined_df_CA)
# result = model.fit()
# print(result.summary())

# Assuming 'result' is the fitted model object
cov = result.cov_params()
diff_se = np.sqrt(
    cov.loc['TrialType[T.CA]', 'TrialType[T.CA]'] + cov.loc['TrialType[T.AD]', 'TrialType[T.AD]'] - 2 * cov.loc[
        'TrialType[T.CA]', 'TrialType[T.AD]'])
z_score = (-1.5318 - 0.2048) / diff_se
p_value = norm.sf(abs(z_score)) * 2  # two-tailed p-value

print("Z-score for difference:", z_score)
print("P-value for difference:", p_value)


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


# # ====================================================================================================================
# # Below is the code for the uncertainty dataset which is not used in the paper (comment out if not needed)
# # ====================================================================================================================

# # import the data
# uncertainty_data = pd.read_csv('./data/UncertaintyData.csv')
# uncertaintyPropOptimal = pd.read_csv('./data/UncertaintyPropOptimal.csv')
# condition_assignments = uncertainty_data[['Subnum', 'Condition']].drop_duplicates()
#
# uncertainty_uf = uncertainty_data[uncertainty_data['Condition'] == 'S2A1']
# uncertainty_uo = uncertainty_data[uncertainty_data['Condition'] == 'S2A2']
# uncertaintyPropOptimal = uncertaintyPropOptimal.merge(condition_assignments, on='Subnum')

# uncertainty_condition_results = {}
#
# for key, results in fitting_results.items():
#     if 'uncertainty' in key:
#         results.rename(columns={'participant_id': 'Subnum'}, inplace=True)
#         # join the condition assignments to the uncertainty data
#         results = results.merge(condition_assignments, on='Subnum')
#         grouped_results = results.groupby('Condition')
#         for condition, group in grouped_results:
#             uncertainty_condition_results[f"{key}_{condition}"] = group
#         print(f"Model: {key}")
#         print(f"Mean AIC: {grouped_results['AIC'].mean()}")
#         print(f"Mean BIC: {grouped_results['BIC'].mean()}")
#
# for key in uncertainty_condition_results:
#     globals()[key] = uncertainty_condition_results[key]
#
# # create bayes factor matrices
# uncertainty_frequency_results = {k: v for k, v in uncertainty_condition_results.items() if 'S2A1' in k}
# uncertainty_only_results = {k: v for k, v in uncertainty_condition_results.items() if 'S2A2' in k}
#
# # explode ProcessChosen
# uncertainty_uf, process_chosen_uf = process_chosen_prop(Dual_uncertaintyOld_results_S2A1, uncertainty_uf, sub=True)
# uncertainty_uo, process_chosen_uo = process_chosen_prop(Dual_uncertaintyOld_results_S2A2, uncertainty_uo, sub=True)
# uncertainty_data, process_chosen_uncertainty = process_chosen_prop(Dual_uncertaintyOld_results, uncertainty_data,
#                                                                    sub=True)
#
# # uncertainty_data.to_csv('./data/UncertaintyDualProcess.csv', index=False)
#
# # test the uncertainty data
# prop_uf = uncertaintyPropOptimal[uncertaintyPropOptimal['Condition'] == 'S2A1']
# prop_uo = uncertaintyPropOptimal[uncertaintyPropOptimal['Condition'] == 'S2A2']
#
# uncertainty_dfs = [uncertainty_uf, uncertainty_uo]
# uncertainty_process_chosen = [process_chosen_uf, process_chosen_uo]
# prop_optimal_dfs = [prop_uf, prop_uo]
# rfile_names = ['uncertainty_uf', 'uncertainty_uo']
#
# for i in range(len(uncertainty_dfs)):
#     prop_optimal = prop_optimal_dfs[i]
#     prop_optimal.rename(columns={'ChoiceSet': 'TrialType'}, inplace=True)
#     process_chosen_Dir = uncertainty_process_chosen[i][uncertainty_process_chosen[i]['best_process_chosen'] == 'Dir']
#     df_corr = pd.merge(prop_optimal, process_chosen_Dir, on=['Subnum', 'TrialType'], how='outer')
#     df_corr.fillna(0, inplace=True)
#     df_corr = df_corr[df_corr['TrialType'] == 'CA']
#     corr, p = pearsonr(df_corr['PropOptimal'], df_corr['proportion'])
#     print(f"correlation: {corr}, p-value: {p}")

# # tentative analysis
# selected_data = uncertainty_data.iloc[:, 0:21].drop_duplicates()
# # selected_data = selected_data.merge(uncertaintyPropOptimal[uncertaintyPropOptimal['ChoiceSet'] == 'CA'], on='Subnum')
#
# un_df, process_chosen_un = process_chosen_prop(Dual_uncertaintyOld_results, uncertainty_data, sub=True)
# process_chosen_un_Dir = process_chosen_un[process_chosen_un['best_process_chosen'] == 'Dir']
# process_chosen_un_Dir.rename(columns={'TrialType': 'ChoiceSet'}, inplace=True)
# df_corr = pd.merge(uncertaintyPropOptimal, process_chosen_un_Dir, on=['Subnum', 'ChoiceSet'], how='outer')
# df_corr.fillna(0, inplace=True)
# df_corr = df_corr.merge(selected_data, on=['Subnum', 'Condition'])
# df_corr.to_csv('./data/UncertaintyDualProcessParticipant.csv', index=False)



