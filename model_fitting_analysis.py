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
# folder_path = './data/DataFitting/FittingResults/AlternativeModels/'
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
included_models = ['decay', 'delta', 'actr', 'Dual', 'Obj', 'Dir', 'Gau']
indices_to_calculate = ['AIC', 'BIC']
fitting_summary = fitting_summary_generator(fitting_results, included_models, indices_to_calculate)
fitting_summary = fitting_summary.round(3)
# if the value is less than 0.001, replace it with <0.001
numeric_cols = fitting_summary.select_dtypes(include='number').columns
fitting_summary[numeric_cols] = fitting_summary[numeric_cols].map(
    lambda x: '<0.001' if x == 0 else x)
save_df_to_word(fitting_summary, 'FittingSummary.docx')

# calculate the mean AIC and BIC advantage of dual process model over the other models
fitting_summary['model'] = fitting_summary['index'].str.split('_').str[0]
fitting_summary['condition'] = fitting_summary['index'].str.split('_').str[1]
fitting_summary.drop('index', axis=1, inplace=True)

# Specify the reference model to calculate the difference from
reference_model = 'Dual'

# Group by the condition and apply the difference calculation
fitting_summary_diff = fitting_summary.groupby('condition').apply(calculate_difference, reference_model,
                                                                  include_groups=False)
fitting_summary_diff = fitting_summary_diff[
    (fitting_summary_diff['model'] != reference_model) &
    (fitting_summary_diff['model'] != "Obj")
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

individual_param_df.to_csv('./data/IndividualParamResults.csv', index=False)

dual_param = individual_param_df[individual_param_df['model'] == 'Dual'].reset_index()
dual_param.loc[:, 'Subnum'] = dual_param.index + 1

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

individual_indices_df.to_csv('./data/IndividualIndices.csv', index=False)


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
HV_df, process_chosen_HV = process_chosen_prop(Dual_HV_results, HV_df, sub=True, values=['best_weight', 'best_obj_weight'])
MV_df, process_chosen_MV = process_chosen_prop(Dual_MV_results, MV_df, sub=True, values=['best_weight', 'best_obj_weight'])
LV_df, process_chosen_LV = process_chosen_prop(Dual_LV_results, LV_df, sub=True, values=['best_weight', 'best_obj_weight'])

# the proportion of choosing the Dirichlet process
print(HV_df.groupby('TrialType')['best_weight'].mean())


dfs = [HV_df, MV_df, LV_df]
process_chosen_df = [process_chosen_HV, process_chosen_MV, process_chosen_LV]

# combine the AIC and BIC values
columns = ['AIC', 'BIC', 'Model']
hv_results = [delta_HV_results, decay_HV_results, Dual_HV_results, Obj_HV_results, actr_HV_results]
mv_results = [delta_MV_results, decay_MV_results, Dual_MV_results, Obj_MV_results, actr_MV_results]
lv_results = [delta_LV_results, decay_LV_results, Dual_LV_results, Obj_LV_results, actr_LV_results]


def combine_results(results):
    combined_results = []
    model_names = ['Delta', 'Decay', 'Dual', 'Obj', 'ACTR']
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


# combine the dataframes and add a column for the condition
HV_df['Condition'] = 'HV'
MV_df['Condition'] = 'MV'
LV_df['Condition'] = 'LV'
combined_df = pd.concat([HV_df, MV_df, LV_df]).reset_index()
combined_df['Subnum'] = combined_df.index // 250 + 1
combined_df = combined_df.merge(dual_param, on='Subnum')
col_to_drop = ['index_x', 'index_y', 'Unnamed: 0', 'fname', 'AdvChoice', 'BlockCentered', 'subjID', 'Block']
combined_df.drop(col_to_drop, axis=1, inplace=True)
combined_df.rename(columns={'param_1': 't', 'param_2': 'a', 'param_3': 'subj_weight'}, inplace=True)

# save the data
combined_df.to_csv('./data/CombinedVarianceData.csv', index=False)



