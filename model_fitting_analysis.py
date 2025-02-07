import ast
import numpy as np
import pandas as pd
import os
import pingouin as pg
import sm
from scipy.stats import ttest_ind, pearsonr, norm, f_oneway
import statsmodels.formula.api as smf

from plotting import data_CA
from utilities.utility_DataAnalysis import (mean_AIC_BIC, create_bayes_matrix, process_chosen_prop,
                                            calculate_mean_squared_error, fitting_summary_generator,
                                            save_df_to_word, individual_param_generator, calculate_difference)
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.utility_PlottingFunctions import prop
from utils.ComputationalModeling import (vb_model_selection, compute_exceedance_prob, parameter_extractor,
                                         clean_list_string)

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
included_models = ['decay', 'delta', 'actr', 'Dual', 'deltaasym', 'utility', 'Obj', 'Gau', 'Dir']
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
print(individual_param_df['model'].unique())
print(f'max t for all models: {individual_param_df["param_1"].max()}; min t: {individual_param_df["param_1"].min()}')
print(f'max t for dual: {dual_param["param_1"].max()}; min t for dual: {dual_param["param_1"].min()}')
print(f'max a for dual: {dual_param["param_2"].max()}; min a for dual: {dual_param["param_2"].min()}')
print(f'sd a for dual: {dual_param["param_2"].std()}')
print(f'max subj_weight for dual: {dual_param["param_3"].max()}; min subj_weight for dual: {dual_param["param_3"].min()}')

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
hv_results = [Dual_HV_results, delta_HV_results, decay_HV_results, actr_HV_results, deltaasym_HV_results,
              utility_HV_results]
mv_results = [Dual_MV_results, delta_MV_results, decay_MV_results, actr_MV_results, deltaasym_MV_results,
              utility_MV_results]
lv_results = [Dual_LV_results, delta_LV_results, decay_LV_results, actr_LV_results, deltaasym_LV_results,
              utility_LV_results]

# hv_results = [Dual_HV_results, delta_HV_results, decay_HV_results, actr_HV_results, delta_asym_HV_results,
#               utility_HV_results, Dir_HV_results, Gau_HV_results, Obj_HV_results]
# mv_results = [Dual_MV_results, delta_MV_results, decay_MV_results, actr_MV_results, delta_asym_MV_results,
#               utility_MV_results, Dir_MV_results, Gau_MV_results, Obj_MV_results]
# lv_results = [Dual_LV_results, delta_LV_results, decay_LV_results, actr_LV_results, delta_asym_LV_results,
#               utility_LV_results, Dir_LV_results, Gau_LV_results, Obj_LV_results]


def combine_results(results):
    combined_results = []
    model_names = ['Dual', 'Delta', 'Decay', 'ACTR', 'Risk-Sensitive Delta', 'Mean-Variance Utility']
    # model_names = ['Dual', 'Delta', 'Decay', 'ACTR', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Dirichlet',
    #                'Gaussian', 'Objective']
    for i in range(len(results)):
        model_results = results[i]
        model_results['Model'] = model_names[i]
        combined_results.append(model_results[columns])
    df = pd.concat(combined_results)

    index_df = pd.DataFrame()
    for model in df['Model'].unique():
        index_df[f'{model}_AIC'] = df[df['Model'] == model]['AIC'].values
        index_df[f'{model}_BIC'] = df[df['Model'] == model]['BIC'].values

    return df, index_df


_, combined_HV_results = combine_results(hv_results)
_, combined_MV_results = combine_results(mv_results)
_, combined_LV_results = combine_results(lv_results)

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

# --------------------------------------------------------------------------------------------------------------
# Extract the EV Gau
# --------------------------------------------------------------------------------------------------------------
# extract the EV Gau
EV_Gau = []
for result in [Dual_HV_results, Dual_MV_results, Dual_LV_results]:
    for i in range(len(result)):
        EV_Gau.append(result['EV_Gau'].iloc[i])

EV_Gau_df = pd.DataFrame(EV_Gau, columns=['EV_Gau'])
EV_Gau_df['Condition'] = ['HV'] * 100 + ['MV'] * 100 + ['LV'] * 93
EV_Gau_df['Subnum'] = EV_Gau_df.index + 1

# extract Gaussian EV for A and C
for i in range(len(EV_Gau_df)):
    while True:
        try:
            # Try to evaluate the current row
            EV_Gau_df.at[i, 'EV_Gau'] = ast.literal_eval(EV_Gau_df.at[i, 'EV_Gau'])
            break  # If successful, exit the loop
        except (SyntaxError, ValueError):
            # If evaluation fails, clean the string and retry
            EV_Gau_df.at[i, 'EV_Gau'] = clean_list_string(EV_Gau_df.at[i, 'EV_Gau'])

EV_Gau_df['EV_A'] = EV_Gau_df['EV_Gau'].apply(lambda x: x[0])
EV_Gau_df['EV_C'] = EV_Gau_df['EV_Gau'].apply(lambda x: x[2])

# incorporate real data
data_filtered = combined_df[(combined_df['Trial'] <= 150)]
data_filtered = data_filtered.groupby(['Subnum', 'Condition', 'KeyResponse'])[['Reward']].mean().reset_index()
data_filtered = data_filtered[data_filtered['KeyResponse'].isin([0, 2])]
data_filtered = data_filtered.pivot_table(index=['Subnum', 'Condition'], columns='KeyResponse', values='Reward').reset_index()
EV_Gau_df = EV_Gau_df.merge(data_filtered, on=['Subnum', 'Condition'], how='left')

# make the data long
EV_Gau_df = EV_Gau_df.melt(id_vars=['Subnum', 'Condition'], value_vars=['EV_A', 'EV_C', 0, 2], var_name='Option',
                            value_name='EV')

# visualize the data
plt.figure()
sns.barplot(data=EV_Gau_df, x='Condition', y='EV', hue='Option', errorbar='se', hue_order=['EV_A', 0, 'EV_C', 2],
            palette=sns.color_palette('deep')[:4])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=['EV(A)', 'Reward(A)', 'EV(C)', 'Reward(C)'], title='')
sns.despine()
plt.xlabel('')
plt.tight_layout()
plt.savefig('./figures/EV_Gau.png', dpi=1000)

# --------------------------------------------------------------------------------------------------------------
# Examine the parameters
# --------------------------------------------------------------------------------------------------------------
# anova
print(individual_param_df['model'].unique())
model_of_interest = 'deltaasym'

# t
print(pg.anova(data=individual_param_df[individual_param_df['model'] == model_of_interest], dv='param_1', between='condition'))
pairwise = pg.pairwise_tukey(data=individual_param_df[individual_param_df['model'] == model_of_interest], dv='param_1', between='condition')
# a
print(pg.anova(data=individual_param_df[individual_param_df['model'] == model_of_interest], dv='param_2', between='condition'))
# subj_weight
print(pg.anova(data=individual_param_df[individual_param_df['model'] == model_of_interest], dv='param_3', between='condition'))

# mediation analysis
fre_rate = combined_df.groupby(['Subnum', 'TrialType'])[['bestOption', 'best_weight']].mean().reset_index()
fre_rate = fre_rate[fre_rate['TrialType'] == 'CA']

dual_param = dual_param.merge(fre_rate, on='Subnum')
dual_param['best_weight'] = dual_param['best_weight'].astype(float)

results = pg.mediation_analysis(data=dual_param, x='best_weight', m='param_1', y='bestOption', alpha=0.05)

# generalized linear model
data_CA = combined_df[combined_df['TrialType'] == 'CA']
model = smf.mixedlm("best_obj_weight ~ C(condition)", data_CA, groups=data_CA["Subnum"])

# --------------------------------------------------------------------------------------------------------------
# Perform variational Bayesian model selection
# --------------------------------------------------------------------------------------------------------------
K = 6 # number of models

# select columns that end with BIC
condition_of_interest = combined_LV_results
bic_cols = [col for col in condition_of_interest.columns if col.endswith('AIC')]
condition_of_interest['best_model'] = condition_of_interest[bic_cols].idxmin(axis=1)
print(condition_of_interest['best_model'].value_counts() / len(condition_of_interest))
log_evidences = condition_of_interest[bic_cols].values / (-2)

# Run VB model selection
alpha0 = np.ones(K)  # uniform prior
alpha_est, g_est = vb_model_selection(log_evidences, alpha0=alpha0, tol=1e-12, max_iter=50000)

# alpha_est: Dirichlet parameters of the approximate posterior q(r)
# g_est: posterior probabilities that each subject was generated by each model
print("Final alpha (Dirichlet parameters):", alpha_est)
# print("Posterior model probabilities per subject:\n", g_est)
print("Expected model frequencies:", alpha_est / np.sum(alpha_est))

# calculate the exceedance probabilities
ex_probs = compute_exceedance_prob(alpha_est, n_samples=100000)
print("Exceedance probabilities:", ex_probs.round(3))

# ======================================================================================================================
# Additional Model Fitting Analysis as Requested by Reviewers
# ======================================================================================================================
folder_50_path = './data/DataFitting/FittingResults/AlternativeModels/'
fitting_50_results = {}

for file in os.listdir(folder_50_path):
    if (file.endswith('.csv')) and ('50' in file):
        file_path = os.path.join(folder_50_path, file)
        df_name = os.path.splitext(file)[0]
        fitting_50_results[df_name] = pd.read_csv(file_path)

# unnest the dictionary into dfs
for key in fitting_50_results:
    print(key)
    mean_AIC_BIC(fitting_50_results[key])
    globals()[key] = fitting_50_results[key]

# extract the weights
HV_50 = HV_df[(HV_df['TrialType'] == 'AB') | (HV_df['TrialType'] == 'CD')].groupby(['Subnum', 'TrialType']).head(50)
MV_50 = MV_df[(MV_df['TrialType'] == 'AB') | (MV_df['TrialType'] == 'CD')].groupby(['Subnum', 'TrialType']).head(50)
LV_50 = LV_df[(LV_df['TrialType'] == 'AB') | (LV_df['TrialType'] == 'CD')].groupby(['Subnum', 'TrialType']).head(50)

_, dual_50_HV = process_chosen_prop(dual_50_HV_results, HV_50, sub=True, values=['best_weight', 'best_obj_weight'])
_, dual_50_MV = process_chosen_prop(dual_50_MV_results, MV_50, sub=True, values=['best_weight', 'best_obj_weight'])
_, dual_50_LV = process_chosen_prop(dual_50_LV_results, LV_50, sub=True, values=['best_weight', 'best_obj_weight'])

# extract the parameters
dual_param_HV = parameter_extractor(dual_50_HV_results).reset_index(drop=True)
dual_param_MV = parameter_extractor(dual_50_MV_results).reset_index(drop=True)
dual_param_LV = parameter_extractor(dual_50_LV_results).reset_index(drop=True)

# add condition labels
dual_param_HV['Condition'] = 'HV'
dual_param_MV['Condition'] = 'MV'
dual_param_LV['Condition'] = 'LV'

# combine the dataframes
dual_param_50 = pd.concat([dual_param_LV, dual_param_MV, dual_param_HV], axis=0).reset_index(drop=True)
dual_param_50.rename(columns={'participant_id': 'Subnum'}, inplace=True)

dual_param_50 = dual_param_50[['Subnum', 't', 'alpha', 'subj_weight', 'Condition']]
dual_data_50 = pd.concat([dual_50_LV, dual_50_MV, dual_50_HV], axis=0).reset_index(drop=True)

dual_param_50.loc[:, 'Subnum'] = dual_param_50.index + 1
dual_data_50.loc[:, 'Subnum'] = dual_data_50.index // 100 + 1

dual_50 = pd.merge(dual_data_50, dual_param_50, on='Subnum')

# visualize the distribution
bins = np.linspace(0, 1, 11)
plt.figure()
g = sns.FacetGrid(dual_50, col='Condition', margin_titles=False)
g.map(sns.histplot, 'best_obj_weight', kde=True, color=sns.color_palette('deep')[0], stat='probability', bins=bins)
g.set_axis_labels('', '% of Participants', fontproperties=prop, fontsize=15)
g.set_titles(col_template="{col_name}", fontproperties=prop, size=20)
g.set(xlim=(0, 1))
g.set_xticklabels(fontproperties=prop)
g.set_yticklabels(fontproperties=prop)
g.fig.text(0.5, 0.05, 'Objective Dirichlet Weight', ha='center', fontproperties=prop, fontsize=15)
plt.savefig('./figures/Weight_Distribution_50.png', dpi=1000)
plt.show()
