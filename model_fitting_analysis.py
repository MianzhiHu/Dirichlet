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
from utilities.utility_PlottingFunctions import prop
from scipy.special import psi  # Digamma function

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
included_models = ['decay', 'delta', 'actr', 'Dual', 'Obj', 'Dir', 'Gau', 'delta_asymmetric', 'mean_var_utility']
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
hv_results = [Dual_HV_results, delta_HV_results, decay_HV_results, actr_HV_results]
mv_results = [Dual_MV_results, delta_MV_results, decay_MV_results, actr_MV_results]
lv_results = [Dual_LV_results, delta_LV_results, decay_LV_results, actr_LV_results]


def combine_results(results):
    combined_results = []
    model_names = ['Dual', 'Delta', 'Decay', 'ACTR']
    for i in range(len(results)):
        model_results = results[i]
        model_results['Model'] = model_names[i]
        combined_results.append(model_results[columns])
    return pd.concat(combined_results)


combined_HV_results = combine_results(hv_results)
combined_MV_results = combine_results(mv_results)
combined_LV_results = combine_results(lv_results)

# plot the AIC and BIC values
sns.set_theme(style='white')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(x='Model', y='BIC', data=combined_LV_results, ax=ax[0], errorbar=None, color='darkred')
ax[0].set_title('LV')
sns.barplot(x='Model', y='BIC', data=combined_MV_results, ax=ax[1], errorbar=None, color='darkred')
ax[1].set_title('MV')
sns.barplot(x='Model', y='BIC', data=combined_HV_results, ax=ax[2], errorbar=None, color='darkred')
ax[2].set_title('HV')
for i in range(3):
    # remove the x label
    ax[i].set_xlabel('')
    # set lower y limit
    dfs = [combined_LV_results, combined_MV_results, combined_HV_results]
    y_lower = dfs[i].groupby('Model')['BIC'].mean().min() - 10
    ax[i].set_ylim(bottom=y_lower)
    # set the font properties
    ax[i].set_xticklabels(ax[i].get_xticklabels(), fontproperties=prop, fontsize=15)
    ax[i].set_yticklabels(ax[i].get_yticks(), fontproperties=prop, fontsize=15)
    # set the title
    ax[i].set_title(ax[i].get_title(), fontproperties=prop, fontsize=25)
for i in (1, 2):
    ax[i].set_ylabel('')
ax[0].set_ylabel('BIC', fontproperties=prop, fontsize=20)
sns.despine()
plt.tight_layout()
plt.savefig('./figures/BICValues.png', dpi=600)
plt.show()


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


# ======================================================================================================================
# Implement Bayesian Model Selection (BMS)
# ======================================================================================================================
def vb_model_selection(log_evidences, alpha0=None, tol=1e-6, max_iter=1000):
    """
    Variational Bayesian Model Selection for multiple models and multiple subjects.

    Implements the iterative VB algorithm described by:
    - Equations 3, 7, 9, 11, 12, 13, and the final pseudo-code in Equation 14.

    Parameters
    ----------
    log_evidences : np.ndarray, shape (N, K)
        Matrix of log model evidences for N subjects and K models.
        log_evidences[n, k] = ln p(y_n | m_{nk})
    alpha0 : np.ndarray, shape (K,)
        Initial Dirichlet prior parameters. If None, set to 1 for all models.
    tol : float
        Tolerance for convergence based on changes in alpha.
    max_iter : int
        Maximum number of VB iterations.

    Returns
    -------
    alpha : np.ndarray, shape (K,)
        Final Dirichlet parameters of the approximate posterior q(r).
    g : np.ndarray, shape (N, K)
        Posterior model assignment probabilities per subject.
    """

    N, K = log_evidences.shape
    if alpha0 is None:
        alpha0 = np.ones(K)

    # Initialize alpha
    alpha = alpha0.copy()

    for iteration in range(max_iter):
        alpha_sum = np.sum(alpha)

        # Compute unnormalized posterior assignments u_nk
        # u_nk = exp(ln p(y_n | m_nk) + Psi(alpha_k) - Psi(alpha_sum))
        u = np.exp(log_evidences + psi(alpha) - psi(alpha_sum))  # shape (N,K)

        # Normalize to get g_nk
        u_sum = np.sum(u, axis=1, keepdims=True)
        g = u / u_sum  # shape (N,K)

        # Update beta_k = sum_n g_nk
        beta = np.sum(g, axis=0)  # shape (K,)

        # Update alpha
        alpha_new = alpha0 + beta

        # Check convergence
        diff = np.linalg.norm(alpha_new - alpha)
        alpha = alpha_new
        if diff < tol:
            break

    return alpha, g


# -----------------------------------------
# Example usage:
# Suppose we have N=10 subjects and K=3 models.
# We assume we have computed log model evidences for each subject and model.
np.random.seed(42)

N = 10  # number of subjects
K = 3  # number of models

# Simulated log-evidences (e.g., from model fitting)
# In practice, these would be obtained from a separate model estimation procedure per subject and model.
log_evidences = np.random.randn(N, K)

# Run VB model selection
alpha0 = np.ones(K)  # uniform prior
alpha_est, g_est = vb_model_selection(log_evidences, alpha0=alpha0, tol=1e-6, max_iter=500)

# alpha_est: Dirichlet parameters of the approximate posterior q(r)
# g_est: posterior probabilities that each subject was generated by each model
print("Final alpha (Dirichlet parameters):", alpha_est)
print("Posterior model probabilities per subject:\n", g_est)
print("Expected model frequencies:", alpha_est / np.sum(alpha_est))


from scipy.stats import dirichlet

def compute_exceedance_prob(alpha, n_samples=100000):
    """
    Compute exceedance probabilities for each model by Monte Carlo approximation.

    Parameters
    ----------
    alpha : array-like of shape (K,)
        The Dirichlet parameters for the posterior q(r).
    n_samples : int
        Number of samples to draw from Dirichlet.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    exceedance_probs : np.ndarray of shape (K,)
        The exceedance probability for each model.
    """
    samples = dirichlet.rvs(alpha, size=n_samples)
    winners = np.argmax(samples, axis=1)  # indices of best model per draw

    K = len(alpha)
    exceedance_probs = np.bincount(winners, minlength=K) / n_samples
    return exceedance_probs

# Example usage:
ex_probs = compute_exceedance_prob(alpha_est[:2], n_samples=100000)
print("Exceedance probabilities:", ex_probs)

