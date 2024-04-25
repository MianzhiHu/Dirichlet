import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests

# import the data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")
data_process = pd.read_csv("./data/CombinedVarianceData.csv")
uncertainty_data = pd.read_csv('./data/UncertaintyData.csv')
uncertaintyPropOptimal = pd.read_csv('./data/UncertaintyDualProcessParticipant.csv')
condition_assignments = uncertainty_data[['Subnum', 'Condition']].drop_duplicates()

LV = data[data['Condition'] == 'LV']
MV = data[data['Condition'] == 'MV']
HV = data[data['Condition'] == 'HV']

var_condition = [LV, MV, HV]

mean_CA = []
se_CA = []

for condition in var_condition:
    propoptimal_CA = condition[condition['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean()
    mean_CA.append(propoptimal_CA.mean())
    # calculate the standard error
    propoptimal_CA_se = propoptimal_CA.std() / np.sqrt(len(propoptimal_CA))
    se_CA.append(propoptimal_CA_se)

# Define colors for each bar
palette = sns.color_palette("pastel", 3)

# Plot the percentage of choosing the best option only for CA pair
plt.bar(['LV', 'MV', 'HV'], mean_CA, yerr=se_CA, color=palette)
plt.ylim(0, 0.7)
plt.ylabel('Percentage of Selecting C in CA Pair')
plt.xlabel('Condition')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
# remove the top and right spines
sns.despine()
plt.show()


# conduct an ANOVA
f_oneway(*[var[var['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean() for var in var_condition])

# post-hoc t-tests
LM_t, LM_p = ttest_ind(LV[LV['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean(),
            MV[MV['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean())

LH_t, LH_p = ttest_ind(LV[LV['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean(),
            HV[HV['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean())

MH_t, MH_p = ttest_ind(MV[MV['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean(),
            HV[HV['TrialType'] == 'CA'].groupby('subnum')['bestOption'].mean())

print(f"LM_t: {LM_t}, LM_p: {LM_p}")
print(f"LH_t: {LH_t}, LH_p: {LH_p}")
print(f"MH_t: {MH_t}, MH_p: {MH_p}")

# correct for multiple comparisons
t_values = [LM_t, LH_t, MH_t]
p_values = [LM_p, LH_p, MH_p]

reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# dealing with uncertainty data
uncertainty = uncertaintyPropOptimal[(uncertaintyPropOptimal['Condition'] == 'S2A1')]

mean_uncertainty = []
se_uncertainty = []

for trial in uncertainty['ChoiceSet'].unique():
    propoptimal_CA = uncertainty[uncertainty['ChoiceSet'] == trial].groupby('Subnum')['PropOptimal'].mean()
    mean_uncertainty.append(propoptimal_CA.mean())
    # calculate the standard error
    propoptimal_CA_se = propoptimal_CA.std() / np.sqrt(len(propoptimal_CA))
    se_uncertainty.append(propoptimal_CA_se)

plt.bar(uncertainty['ChoiceSet'].unique(), mean_uncertainty, yerr=se_uncertainty)
plt.title('Percentage of Choosing the Best Option for Uncertainty Pair')
plt.ylim(0, 0.9)
plt.ylabel('Percentage')
plt.xlabel('Condition')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.show()

# plot the percentage of best process chosen
# set the condition order for plotting
condition_order = ['LV', 'MV', 'HV']
option_order = ['AB', 'CD', 'CA', 'CB', 'AD', 'BD']

data_process['Condition'] = pd.Categorical(data_process['Condition'], condition_order)
data_process['TrialType'] = pd.Categorical(data_process['TrialType'], option_order)

process_chosen = data_process.groupby(['Subnum', 'Condition', 'TrialType'])['best_process_chosen'].mean().reset_index()
process_chosen_CA = data_process[data_process['TrialType'] == 'CA']
process_chosen_CA = process_chosen_CA.groupby(['Subnum', 'Condition'])['best_process_chosen'].mean().reset_index()
process_chosen_CA['best_process_chosen'] = 1 - process_chosen_CA['best_process_chosen']

best_option = data_process.groupby(['Subnum', 'Condition', 'TrialType'])['bestOption'].mean().reset_index()

# Create a bar plot
sns.barplot(x='Condition', y='bestOption', hue='TrialType', data=best_option)
plt.title('Percentage of Best Option Chosen')
plt.ylabel('Percentage')
plt.xlabel('Condition')
plt.show()

palette = sns.color_palette("pastel", 3)

sns.barplot(x='Condition', y='best_process_chosen', data=process_chosen_CA, palette=palette)
plt.ylabel('Percentage')
plt.xlabel('Condition')
sns.despine()
plt.show()

# plot for RT
RT = data_process.groupby(['Subnum', 'best_process_chosen'])['RT'].mean().reset_index()

sns.barplot(x='best_process_chosen', y='RT', data=RT, palette=palette)
plt.xticks([0, 1], ['Gaussian', 'Dirichlet'])
plt.ylabel('RT')
plt.xlabel('')
sns.despine()
plt.show()





