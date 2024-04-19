import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests

# import the data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")
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

plt.bar(['LV', 'MV', 'HV'], mean_CA, yerr=se_CA)
plt.title('Percentage of Choosing the Best Option for CA Pair')
plt.ylim(0, 0.7)
plt.ylabel('Percentage')
plt.xlabel('Condition')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
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






