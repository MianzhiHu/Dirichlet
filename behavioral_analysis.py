import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import f_oneway, ttest_ind, ttest_1samp, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from utilities.utility_DataAnalysis import option_mean_calculation, count_choices, summary_choices
from utilities.utility_PlottingFunctions import prop

# import the data
data = pd.read_csv("./data/ABCDContRewardsAllData.csv")
data_process = pd.read_csv("./data/CombinedVarianceData.csv")
summary = data_process[(data_process['TrialType'] == 'CA') & (data_process['model'] == 'Dual')].groupby(
    ['Subnum', 'Condition']).agg(
    bestOption=('bestOption', 'mean'),
    t=('t', 'mean'),
    a=('a', 'mean'),
    obj_weight=('best_obj_weight', 'mean'),
    subj_weight=('subj_weight', 'mean'),
    best_weight=('best_weight', 'mean')
).reset_index()
summary.to_csv('./data/CA_summary.csv', index=False)

sub_hv_data = data_process[(data_process['Condition'] == 'HV') & (data_process['TrialType'].isin(['AB', 'CD']))]
sub_mv_data = data_process[(data_process['Condition'] == 'MV') & (data_process['TrialType'].isin(['AB', 'CD']))]
sub_lv_data = data_process[(data_process['Condition'] == 'LV') & (data_process['TrialType'].isin(['AB', 'CD']))]

sub_hv_summary = summary_choices(sub_hv_data)
sub_hv_count = count_choices(sub_hv_data)
sub_mv_summary = summary_choices(sub_mv_data)
sub_mv_count = count_choices(sub_mv_data)
sub_lv_summary = summary_choices(sub_lv_data)
sub_lv_count = count_choices(sub_lv_data)
sub_all_summary = summary_choices(data_process[data_process['TrialType'].isin(['AB', 'CD'])])
sub_all_count = count_choices(data_process[data_process['TrialType'].isin(['AB', 'CD'])])


if __name__ == '__main__':
    LV = data[data['Condition'] == 'LV']
    MV = data[data['Condition'] == 'MV']
    HV = data[data['Condition'] == 'HV']

    # calculate the average reward for each condition
    LV_option = option_mean_calculation(LV)
    MV_option = option_mean_calculation(MV)
    HV_option = option_mean_calculation(HV)

    training_data = data[data['Trial'] <= 150]
    mean = training_data.groupby('choice')['points'].mean()

    # keep track of the dynamic mean for each participant
    training_data.loc[:, 'cumulative_mean'] = (training_data.groupby('subnum')['points']
                                               .expanding().mean().reset_index(level=0, drop=True))
    training_data.loc[:, 'above_average'] = training_data['points'] > training_data['cumulative_mean']

    var_condition = [LV, MV, HV]

    propoptimal_mean = pd.DataFrame()
    se = pd.DataFrame()

    propoptimal = LV.groupby(['subnum', 'TrialType'])['bestOption'].mean()

    for condition in var_condition:
        propoptimal = condition.groupby(['subnum', 'TrialType'])['bestOption'].mean()
        if propoptimal_mean.empty:
            propoptimal_mean = propoptimal.reset_index().groupby('TrialType')['bestOption'].mean()
            se = propoptimal.reset_index().groupby('TrialType')['bestOption'].std() / np.sqrt(len(propoptimal))
        else:
            propoptimal_mean = pd.concat(
                [propoptimal_mean, propoptimal.reset_index().groupby('TrialType')['bestOption'].mean()],
                axis=1)
            se = pd.concat(
                [se, propoptimal.reset_index().groupby('TrialType')['bestOption'].std() / np.sqrt(len(propoptimal))],
                axis=1)

    # put the data in the right format
    propoptimal_mean.columns = ['LV', 'MV', 'HV']
    se.columns = ['LV', 'MV', 'HV']

    # Conduct one-sample t-tests
    reward_ratio = 0.75 / (0.75 + 0.65)
    for condition in ['LV', 'MV', 'HV']:
        CA_trials = data[(data['Condition'] == condition) & (data['TrialType'] == 'CA')]
        CA_trials = CA_trials.groupby('subnum')['bestOption'].mean()
        t, p = ttest_1samp(CA_trials, 0.5)
        t1, p1 = ttest_1samp(CA_trials, reward_ratio)
        print(f"Condition: {condition}")
        print(f"Against 0.5: {t}, {p}")
        print(f"Against reward ratio: {t1}, {p1}")

    # Conduct Kruskal-Wallis test
    summary_hv = summary[summary['Condition'] == 'HV']
    summary_mv = summary[summary['Condition'] == 'MV']
    summary_lv = summary[summary['Condition'] == 'LV']

    # analysis for the proportion of optimal choices
    sub_all_count = pd.pivot_table(sub_all_count, index='Subnum', columns=['TrialType'],
                                   values='optimal_ratio').reset_index()
    sub_all_count.loc[:, 'total_ratio'] = (sub_all_count['AB'] * 2 + sub_all_count['CD']) / 3
    sub_all_count = pd.merge(sub_all_count, summary, on='Subnum', how='outer')
    sub_all_count.to_csv('./data/sub_all_count.csv', index=False)

    # Get demographics
    sex = [1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2,
           1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2,
           0, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2,
           2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1,
           2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1,
           1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 1, 2, 2]
    age = [19, 20, 20, 18, 18, 19, 32, 19, 18, 18, 18, 18, 19, 19, 20, 18, 19, 19, 20, 20, 19, 20, 18, 18, 19, 19, 20,
           18, 19, 18, 19, 19, 18, 19, 19, 18, 21, 19, 19, 20, 20, 18, 18, 18, 18, 19, 19, 19, 18, 19, 18, 19, 18, 18,
           18, 20, 19, 19, 19, 19, 20, 25, 18, 19, 19, 19, 20, 18, 18, 20, 18, 19, 19, 18, 20, 19, 19, 19, 19, 18, 19,
           18, 19, 19, 19, 19, 18, 19, 19, 19, 18, 19, 20, 18, 19, 18, 27, 19, 19, 19, 19, 19, 18, 19, 19, 19, 18, 18,
           19, 19, 18, 18, 20, 19, 21, 18, 19, 18, 19, 20, 19, 19, 19, 20, 18, 30, 19, 18, 18, 19, 18, 20, 18, 19, 18,
           19, 20, 20, 19, 18, 25, 20, 19, 19, 20, 19, 18, 20, 19, 18, 18, 20, 18, 18, 20, 19, 19, 19, 19, 24, 19, 20,
           19, 19, 19, 18, 19, 20, 23, 20, 18, 18, 22, 19, 19, 18, 19, 18, 20, 19, 18, 19, 19, 21, 19, 20, 21, 18, 19,
           19, 19, 19, 18, 19, 18, 18, 18, 18, 19, 18, 19, 19, 19, 18, 20, 18, 18, 18, 19, 19, 19, 18, 21, 18, 19, 19,
           20, 20, 20, 19, 19, 18, 19, 18, 18, 18, 19, 18, 18, 18, 22, 20, 20, 19, 21, 19, 20, 22, 19, 18, 19, 20, 19,
           22, 21, 19, 19, 18, 19, 19, 19, 18, 18, 20, 18, 19, 18, 20, 18, 19]

    print(f"Age: {np.mean(age)}, {np.std(age)}")
    print(sex.count(2) / 260)



