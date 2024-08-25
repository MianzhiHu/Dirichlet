import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from fontTools.ttLib.woff2 import bboxFormat
from matplotlib.pyplot import errorbar
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import dirichlet, multivariate_normal
import matplotlib.tri as mtri
from matplotlib import cm, ticker
from utilities.utility_PlottingFunctions import (scatter_Dirichlet, bar_Dirichlet, scatter_Gaussian, bar_Gaussian,
                                                 plot_planes, prop, three_planes)
from behavioral_analysis import sub_hv_count, sub_mv_count, sub_lv_count
from sklearn.linear_model import LinearRegression

# ======================================================================================================================
# Plot for theoretical comparison between Dirichlet and Gaussian distributions
# ======================================================================================================================
# plot illustrations of Dirichlet distribution
alpha = [2, 102, 1]
# scatter_Dirichlet(alpha)
# bar_Dirichlet(alpha, resolution=500, elev=45, azim=60)

# do the same for multivariate Gaussian distribution
# ========== Starting point ==========
mean = [0.5, 0.5, 0.5]
var = [0.01, 0.01, 0.01]

# ========== When B is selected again ==========
b_history = [0.5, 0.6]
mean_b = np.mean(b_history)
var_b = np.var(b_history)

# ========== When B is selected 100 times ==========
b_history_hypo = np.repeat(0.6, 102)
b_history_hypo[0] = 0.5
mean_b = np.mean(b_history_hypo)
var_b = np.var(b_history_hypo)

# ========== When A is suddenly selected ==========
a_history = [0.5, 0.95]
mean_a = np.mean(a_history)
var_a = np.var(a_history)
mean = [mean_a, mean_b, 0.5]
var = [var_a, var_b, 0.01]

cov = np.diag(var)

# scatter_Gaussian(mean, cov)
# bar_Gaussian(mean, cov, resolution=500, elev=45, azim=60)

# ======================================================================================================================
# Plot for empirical data
# ======================================================================================================================
# Load the data
data = pd.read_csv("./data/CombinedVarianceData.csv")
sub_all_count = pd.read_csv('./data/sub_all_count.csv')

data_CA = data[(data['TrialType'] == 'CA') & (data['model'] == 'Dual')]
data_CA.loc[:, 'Condition'] = data_CA['Condition'].map({'HV': 3, 'MV': 2, 'LV': 1})
summary = data_CA.groupby(['Subnum', 'Condition']).agg(
    bestOption=('bestOption', 'mean'),
    t=('t', 'mean'),
    a=('a', 'mean'),
    obj_weight=('best_obj_weight', 'mean'),
    subj_weight=('subj_weight', 'mean'),
    best_weight=('best_weight', 'mean')
).reset_index()

bins = np.linspace(0, 1, 11)

# Create a 3D plot to show the relationship between weight, best_option, and condition
three_planes(summary, 'best_weight', x_label='Overall Dirichlet Weight', name='Overall_Weight_Optimal')


# plot how subjective and objective weights change across conditions
plt.figure()
sns.barplot(x='Condition', y='best_weight', data=summary, color=sns.color_palette('pastel')[3], errorbar='se')
plt.ylabel('Dirichlet Weight', fontproperties=prop, fontsize=15)
plt.xlabel('')
plt.xticks([0, 1, 2], ['LV', 'MV', 'HV'], fontproperties=prop, fontsize=15)
plt.yticks(fontproperties=prop)
sns.despine()
plt.savefig('./figures/Weight_Condition.png', dpi=1000)
plt.show()


# revert the condition to the original order
summary['Condition'] = summary['Condition'].map({3: 'HV', 2: 'MV', 1: 'LV'})

# sort the summary by condition from LV to HV
order = ['LV', 'MV', 'HV']
summary = summary.sort_values(by='Condition', key=lambda x: x.map({v: i for i, v in enumerate(order)}))

# plot the distribution of the weights for each condition and each weight using facet grid
plt.figure()
g = sns.FacetGrid(summary, col='Condition', margin_titles=False)
g.map(sns.histplot, 'best_weight', kde=True, color=sns.color_palette('deep')[0], stat='probability', bins=bins)
g.set_axis_labels('', '% of Participants', fontproperties=prop, fontsize=15)
g.set_titles(col_template="{col_name}", fontproperties=prop, size=20)
g.set(xlim=(0, 1))
g.set_xticklabels(fontproperties=prop)
g.set_yticklabels(fontproperties=prop)
g.fig.text(0.5, 0.05, 'Overall Dirichlet Weight', ha='center', fontproperties=prop, fontsize=15)
plt.savefig('./figures/Weight_Distribution.png', dpi=1000)
plt.show()

# plot for RT
RT = data[data['TrialType'] == 'CA'].groupby(['Subnum', 'best_weight', 'Condition'])['RT'].mean().reset_index()

pos_bins = np.linspace(0, 1, 21)

red_color = sns.color_palette("bright", 6)[3]

sns.regplot(x='best_weight', y='RT', data=RT, x_ci='ci', ci=95, fit_reg=True, order=4, line_kws={'color': red_color, 'lw': 3},
            scatter_kws={'alpha': 0.5}, x_bins=pos_bins, n_boot=10000)
plt.ylabel('Reaction Time', fontproperties=prop, fontsize=15)
plt.xlabel('Dirichlet Weight', fontproperties=prop, fontsize=15)
plt.xticks(fontproperties=prop)
plt.yticks(fontproperties=prop)
sns.despine()
plt.savefig('./figures/RT.png', dpi=600)
plt.show()

# plot for all
# Define colors for each bar
palette6 = sns.color_palette("pastel", 6)
palette3 = sns.color_palette("pastel", 3)

all_mean = data.groupby(['Subnum', 'Condition', 'TrialType'])['bestOption'].mean().reset_index()
all_mean.loc[:, 'Condition'] = pd.Categorical(all_mean['Condition'], categories=['LV', 'MV', 'HV'], ordered=True)
all_mean_CA = all_mean[all_mean['TrialType'] == 'CA']
all_mean_CA.loc[:, 'Condition'] = pd.Categorical(all_mean_CA['Condition'], categories=['LV', 'MV', 'HV'], ordered=True)

hue_order = ['AB', 'CD', 'CA', 'BD', 'AD', 'CB']

sns.barplot(x='Condition', y='bestOption', hue='TrialType', data=all_mean, palette=palette6, hue_order=hue_order,
            order=['LV', 'MV', 'HV'])

# Plot the overall trend line on top of the bar plot
sns.lineplot(x='Condition', y='bestOption', data=all_mean_CA, linewidth=2, errorbar='se',
             color=sns.color_palette("dark")[8], marker='o', markersize=5)

plt.ylabel('% Selecting the Optimal Option', fontproperties=prop, fontsize=20)
plt.xlabel('')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.xticks(fontproperties=prop, fontsize=15)
plt.yticks(fontproperties=prop, fontsize=15)
legend = plt.legend(title='Trial Type', prop=prop, framealpha=0.5)
legend.get_title().set_fontproperties(prop)
sns.despine()
plt.savefig('./figures/all_behavioral.png', dpi=1000)
plt.show()

# plot for CA
sns.barplot(x='Condition', y='bestOption', data=all_mean_CA, color=palette6[2], order=['LV', 'MV', 'HV'])
plt.ylabel('% Selecting C in CA Trials', fontproperties=prop, fontsize=20)
plt.xlabel('')
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.xticks(fontproperties=prop, fontsize=15)
plt.yticks(fontproperties=prop, fontsize=15)
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
sns.despine()
plt.savefig('./figures/CA_behavioral.png', dpi=1000)
plt.show()

# plot the distribution of counts for MV and LV
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
sns.histplot(sub_lv_count['optimal_ratio'], kde=True, stat='probability', bins=bins, color=sns.color_palette('deep')[3], ax=ax[0])
sns.histplot(sub_mv_count['optimal_ratio'], kde=True, stat='probability', bins=bins, color=sns.color_palette('deep')[3], ax=ax[1])
sns.histplot(sub_hv_count['optimal_ratio'], kde=True, stat='probability', bins=bins, color=sns.color_palette('deep')[3], ax=ax[2])
ax[0].set_title('LV', fontproperties=prop, fontsize=20)
ax[1].set_title('MV', fontproperties=prop, fontsize=20)
ax[2].set_title('HV', fontproperties=prop, fontsize=20)
for i in range(3):
    ax[i].set_xlabel('')
    ax[i].set_ylabel('% of Participants', fontproperties=prop, fontsize=15, labelpad=10)
    for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        label.set_fontproperties(prop)
    ax[i].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax[i].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[i].tick_params(axis='both', which='major', labelsize=10)

fig.text(0.5, 0.02, '% of Choosing the Optimal Option During Training', ha='center', fontproperties=prop, fontsize=15)
sns.despine()
plt.savefig('./figures/HighLowDirichlet.png', dpi=1000)
plt.show()

# plot a plot to predict objective weight from proportion of optimal choices
sub_all_count= sub_all_count[sub_all_count['total_ratio'] > 0.1]
plt.figure()
sns.lmplot(x='total_ratio', y='obj_weight', hue='Condition', data=sub_all_count, ci=95, palette=sns.color_palette('deep'),
           scatter_kws={'alpha': 0.3}, legend=False, aspect=1.25, line_kws={'lw': 3})
plt.ylabel('Objective Dirichlet Weight', fontproperties=prop, fontsize=15)
plt.xlabel('% of Choosing the Optimal Option During Training', fontproperties=prop, fontsize=15)
plt.xticks(fontproperties=prop)
plt.yticks(fontproperties=prop)
plt.legend(title='Condition', prop=prop, title_fontproperties=prop, loc='lower left', fontsize='xx-large')
plt.savefig('./figures/Weight_Prediction.png', dpi=1000)
plt.show()


# plot the distribution of counts for MV and LV
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
sns.histplot(sub_lv_count['optimal_ratio'], kde=True, stat='probability', bins=bins, color=sns.color_palette('deep')[3], ax=ax[0])
sns.histplot(sub_mv_count['optimal_ratio'], kde=True, stat='probability', bins=bins, color=sns.color_palette('deep')[3], ax=ax[1])
sns.histplot(sub_hv_count['optimal_ratio'], kde=True, stat='probability', bins=bins, color=sns.color_palette('deep')[3], ax=ax[2])
ax[0].set_title('LV', fontproperties=prop, fontsize=20)
ax[1].set_title('MV', fontproperties=prop, fontsize=20)
ax[2].set_title('HV', fontproperties=prop, fontsize=20)
for i in range(3):
    ax[i].set_xlabel('')
    ax[i].set_ylabel('% of Participants', fontproperties=prop, fontsize=15, labelpad=10)
    for label in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
        label.set_fontproperties(prop)
    ax[i].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax[i].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[i].tick_params(axis='both', which='major', labelsize=10)

fig.text(0.5, 0.02, '% of Choosing the Optimal Option During Training', ha='center', fontproperties=prop, fontsize=15)
sns.despine()
plt.savefig('./figures/HighLowDirichlet.png', dpi=1000)
plt.show()


