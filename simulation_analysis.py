import numpy as np
import pandas as pd
import os
import ast
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from utilities.utility_PlottingFunctions import visualization_3D, prop
from scipy.stats import f_oneway, ttest_1samp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from scipy.interpolate import griddata

# after the simulation has been completed, we can just load the simulated data from the folder
dual = pd.read_csv('./data/Simulation/random_dual.csv')
delta = pd.read_csv('./data/Simulation/random_delta.csv')
decay = pd.read_csv('./data/Simulation/random_decay.csv')
actr = pd.read_csv('./data/Simulation/random_actr.csv')

# Generate a summary document for the simulation
sim_summary = []

for i, data in enumerate([dual, delta, decay, actr]):
    # Remove the simulation number column and take the mean of the rest
    data = data.drop(columns='simulation_num')
    data.loc[:, 'proportion'] = (data['choice'] < data['reward_ratio']).astype(int)
    summary = data.groupby(['diff', 'var']).mean().reset_index()
    summary.loc[:, 'model'] = ['dual', 'delta', 'decay', 'actr'][i]
    # print(f'Minimum proportion of frequency effects for {summary["model"].iloc[0]}: {summary["proportion"].min()}')
    # print(f'Maximum proportion of frequency effects for {summary["model"].iloc[0]}: {summary["proportion"].max()}')
    # print(f'Minimum proportion of C choices for {summary["model"].iloc[0]}: {summary["choice"].min()}')
    # print(f'Maximum proportion of C choices for {summary["model"].iloc[0]}: {summary["choice"].max()}')
    sim_summary.append(summary)

sim_summary_df = pd.concat(sim_summary)
sim_summary_df.to_csv('./data/Simulation/sim_summary.csv', index=False)

# Generate a 3D visualization of the simulation results
visualization_3D(sim_summary, plot_type='surface', z_label='% of Frequency Effects')
visualization_3D(sim_summary, plot_type='contourf', elev=None, azim=None, z_label='% of Frequency Effects')

# # Generate a heatmap to show the relationship between the reward ratio, variance, and the objective weight of the
# # Dirichlet process
# x = sim_summary[0]['reward_ratio']
# y = sim_summary[0]['var']
# z = sim_summary[0]['obj_weight']
#
# xi = np.linspace(x.min(), x.max(), 100)
# yi = np.linspace(y.min(), y.max(), 100)
# zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
#
# fig, ax = plt.subplots()
# c = ax.pcolormesh(xi, yi, zi, cmap='Oranges')
# ax.set_xlabel('Reward Ratio', fontproperties=prop, fontsize=12)
# ax.set_ylabel('Variance', fontproperties=prop, fontsize=12)
# cbar = plt.colorbar(c, ax=ax)
# cbar.set_label('Objective Weight of the Dirichlet Process', fontproperties=prop, fontsize=12, labelpad=10)
# sns.despine()
# plt.savefig('./figures/heatmap.png', dpi=600)
# plt.show()
#
#
# Finally, generate a linear plot to show the relationship between the objective weight of the Dirichlet process and
# the proportion of frequency effects
# sns.set(style='white')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# sns.regplot(data=sim_summary[0], x='obj_weight', y='proportion', ax=ax, x_ci='ci', ci=95, color='darkorange',
#             scatter_kws={'s': 10, 'alpha': 0.3})
# ax.set_xlabel('Objective Weight of the Dirichlet Process', fontproperties=prop, fontsize=12)
# ax.set_ylabel('% of Frequency Effects', fontproperties=prop, fontsize=12)
# sns.despine()
# plt.savefig('./figures/obj_weight_vs_proportion.png', dpi=600)
# plt.show()

