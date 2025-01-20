import ast
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utilities.utility_PlottingFunctions import visualization_3D, prop, crop_colormap, visualization_3D_prop

# after the simulation has been completed, we can just load the simulated data from the folder
dual = pd.read_csv('./data/Simulation/random_dual.csv')
delta = pd.read_csv('./data/Simulation/random_delta.csv')
delta_asym = pd.read_csv('./data/Simulation/random_delta_asym.csv')
decay = pd.read_csv('./data/Simulation/random_decay.csv')
utility = pd.read_csv('./data/Simulation/random_utility.csv')
actr = pd.read_csv('./data/Simulation/random_actr.csv')

palette3 = sns.color_palette("pastel", 3)

# Generate a summary document for the simulation
sim_summary = []

for i, data in enumerate([dual, delta, delta_asym, utility, decay, actr]):
    # Remove the simulation number column and take the mean of the rest
    data = data.drop(columns='simulation_num')
    data.loc[:, 'proportion'] = (data['choice'] < data['reward_ratio']).astype(int)
    summary = data.groupby(['diff', 'var']).mean().reset_index()
    summary.loc[:, 'model'] = ['dual', 'delta', 'delta_asym', 'utility', 'decay', 'actr'][i]
    # print(f'Minimum proportion of frequency effects for {summary["model"].iloc[0]}: {summary["proportion"].min()}')
    # print(f'Maximum proportion of frequency effects for {summary["model"].iloc[0]}: {summary["proportion"].max()}')
    # print(f'Minimum proportion of C choices for {summary["model"].iloc[0]}: {summary["choice"].min()}')
    # print(f'Maximum proportion of C choices for {summary["model"].iloc[0]}: {summary["choice"].max()}')
    sim_summary.append(summary)

sim_summary_df = pd.concat(sim_summary)
sim_summary_df.to_csv('./data/sim_summary.csv', index=False)

# Generate a 3D visualization of the simulation results
visualization_3D(sim_summary, plot_type='surface')
visualization_3D_prop(sim_summary, plot_type='surface')

# Generate a heatmap to show the relationship between the reward ratio, variance, and the objective weight of the
# Dirichlet process
x = sim_summary[0]['reward_ratio']
y = sim_summary[0]['var']
z = sim_summary[0]['obj_weight']

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')

fig, ax = plt.subplots()
c = ax.pcolormesh(xi, yi, zi, cmap='Oranges')
ax.set_xlabel('Reward Ratio', fontproperties=prop, fontsize=20)
ax.set_ylabel('Variance', fontproperties=prop, fontsize=20)
plt.xticks(fontproperties=prop, fontsize=15)
plt.yticks(fontproperties=prop, fontsize=15)
cbar = plt.colorbar(c, ax=ax)
cbar.set_label('Objective Weight of the Dirichlet Process', fontproperties=prop, fontsize=15, labelpad=10)
plt.tight_layout()
sns.despine()
plt.savefig('./figures/heatmap.png', dpi=600)
plt.show()


# Finally, generate a linear plot to show the relationship between the objective weight of the Dirichlet process and
# the proportion of frequency effects
sns.set(style='white')
fig = plt.figure()
ax = fig.add_subplot(111)
sns.regplot(data=sim_summary[0], x='obj_weight', y='proportion', ax=ax, x_ci='ci', ci=95, color='darkorange',
            scatter_kws={'s': 10, 'alpha': 0.3})
ax.set_xlabel('Objective Weight of the Dirichlet Process', fontproperties=prop, fontsize=20)
ax.set_ylabel('% of Frequency Effects', fontproperties=prop, fontsize=20)
plt.xticks(fontproperties=prop, fontsize=15)
plt.yticks(fontproperties=prop, fontsize=15)
sns.despine()
plt.tight_layout()
plt.savefig('./figures/obj_weight_vs_proportion.png', dpi=600)
plt.show()

# ======================================================================================================================
# Load traditional simulation data
# ======================================================================================================================
# Load the data
folder_path = './data/Simulation/Traditional Simulations/'
best_option_mappping = {
    'AB': 'A',
    'CD': 'C',
    'CA': 'C',
    'BD': 'B',
    'AD': 'A',
    'CB': 'C'
}
all_tra_sim = {}

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        all_tra_sim[file] = pd.read_csv(file_path)
        all_tra_sim[file].loc[:, 'Model'] = os.path.splitext(file)[0].split('_')[0]
        all_tra_sim[file].loc[:, 'Condition'] = os.path.splitext(file)[0].split('_')[1]
        all_tra_sim[file].loc[:, 'pair'] = all_tra_sim[file]['pair'].apply(ast.literal_eval)
        all_tra_sim[file].loc[:, 'pair'] = all_tra_sim[file]['pair'].apply(lambda x: ''.join(x))
        all_tra_sim[file].loc[:, 'mapping'] = all_tra_sim[file]['pair'].apply(lambda x: best_option_mappping[x])
        all_tra_sim[file].loc[:, 'bestoption'] = (all_tra_sim[file]['choice'] == all_tra_sim[file]['mapping']).astype(int)

# combine the data
all_tra_sim_df = pd.concat(all_tra_sim.values())
print(all_tra_sim_df['Model'].unique())

# generate summary
dual_sim = all_tra_sim_df[all_tra_sim_df['Model'] == 'dual']
dual_summary = dual_sim.groupby(['Condition', 'pair'])['bestoption'].mean().reset_index()

# plot
sns.set(style='white')
fig = plt.figure()
ax = fig.add_subplot(111)
sns.barplot(data=dual_summary, x='Condition', y='bestoption', hue='pair', ax=ax,
            order=['lv', 'mv', 'hv'], hue_order=['AB', 'CD', 'CA', 'BD', 'AD', 'CB'])
sns.lineplot(data=dual_summary, x='Condition', y='bestoption', ax=ax, color=sns.color_palette("dark")[8], markers='o',
             markersize=5, linewidth=2, errorbar=None)
plt.axhline(y=0.5, color='black', linestyle='--', label='Random Choice')
plt.axhline(y=0.75/1.4, color='black', linestyle='-', label='Reward Ratio')
ax.set_xlabel('Pair', fontproperties=prop, fontsize=20)
ax.set_ylabel('Proportion of Best Option Choices', fontproperties=prop, fontsize=20)
plt.xticks(fontproperties=prop, fontsize=15, ticks=[0, 1, 2], labels=['LV', 'MV', 'HV'])
plt.yticks(fontproperties=prop, fontsize=15)
legend = plt.legend(title='Trial Type', prop=prop, framealpha=0.5)
legend.get_title().set_fontproperties(prop)
sns.despine()
plt.tight_layout()
plt.savefig('./figures/best_option_choices.png', dpi=600)
plt.show()

# plot for all models for CA pair
CA = all_tra_sim_df[all_tra_sim_df['pair'] == 'CA']
CA_summary = CA.groupby(['Condition', 'Model'])['bestoption'].mean().reset_index()
CA_summary.loc[:, 'Condition'] = CA_summary['Condition'].apply(lambda x: x.upper())

sns.set(style='white')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
sns.barplot(data=CA_summary, x='Model', y='bestoption', hue='Condition', ax=ax,
            order=['dual', 'delta', 'deltaasym', 'utility', 'decay', 'actr'], hue_order=['LV', 'MV', 'HV'])
plt.axhline(y=0.5, color='black', linestyle='--', label='Random Choice')
plt.axhline(y=0.75/1.4, color='black', linestyle='-', label='Reward Ratio')
ax.set_xlabel('')
ax.set_ylabel('Proportion of C Choices in CA trials', fontproperties=prop, fontsize=20)
plt.xticks(fontproperties=prop, fontsize=15, ticks=[0, 1, 2, 3, 4, 5],
           labels=['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility', 'Decay', 'ACT-R'],
            rotation=90)
plt.yticks(fontproperties=prop, fontsize=15)
legend = plt.legend(title='Condition', prop=prop, framealpha=0.5, loc='lower left')
legend.get_title().set_fontproperties(prop)
sns.despine()
plt.tight_layout()
plt.savefig('./figures/CA_sim_all_models.png', dpi=600)
plt.show()
