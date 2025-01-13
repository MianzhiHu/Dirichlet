import ast
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm

from model_fitting_analysis import labels
from utilities.utility_PlottingFunctions import visualization_3D, prop, crop_colormap, visualization_3D_prop
from scipy.interpolate import griddata

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
sim_summary_df.to_csv('./data/Simulation/sim_summary.csv', index=False)

def visualization_3D_prop(sim_summary, x_var='reward_ratio', y_var='var', z_var='proportion',
                     x_label='Reward Ratio', y_label='Variance', z_label='% of Frequency Effects',
                     plot_type='surface', cmap='coolwarm', color='skyblue', elev=20, azim=-135, title=True):
    fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(16, 8))
    axs = axs.flatten()
    cmap = plt.get_cmap(cmap)
    cmap_x = 'OrRd'
    cmap_y = 'PuBu'

    fig.subplots_adjust(hspace=0.25, wspace=-0.1)

    max_z = float('-inf')
    min_z = float('inf')
    for data in sim_summary:
        z = data[z_var]
        max_z = max(max_z, z.max())
        min_z = min(min_z, z.min())

    print(f'min: {min_z}, max: {max_z}')

    norm = Normalize(vmin=min_z, vmax=max_z)

    # Plot each dataset in its own subplot
    for i in range(len(sim_summary)):
        x = sim_summary[i][x_var]
        y = sim_summary[i][y_var]
        z = sim_summary[i][z_var]
        grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

        if plot_type == 'surface':
            axs[i].plot_surface(grid_x, grid_y, grid_z, cmap=cmap, norm=norm, alpha=0.99, rstride=1, cstride=1)
        elif plot_type == 'wireframe':
            axs[i].plot_wireframe(grid_x, grid_y, grid_z, color=color)
        elif plot_type == 'contour':
            axs[i].contour(grid_x, grid_y, grid_z, zdir='x', offset=x.min(), cmap=cmap, norm=norm)
            axs[i].contour(grid_x, grid_y, grid_z, zdir='y', offset=y.max(), cmap=cmap, norm=norm)
        elif plot_type == 'contourf':
            axs[i].contourf(grid_x, grid_y, grid_z, zdir='x', offset=x.min(), cmap=cmap_y)
            axs[i].contourf(grid_x, grid_y, grid_z, zdir='y', offset=y.max(), cmap=cmap_x)

        if title:
            axs[i].set_title(['Dual-Process', 'Delta', 'Risk-Sensitive Delta', 'Mean-Variance Utility',
                              'Decay', 'ACT-R'][i], fontproperties=prop, fontsize=25, pad=5)

        axs[i].set_xlabel(x_label, fontproperties=prop, fontsize=15)
        axs[i].set_ylabel(y_label, fontproperties=prop, fontsize=15)
        axs[i].set_zlabel(z_label, fontproperties=prop, fontsize=15)
        # set font for tick labels
        for label in (axs[i].get_xticklabels() + axs[i].get_yticklabels() + axs[i].get_zticklabels()):
            label.set_fontproperties(prop)
        axs[i].set_zlim(0, 1)

        # Set elevation and azimuth angles
        if elev is not None or azim is not None:
            axs[i].view_init(elev=elev if elev is not None else axs[i].elev,
                             azim=azim if azim is not None else axs[i].azim)

    if plot_type == 'surface':
        # Create a single colorbar for all subplots
        cbar_ax = fig.add_axes([0.93, 0.25, 0.01, 0.5])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical')
        cbar.set_label(z_label, fontproperties=prop, fontsize=15, labelpad=15)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontproperties(prop)

    plt.savefig(f'./figures/simulation_{plot_type}_percentage.png', dpi=1000)
    plt.show(dpi=600)


# Generate a 3D visualization of the simulation results
# visualization_3D(sim_summary, plot_type='surface')
visualization_3D_prop(sim_summary, plot_type='surface')
# visualization_3D(sim_summary, plot_type='contourf', elev=None, azim=None, z_label='% of Frequency Effects', z_var='choice')

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
# ax.set_xlabel('Reward Ratio', fontproperties=prop, fontsize=20)
# ax.set_ylabel('Variance', fontproperties=prop, fontsize=20)
# plt.xticks(fontproperties=prop, fontsize=15)
# plt.yticks(fontproperties=prop, fontsize=15)
# cbar = plt.colorbar(c, ax=ax)
# cbar.set_label('Objective Weight of the Dirichlet Process', fontproperties=prop, fontsize=15, labelpad=10)
# plt.tight_layout()
# sns.despine()
# plt.savefig('./figures/heatmap.png', dpi=600)
# plt.show()
#
#
# # Finally, generate a linear plot to show the relationship between the objective weight of the Dirichlet process and
# # the proportion of frequency effects
# sns.set(style='white')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# sns.regplot(data=sim_summary[0], x='obj_weight', y='proportion', ax=ax, x_ci='ci', ci=95, color='darkorange',
#             scatter_kws={'s': 10, 'alpha': 0.3})
# ax.set_xlabel('Objective Weight of the Dirichlet Process', fontproperties=prop, fontsize=20)
# ax.set_ylabel('% of Frequency Effects', fontproperties=prop, fontsize=20)
# plt.xticks(fontproperties=prop, fontsize=15)
# plt.yticks(fontproperties=prop, fontsize=15)
# sns.despine()
# plt.tight_layout()
# plt.savefig('./figures/obj_weight_vs_proportion.png', dpi=600)
# plt.show()

# # ======================================================================================================================
# # Load traditional simulation data
# # ======================================================================================================================
# # Load the data
# folder_path = './data/Simulation/Traditional Simulations/'
# best_option_mappping = {
#     'AB': 'A',
#     'CD': 'C',
#     'CA': 'C',
#     'BD': 'B',
#     'AD': 'A',
#     'CB': 'C'
# }
# dual_sim = {}
#
# for file in os.listdir(folder_path):
#     if 'dual' in file:
#         file_path = os.path.join(folder_path, file)
#         dual_sim[file] = pd.read_csv(file_path)
#         dual_sim[file].loc[:, 'Condition'] = os.path.splitext(file)[0].split('_')[1]
#         dual_sim[file].loc[:, 'pair'] = dual_sim[file]['pair'].apply(ast.literal_eval)
#         dual_sim[file].loc[:, 'pair'] = dual_sim[file]['pair'].apply(lambda x: ''.join(x))
#         dual_sim[file].loc[:, 'mapping'] = dual_sim[file]['pair'].apply(lambda x: best_option_mappping[x])
#         dual_sim[file].loc[:, 'bestoption'] = (dual_sim[file]['choice'] == dual_sim[file]['mapping']).astype(int)
#
# # combine the data
# dual_sim_df = pd.concat(dual_sim.values())
#
# # generate summary
# dual_summary = dual_sim_df.groupby(['Condition', 'pair'])['bestoption'].mean().reset_index()
#
# # plot
# sns.set(style='white')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# sns.barplot(data=dual_summary, x='Condition', y='bestoption', hue='pair', ax=ax, palette=sns.color_palette('pastel')[:6],
#             order=['lv', 'mv', 'hv'], hue_order=['AB', 'CD', 'CA', 'BD', 'AD', 'CB'])
# sns.lineplot(data=dual_summary, x='Condition', y='bestoption', ax=ax, color=sns.color_palette("dark")[8], markers='o',
#              markersize=5, linewidth=2, errorbar=None)
# plt.axhline(y=0.5, color='black', linestyle='--', label='Random Choice')
# plt.axhline(y=0.75/1.4, color='black', linestyle='-', label='Reward Ratio')
# ax.set_xlabel('Pair', fontproperties=prop, fontsize=20)
# ax.set_ylabel('Proportion of Best Option Choices', fontproperties=prop, fontsize=20)
# plt.xticks(fontproperties=prop, fontsize=15, ticks=[0, 1, 2], labels=['LV', 'MV', 'HV'])
# plt.yticks(fontproperties=prop, fontsize=15)
# legend = plt.legend(title='Trial Type', prop=prop, framealpha=0.5)
# legend.get_title().set_fontproperties(prop)
# sns.despine()
# plt.tight_layout()
# plt.savefig('./figures/best_option_choices.png', dpi=600)
# plt.show()

