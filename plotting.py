import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import dirichlet, multivariate_normal
import matplotlib.tri as mtri
from matplotlib import cm
from utilities.utility_PlottingFunctions import scatter_Dirichlet, bar_Dirichlet, scatter_Gaussian, bar_Gaussian


# plot illustrations of Dirichlet distribution
alpha = [1, 2, 1]
# scatter_Dirichlet(alpha)
# bar_Dirichlet(alpha, resolution=500, elev=45, azim=60)

# do the same for multivariate Gaussian distribution
b_history = [0.5, 0.0]
mean_b = np.mean(b_history)
var_b = np.var(b_history)
mean = [0.5, mean_b, 0.5]
var = [0.01, var_b, 0.01]

mean = [0.5, 0.25, 0.5]
var = [0.01, 0.01, 0.01]
cov = np.diag(var)

scatter_Gaussian(mean, cov)
bar_Gaussian(mean, cov, resolution=500, elev=45, azim=60)

# plot for empirical data
data = pd.read_csv("./data/CombinedVarianceData.csv")
data_CA = data[data['TrialType'] == 'CA']

condition_list = data_CA.groupby('Subnum')['Condition'].first().reset_index()
process_chosen = data_CA.groupby('Subnum')['best_process_chosen'].mean().reset_index()
prop_optimal = data_CA.groupby('Subnum')['bestOption'].mean().reset_index()
# merge all three dataframes
df = pd.merge(condition_list, process_chosen, on='Subnum')
df = pd.merge(df, prop_optimal, on='Subnum')

# test the correlation between the proportion of choosing the best option and the proportion of choosing the dual process

# Create a scatter plot
scatter = plt.scatter(df['best_process_chosen'], df['bestOption'], c=df['Condition'].map({'LV': 'g', 'MV': 'b', 'HV': 'r'}), alpha=0.4)

# Create a list of Line2D objects to represent the legend entries
colors = ['g', 'b', 'r']
labels = ['LV', 'MV', 'HV']
lines = [mlines.Line2D([], [], color=c, marker='o', markersize=10, label=l, linestyle='None') for c, l in zip(colors, labels)]
plt.legend(handles=lines)
# add linear regression
m, b = np.polyfit(df['best_process_chosen'], df['bestOption'], 1)
plt.plot(df['best_process_chosen'], m * df['best_process_chosen'] + b, color='gray')
plt.xlabel('Proportion of Dirichlet-Based Choices')
plt.ylabel('Proportion of Choosing Best Option')
plt.title('Correlation Between Process Chosen and Best Option Chosen')
plt.show()