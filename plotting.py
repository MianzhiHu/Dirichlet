import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import dirichlet, multivariate_normal
import matplotlib.tri as mtri
from matplotlib import cm
from utilities.utility_PlottingFunctions import scatter_Dirichlet, bar_Dirichlet, scatter_Gaussian, bar_Gaussian


# plot illustrations of Dirichlet distribution
alpha = [2, 102, 1]
scatter_Dirichlet(alpha)
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
df['Condition'] = df['Condition'].astype('category')

# test the correlation between the proportion of choosing the best option and the proportion of choosing Dirichlet
sns.lmplot(data=df, x='best_process_chosen', y='bestOption', hue='Condition', hue_order=['LV', 'MV', 'HV'],
           palette='pastel', markers=['o', 's', 'D'], scatter_kws={'alpha': 0.7}, scatter=True, fit_reg=True)
plt.xlabel('Proportion of Dirichlet-Based Choices')
plt.ylabel('Proportion of Choosing C in CA')
plt.legend(title='Condition', loc='lower left')
plt.show(dp=600)
