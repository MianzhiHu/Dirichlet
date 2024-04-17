import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import dirichlet, multivariate_normal
import matplotlib.tri as mtri
from matplotlib import cm
from utilities.utility_PlottingFunctions import scatter_Dirichlet, bar_Dirichlet, scatter_Gaussian, bar_Gaussian


# plot illustrations of Dirichlet distribution
alpha = [2, 1, 1]
scatter_Dirichlet(alpha)
bar_Dirichlet(alpha, option="BC", resolution=50)

# do the same for multivariate Gaussian distribution
b_history = [0.5, 0.25]
mean_b = np.mean(b_history)
var_b = np.var(b_history)
mean = [0.3, mean_b, 0.7]
var = [0.01, var_b, 0.01]

mean = [0.3, 0.5, 0.7]
var = [0.01, 0.01, 0.01]
cov = np.diag(var)

scatter_Gaussian(mean, cov)

# Example usage
bar_Gaussian(mean, cov, option="AB", resolution=50)

