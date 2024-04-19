import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import dirichlet, multivariate_normal
import matplotlib.tri as mtri
from matplotlib import cm


def scatter_Dirichlet(alpha, num_samples=5000):
    # Sample from a Dirichlet distribution
    samples = dirichlet.rvs(alpha, num_samples)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)  # Hide gridlines
    ax.xaxis.pane.fill = False  # Hide the spines
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Scatter plot of samples with colormap
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, s=5)

    # Labels and title with a custom font
    ax.set_xlabel('Option A', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_zlabel('Option C', fontweight='bold', fontsize=12, labelpad=15)

    # Viewing angle
    ax.view_init(elev=30, azim=45)

    # Show the plot
    plt.show()


def bar_Dirichlet(alpha, resolution=50, elev=30, azim=135):
    # Reduce resolution to make the 3D bar plot manageable
    x = np.linspace(0, 1, resolution)
    X1, X2 = np.meshgrid(x, x)
    X1, X2 = X1.flatten(), X2.flatten()
    X3 = 1 - X1 - X2

    # Filter out the invalid points
    valid = X3 >= 0
    X1, X2, X3 = X1[valid], X2[valid], X3[valid]
    points = np.vstack((X1, X2, X3)).T

    # Compute the PDF values
    pdf_values = dirichlet(alpha).pdf(points.T)

    # Normalize the PDF values to use in color mapping
    pdf_norm = pdf_values / pdf_values.max()

    # Create a colormap
    colormap = cm.viridis

    # Map normalized PDF values to colors
    colors = colormap(pdf_norm)

    # Create a 3D plot with increased size for better visibility
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)  # Turn off the grid
    ax.xaxis.pane.fill = False  # Turn off the pane filling
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # The width of the bars
    dx = dy = 1 / resolution

    # Create a 3D bar plot with color
    ax.bar3d(X1, X2, np.zeros_like(X1), dx, dy, pdf_values, color=colors, shade=True)

    # Labels and title with custom fonts
    ax.set_xlabel('Option A/Option C', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_zlabel('PDF', fontweight='bold', fontsize=12, rotation=-90)

    # Set up the colorbar explicitly with the 'ax' argument
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    mappable = cm.ScalarMappable(cmap=colormap)
    mappable.set_array(pdf_values)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('PDF Value', fontweight='bold', fontsize=12, rotation=-90, labelpad=15)

    # Adjust the viewing angle for better perception
    ax.view_init(elev=elev, azim=azim)

    # Show the plot
    plt.show()


def scatter_Gaussian(mean, cov, num_samples=5000):
    # Sample from a multivariate Gaussian distribution
    samples = multivariate_normal.rvs(mean, cov, num_samples)

    # filter out the invalid points
    valid = np.all(samples >= 0, axis=1) & np.all(samples <= 1, axis=1)
    samples = samples[valid]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)  # Hide gridlines
    ax.xaxis.pane.fill = False  # Hide the spines
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Scatter plot of samples
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, s=5)

    # set the axis limits
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_zlim(0, 0.8)

    # Labels and title with a custom font
    ax.set_xlabel('Option A', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_zlabel('Option C', fontweight='bold', fontsize=12, labelpad=15)

    # Viewing angle
    ax.view_init(elev=30, azim=45)

    # Show the plot
    plt.show()


def bar_Gaussian(mean, cov, resolution=50, elev=30, azim=135):
    # Create a grid of points in 2D space
    x = np.linspace(0, 1, resolution)
    X1, X2 = np.meshgrid(x, x)
    X1, X2 = X1.flatten(), X2.flatten()
    X3 = 1 - X1 - X2

    # Filter out the invalid points
    valid = X3 >= 0
    X1, X2, X3 = X1[valid], X2[valid], X3[valid]
    points = np.vstack((X1, X2, X3)).T
    # Compute the PDF values
    pdf_values = multivariate_normal(mean, cov).pdf(points)

    # Normalize the PDF values to use in color mapping
    pdf_norm = pdf_values / pdf_values.max()

    # Create a colormap
    colormap = cm.viridis

    # Map normalized PDF values to colors
    colors = colormap(pdf_norm)

    # Create a 3D plot with increased size for better visibility
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)  # Turn off the grid
    ax.xaxis.pane.fill = False  # Turn off the pane filling
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # The width of the bars
    dx = dy = 1 / resolution  # Width of bars based on the range used above

    # Create a 3D bar plot with color
    ax.bar3d(X1, X2, np.zeros_like(X1), dx, dy, pdf_values, color=colors, shade=True)

    # Labels and title with custom fonts
    ax.set_xlabel('Option A/Option C', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=12, labelpad=15)
    ax.set_zlabel('PDF', fontweight='bold', fontsize=12, rotation=-90, labelpad=15)


    # Set up the colorbar explicitly with the 'ax' argument
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    mappable = cm.ScalarMappable(cmap=colormap)
    mappable.set_array(pdf_values)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('PDF Value', fontweight='bold', fontsize=12, rotation=-90, labelpad=15)

    # Adjust the viewing angle for better perception
    ax.view_init(elev=elev, azim=azim)

    # Show the plot
    plt.show()
