import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import dirichlet, multivariate_normal
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib import font_manager as fm


# load the font
font_path = 'utilities/AbhayaLibre-ExtraBold.ttf'
prop = fm.FontProperties(fname=font_path)


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
    ax.set_xlabel('Option A', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_zlabel('Option C', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)

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
    ax.set_xlabel('Option A/Option C', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_zlabel('PDF', fontweight='bold', fontsize=20, rotation=-90, fontproperties=prop)

    # Set up the colorbar explicitly with the 'ax' argument
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    mappable = cm.ScalarMappable(cmap=colormap)
    mappable.set_array(pdf_values)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('PDF Value', fontweight='bold', fontsize=20, rotation=-90, labelpad=15, fontproperties=prop)

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
    ax.set_xlabel('Option A', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_zlabel('Option C', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)

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
    ax.set_xlabel('Option A/Option C', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_zlabel('PDF', fontweight='bold', fontsize=20, rotation=-90, labelpad=15, fontproperties=prop)


    # Set up the colorbar explicitly with the 'ax' argument
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    mappable = cm.ScalarMappable(cmap=colormap)
    mappable.set_array(pdf_values)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('PDF Value', fontweight='bold', fontsize=20, rotation=-90, labelpad=15, fontproperties=prop)

    # Adjust the viewing angle for better perception
    ax.view_init(elev=elev, azim=azim)

    # Show the plot
    plt.show()


# Function to crop colormap
def crop_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'cropped',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


# Generate a 3D visualization of the simulation results
def visualization_3D(sim_summary, x_var='reward_ratio', y_var='var', z_var='proportion',
                     x_label='Reward Ratio', y_label='Reward Variance', z_label='Proportion of Frequency Effects',
                     minval=0.09, maxval=0.81, plot_type='surface', cmap='coolwarm', color='skyblue', elev=20,
                     azim=-135, title=True):

    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 8))
    axs = axs.flatten()
    cmap = plt.get_cmap(cmap)
    cmap_x = 'OrRd'
    cmap_y = 'PuBu'

    cropped_cmap = crop_colormap(cmap, minval=minval, maxval=maxval)
    norm = Normalize(vmin=minval, vmax=maxval)

    fig.subplots_adjust(hspace=0.25, wspace=-0.1)

    # Plot each dataset in its own subplot
    for i in range(len(sim_summary)):
        x = sim_summary[i][x_var]
        y = sim_summary[i][y_var]
        z = sim_summary[i][z_var]
        grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        if plot_type == 'surface':
            axs[i].plot_surface(grid_x, grid_y, grid_z, cmap=cropped_cmap, alpha=0.99)
        elif plot_type == 'wireframe':
            axs[i].plot_wireframe(grid_x, grid_y, grid_z, color=color)
        elif plot_type == 'contour':
            axs[i].contour(grid_x, grid_y, grid_z, zdir='x', offset=x.min(), cmap=cropped_cmap, norm=norm)
            axs[i].contour(grid_x, grid_y, grid_z, zdir='y', offset=y.max(), cmap=cropped_cmap, norm=norm)
        elif plot_type == 'contourf':
            axs[i].contourf(grid_x, grid_y, grid_z, zdir='x', offset=x.min(), cmap=cmap_y)
            axs[i].contourf(grid_x, grid_y, grid_z, zdir='y', offset=y.max(), cmap=cmap_x)

        if title:
            axs[i].set_title(['Dual-Process', 'Delta', 'Decay', 'ACT-R'][i], fontproperties=prop, fontsize=20, pad=5)

        axs[i].set_xlabel(x_label, fontproperties=prop)
        axs[i].set_ylabel(y_label, fontproperties=prop)
        axs[i].set_zlabel(z_label, fontproperties=prop)
        axs[i].set_zlim(0, 1)

        # Set elevation and azimuth angles
        if elev is not None or azim is not None:
            axs[i].view_init(elev=elev if elev is not None else axs[i].elev,
                             azim=azim if azim is not None else axs[i].azim)

    if plot_type == 'surface':
        # Create a single colorbar for all subplots
        cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cropped_cmap), cax=cbar_ax, orientation='vertical')

    plt.savefig(f'./figures/simulation_{plot_type}.png', dpi=600)
    plt.show(dpi=600)
