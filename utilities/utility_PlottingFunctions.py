import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import dirichlet, multivariate_normal
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib import font_manager as fm
from sklearn.linear_model import LinearRegression

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


def scatter_Dirichlet_2D(alpha, num_samples=5000):
    # Sample from a Dirichlet distribution
    samples = dirichlet.rvs(alpha, num_samples)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.grid(False)  # Hide gridlines\

    # Scatter plot of samples with colormap
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=5)

    # Labels and title with a custom font
    ax.set_xlabel('Option A', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)
    ax.set_ylabel('Option B', fontweight='bold', fontsize=20, labelpad=15, fontproperties=prop)

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
def visualization_3D(sim_summary, x_var='reward_ratio', y_var='var', z_var='choice',
                     x_label='Reward Ratio', y_label='Variance', z_label='% of C choices',
                     plot_type='surface', cmap='coolwarm', color='skyblue', elev=20, azim=-135, title=True):
    fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(16, 8))
    axs = axs.flatten()
    cmap = plt.get_cmap(cmap)
    cmap_x = 'OrRd'
    cmap_y = 'PuBu'

    fig.subplots_adjust(hspace=0.25, wspace=-0.1)

    global_z_diff_min = float('inf')
    global_z_diff_max = float('-inf')
    for data in sim_summary:
        x = data[x_var]
        z = data[z_var]
        z_diff = z - x
        global_z_diff_min = min(global_z_diff_min, z_diff.min())
        global_z_diff_max = max(global_z_diff_max, z_diff.max())

    print(f'min: {global_z_diff_min}, max: {global_z_diff_max}')

    norm = TwoSlopeNorm(
        vmin=global_z_diff_min,
        vcenter=0,
        vmax=global_z_diff_max
    )

    # Plot each dataset in its own subplot
    for i in range(len(sim_summary)):
        x = sim_summary[i][x_var]
        y = sim_summary[i][y_var]
        z = sim_summary[i][z_var]
        z_diff = z - x
        grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        grid_z_diff = griddata((x, y), z_diff, (grid_x, grid_y), method='linear')  # For colormap
        facecolors = cmap(norm(grid_z_diff))

        if plot_type == 'surface':
            axs[i].plot_surface(grid_x, grid_y, grid_z, facecolors=facecolors, shade=False, rstride=1, cstride=1,
                                alpha=0.99)
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
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])  # so colorbar knows the data range
        cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='vertical')
        cbar.set_label('% of Choosing C - Reward Ratio', fontproperties=prop, fontsize=15, labelpad=15)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontproperties(prop)

    plt.savefig(f'./figures/simulation_{plot_type}.png', dpi=1000)
    plt.show(dpi=1000)


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
    plt.show(dpi=1000)


# Function to plot planes with filled colors
def plot_planes(ax, fixed_condition, x_range, z_range, color, alpha=0.3):
    """Create a plane at a fixed condition (y-value) that covers the entire x and z ranges."""
    X_plane, Z_plane = np.meshgrid(np.linspace(x_range[0], x_range[1], 10),
                                   np.linspace(z_range[0], z_range[1], 10))
    Y_plane = np.full_like(X_plane, fixed_condition)

    ax.plot_surface(X_plane, Y_plane, Z_plane, color=color, alpha=alpha, rstride=1000, cstride=1000, edgecolors='none')


def three_planes(summary, var, x_label='Dirichlet Weight', name='Weight_Optimal'):
    # Create a 3D plot to show the relationship between weight, best_option, and condition
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set conditions and colors
    conditions = summary['Condition'].unique()
    condition_names = ['HV', 'MV', 'LV']
    color = sns.color_palette('deep', len(conditions))

    # Get the min and max of the x (Dirichlet Weight) and z (% of Choosing C) axes
    x_min, x_max = summary[var].min(), summary[var].max()
    z_min, z_max = summary['bestOption'].min(), summary['bestOption'].max()

    # Loop through each condition, fit a linear regression, and plot the line
    for condition in conditions:
        subset = summary[summary['Condition'] == condition]

        # Extract the variables for regression
        X = subset[[var]].values  # Reshaped to fit the model
        y = subset['bestOption'].values

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Generate the regression line
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = model.predict(x_vals.reshape(-1, 1))

        # Plot the original data points
        condition_index = 3 - condition
        ax.scatter(subset[var], subset['Condition'], subset['bestOption'], marker='o',
                   color=color[condition_index], alpha=0.4)

        # Plot the regression line
        ax.plot(x_vals, [condition] * len(x_vals), y_vals, label=f'{condition_names[condition_index]}',
                color=color[condition_index])

        # Plot a plane at each condition
        plot_planes(ax, condition, (x_min, x_max), (z_min, z_max), color=color[condition_index], alpha=0.12)

    # Set the labels and title
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['LV', 'MV', 'HV'], fontproperties=prop)
    ax.set_box_aspect([1, 2, 1])
    ax.set_ylim(1, 3)
    ax.set_xlabel(x_label, fontproperties=prop)
    ax.set_ylabel('Condition', fontproperties=prop, labelpad=10)
    ax.set_zlabel('% of Choosing C in CA Trials', fontproperties=prop)
    # set font for tick labels
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontproperties(prop)
    ax.view_init(azim=-55)
    plt.legend(title='Condition', prop=prop, title_fontproperties=prop, loc='upper left')
    plt.savefig(f'./figures/{name}.png', dpi=1000)
    plt.show()

