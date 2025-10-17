import matplotlib.pyplot as plt
from itertools import cycle
from border_ownership.ploter import error_bar_plot, compute_y_and_ybound

def get_mse(data_dict, is_bo, module):
    '''
    Get the mse for a specific module and is_bo
    data_dict: dict, with keys in the form of f'{is_bo}_{module}_{n_units}'
    is_bo: bool, whether to use bo or not
    module: str, the module to get the mse for
    '''
    keys = [key for key in data_dict.keys() if key.startswith(f'{is_bo}_{module}_')]
    n_units = [int(key.split('_')[-1]) for key in keys]
    mse = [data_dict[key] for key in keys]

    # zipping, sorting by n_units, and unzipping
    n_units, mse = zip(*sorted(zip(n_units, mse)))

    return n_units, mse

def plot_module(data, module, fig, ax, color=None):
    if color is None:
        color = ['tab:blue', 'tab:orange']

    # Plots for is_bo=True, solid line
    n_units, mse = get_mse(data, True, module)
    error_bar_plot(n_units, mse, fig=fig, ax=ax, label=f'BO', error_mode='quantile', mean_mode='median', line_style='-', color=color[0])
    
    # Plots for is_bo=False, dashed line
    n_units, mse = get_mse(data, False, module)
    error_bar_plot(n_units, mse, fig=fig, ax=ax, label=f'Non-BO', error_mode='quantile', mean_mode='median', line_style='-', color=color[1])

def plot_modules(data, modules, fig=None, ax=None):
    colors = cycle(plt.cm.viridis(np.linspace(0, 1, len(modules))))  # Cycle through different colors

    for module in modules:
        plot_module(data, module, ax, next(colors))

    ax.legend()
