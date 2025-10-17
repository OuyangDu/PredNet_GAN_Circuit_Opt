import matplotlib.pyplot as plt
from typing import Optional, Tuple
from border_ownership.ploter import error_bar_plot
import numpy as np
import matplotlib.ticker as mticker
from kitti_settings import *

def format_axis(ax, title, xlabel='Relative Time', ylabel='Response Difference (a.u.)', is_first_ax=False, is_last_row=False, sci_ytick=True, title_size=15):
    """
    Formats the given axis with appropriate labels, legends, and title.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes to format.
    - title (str): The title to set for the axes.
    - is_first_row (bool): True if the axes are in the first row.
    - is_last_row (bool): True if the axes are in the last row.
    """
    if sci_ytick:
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_size(10 if not is_last_row else 13)
        offset_text.set_verticalalignment('bottom')
    ax.set_title(title, fontsize=title_size)

    if is_first_ax:
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 1))
        ax.set_ylabel(ylabel)

    if is_last_row:
        ax.set_xlabel(xlabel)

def plot_all_modules(data, module_list, plot_keys, plot_colors, plot_labels, error_mode='se', mean_mode='mean', remove_outlier=False, fig_size=(10, 5), normalize=False, xlabel='Relative Time', ylabel='Response Difference (a.u.)'):
    """
    Plots data for all modules and a single specified module.
    ! bad function name

    Parameters:
    - data (dict): The data to plot.
    - module_list (list): List of modules to plot.
    - fig_size (tuple): The size of the figure for all modules.
    - single_fig_size (tuple): The size of the figure for a single module.
    """
    # Plot for all modules
    fig, axes = plt.subplots(2, 4, figsize=fig_size)
    for i, module in enumerate(module_list):
        ax = axes.flatten()[i]
        fig, ax = plot_res_diff_traces_module(data, module, plot_keys, plot_colors, plot_labels, error_mode=error_mode, remove_outlier=remove_outlier, ax=ax, fig=fig, normalize=normalize, mean_mode=mean_mode)
        n_unit = np.array(data[list(plot_keys)[0]][module]).shape[0]
        format_axis(ax, f'{module}: {n_unit}', is_first_ax=i==0, is_last_row=i>=len(axes.flatten())/2, xlabel=xlabel, ylabel=ylabel)

    fig.tight_layout()
    return fig, axes

def plot_one_module(data, module, plot_keys, plot_colors, plot_labels, error_mode='se', mean_mode='mean', remove_outlier=False, normalize=False, fig_size=(3, 3), fig_module=None, ax_module=None, do_format_axis=True, sci_ytick=True, xlabel='Relative Time', ylabel='Response Difference (a.u.)'):
    if fig_module is None or ax_module is None:
        fig_module, ax_module = plt.subplots(figsize=fig_size)
    fig_module, ax_module = plot_res_diff_traces_module(data, module, plot_keys, plot_colors, plot_labels, error_mode=error_mode, remove_outlier=remove_outlier, ax=ax_module, fig=fig_module, normalize=normalize, mean_mode=mean_mode)
    if do_format_axis:
        format_axis(ax_module, module, is_last_row=True, is_first_ax=True, sci_ytick=sci_ytick, ylabel='Relative and Rescaled \n Response Difference (a.u.)' if normalize else 'Relative Response Difference (a.u.)')
    return fig_module, ax_module

def plot_res_diff_traces_module(
    data: dict, 
    module: str, 
    plot_keys: list,
    plot_colors: list,
    plot_labels: list,
    fig: Optional[plt.Figure] = None, 
    ax: Optional[plt.Axes] = None, 
    error_mode: str = 'se', 
    remove_outlier: bool = False, 
    mean_mode: str = 'mean',
    normalize: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots traces for a specific module from the provided data.

    Parameters:
    - data (dict): The dataset containing the traces.
    - module (str): The specific module to plot.
    - fig (Optional[plt.Figure]): Matplotlib figure object, creates a new one if None.
    - ax (Optional[plt.Axes]): Matplotlib axes object, creates new axes if None.
    - error_mode (str): The mode of error representation in the plot. Default is 'se'.
    - remove_outlier (bool): Whether to remove outliers in the plot. Default is False.
    - mean_mode (str): The method used for computing the mean. Default is 'median'.
    - normalize (bool): Whether to normalize the data, Default is True.

    Returns:
    - Tuple[plt.Figure, plt.Axes]: The matplotlib figure and axes objects with the plot.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))

    time, switch_time = np.array(data['time']), data['switch_time']
    time = time[switch_time - 1:] - time[switch_time]

    for key, color, label in zip(plot_keys, plot_colors, plot_labels):
        res_diff = np.array(data[key][module])[:, switch_time - 1:]
        if normalize:
            # res_diff = np.abs(res_diff)
            res_diff = res_diff / np.abs(np.mean(res_diff, axis=0)).max()

        error_bar_plot(time, res_diff.T, fig=fig, ax=ax, color=color, label=label, error_band=True, remove_outlier=remove_outlier, error_mode=error_mode, mean_mode=mean_mode)

    ax.axvline(x=0, color='black', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--')

    return fig, ax
