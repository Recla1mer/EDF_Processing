import copy
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns
import bitsandbobs as bnb
import pickle

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    "color", bnb.plt.get_default_colors()
)
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 2  # padding between text and the tick
matplotlib.rcParams["ytick.major.pad"] = 2  # default 3.5
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 200
#matplotlib.rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)  # transparent figure bg
matplotlib.rcParams["axes.facecolor"] = (1.0, 0.0, 0.0, 0.0)


def seperate_plots_from_bib(file_name, save_path):
    """
    Read a pickle file containing a dictionary of plots and save them as individual figures.
    """
    with open(file_name, "rb") as f:
        plots = pickle.load(f)

    for key, value in plots.items():
        plt.figure()
        plt.imshow(value)
        plt.axis("off")
        plt.savefig(save_path + key + ".png", bbox_inches="tight", pad_inches=0)
        plt.close()


def seperate_plots(data, time, signal, save_path, **kwargs):
    """
    Save the plots in the dictionary as individual figures.
    """
    kwargs.setdefault("xlim", [2700,2720])
    
    fig, ax = plt.subplots()

    ax.plot(time, data, label=signal)
    ax.legend(loc="best")
    plt.xlim(kwargs["xlim"])
    plt.savefig(save_path + "_" + signal + ".png")


def simple_plot(data_y, data_x, save_path, **kwargs):
    """
    Create plot and save it.
    """
    fig, ax = plt.subplots()

    ax.plot(data_x, data_y)
    plt.savefig(save_path)


def plot_valid_regions(data: list, valid_regions: list, **kwargs):
    """
    Plot the valid regions of the ECG data.
    """
    # Set default values
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "ECG Data")
    kwargs.setdefault("x_label", "time (in iterations)")
    kwargs.setdefault("y_label", "uV")
    kwargs.setdefault("legend", ["Valid", "Invalid"])
    kwargs.setdefault("color", ["green", "red"])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("line_alpha", 1)
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("xlim", [0, len(data)])

    y_min = min(data[kwargs["xlim"][0]:kwargs["xlim"][1]])
    y_max = max(data[kwargs["xlim"][0]:kwargs["xlim"][1]])
    kwargs.setdefault("ylim", [y_min-abs(0.2*y_max), y_max+abs(0.2*y_max)])

    local_plot_kwargs = dict()
    local_plot_kwargs["linewidth"] = kwargs["linewidth"]
    local_plot_kwargs["alpha"] = kwargs["line_alpha"]
    local_plot_kwargs["linestyle"] = kwargs["linestyle"]

    # calculate invalid regions
    invalid_regions = []
    if valid_regions[0][0] != 0:
        invalid_regions.append([0, valid_regions[0][0]])
    for i in range(1, len(valid_regions)):
        invalid_regions.append([valid_regions[i - 1][1], valid_regions[i][0]])
    if valid_regions[-1][1] != len(data):
        invalid_regions.append([valid_regions[-1][1], len(data)])
    
    # crop data to fasten plotting
    cropped_valid_regions = []
    cropped_invalid_regions = []
    for region in valid_regions:
        if region[1] < kwargs["xlim"][0] or region[0] > kwargs["xlim"][1]:
            continue
        if region[0] >= kwargs["xlim"][0] and region[1] <= kwargs["xlim"][1]:
            cropped_valid_regions.append(region)
        elif region[0] >= kwargs["xlim"][0] and region[1] > kwargs["xlim"][1]:
            cropped_valid_regions.append([region[0], kwargs["xlim"][1]])
        elif region[0] < kwargs["xlim"][0] and region[1] <= kwargs["xlim"][1]:
            cropped_valid_regions.append([kwargs["xlim"][0], region[1]])
    for region in invalid_regions:
        if region[1] < kwargs["xlim"][0] or region[0] > kwargs["xlim"][1]:
            continue
        if region[0] >= kwargs["xlim"][0] and region[1] <= kwargs["xlim"][1]:
            cropped_invalid_regions.append(region)
        elif region[0] >= kwargs["xlim"][0] and region[1] > kwargs["xlim"][1]:
            cropped_invalid_regions.append([region[0], kwargs["xlim"][1]])
        elif region[0] < kwargs["xlim"][0] and region[1] <= kwargs["xlim"][1]:
            cropped_invalid_regions.append([kwargs["xlim"][0], region[1]])
    
    del valid_regions
    del invalid_regions
    valid_regions = copy.deepcopy(cropped_valid_regions)
    invalid_regions = copy.deepcopy(cropped_invalid_regions)
    
    # plot the data
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    skip_label = False
    for region in valid_regions:
        if skip_label:
            ax.plot(
                np.arange(region[0], region[1]), 
                data[region[0] : region[1]], 
                color=kwargs["color"][0],
                **local_plot_kwargs
            )
        else:
            ax.plot(
                np.arange(region[0], region[1]), 
                data[region[0] : region[1]], 
                label=kwargs["legend"][0], 
                color=kwargs["color"][0],
                **local_plot_kwargs
            )
            skip_label = True
    
    skip_label = False
    for region in invalid_regions:
        if skip_label:
            ax.plot(
                np.arange(region[0], region[1]), 
                data[region[0] : region[1]], 
                color=kwargs["color"][1],
                **local_plot_kwargs
            )
        else:
            ax.plot(
                np.arange(region[0], region[1]), 
                data[region[0] : region[1]], 
                label=kwargs["legend"][1], 
                color=kwargs["color"][1],
                **local_plot_kwargs
            )
            skip_label = True

    ax.legend(loc="best")
    ax.set_xlim(kwargs["xlim"])
    ax.set_ylim(kwargs["ylim"])
    plt.show()


def plot_rpeak_detection(
        data: list, 
        certain_peaks: list, 
        uncertain_primary_peaks: list, 
        uncertain_secondary_peaks: list, 
        name_primary: str,
        name_secondary: str,
        **kwargs
    ):
    """
    Plot the R-peak detection results.
    """
    # Set default values
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "ECG Data")
    kwargs.setdefault("x_label", "time (in iterations)")
    kwargs.setdefault("y_label", "uV")
    kwargs.setdefault("legend", ["ECG", "both", name_primary, name_secondary])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("line_alpha", 1)
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("xlim", [0, len(data)])
    kwargs.setdefault("marker_size", 10)
    kwargs.setdefault("scatter_alpha", 1)
    kwargs.setdefault("scatter_zorder", 2)
    
    y_min = min(data[kwargs["xlim"][0]:kwargs["xlim"][1]])
    y_max = max(data[kwargs["xlim"][0]:kwargs["xlim"][1]])
    kwargs.setdefault("ylim", [y_min-abs(0.2*y_max), y_max+abs(0.2*y_max)])

    local_plot_kwargs = dict()
    local_plot_kwargs["linewidth"] = kwargs["linewidth"]
    local_plot_kwargs["alpha"] = kwargs["line_alpha"]
    local_plot_kwargs["linestyle"] = kwargs["linestyle"]

    local_scatter_kwargs = dict()
    local_scatter_kwargs["s"] = kwargs["marker_size"]
    local_scatter_kwargs["alpha"] = kwargs["scatter_alpha"]
    local_scatter_kwargs["zorder"] = kwargs["scatter_zorder"]

    # plot the data
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    ax.plot(
        np.arange(kwargs["xlim"][0], kwargs["xlim"][1]), 
        data[kwargs["xlim"][0]:kwargs["xlim"][1]], 
        label=kwargs["legend"][0], 
        **local_plot_kwargs
        )
    ax.scatter([],[])
    ax.scatter(
        certain_peaks, 
        data[certain_peaks], 
        label=kwargs["legend"][1],
        **local_scatter_kwargs
        )
    ax.scatter(
        uncertain_primary_peaks, 
        data[uncertain_primary_peaks], 
        label=kwargs["legend"][2],
        **local_scatter_kwargs
        )
    ax.scatter(
        uncertain_secondary_peaks, 
        data[uncertain_secondary_peaks], 
        label=kwargs["legend"][3],
        **local_scatter_kwargs
        )
    ax.legend(loc="best")
    ax.set_xlim(kwargs["xlim"])
    ax.set_ylim(kwargs["ylim"])
    ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.show()