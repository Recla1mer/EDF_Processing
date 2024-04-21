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


def plot_valid_regions(data, valid_regions, xlim=None):
    """
    Plot the valid regions of the ECG data.
    """
    fig, ax = plt.subplots()

    ax.plot(data, label="Invalid", color="red")
    for region in valid_regions:
        ax.plot(
            np.arange(region[0], region[1]), data[region[0] : region[1]], color="green"
        )
    ax.legend(loc="best")
    ax.set_xlim(xlim)
    plt.show()