"""
Author: Johannes Peter Knoll

Python file containing functions that plot data for this project.
"""

# IMPORTS
import read_edf
from main import *

import copy
import os
import numpy as np
import random
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns
import bitsandbobs as bnb

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler( # type: ignore
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


def simple_plot(data_x, data_y, **kwargs):
    """
    Create a simple plot.
    """

    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "")
    kwargs.setdefault("label", [])
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        alpha = kwargs["alpha"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )
    
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])
    if len(kwargs["label"]) > 0:
        ax.legend(kwargs["label"], loc=kwargs["loc"])

    if isinstance(data_x[0], list):
        for i in range(len(data_x)):
            ax.plot(data_x[i], data_y[i], **plot_args)
    else:
        ax.plot(data_x, data_y, **plot_args)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    plt.show()


def plot_ecg(**kwargs):
    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("label", [])
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        alpha = kwargs["alpha"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    # choose a random edf file
    file_data_name = "Somnowatch_Messung.edf"
    file_data_path = "Data/" + file_data_name

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    interval_size = 2*256 - int(0.3*256)# x seconds for 256 Hz

    x = 30
    lower_border = int(2091000 + random.randint(-x, x)*256) # normal 2087928, 2085880
    lower_border = 2085880
    lower_border = 2087928 + int(0.3*256)
    # lower_border = 6292992 + 640 # normal with fluktuations
    # lower_border = 1781760 # normal but negative peaks
    # lower_border = 2156544 # normal but noisier

    # lower_border = 17752064 # hard noise
    # lower_border = 18344704 # not as extreme overkill
    # lower_border = 10788096 + 640 # continous flat, one large spike 
    # lower_border = 19059968 # extreme overkill

    xlim = [lower_border, lower_border + interval_size]

    data_y = ECG[lower_border:lower_border + interval_size]

    # plot ECG
    simple_plot(
        data_x = np.array([i for i in range(len(data_y))]) / frequency, 
        data_y = data_y, 
        xlabel = r"Time (s)",
        ylabel = r"Voltage ($\mu$V)",
        **kwargs
        )


def plot_calibration_data(data_y, data_x, save_path, **kwargs):
    """
    Create plot and save it.

    ARGUMENTS:
    --------------------------------
    data_y: list
        list of y-values
    data_x: list
        list of x-values
    save_path: str
        path to save the plot

    RETURNS:
    --------------------------------
    None, but the plot is saved
    """
    fig, ax = plt.subplots()

    ax.plot(data_x, data_y)
    plt.savefig(save_path)


def plot_valid_regions(ECG: list, sampling_frequency: int, valid_regions: list, **kwargs):
    """
    Plot the valid regions of the ECG data.

    ARGUMENTS:
    --------------------------------
    ECG: list
        list of ECG data
    sampling_frequency: int
        sampling frequency of the ECG data
    valid_regions: list
        list of valid regions: valid_regions[i] = [start, end]
    
    RETURNS:
    --------------------------------
    None, but the plot is shown
    """

    # Set default values

    # figure
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "ECG Data")
    kwargs.setdefault("x_label", r"Time $\left(\text{in }\dfrac{1}{%i} \text{s}\right)$" % sampling_frequency)
    kwargs.setdefault("y_label", r"Voltage $\left(\text{in } \mu\text{V} \right)$")
    kwargs.setdefault("legend", ["Valid", "Invalid"])
    kwargs.setdefault("color", ["green", "red"])

    # line plot
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("line_alpha", 1)
    kwargs.setdefault("linestyle", "-")

    # xlim and ylim
    kwargs.setdefault("xlim", [0, len(ECG)])

    y_min = min(ECG[kwargs["xlim"][0]:kwargs["xlim"][1]])
    y_max = max(ECG[kwargs["xlim"][0]:kwargs["xlim"][1]])
    kwargs.setdefault("ylim", [y_min-abs(0.2*y_max), y_max+abs(0.2*y_max)])

    # create arguments for plotting
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
    if valid_regions[-1][1] != len(ECG):
        invalid_regions.append([valid_regions[-1][1], len(ECG)])
    
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
        elif region[0] < kwargs["xlim"][0] and region[1] > kwargs["xlim"][1]:
            cropped_valid_regions.append([kwargs["xlim"][0], kwargs["xlim"][1]])
    for region in invalid_regions:
        if region[1] < kwargs["xlim"][0] or region[0] > kwargs["xlim"][1]:
            continue
        if region[0] >= kwargs["xlim"][0] and region[1] <= kwargs["xlim"][1]:
            cropped_invalid_regions.append(region)
        elif region[0] >= kwargs["xlim"][0] and region[1] > kwargs["xlim"][1]:
            cropped_invalid_regions.append([region[0], kwargs["xlim"][1]])
        elif region[0] < kwargs["xlim"][0] and region[1] <= kwargs["xlim"][1]:
            cropped_invalid_regions.append([kwargs["xlim"][0], region[1]])
        elif region[0] < kwargs["xlim"][0] and region[1] > kwargs["xlim"][1]:
            cropped_valid_regions.append([kwargs["xlim"][0], kwargs["xlim"][1]])
    del valid_regions
    del invalid_regions
    valid_regions = copy.deepcopy(cropped_valid_regions)
    invalid_regions = copy.deepcopy(cropped_invalid_regions)
    
    # create plot
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    # plot the ECG data
    skip_label = False
    for region in valid_regions:
        if skip_label:
            ax.plot(
                np.arange(region[0], region[1]), 
                ECG[region[0] : region[1]], 
                color=kwargs["color"][0],
                **local_plot_kwargs
            )
        else:
            ax.plot(
                np.arange(region[0], region[1]), 
                ECG[region[0] : region[1]], 
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
                ECG[region[0] : region[1]], 
                color=kwargs["color"][1],
                **local_plot_kwargs
            )
        else:
            ax.plot(
                np.arange(region[0], region[1]), 
                ECG[region[0] : region[1]], 
                label=kwargs["legend"][1], 
                color=kwargs["color"][1],
                **local_plot_kwargs
            )
            skip_label = True

    ax.legend(loc="best")
    ax.set_xlim(kwargs["xlim"])
    ax.set_ylim(kwargs["ylim"])
    plt.show()


def plot_ecg_validation_comparison(ECG: list, valid_regions: list, ecg_classification: dict, **kwargs):
    """
    """
    # Set default values

    # figure
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "ECG Data")
    kwargs.setdefault("x_label", "time (in iterations)")
    kwargs.setdefault("y_label", "uV")
    kwargs.setdefault("legend", ["Valid", "Invalid"])
    kwargs.setdefault("color", ["green", "red"])

    # line plot
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("line_alpha", 1)
    kwargs.setdefault("linestyle", "-")

    # xlim and ylim
    kwargs.setdefault("xlim", [0, len(ECG)])

    y_min = min(ECG[kwargs["xlim"][0]:kwargs["xlim"][1]])
    y_max = max(ECG[kwargs["xlim"][0]:kwargs["xlim"][1]])
    kwargs.setdefault("ylim", [y_min-abs(0.2*y_max), y_max+abs(0.2*y_max)])

    # create arguments for plotting
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
    if valid_regions[-1][1] != len(ECG):
        invalid_regions.append([valid_regions[-1][1], len(ECG)])

    # get points considered valid and invalid by the ECG classification
    try:
        classification_invalid_points = ecg_classification["1"]
    except:
        classification_invalid_points = []
    try:
        classification_valid_points = ecg_classification["0"]
    except:
        classification_valid_points = []
    
    # create plot
    fig, ax = plt.subplots(2, figsize=kwargs["figsize"])
    # ax.set_xlabel(kwargs["x_label"])
    # ax.set_ylabel(kwargs["y_label"])
    # ax.set_title(kwargs["title"])

    # plot classified points
    ax[1].plot(classification_valid_points, 
        ECG[classification_valid_points], # type: ignore
        color=kwargs["color"][0],
        label=kwargs["legend"][0],
        **local_plot_kwargs
        )
    ax[1].plot(classification_invalid_points, 
        ECG[classification_invalid_points], # type: ignore
        color=kwargs["color"][1],
        label=kwargs["legend"][1],
        **local_plot_kwargs
        )

    # plot the ECG data
    skip_label = False
    for region in valid_regions:
        if skip_label:
            ax[0].plot(
                np.arange(region[0], region[1]), 
                ECG[region[0] : region[1]], 
                color=kwargs["color"][0],
                **local_plot_kwargs
            )
        else:
            ax[0].plot(
                np.arange(region[0], region[1]), 
                ECG[region[0] : region[1]], 
                label=kwargs["legend"][0], 
                color=kwargs["color"][0],
                **local_plot_kwargs
            )
            skip_label = True
    
    skip_label = False
    for region in invalid_regions:
        if skip_label:
            ax[0].plot(
                np.arange(region[0], region[1]), 
                ECG[region[0] : region[1]], 
                color=kwargs["color"][1],
                **local_plot_kwargs
            )
        else:
            ax[0].plot(
                np.arange(region[0], region[1]), 
                ECG[region[0] : region[1]], 
                label=kwargs["legend"][1], 
                color=kwargs["color"][1],
                **local_plot_kwargs
            )
            skip_label = True

    # ax.legend(loc="best")
    # ax.set_xlim(kwargs["xlim"])
    # ax.set_ylim(kwargs["ylim"])
    plt.show()


def plot_rpeak_detection(
        time: list,
        ECG: list, 
        sampling_frequency: int,
        rpeaks: list, 
        rpeaks_name: list,
        **kwargs
    ):
    """
    Plot the R-peak detection results.

    ARGUMENTS:
    --------------------------------
    time: list
        list of time stamps
    ECG: list
        list of ECG data
    sampling_frequency: int
        sampling frequency of the ECG data
    rpeaks: list
        nested list of R-peaks: rpeaks[i] = [R-peak indices]
    rpeaks_name: list
        list of names for the R-peak detection methods

    RETURNS:
    --------------------------------
    None, but the plot is shown
    """

    # Set default values

    # figure
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", None)
    kwargs.setdefault("x_label", r"Time $\left(\text{in }\dfrac{1}{%i} \text{s}\right)$" % sampling_frequency)
    kwargs.setdefault("y_label", r"Voltage $\left(\text{in } \mu\text{V} \right)$")
    kwargs.setdefault("legend", ["ECG"])
    kwargs.setdefault("loc", "best")

    for name in rpeaks_name:
        kwargs["legend"].append(name)

    # line plot
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("line_alpha", 1)
    kwargs.setdefault("linestyle", "-")

    # scatter plot
    kwargs.setdefault("scatter_s", 10)
    kwargs.setdefault("scatter_alpha", 1)
    kwargs.setdefault("scatter_zorder", 2)
    kwargs.setdefault("scatter_markers", ["s", "D", "^", "v", "o", "x", "<", ">", "p", "P", "*", "h", "H", "+", "X", "|", "_"])
    kwargs.setdefault("scatter_marker_resize", 0.6)

    # create arguments for line plotting
    local_plot_kwargs = dict()
    local_plot_kwargs["linewidth"] = kwargs["linewidth"]
    local_plot_kwargs["alpha"] = kwargs["line_alpha"]
    local_plot_kwargs["linestyle"] = kwargs["linestyle"]

    # create arguments for scatter plotting
    local_scatter_kwargs = dict()
    local_scatter_kwargs["alpha"] = kwargs["scatter_alpha"]
    local_scatter_kwargs["zorder"] = kwargs["scatter_zorder"]

    # create plot
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    # plot the ECG data
    ax.plot(
        # np.arange(kwargs["xlim"][0], kwargs["xlim"][1]), 
        # ECG[kwargs["xlim"][0]:kwargs["xlim"][1]], 
        time,
        ECG,
        label=kwargs["legend"][0], 
        **local_plot_kwargs
        )
    
    # plot empty scatter to shift the color cycle
    ax.scatter([],[])

    # plot the R-peaks
    for i in range(len(rpeaks)):
        ax.scatter(
            rpeaks[i], 
            ECG[rpeaks[i]], 
            label=kwargs["legend"][i+1],
            marker=kwargs["scatter_markers"][i],
            s = int(kwargs["scatter_s"] * (kwargs["scatter_marker_resize"] ** i)),
            **local_scatter_kwargs
            )
    
    # xlim and ylim
    kwargs.setdefault("xlim", plt.xlim())
    kwargs.setdefault("ylim", plt.ylim())
    
    ax.legend(loc=kwargs["loc"])
    ax.set_xlim(kwargs["xlim"])
    ax.set_ylim(kwargs["ylim"])
    ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.show()


def plot_MAD_values(
        acceleration_data: list,
        frequency: int,
        MAD_values: list,
        mad_time_period_seconds: int,
        **kwargs
    ):
    """
    Plot the MAD values.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequencies: dict
        dictionary containing the frequencies of the data arrays
    wrist_acceleration_keys: list
        list of keys of data dictionary that are relevant for MAD calculation
    MAD_values: list
        list of MAD values for each interval: MAD[i] = MAD in interval i
    mad_time_period_seconds: int
        length of the time period in seconds over which the MAD will be calculated
    
    RETURNS:
    --------------------------------
    None, but the plot is shown
    """
    # Set default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", None)
    kwargs.setdefault("grid", False)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("xlabel", r"Time (s)")
    kwargs.setdefault("ylabel", r"Acceleration / MAD $\left(1g~/~\frac{1}{10}g\right)$")
    kwargs.setdefault("label", [r"$x$", r"$y$", r"$z$", r"$r$", "MAD"])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("s", 10)
    kwargs.setdefault("fmt", "o")
    kwargs.setdefault("zorder", 2)
    kwargs.setdefault("capsize", 0)
    kwargs.setdefault("capthick", 1.5)
    kwargs.setdefault("elinewidth", 1.5)

    # create arguments for line plotting
    local_plot_kwargs = dict()
    local_plot_kwargs["linewidth"] = kwargs["linewidth"]
    local_plot_kwargs["linestyle"] = kwargs["linestyle"]

    # create arguments for scatter plotting
    local_scatter_kwargs = dict()
    local_scatter_kwargs["s"] = kwargs["s"]
    local_scatter_kwargs["zorder"] = 2

    # create arguments for errorbar plotting
    local_errorbar_kwargs = dict()
    local_errorbar_kwargs["fmt"] = kwargs["fmt"]
    local_errorbar_kwargs["zorder"] = kwargs["zorder"]
    local_errorbar_kwargs["capsize"] = kwargs["capsize"]
    local_errorbar_kwargs["capthick"] = kwargs["capthick"]
    local_errorbar_kwargs["elinewidth"] = kwargs["elinewidth"]

    acc_colors = [plt.rcParams["axes.prop_cycle"].by_key()['color'][2], plt.rcParams["axes.prop_cycle"].by_key()['color'][3], plt.rcParams["axes.prop_cycle"].by_key()['color'][4]]
    res_acc_color = plt.rcParams["axes.prop_cycle"].by_key()['color'][0]
    mad_color = plt.rcParams["axes.prop_cycle"].by_key()['color'][1]

    # plot the data
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])

    resultant_acceleration = [np.sqrt((acceleration_data[0][i]**2 + acceleration_data[1][i]**2 + acceleration_data[2][i]**2)) for i in range(len(acceleration_data[0]))]

    line_x_values = np.array(range(0, len(acceleration_data[0])))/128

    legend_label_counter = 0
    for acc_data in acceleration_data:
        ax.plot(
            line_x_values,
            acc_data,
            label=kwargs["label"][legend_label_counter],
            color = acc_colors[legend_label_counter],
            alpha = 1,
            **local_plot_kwargs
            )
        legend_label_counter += 1
    
    ax.plot(
        line_x_values,
        resultant_acceleration,
        label=kwargs["label"][legend_label_counter],
        color = res_acc_color,
        alpha = 1,
        **local_plot_kwargs
    )
    legend_label_counter += 1

    # calculate time period in samples
    mad_time_period_intervals = int(mad_time_period_seconds)
    mad_x_values = np.arange(mad_time_period_intervals/2, len(MAD_values), mad_time_period_intervals)

    ax.scatter(
        mad_x_values, 
        MAD_values, 
        label=kwargs["label"][legend_label_counter],
        color = mad_color,
        **local_scatter_kwargs
        )
    
    # first_entry = True
    # for i in range(len(MAD_values)):
    #     if first_entry:
    #         ax.errorbar(
    #             mad_x_values[i], 
    #             MAD_values[i], 
    #             xerr=mad_time_period_intervals/2, 
    #             label=kwargs["label"][legend_label_counter],
    #             color = mad_color,
    #             **local_errorbar_kwargs
    #             )
    #         first_entry = False
    #     else:
    #         ax.errorbar(
    #             mad_x_values[i], 
    #             MAD_values[i], 
    #             xerr=mad_time_period_intervals/2,
    #             color = mad_color,
    #             **local_errorbar_kwargs
    #             )
    
    if len(kwargs["label"]) > 0:
        ax.legend(kwargs["label"], loc=kwargs["loc"])
    
    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


def plot_acc_mad(**kwargs):
    # choose a random file
    data_directory = "Data/"
    file_data_name = "Somnowatch_Messung.edf"

    file_data_path = data_directory + file_data_name

    # create lists to save the acceleration data and frequencies for each axis
    acceleration_data = []
    acceleration_data_frequencies = []

    # get the acceleration data and frequency for each axis
    for possible_axis_keys in parameters["wrist_acceleration_keys"]:
        this_axis_signal, this_axis_frequency = read_edf.get_data_from_edf_channel(
            file_path = file_data_path,
            possible_channel_labels = possible_axis_keys,
            physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

        # append data to corresponding lists
        acceleration_data.append(this_axis_signal)
        acceleration_data_frequencies.append(this_axis_frequency)

    # load data and choose interval
    total_length = len(acceleration_data[0])
    frequency = acceleration_data_frequencies[0]

    # choose size of interval
    interval_size = 10 * 128 # 6 seconds for 128 Hz
    lower_border = random.randint(0, total_length - interval_size) # 518241, 961920, 863046 (nice),
    lower_border = 862904
    upper_border = lower_border + interval_size

    g = 9.80665 # m/s^2
    g = 1

    for i in range(len(acceleration_data)):
        acceleration_data[i] = np.array(acceleration_data[i][lower_border:upper_border])/1000*g # convert to m/s^2

    # calculate MAD values
    this_files_MAD_values = MAD.calc_mad(
        acceleration_data_lists = acceleration_data,
        frequencies = acceleration_data_frequencies, 
        time_period = 1,
    )
    this_files_MAD_values = np.array(this_files_MAD_values)*10

    plot_MAD_values(
        acceleration_data = acceleration_data,
        frequency = frequency,
        MAD_values = this_files_MAD_values, # type: ignore
        mad_time_period_seconds = parameters["mad_time_period_seconds"],
        )


def plot_simple_histogram(
        data: list,
        **kwargs
    ):

    # Set default values

    # figure
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "")
    kwargs.setdefault("y_label", "count")
    kwargs.setdefault("label", [""])
    kwargs.setdefault("label_title", "")
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", True)

    # histogram
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", False)
    kwargs.setdefault("bw_adjust", 2)
    kwargs.setdefault("binwidth", 0.05)
    kwargs.setdefault("binrange", None)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("y_scale", "linear")
    kwargs.setdefault("zorder", 2)
    
    # xlim and ylim
    x_min = min(data[0])
    x_max = max(data[0])
    for i in range(1, len(data)):
        x_min = min(x_min, min(data[i]))
        x_max = max(x_max, max(data[i]))
    
    kwargs.setdefault("xlim", (x_min-0.01*abs(x_max), x_max+0.01*abs(x_max)))

    sns_args = dict(
        kde=kwargs["kde"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"],
        zorder=kwargs["zorder"]
    )
    if kwargs["binrange"] is not None:
        sns_args["binrange"] = kwargs["binrange"]

    # create data dictionary
    data_dict = dict(
        data = [],
        name = []
    )
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_dict["data"].append(data[i][j])
            data_dict["name"].append(kwargs["label"][i])

    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax = sns.histplot(data=data_dict, x="data", hue="name", **sns_args)
    ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
    ax.set_yscale(kwargs["y_scale"])
    ax.set_xlim(kwargs["xlim"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)
    ax.legend(title=kwargs["label_title"], labels=kwargs["label"], loc=kwargs["loc"])
    ax.set_title(kwargs["title"])
    plt.show()


def plot_eeg(plot_stage: str = "w", include_eeg = ["f4-m1", "c4-m1", "o2-m1", "c3-m2"], **kwargs):

    f = read_edf.pyedflib.EdfReader("Data/SN001.edf")
    signal_labels = f.getSignalLabels()
    
    start_time = f.getStartdatetime()
    
    eeg_f4_m1 = f.readSignal(signal_labels.index("EEG F4-M1"))
    eeg_c4_m1 = f.readSignal(signal_labels.index("EEG C4-M1"))
    eeg_o2_m1 = f.readSignal(signal_labels.index("EEG O2-M1"))
    eeg_c3_m2 = f.readSignal(signal_labels.index("EEG C3-M2"))

    eeg_f4_m1_frequency = f.getSampleFrequency(signal_labels.index("EEG F4-M1"))
    eeg_c4_m1_frequency = f.getSampleFrequency(signal_labels.index("EEG C4-M1"))
    eeg_o2_m1_frequency = f.getSampleFrequency(signal_labels.index("EEG O2-M1"))
    eeg_c3_m2_frequency = f.getSampleFrequency(signal_labels.index("EEG C3-M2"))
    f.close()

    # plot a segment within desired time range
    w_ranges = [["2001-01-01 23:59:30", "2001-01-02 00:03:00"], ["2001-01-02 00:17:00", "2001-01-02 00:18:30"], ["2001-01-02 01:29:30", "2001-01-02 01:35:00"]]
    n1_ranges = [["2001-01-02 00:03:30", "2001-01-02 00:07:30"], ["2001-01-02 01:35:30", "2001-01-02 01:41:30"], ["2001-01-02 01:35:30", "2001-01-02 01:38:00"]]
    n2_ranges = [["2001-01-02 00:08:30", "2001-01-02 00:11:30"], ["2001-01-02 00:25:00", "2001-01-02 00:51:30"], ["2001-01-02 00:59:30", "2001-01-02 01:16:00"], ["2001-01-02 01:38:30", "2001-01-02 01:59:30"]]
    n3_ranges = [["2001-01-02 00:52:00", "2001-01-02 00:59:30"], ["2001-01-02 05:21:30", "2001-01-02 05:23:30"]]
    rem_ranges = [["2001-01-02 01:17:00", "2001-01-02 01:29:30"], ["2001-01-02 04:18:30", "2001-01-02 04:46:30"], ["2001-01-02 05:47:30", "2001-01-02 06:16:30"]]

    if plot_stage == "w":
        plot_ranges = random.choice(w_ranges)
    elif plot_stage == "n1":
        plot_ranges = random.choice(n1_ranges)
    elif plot_stage == "n2":
        plot_ranges = random.choice(n2_ranges)
    elif plot_stage == "n3":
        plot_ranges = random.choice(n3_ranges)
    elif plot_stage == "rem":
        plot_ranges = random.choice(rem_ranges)

    plot_start_index = int((datetime.strptime(plot_ranges[0], "%Y-%m-%d %H:%M:%S") - start_time).total_seconds() * eeg_f4_m1_frequency)
    plot_end_index = int((datetime.strptime(plot_ranges[1], "%Y-%m-%d %H:%M:%S") - start_time).total_seconds() * eeg_f4_m1_frequency)

    simple_x_axis = range(0, len(eeg_f4_m1[plot_start_index:plot_end_index]))
    
    if "f4-m1" in include_eeg:
        simple_plot(
            data_x=simple_x_axis,
            data_y=eeg_f4_m1[plot_start_index:plot_end_index],
            title = "EEG F4-M1",
            **kwargs
        )

    if "c4-m1" in include_eeg:
        simple_plot(
            data_x=simple_x_axis,
            data_y=eeg_c4_m1[plot_start_index:plot_end_index],
            title = "EEG C4-M1",
        **kwargs
    )

    if "o2-m1" in include_eeg:
        simple_plot(
            data_x=simple_x_axis,
            data_y=eeg_o2_m1[plot_start_index:plot_end_index],
            title = "EEG O2-M1",
            **kwargs
        )

    if "c3-m2" in include_eeg:
        simple_plot(
            data_x=simple_x_axis,
            data_y=eeg_c3_m2[plot_start_index:plot_end_index],
            title = "EEG C3-M2",
            **kwargs
        )


def eeg_plotting():

    plot_stage = "n1"

    f = read_edf.pyedflib.EdfReader("Data/SN001.edf")
    signal_labels = f.getSignalLabels()
    print(signal_labels)
    
    start_time = f.getStartdatetime()
    
    eeg_f4_m1 = f.readSignal(signal_labels.index("EEG F4-M1"))
    eeg_c4_m1 = f.readSignal(signal_labels.index("EEG C4-M1"))
    eeg_o2_m1 = f.readSignal(signal_labels.index("EEG O2-M1"))
    eeg_c3_m2 = f.readSignal(signal_labels.index("EEG C3-M2"))

    eeg_f4_m1_frequency = f.getSampleFrequency(signal_labels.index("EEG F4-M1"))
    eeg_c4_m1_frequency = f.getSampleFrequency(signal_labels.index("EEG C4-M1"))
    eeg_o2_m1_frequency = f.getSampleFrequency(signal_labels.index("EEG O2-M1"))
    eeg_c3_m2_frequency = f.getSampleFrequency(signal_labels.index("EEG C3-M2"))
    f.close()

    # plot a segment within desired time range
    w_ranges = [["2001-01-01 23:59:30", "2001-01-02 00:03:00"], ["2001-01-02 00:17:00", "2001-01-02 00:18:30"], ["2001-01-02 01:29:30", "2001-01-02 01:35:00"]]
    n1_ranges = [["2001-01-02 00:03:30", "2001-01-02 00:07:30"], ["2001-01-02 01:35:30", "2001-01-02 01:41:30"], ["2001-01-02 01:35:30", "2001-01-02 01:38:00"]]
    n2_ranges = [["2001-01-02 00:08:30", "2001-01-02 00:11:30"], ["2001-01-02 00:25:00", "2001-01-02 00:51:30"], ["2001-01-02 00:59:30", "2001-01-02 01:16:00"], ["2001-01-02 01:38:30", "2001-01-02 01:59:30"]]
    n3_ranges = [["2001-01-02 00:52:00", "2001-01-02 00:59:30"], ["2001-01-02 05:21:30", "2001-01-02 05:23:30"]]
    rem_ranges = [["2001-01-02 01:17:00", "2001-01-02 01:29:30"], ["2001-01-02 04:18:30", "2001-01-02 04:46:30"], ["2001-01-02 05:47:30", "2001-01-02 06:16:30"]]

    if plot_stage == "w":
        plot_ranges = random.choice(w_ranges)
    elif plot_stage == "n1":
        plot_ranges = random.choice(n1_ranges)
    elif plot_stage == "n2":
        plot_ranges = random.choice(n2_ranges)
    elif plot_stage == "n3":
        plot_ranges = random.choice(n3_ranges)
    elif plot_stage == "rem":
        plot_ranges = random.choice(rem_ranges)

    plot_start_index = int((datetime.strptime(plot_ranges[0], "%Y-%m-%d %H:%M:%S") - start_time).total_seconds() * eeg_f4_m1_frequency)
    plot_end_index = int((datetime.strptime(plot_ranges[1], "%Y-%m-%d %H:%M:%S") - start_time).total_seconds() * eeg_f4_m1_frequency)

    simple_x_axis = range(0, len(eeg_f4_m1[plot_start_index:plot_end_index]))
    
    simple_plot(
        data_x=simple_x_axis,
        data_y=eeg_f4_m1[plot_start_index:plot_end_index],
        title = "EEG F4-M1"
    )

    simple_plot(
        data_x=simple_x_axis,
        data_y=eeg_c4_m1[plot_start_index:plot_end_index],
        title = "EEG C4-M1"
    )

    simple_plot(
        data_x=simple_x_axis,
        data_y=eeg_o2_m1[plot_start_index:plot_end_index],
        title = "EEG O2-M1"
    )

    simple_plot(
        data_x=simple_x_axis,
        data_y=eeg_c3_m2[plot_start_index:plot_end_index],
        title = "EEG C3-M2"
    )


def plot_slp_course(**kwargs):
    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "")
    kwargs.setdefault("label", [])
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        alpha = kwargs["alpha"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    slp_files_directory = 'Data/GIF/PSG_GIF/'
    valid_slp_files = [file for file in os.listdir(slp_files_directory) if file.endswith(".slp")]

    for slp_file in valid_slp_files:

        slp_file_path = slp_files_directory + slp_file
        slp_file = open(slp_file_path, "rb")
        slp_file_lines = slp_file.readlines()
        slp_file.close()

        numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

        first_slp_line = slp_file_lines[0].decode("utf-8").strip()
        final_index = len(first_slp_line)
        for char_pos in range(len(first_slp_line)):
            if first_slp_line[char_pos] not in numbers:
                final_index = char_pos
                break
        slp_start_time_psg = first_slp_line[0:final_index]
        slp_start_time_seconds_psg = float(slp_start_time_psg) * 3600

        last_slp_line = slp_file_lines[-1].decode("utf-8").strip()
        final_index = len(last_slp_line)
        for char_pos in range(len(last_slp_line)):
            if last_slp_line[char_pos] not in numbers:
                final_index = char_pos
                break
        slp_end_time_psg = last_slp_line[0:final_index]
        slp_end_time_seconds_psg = float(slp_end_time_psg) * 3600

        del numbers, first_slp_line, last_slp_line, final_index

        slp = list()
        for slp_line in slp_file_lines:
            slp_line = slp_line.decode("utf-8").strip()
            if len(slp_line) > 1:
                continue
            slp.append(int(slp_line))
        
        while True:
            if slp[-1] == 0:
                del slp[-1]
            else:
                break
        
        while True:
            if slp[0] == 0:
                del slp[0]
            else:
                break
        
        better_slp_numbers = []
        for stage in slp:
            if stage == 0:
                better_slp_numbers.append(5) # Wake
            elif stage == 1:
                better_slp_numbers.append(1) # N1
            elif stage == 2:
                better_slp_numbers.append(2) # N2
            elif stage == 3:
                better_slp_numbers.append(3) # N3
            elif stage == 5:
                better_slp_numbers.append(4) # REM
        
        number_stages = len(slp)
        
        rem_before_half = 0
        rem_after_half = 0
        n3_after_half = 0
        changes = 0
        
        index = 0
        
        stage_course = []
        last_stage = -1
        for stage in better_slp_numbers:
            if stage == last_stage:
                stage_course[-1][1] += 1
            else:
                stage_course.append([stage, 1])
                last_stage = stage
                changes += 1
            
            if stage == 4 and index < number_stages / 2:
                rem_before_half += 1
            elif stage == 4 and index >= number_stages / 2:
                rem_after_half += 1
            elif stage == 3 and index >= number_stages * 0.65:
                n3_after_half += 1
            
            index += 1
        
        if n3_after_half != 0 or rem_before_half > rem_after_half or changes > 50:
            continue
        
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
        ax.grid(kwargs["grid"])
        if len(kwargs["label"]) > 0:
            ax.legend(kwargs["label"], loc=kwargs["loc"])

        stage_colors = [None, plt.rcParams["axes.prop_cycle"].by_key()['color'][0], plt.rcParams["axes.prop_cycle"].by_key()['color'][1], plt.rcParams["axes.prop_cycle"].by_key()['color'][2], plt.rcParams["axes.prop_cycle"].by_key()['color'][3], plt.rcParams["axes.prop_cycle"].by_key()['color'][4]]
        stage_counter = 0
        
        for i in range(len(stage_course)):
            better_x = range(stage_counter*30, (stage_counter + stage_course[i][1])*30)
            better_x = np.array(better_x) / 3600

            ax.fill_between(
                x = better_x,
                y1 = 0,
                y2 = stage_course[i][0],
                color = stage_colors[stage_course[i][0]],
                alpha = 0.6,
            )
            stage_counter += stage_course[i][1]

        kwargs.setdefault("ylim", plt.ylim())
        kwargs.setdefault("xlim", plt.xlim())
        plt.ylim(kwargs["ylim"])
        plt.xlim(kwargs["xlim"])

        print(f"SLP file: {slp_file}")
        
        plt.show()

        # answer = input("Show next or stop? (n/s): ")

        # if answer.lower() == "s":
        #     break


tex_correction = 0.5
tex_look = {
    "text.usetex": True,
    # "text.latex.preamble": \usepackage{amsmath}\usepackage{amssymb},
    "font.family": "serif",
    "font.serif": "Computer Modern",
    #
    "legend.fontsize": 10-tex_correction,
    "xtick.labelsize": 10-tex_correction,
    "ytick.labelsize": 10-tex_correction,
    "font.size": 12-tex_correction,
    "axes.titlesize": 12-tex_correction,
    "axes.labelsize": 12-tex_correction,
    #
    "savefig.format": "pdf",
    #
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.dpi": 600,
}

python_correction = 0
python_look = {
    "legend.fontsize": 8+python_correction,
    "xtick.labelsize": 8+python_correction,
    "ytick.labelsize": 8+python_correction,
    "font.size": 10+python_correction,
    "axes.titlesize": 10+python_correction,
    "axes.labelsize": 10+python_correction,
    #
    "savefig.format": "pdf",
    #
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.dpi": 600,
}

pt_to_inch = 1./72.27
cm_to_inch = 1/2.54

# linewidth = 16.2*cm_to_inch
linewidth = 459.6215*pt_to_inch

# fig_ratio = 3.4 / 2.7

if __name__ == "__main__":
    matplotlib.rcParams.update(tex_look)
    
    # multi-plots
    # fig_ratio = 4 / 3
    # linewidth *= 0.48 # 0.48, 0.5, 0.3

    # standalone plots
    fig_ratio = 3 / 2
    fig_ratio = 2 / 1.05
    linewidth *= 0.8
    matplotlib.rcParams["figure.figsize"] = [linewidth, linewidth / fig_ratio]

    # plot_ecg(ylim=[-500, 1500])
    plot_acc_mad()

    # plot_slp_course()
    raise SystemExit

    random_seed = 2
    print("Showing EEG plots for Wake stage...")
    plot_eeg(plot_stage="w", include_eeg = ["f4-m1", "c4-m1", "o2-m1", "c3-m2"], xlim=[random_seed*30*256, (random_seed+0.5)*30*256], ylim=[-100, 100])
    print("Showing EEG plots for N1 stage...")
    plot_eeg(plot_stage="n1", include_eeg = ["f4-m1", "c4-m1", "o2-m1", "c3-m2"], xlim=[random_seed*30*256, (random_seed+0.5)*30*256], ylim=[-100, 100])
    print("Showing EEG plots for N2 stage...")
    plot_eeg(plot_stage="n2", include_eeg = ["f4-m1", "c4-m1", "o2-m1", "c3-m2"], xlim=[random_seed*30*256, (random_seed+0.5)*30*256], ylim=[-100, 100])
    print("Showing EEG plots for N3 stage...")
    plot_eeg(plot_stage="n3", include_eeg = ["f4-m1", "c4-m1", "o2-m1", "c3-m2"], xlim=[random_seed*30*256, (random_seed+0.5)*30*256], ylim=[-100, 100])
    print("Showing EEG plots for REM stage...")
    plot_eeg(plot_stage="rem", include_eeg = ["f4-m1", "c4-m1", "o2-m1", "c3-m2"], xlim=[random_seed*30*256, (random_seed+0.5)*30*256], ylim=[-100, 100])

    # plot_eeg(plot_stage="n2", include_eeg=["c3-m2", "c4-m1"], xlim=[random_seed*30*256, (random_seed+1)*30*256])