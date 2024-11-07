"""
Author: Johannes Peter Knoll

Python file containing functions that plot data for this project.
"""

# IMPORTS
import copy
import numpy as np

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
    kwargs.setdefault("figsize", [3.4, 2.7])
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
    
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
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
    
    return plt


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
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "ECG Data")
    kwargs.setdefault("x_label", "time (in iterations)")
    kwargs.setdefault("y_label", "mg")
    kwargs.setdefault("legend", ["X", "Y", "Z", "MAD"])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("line_alpha", 0.7)
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("xlim", [0, len(acceleration_data[0])])
    kwargs.setdefault("marker_size", 10)
    kwargs.setdefault("scatter_alpha", 1)
    kwargs.setdefault("scatter_zorder", 2)
    kwargs.setdefault("marker_color", "red")
    kwargs.setdefault("errorbar_fmt", "o")
    kwargs.setdefault("errorbar_zorder", 2)
    kwargs.setdefault("errorbar_capsize", 4)
    kwargs.setdefault("errorbar_capthick", 1.5)
    kwargs.setdefault("errorbar_elinewidth", 1.5)


    y_mins = []
    y_maxs = []
    for acc_data in acceleration_data:
        y_mins.append(min(acc_data[kwargs["xlim"][0]:kwargs["xlim"][1]]))
        y_maxs.append(max(acc_data[kwargs["xlim"][0]:kwargs["xlim"][1]]))
    y_min = min(y_mins)
    y_max = max(y_maxs)
    kwargs.setdefault("ylim", [y_min-abs(0.2*y_max), y_max+abs(0.2*y_max)])

    # create arguments for line plotting
    local_plot_kwargs = dict()
    local_plot_kwargs["linewidth"] = kwargs["linewidth"]
    local_plot_kwargs["alpha"] = kwargs["line_alpha"]
    local_plot_kwargs["linestyle"] = kwargs["linestyle"]

    # create arguments for scatter plotting
    local_scatter_kwargs = dict()
    local_scatter_kwargs["s"] = kwargs["marker_size"]
    local_scatter_kwargs["color"] = kwargs["marker_color"]
    local_scatter_kwargs["alpha"] = kwargs["scatter_alpha"]
    local_scatter_kwargs["zorder"] = kwargs["scatter_zorder"]

    # create arguments for errorbar plotting
    local_errorbar_kwargs = dict()
    local_errorbar_kwargs["fmt"] = kwargs["errorbar_fmt"]
    local_errorbar_kwargs["zorder"] = kwargs["errorbar_zorder"]
    local_errorbar_kwargs["capsize"] = kwargs["errorbar_capsize"]
    local_errorbar_kwargs["capthick"] = kwargs["errorbar_capthick"]
    local_errorbar_kwargs["elinewidth"] = kwargs["errorbar_elinewidth"]
    local_errorbar_kwargs["color"] = kwargs["marker_color"]

    # calculate time period in samples
    mad_time_period_intervals = int(mad_time_period_seconds * frequency)
    mad_x_values = np.arange(mad_time_period_intervals/2, len(acceleration_data[0]), mad_time_period_intervals)

    # cut MAD values outside area of interest
    start_interval = 0
    end_interval = len(mad_x_values)
    for i in range(len(mad_x_values)):
        if mad_x_values[i] <= kwargs["xlim"][0]:
            start_interval = i + 1
        if mad_x_values[i] <= kwargs["xlim"][1]:
            end_interval = i + 1

    # plot the data
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    legend_label_counter = 0
    for acc_data in acceleration_data:
        ax.plot(
            np.arange(kwargs["xlim"][0], kwargs["xlim"][1]), 
            acc_data[kwargs["xlim"][0]:kwargs["xlim"][1]], 
            label=kwargs["legend"][legend_label_counter], 
            **local_plot_kwargs
            )
        legend_label_counter += 1
    
    # ax.scatter(
    #     mad_x_values[start_interval:end_interval], 
    #     MAD_values[start_interval:end_interval], 
    #     label=kwargs["legend"],
    #     **local_scatter_kwargs
    #     )
    
    first_entry = True
    for i in range(start_interval, end_interval):
        if first_entry:
            ax.errorbar(
                mad_x_values[i], 
                MAD_values[i], 
                xerr=mad_time_period_intervals/2, 
                label=kwargs["legend"][legend_label_counter],
                **local_errorbar_kwargs
                )
            first_entry = False
        else:
            ax.errorbar(
                mad_x_values[i], 
                MAD_values[i], 
                xerr=mad_time_period_intervals/2,
                **local_errorbar_kwargs
                )
    
    ax.legend(loc="best")
    ax.set_xlim(kwargs["xlim"])
    ax.set_ylim(kwargs["ylim"])
    plt.show()


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