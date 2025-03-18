"""
Author: Johannes Peter Knoll

Same Code as 'Access_Results.ipynb' (needed for work computer)
"""

# IMPORTS
import random
import numpy as np

# LOCAL IMPORTS
import main
import read_edf
import MAD
import rpeak_detection
import check_data
import plot_helper
from side_functions import *

parameters = main.parameters

"""
=============================================================
Implementing Functions (equal to cells of jupyter notebooks)
=============================================================
"""


def show_ecg_validation(
        file_data_name = "SL104_SL104_(1).edf",
        file_data_directory = "Data/GIF/SOMNOwatch/",
        results_path = "Processed_GIF/GIF_Results.pkl",
        xlim_ratio = [0, 1]
    ):
    """
    Plot valid and invalid ECG data
    """

    file_data_path = file_data_directory + file_data_name

    # load the valid regions
    results_generator = load_from_pickle(results_path)
    for generator_entry in results_generator:
        if generator_entry["file_name"] == file_data_name:
            this_files_valid_ecg_regions = generator_entry["valid_ecg_regions"]
            break

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    # calculate the ratio of valid regions to total regions
    valid_regions_ratio = check_data.determine_valid_total_ecg_ratio(
        ECG_length = len(ECG), 
        valid_regions = this_files_valid_ecg_regions
        )
    print("(Valid / Total) Regions Ratio: %f %%" % (round(valid_regions_ratio, 4)*100))

    total_length = len(ECG)
    xlim = [int(xlim_ratio[0]*total_length), int(xlim_ratio[1]*total_length)]

    plot_helper.plot_valid_regions(
        ECG = ECG, 
        sampling_frequency = frequency,
        valid_regions = this_files_valid_ecg_regions,
        xlim = xlim
        )


def show_rpeak_comparison(
        first_function_name = "hamilton",
        second_function_name = "gif_classification",
        results_path = "Processed_GIF/GIF_Results.pkl",
        **kwargs
    ):
    """
    Plot R-Peak comparison
    """

    # Default values
    kwargs.setdefault("label_title", "R-Peak Detection Method")
    kwargs.setdefault("x_label", "Analogue Ratio")
    kwargs.setdefault("y_label", "Count")
    kwargs.setdefault("xlim", (0.95, 1))
    kwargs.setdefault("binrange", (0, 1))
    kwargs.setdefault("binwidth", 0.005)
    kwargs.setdefault("kde", False)

    # find the position in the list
    position_in_list = 0
    for path_index_first in range(len(parameters["rpeak_comparison_function_names"])):
        found_pair = False
        for path_index_second in range(path_index_first+1, len(parameters["rpeak_comparison_function_names"])):
            if parameters["rpeak_comparison_function_names"][path_index_first] == first_function_name and parameters["rpeak_comparison_function_names"][path_index_second] == second_function_name:
                found_pair = True
                break
            if parameters["rpeak_comparison_function_names"][path_index_first] == second_function_name and parameters["rpeak_comparison_function_names"][path_index_second] == first_function_name:
                found_pair = True
                second_function_name = first_function_name
                first_function_name = parameters["rpeak_comparison_function_names"][path_index_second]
                break
            position_in_list += 1
        if found_pair:
            break

    # load the data
    analogue_ratios_first_function = []
    analogue_ratios_second_function = []

    results_generator = load_from_pickle(results_path)
    for generator_entry in results_generator:
        this_files_rpeak_comparison_values = generator_entry["rpeak_comparison"]
        try:
            analogue_ratios_first_function.append(this_files_rpeak_comparison_values[position_in_list][3]/this_files_rpeak_comparison_values[position_in_list][4])
        except:
            pass
        try:
            analogue_ratios_second_function.append(this_files_rpeak_comparison_values[position_in_list][3]/this_files_rpeak_comparison_values[position_in_list][5])
        except:
            pass

    # plot the data
    plot_helper.plot_simple_histogram(
        data = [analogue_ratios_first_function, analogue_ratios_second_function],
        label = [first_function_name, second_function_name],
        label_title = kwargs["label_title"],
        x_label = kwargs["x_label"],
        y_label = kwargs["y_label"],
        xlim = kwargs["xlim"],
        binrange = kwargs["binrange"],
        binwidth = kwargs["binwidth"],
        kde = kwargs["kde"]
    )


def plot_non_intersecting_rpeaks(
        data_directory = "Data/GIF/SOMNOwatch/",
        file_data_name = "SL001_SL001_(1).edf",
        results_path = "Processed_GIF/GIF_Results.pkl",
        first_rpeak_function_name = "hamilton", # "wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification"
        second_rpeak_function_name = "gif_classification",
        interval_size = 2560,
        random_rpeak = None
    ):
    """
    Plot non-intersecting R-Peaks
    """

    file_data_path = data_directory + file_data_name

    # load the valid regions
    results_generator = load_from_pickle(results_path)
    for generator_entry in results_generator:
        if generator_entry["file_name"] == file_data_name:
            first_rpeaks = generator_entry[first_rpeak_function_name]
            second_rpeaks = generator_entry[second_rpeak_function_name]
            break

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    # combine the r-peaks, retrieve the intersected r-peaks and the r-peaks that are only in the first or second list
    rpeaks_intersected, rpeaks_only_primary, rpeaks_only_secondary = main.rpeak_detection.combine_rpeaks(
        rpeaks_primary = first_rpeaks,
        rpeaks_secondary = second_rpeaks,
        frequency = frequency,
        rpeak_distance_threshold_seconds = parameters["rpeak_distance_threshold_seconds"]
    )

    # choose random r-peak for plotting
    random_first_rpeak = random.choice(rpeaks_only_primary)
    random_second_rpeak = random.choice(rpeaks_only_secondary)

    if random_rpeak is None:
        random_rpeak = random.choice([random_first_rpeak, random_second_rpeak])

    print("Random r-peak location: %d" % random_rpeak)

    # the exact time of the ECG sequence is unnecessary, thats why we want it to be easily readable
    # to do this, we need to the following code to correct the rpeaks, ECG and time values

    xlim = [int(random_rpeak-interval_size/2), int(random_rpeak+interval_size/2)]

    ecg_signal = ECG[xlim[0]:xlim[1]+2600]

    rpeak_lists = []
    for rpeaks in [rpeaks_only_primary, rpeaks_only_secondary, rpeaks_intersected]:
        rpeaks = list(rpeaks)
        for i in range(len(rpeaks)-1, -1, -1):
            if rpeaks[i] < xlim[0] or rpeaks[i] > xlim[1]:
                del rpeaks[i]
        
        if len(rpeaks) == 0:
            rpeak_lists.append(np.array([len(ecg_signal)-1]))
        else:
            rpeak_lists.append(np.array(rpeaks)-xlim[0])

    xlim = [0, len(ecg_signal)-2600]
    time = np.array([i for i in range(len(ecg_signal))])

    # plot the data
    plot_helper.plot_rpeak_detection(
        time = time, # type: ignore
        ECG = ecg_signal,
        sampling_frequency = frequency,
        rpeaks = rpeak_lists,
        rpeaks_name = [first_rpeak_function_name, second_rpeak_function_name, "both"],
        xlim = xlim,
        loc = "lower left"
        )