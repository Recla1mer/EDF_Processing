"""
Author: Johannes Peter Knoll

Same Code as 'Access_Results.ipynb' (needed for work computer)
"""

# IMPORTS
import random

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

"""
--------------------------
ECG VALIDATION
--------------------------
"""

def straighten_the_ecg_signal(
        data_directory: str, 
        file_data_name: str,
        lower_border: int,
        interval_size: int
        ):
    # choose a random file
    file_data_path = data_directory + file_data_name

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    ecg_signal = ECG[lower_border:lower_border + interval_size]

    plot_helper.simple_plot([i for i in range(len(ecg_signal))], ecg_signal)

    straightened_ecg = check_data.straighten_ecg(ecg_signal, frequency)

    plot_helper.simple_plot([i for i in range(len(straightened_ecg))], straightened_ecg)


def evaluate_and_show_valid_ecg_regions(
        data_directory: str,
        file_data_name: str,
        x_lim_ratio: list,
):
    # choose a random file
    file_data_path = data_directory + file_data_name

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    results = check_data.check_ecg(
        ECG=ECG, 
        frequency=frequency, 
        check_ecg_time_interval_seconds=5,  
        straighten_ecg_signal=True, 
        check_ecg_overlapping_interval_steps=1, 
        check_ecg_validation_strictness=[0.4, 0.6, 0.8],
        check_ecg_removed_peak_difference_threshold=0.3,
        check_ecg_std_min_threshold=80,
        check_ecg_std_max_threshold=800,
        check_ecg_distance_std_ratio_threshold=5,
        check_ecg_allowed_invalid_region_length_seconds=30,
        check_ecg_min_valid_length_minutes=5,
        )

    valid_regions_for_one_val_strictness = results[1] # looks weird I know, but it is the way it is (reason is behind ecg validation comparison)

    # calculate the ratio of valid regions to total regions
    valid_regions_ratio = check_data.determine_valid_total_ecg_ratio(
        ECG_length = len(ECG), 
        valid_regions = valid_regions_for_one_val_strictness
        )
    print("(Valid / Total) Regions Ratio: %f %%" % (round(valid_regions_ratio, 4)*100))

    # choose region to plot
    total_length = len(ECG)
    x_lim = [int(x_lim_ratio[0]*total_length), int(x_lim_ratio[1]*total_length)]

    plot_helper.plot_valid_regions(
        ECG = ECG, 
        valid_regions = valid_regions_for_one_val_strictness,
        xlim = x_lim
        )


def show_evaluated_valid_ecg_regions(
        data_directory: str,
        file_data_name: str,
        results_path: str,
        x_lim_ratio: list,
):
    # choose a random file
    file_data_path = data_directory + file_data_name

    # load the valid regions
    results_generator = load_from_pickle(results_path)
    for generator_entry in results_generator:
        if generator_entry[parameters["file_name_dictionary_key"]] == file_data_name:
            this_files_valid_ecg_regions = generator_entry[parameters["valid_ecg_regions_dictionary_key"]]
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
    x_lim = [int(x_lim_ratio[0]*total_length), int(x_lim_ratio[1]*total_length)]

    plot_helper.plot_valid_regions(
        ECG = ECG, 
        valid_regions = this_files_valid_ecg_regions,
        xlim = x_lim
        )


"""
--------------------------
ECG VALIDATION COMPARISON
--------------------------
"""


def compare_ecg_validation_and_gif_classification(
        data_directory: str,
        class_directory: str,
        file_data_name: str,
        results_path: str,
        validation_strictness: float,
        x_lim_ratio = list,
):
    # choose random file
    file_class_name = file_data_name[:-4] + "Somno.txt" # that's how the gif results are saved, don't blame me
    file_data_path = data_directory + file_data_name
    file_class_path = class_directory + file_class_name

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    # get the classification values
    ecg_classification_dictionary = check_data.get_ecg_classification_from_txt_file(file_class_path)

    # retrieve evaluated valid regions
    results_generator = load_from_pickle(results_path)
    for generator_entry in results_generator:
        if generator_entry[parameters["file_name_dictionary_key"]] == file_data_name:
            this_files_valid_ecg_regions = generator_entry[parameters["valid_ecg_regions_dictionary_key"] + "_" + str(validation_strictness)]
            break

    # plot the valid regions for easy comparison
    plot_helper.plot_ecg_validation_comparison(
        ECG = ECG, 
        valid_regions = this_files_valid_ecg_regions,
        ecg_classification = ecg_classification_dictionary,
    )

    # afterwards look at a specific region that does not match, if necessary:
    total_length = len(ECG)
    x_lim = [int(x_lim_ratio[0]*total_length), int(x_lim_ratio[1]*total_length)] # type: ignore

    plot_helper.plot_valid_regions(
        ECG = ECG, 
        valid_regions = this_files_valid_ecg_regions,
        xlim = x_lim
        )

"""
--------------------------
CALCULATING R_PEAKS
--------------------------
"""


def calculating_rpeaks_from_scratch(
        data_directory: str,
        file_data_name: str,
        interval_size: int,
        lower_bound: int
    ):

    # choose interval
    interval = [lower_bound, lower_bound + interval_size]

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = data_directory + file_data_name,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    ecg_signal = ECG[interval[0]:interval[1]]

    # calculate r-peaks
    rpeaks_hamilton = rpeak_detection.get_rpeaks_hamilton(ecg_signal, frequency, None) # type: ignore
    rpeaks_christov = rpeak_detection.get_rpeaks_christov(ecg_signal, frequency, None) # type: ignore
    rpeaks_wfdb = rpeak_detection.get_rpeaks_wfdb(ecg_signal, frequency, None) # type: ignore
    rpeaks_ecgdet = rpeak_detection.get_rpeaks_ecgdetectors(ecg_signal, frequency, None) # type: ignore

    # plot the r-peaks
    plot_helper.plot_rpeak_detection(
        ECG = ecg_signal,
        rpeaks = [rpeaks_hamilton,rpeaks_christov,rpeaks_wfdb],
        rpeaks_name = ["Hamilton","Christov","WFDB"],
    )


"""
--------------------------
R-PEAK DETECTION COMPARISON
--------------------------
"""


def visualize_rpeak_comparison(
        first_function_name: str,
        second_function_name: str,
        results_path: str,
    ):

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
        this_files_rpeak_comparison_values = generator_entry[parameters["rpeak_comparison_dictionary_key"]]
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
        label_title = "R-Peak Detection Method",
        x_label = "Analogue Ratio",
        y_label = "Count",
        xlim = (0.95, 1),
        binrange = (0, 1),
        binwidth = 0.005,
        kde=False,
    )


def plot_non_intersecting_r_peaks(
        data_directory: str,
        file_data_name: str,
        interval_size: int,
        results_path: str,
    ):

    file_data_path = data_directory + file_data_name

    # get r-peak function names ("wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification")
    first_rpeak_function_name = "hamilton"
    second_rpeak_function_name = "gif_classification"

    # load the valid regions
    results_generator = load_from_pickle(results_path)
    for generator_entry in results_generator:
        if generator_entry[parameters["file_name_dictionary_key"]] == file_data_name:
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
    random_rpeak = random.choice([random_first_rpeak, random_second_rpeak])
    print("Random r-peak location: %d" % random_rpeak)

    # nice values for plotting
    # random_rpeak = 10625000 # for Data/GIF/SOMNOwatch/SL001_SL001_(1).edf, wfdb, gif_classification
    # random_rpeak = 10598685

    x_lim = [int(random_rpeak-interval_size/2), int(random_rpeak+interval_size/2)]

    plot_helper.plot_rpeak_detection(
        ECG = ECG,
        rpeaks = [rpeaks_only_primary, rpeaks_only_secondary, rpeaks_intersected],
        rpeaks_name = ["only " + first_rpeak_function_name, "only " + second_rpeak_function_name, "intersected"],
        xlim = x_lim)
    
"""
==========================
Running Functions
==========================
"""


"""
--------------------------
ECG VALIDATION
--------------------------
"""

# straighten_the_ecg_signal(
#     data_directory =  "Data/",
#     file_data_name = "Somnowatch_Messung.edf",
#     lower_border = 1012500,
#     interval_size = 1280
# )

# evaluate_and_show_valid_ecg_regions(
#     data_directory = "Data/GIF/SOMNOwatch/",
#     file_data_name = "SL104_SL104_(1).edf",
#     x_lim_ratio = [0, 1]
# )

# show_evaluated_valid_ecg_regions(
#     data_directory = "Data/GIF/SOMNOwatch/",
#     file_data_name = "SL104_SL104_(1).edf",
#     results_path = "Processed_GIF/GIF_Results.pkl",
#     x_lim_ratio = [0, 1]
# )

"""
--------------------------
ECG VALIDATION COMPARISON
--------------------------
"""

# compare_ecg_validation_and_gif_classification(
#     data_directory = "Data/GIF/SOMNOwatch/",
#     class_directory = "Data/GIF/Analyse_Somno_TUM/Noise/",
#     file_data_name = "SL001_SL001_(1).edf",
#     results_path = "Processed_GIF/GIF_Results.pkl",
#     validation_strictness = 0.5,
#     x_lim_ratio = [0.89, 11/12] # type: ignore
# )

"""
--------------------------
CALCULATING R_PEAKS
--------------------------
"""

# calculating_rpeaks_from_scratch(
#     data_directory = "Data/",
#     file_data_name = "Somnowatch_Messung.edf",
#     interval_size = 2560,
#     lower_bound = 1781760
# )

"""
--------------------------
R-PEAK DETECTION COMPARISON
--------------------------
"""

# get r-peak function names ("wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification")
first_function_name = "wfdb"
second_function_name = "gif_classification"

# visualize_rpeak_comparison(
#     first_function_name = first_function_name,
#     second_function_name = second_function_name,
#     results_path = "Processed_GIF/GIF_Results.pkl"
# )

# plot_non_intersecting_r_peaks(
#     data_directory = "Data/GIF/SOMNOwatch/",
#     file_data_name = "SL001_SL001_(1).edf",
#     interval_size = 2560,
#     results_path = "Processed_GIF/GIF_Results.pkl"
# )