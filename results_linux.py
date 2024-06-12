import random

# import secondary python files
import main
import read_edf
import MAD
import rpeak_detection
import check_data
import plot_helper
from side_functions import *

PREPARATION_DIRECTORY = main.PREPARATION_DIRECTORY
PREPARATION_RESULTS_NAME = main.PREPARATION_RESULTS_NAME

ADDITIONS_RAW_DATA_DIRECTORY = main.ADDITIONS_RAW_DATA_DIRECTORY

parameters = main.parameters

additions_results_path = parameters["additions_results_path"]

"""
==========================
PREPARATION AND ADDITIONS
==========================
"""

"""
--------------------------
ECG VALIDATION
--------------------------
"""

def straighten_the_ecg_signal():
    # choose a random file
    data_directory = "Data/"
    file_data_name = "Somnowatch_Messung.edf"
    file_data_path = data_directory + file_data_name

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    lower_border = 1012500
    interval_size = 1280
    interval = [lower_border, lower_border + interval_size]
    ecg_signal = ECG[lower_border:lower_border + interval_size]

    plot_helper.simple_plot(ecg_signal)

    straightened_ecg = check_data.straighten_ecg(ecg_signal, frequency)

    plot_helper.simple_plot(straightened_ecg)


def evaluate_and_show_valid_ecg_regions():
    # choose a random file
    data_directory = "Data/GIF/SOMNOwatch/"
    file_data_name = "Somnowatch_Messung.edf"
    file_data_name = "SL088_SL088_(1).edf"
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
        check_ecg_validation_strictness=[0.5],
        check_ecg_removed_peak_difference_threshold=0.3,
        check_ecg_std_min_threshold=80,
        check_ecg_std_max_threshold=800,
        check_ecg_distance_std_ratio_threshold=5,
        check_ecg_allowed_invalid_region_length_seconds=30,
        check_ecg_min_valid_length_minutes=5,
        )
    valid_regions = results[0] # looks weird I know, but it is the way it is (reason is behind ecg validation comparison)

    # calculate the ratio of valid regions to total regions
    valid_regions_ratio = check_data.determine_valid_total_ecg_ratio(
        ECG_length = len(ECG), 
        valid_regions = valid_regions
        )
    print("(Valid / Total) Regions Ratio: %f %%" % (round(valid_regions_ratio, 4)*100))

    # choose region to plot
    total_length = len(ECG)
    x_lim = [int(0*total_length), int(1*total_length)]

    plot_helper.plot_valid_regions(
        ECG = ECG, 
        valid_regions = valid_regions,
        xlim = x_lim
        )


def show_evaluated_valid_ecg_regions():
    # choose a random file
    data_directory = "Data/GIF/SOMNOwatch/"
    file_data_name = "SL214_SL214_(1).edf"

    file_data_path = data_directory + file_data_name

    # load the valid regions
    preparation_results_generator = load_from_pickle(additions_results_path)
    for generator_entry in preparation_results_generator:
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
    x_lim = [int(0.2*total_length), int(0.3*total_length)]

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


def compare_ecg_validation_and_gif_classification():
    # choose random file
    file_data_name = "SL088_SL088_(1).edf"
    file_class_name = file_data_name[:-4] + "Somno.txt"
    file_data_path = ADDITIONS_RAW_DATA_DIRECTORY + file_data_name
    file_class_path = parameters["ecg_classification_values_directory"] + file_class_name

    # choose validation_strictness
    validation_strictness = 0.5

    # load the ECG data
    ECG, frequency = read_edf.get_data_from_edf_channel(
        file_path = file_data_path,
        possible_channel_labels = parameters["ecg_keys"],
        physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
        )

    # get the classification values
    ecg_classification_dictionary = check_data.get_ecg_classification_from_txt_file(file_class_path)

    # retrieve evaluated valid regions
    additions_results_generator = load_from_pickle(additions_results_path)
    for generator_entry in additions_results_generator:
        if generator_entry[parameters["file_name_dictionary_key"]] == file_data_name:
            this_files_valid_ecg_regions = generator_entry[parameters["valid_ecg_regions_dictionary_key"] + "_" + str(validation_strictness)]
            break

    # plot the valid regions for easy comparison
    plot_helper.plot_ecg_validation_comparison(
        ECG = ECG, 
        valid_regions = this_files_valid_ecg_regions,
        ecg_classification = ecg_classification_dictionary,
    )

    # afterwards look at a specific region that does not match
    total_length = len(ECG)
    x_lim = [int(0.1*total_length), int(0.5*total_length)]

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


def calculating_rpeaks_from_scratch():
    # choose a random file
    data_directory = "Calibration_Data/"
    file_data_name = "Somnowatch_Messung.edf"

    # choose interval
    interval_size = 2560
    lower_bound = 1781760
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


def visualize_rpeak_comparison():
    # choose a random detection method pair
    first_function_name = "wfdb"
    second_function_name = "gif_classification"

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

    low_ratio_threshold = 0.75
    analogue_ratios_first_function = []
    analogue_ratios_second_function = []
    low_ratio_file_names = []
    low_ratio_values = []

    # load the data
    additions_results_generator = load_from_pickle(additions_results_path)
    for generator_entry in additions_results_generator:
        if parameters["rpeak_comparison_dictionary_key"] not in generator_entry:
            continue
        this_files_rpeak_comparison_values = generator_entry[parameters["rpeak_comparison_dictionary_key"]]
        try:
            first_ratio = this_files_rpeak_comparison_values[position_in_list][3]/this_files_rpeak_comparison_values[position_in_list][4]
            second_ratio = this_files_rpeak_comparison_values[position_in_list][3]/this_files_rpeak_comparison_values[position_in_list][5]
            analogue_ratios_first_function.append(first_ratio)
            analogue_ratios_second_function.append(second_ratio)
            if first_ratio < low_ratio_threshold or second_ratio < low_ratio_threshold:
                low_ratio_values.append([first_ratio, second_ratio])
                low_ratio_file_names.append(generator_entry[parameters["file_name_dictionary_key"]])
        except:
            pass
    
    print("Low Ratio Files: [" + first_function_name + ", " + second_function_name+ "], Total: " + str(len(low_ratio_file_names)) + " Files")
    for i in range(len(low_ratio_file_names)):
        print(low_ratio_file_names[i] + ": " + str(low_ratio_values[i]))

    # plot the data
    plot_helper.plot_simple_histogram(
        data = [analogue_ratios_first_function, analogue_ratios_second_function],
        label = [first_function_name, second_function_name],
        label_title = "R-Peak Detection Method",
        x_label = "Analogue Ratio",
        y_label = "Count",
        xlim = [-0.1, 1.1],
        #binrange = (0.75, 1),
        kde=False,
    )


def plot_non_intersecting_r_peaks():
    # choose a random file
    data_directory = "Data/GIF/SOMNOwatch/"
    file_data_name = "SL088_SL088_(1).edf"

    file_data_path = data_directory + file_data_name

    # choose size of interval
    interval_size = 1280

    # get r-peak function names ("wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification")
    first_rpeak_function_name = "wfdb"
    second_rpeak_function_name = "gif_classification"

    # load the valid regions
    additions_results_generator = load_from_pickle(additions_results_path)
    for generator_entry in additions_results_generator:
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
    random_rpeak = 9040256

    x_lim = [int(random_rpeak-interval_size/2), int(random_rpeak+interval_size/2)]

    plot_helper.plot_rpeak_detection(
        ECG = ECG,
        rpeaks = [rpeaks_only_primary, rpeaks_only_secondary, rpeaks_intersected],
        rpeaks_name = ["only " + first_rpeak_function_name, "only " + second_rpeak_function_name, "intersected"],
        xlim = x_lim)


"""
==========================
PREPARATION AND ADDITIONS
==========================
"""

"""
--------------------------
ECG VALIDATION
--------------------------
"""

# straighten_the_ecg_signal()

evaluate_and_show_valid_ecg_regions()

# show_evaluated_valid_ecg_regions()

"""
--------------------------
ECG VALIDATION COMPARISON
--------------------------
"""

# compare_ecg_validation_and_gif_classification()

"""
--------------------------
CALCULATING R_PEAKS
--------------------------
"""

# calculating_rpeaks_from_scratch()

"""
--------------------------
R-PEAK DETECTION COMPARISON
--------------------------
"""

# visualize_rpeak_comparison()

# plot_non_intersecting_r_peaks()