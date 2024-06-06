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
Plot valid ecg regions
"""
def evaluate_and_plot_valid_ecg_regions():
    # choose a random file
    data_directory = "Data/GIF/SOMNOwatch/"
    file_data_name = "Somnowatch_Messung.edf"
    file_data_name = "SL104_SL104_(1).edf"
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
        check_ecg_validation_strictness=0.5,
        check_ecg_removed_peak_difference_threshold=0.3,
        check_ecg_std_min_threshold=80,
        check_ecg_std_max_threshold=800,
        check_ecg_distance_std_ratio_threshold=5,
        check_ecg_allowed_invalid_region_length_seconds=30,
        check_ecg_min_valid_length_minutes=5,
        ecg_comparison_mode = False,
        )
    valid_regions = results[1][0] # looks weird I know, but it is the way it is (reason is behind ecg validation comparison)

    # calculate the ratio of valid regions to total regions
    valid_regions_ratio = check_data.valid_total_ratio(
        ECG = ECG, 
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


def show_valid_ecg_regions():
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
    valid_regions_ratio = check_data.valid_total_ratio(
        ECG = ECG, 
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


def visualize_ecg_val_comparison():
    # choose random file
    file_data_name = "SL001_SL001_(1).edf"
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
Plot Non-Intersecting R-Peaks
"""

def plot_non_intersecting_r_peaks():
    # choose a random file
    data_directory = "Data/GIF/SOMNOwatch/"
    file_data_name = "SL001_SL001_(1).edf"

    file_data_path = data_directory + file_data_name

    # choose size of interval
    interval_size = 2560

    # get r-peak function names ("wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification")
    first_rpeak_function_name = "wfdb"
    second_rpeak_function_name = "gif_classification"

    # load the valid regions
    additions_results_generator = main.load_from_pickle(additions_results_path)
    for generator_entry in additions_results_generator:
        if generator_entry[parameters["file_name_dictionary_key"]] == file_data_name:
            first_rpeaks = generator_entry[first_rpeak_function_name]
            second_rpeaks = generator_entry[second_rpeak_function_name]
            break

    # load the ECG data
    ECG, frequency = main.read_edf.get_data_from_edf_channel(
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

    x_lim = [int(random_rpeak-interval_size/2), int(random_rpeak+interval_size/2)]

    main.plot_helper.plot_rpeak_detection(
        ECG = ECG,
        rpeaks = [rpeaks_only_primary, rpeaks_only_secondary, rpeaks_intersected],
        rpeaks_name = ["only " + first_rpeak_function_name, "only " + second_rpeak_function_name, "intersected"],
        xlim = x_lim)

"""
Visualizing R-Peak Comparison
"""

def visualizing_r_peak_comparison():
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

    # load the data
    analogue_ratios_first_function = []
    analogue_ratios_second_function = []

    additions_results_generator = load_from_pickle(additions_results_path)
    for generator_entry in additions_results_generator:
        this_files_rpeak_comparison_values = generator_entry[parameters["rpeak_comparison_dictionary_key"]]
        try:
            analogue_ratios_first_function.append(this_files_rpeak_comparison_values[position_in_list][3]/this_files_rpeak_comparison_values[position_in_list][4])
        except ZeroDivisionError:
            pass
        try:
            analogue_ratios_second_function.append(this_files_rpeak_comparison_values[position_in_list][3]/this_files_rpeak_comparison_values[position_in_list][5])
        except ZeroDivisionError:
            pass

    # plot the data
    plot_helper.plot_simple_histogram(
        data = [analogue_ratios_first_function, analogue_ratios_second_function],
        label = [first_function_name, second_function_name],
        label_title = "R-Peak Detection Method",
        x_label = "Analogue Ratio",
        y_label = "Count",
        kde=True,
        binwidth = 0.1,
        x_lim = [0, 1],
    )


#show_valid_ecg_regions()
# visualize_ecg_val_comparison()
#plot_non_intersecting_r_peaks()
#visualizing_r_peak_comparison()
evaluate_and_plot_valid_ecg_regions()