"""
Author: Johannes Peter Knoll

Main python file for the neural network project.
"""

# import libraries
import numpy as np
import os
import pickle
import time

# import secondary python files
import read_edf
import MAD
import rpeak_detection
import check_data
import plot_helper


"""
--------------------------------
PARAMETERS AND FILE SECTION
--------------------------------

In this section we set the parameters for the project and define the file/directory names.
"""

# define directory and file names (will always be written in capital letters)
DATA_DIRECTORY = "Data/"

TEMPORARY_PICKLE_DIRECTORY_NAME = "Temporary_Pickles/"
TEMPORARY_FIGURE_DIRECTORY_PATH = "Temporary_Figures/"

PREPARATION_DIRECTORY = "Preparation/"

CHECK_ECG_DATA_THRESHOLDS_PATH = PREPARATION_DIRECTORY + "Check_ECG_Data_Thresholds.pkl"
VALID_ECG_REGIONS_PATH = PREPARATION_DIRECTORY + "Valid_ECG_Regions.pkl"

RPEAK_ACCURACY_EVALUATION_PATH = PREPARATION_DIRECTORY + "RPeak_Accuracy_Evaluation.pkl"
RPEAK_ACCURACY_PRINT_PATH = PREPARATION_DIRECTORY + "RPeak_Accuracy.txt"
GIF_RPEAKS_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/RRI/"
GIF_DATA_DIRECTORY = "Data/GIF/SOMNOwatch/"

CERTAIN_RPEAKS_PATH = PREPARATION_DIRECTORY + "Certain_Rpeaks.pkl"
UNCERTAIN_PRIMARY_RPEAKS_PATH = PREPARATION_DIRECTORY + "Uncertain_Primary_Rpeaks.pkl"
UNCERTAIN_SECONDARY_RPEAKS_PATH = PREPARATION_DIRECTORY + "Uncertain_Secondary_Rpeaks.pkl"

MAD_VALUES_PATH = PREPARATION_DIRECTORY + "MAD_Values.pkl"

CALIBRATION_DATA_PATH = "Calibration_Data/Somnowatch_Messung.edf"

# create directories if they do not exist
if not os.path.isdir(TEMPORARY_PICKLE_DIRECTORY_NAME):
    os.mkdir(TEMPORARY_PICKLE_DIRECTORY_NAME)
if not os.path.isdir(TEMPORARY_FIGURE_DIRECTORY_PATH):
    os.mkdir(TEMPORARY_FIGURE_DIRECTORY_PATH)
if not os.path.isdir(PREPARATION_DIRECTORY):
    os.mkdir(PREPARATION_DIRECTORY)

# set the parameters for the project
parameters = dict()

file_params = {
    "file_path": CALIBRATION_DATA_PATH, # path to the EDF file for threshold calibration
    "data_directory": DATA_DIRECTORY, # directory where the data is stored
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_key": "ECG", # key for the ECG data in the data dictionary
    "wrist_acceleration_keys": ["X", "Y", "Z"], # keys for the wrist acceleration data in the data dictionary
}

check_ecg_params = {
    "determine_valid_ecg_regions": True, # if True, the valid regions for the ECG data will be determined
    "calculate_thresholds": True, # if True, you will have the option to recalculate the thresholds for various functions
    "show_calibration_data": False, # if True, the calibration data in the manually chosen intervals will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
    "ecg_threshold_multiplier": 0.5, # multiplier for the thresholds in check_data.check_ecg() (between 0 and 1)
    "check_ecg_threshold_dezimal_places": 2, # number of dezimal places for the check ecg thresholds in the pickle files
    "check_ecg_time_interval_seconds": 10, # time interval considered when determining the valid regions for the ECG data
    "min_valid_length_minutes": 5, # minimum length of valid data in minutes
    "allowed_invalid_region_length_seconds": 30, # data region (see above) still considered valid if the invalid part is shorter than this
}

detect_rpeaks_params = {
    "detect_rpeaks": True, # if True, the R peaks will be detected in the ECG data
    "rpeak_primary_function": rpeak_detection.get_rpeaks_wfdb, # primary R peak detection function
    "rpeak_secondary_function": rpeak_detection.get_rpeaks_old, # secondary R peak detection function
    "rpeak_name_primary": "wfdb", # name of the primary R peak detection function
    "rpeak_name_secondary": "ecgdetectors", # name of the secondary R peak detection function
    "rpeak_distance_threshold_seconds": 0.05, # max 50ms
}

calculate_MAD_params = {
    "calculate_MAD": True, # if True, the MAD will be calculated for the wrist acceleration data
    "mad_time_period_seconds": 10, # time period in seconds over which the MAD will be calculated
}

parameters.update(file_params)
parameters.update(check_ecg_params)
parameters.update(detect_rpeaks_params)
parameters.update(calculate_MAD_params)

del file_params
del check_ecg_params
del detect_rpeaks_params

# following parameters are calculated in the PREPARATION section. They are written here for better overview and explanation
params_to_be_calculated = {
    "check_ecg_std_min_threshold": 97.84, # if the standard deviation of the ECG data is below this threshold, the data is considered invalid
    "check_ecg_std_max_threshold": 530.62, # if the standard deviation of the ECG data is above this threshold, the data is considered invalid
    "check_ecg_distance_std_ratio_threshold": 1.99, # if the ratio of the distance between two peaks and twice the standard deviation of the ECG data is above this threshold, the data is considered invalid
    "valid_ecg_regions": dict() # dictionary containing the valid regions for the ECG data
}

# check the parameters
if not isinstance(parameters["file_path"], str):
    raise ValueError("'file_path' parameter must be a string.")
if not isinstance(parameters["data_directory"], str):
    raise ValueError("'data_directory' parameter must be a string.")
if not isinstance(parameters["valid_file_types"], list):
    raise ValueError("'valid_file_types' parameter must be a list.")
if not isinstance(parameters["ecg_key"], str):
    raise ValueError("'ecg_key' parameter must be a string.")
if not isinstance(parameters["wrist_acceleration_keys"], list):
    raise ValueError("'wrist_acceleration_keys' parameter must be a list.")

if not isinstance(parameters["determine_valid_ecg_regions"], bool):
    raise ValueError("'determine_valid_ecg_regions' parameter must be a boolean.")
if not isinstance(parameters["calculate_thresholds"], bool):
    raise ValueError("'calculate_thresholds' parameter must be a boolean.")
if not isinstance(parameters["show_calibration_data"], bool):
    raise ValueError("'show_calibration_data' parameter must be a boolean.")
if parameters["show_calibration_data"] and parameters["calculate_thresholds"]:
    raise ValueError("'show_calibration_data' and 'calculate_thresholds' parameter cannot both be True at the same time.")
if not isinstance(parameters["ecg_threshold_multiplier"], (int, float)):
    raise ValueError("'ecg_threshold_multiplier' parameter must be an integer or a float.")
if parameters["ecg_threshold_multiplier"] <= 0 or parameters["ecg_threshold_multiplier"] > 1:
    raise ValueError("'ecg_threshold_multiplier' parameter must be between 0 and 1.")
if not isinstance(parameters["check_ecg_threshold_dezimal_places"], int):
    raise ValueError("'check_ecg_threshold_dezimal_places' parameter must be an integer.")
if not isinstance(parameters["check_ecg_time_interval_seconds"], int):
    raise ValueError("'check_ecg_time_interval_seconds' parameter must be an integer.")
if not isinstance(parameters["min_valid_length_minutes"], int):
    raise ValueError("'min_valid_length_minutes' parameter must be an integer.")
if not isinstance(parameters["allowed_invalid_region_length_seconds"], int):
    raise ValueError("'allowed_invalid_region_length_seconds' parameter must be an integer.")

if not isinstance(parameters["detect_rpeaks"], bool):
    raise ValueError("'detect_rpeaks' parameter must be a boolean.")
if not callable(parameters["rpeak_primary_function"]):
    raise ValueError("'rpeak_primary_function' parameter must be a function.")
if not callable(parameters["rpeak_secondary_function"]):
    raise ValueError("'rpeak_secondary_function' parameter must be a function.")
if not isinstance(parameters["rpeak_name_primary"], str):
    raise ValueError("'rpeak_name_primary' parameter must be a string.")
if not isinstance(parameters["rpeak_name_secondary"], str):
    raise ValueError("'rpeak_name_secondary' parameter must be a string.")
if not isinstance(parameters["rpeak_distance_threshold_seconds"], float):
    raise ValueError("'rpeak_distance_threshold_seconds' parameter must be a float.")


# Calibration intervals for check_data.check_ecg()
interval_size = 2560 # 10 seconds for 256 Hz
lower_borders = [
    2091000, # 2h 17min 10sec for 256 Hz
    6292992, # 6h 49min 41sec for 256 Hz
    2156544, # 2h 20min 24sec for 256 Hz
    1781760 # 1h 56min 0sec for 256 Hz
    ]
detection_intervals = [(border, border + interval_size) for border in lower_borders]

# Plot the data if show_graphs is True
# if show_calibration_data:
#     names = ["perfect_ecg", "fluctuating_ecg", "noisy_ecg", "negative_peaks"]
#     for interval in detection_intervals:
#         plot_helper.simple_plot(sigbufs[ecg_key][interval[0]:interval[1]], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + names[detection_intervals.index(interval)] + "_ten_sec.png")


"""
--------------------------------
HELPER FUNCTIONS SECTION
--------------------------------

In this section we provide small functions to keep the code a little cleaner.
"""


"""
Following functions are needed in the PREPARATION section of the project.

These are used to calculate thresholds and evaluate valid regions for the ECG data.
They also detect R peaks in the ECG data and calculate the MAD value.

ATTENTION:
Check that the test data and the intervals in which it is used align with the purpose.
Also examine whether the test data used is suitable for the actual data, e.g. the physical
units match, etc.
"""

def create_rpeaks_pickle_path(rpeak_function_name):
    """
    Create the path for the pickle file where the rpeaks are saved for each method.
    """
    return PREPARATION_DIRECTORY + "RPeaks_" + rpeak_function_name + ".pkl"


def detect_rpeaks(
        data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        rpeak_function,
        rpeak_function_name: str,
    ):
    """
    Detect R peaks in the ECG data.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    ecg_key: str
        key for the ECG data in the data dictionary
    rpeak_function: function
        function to detect the R peaks
    rpeak_function_name: str
        name of the R peak detection function

    RETURNS:
    --------------------------------
    None, but the rpeaks are saved to a pickle file
    """
    pickle_path = create_rpeaks_pickle_path(rpeak_function_name)
    user_answer = ask_for_permission_to_override(file_path = pickle_path,
                            message = "With " + rpeak_function_name + " detected R peaks")
    
    if user_answer == "n":
        return

    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    total_files = len(valid_files)
    progressed_files = 0

    all_rpeaks = dict()

    # load valid ecg regions
    valid_ecg_regions = load_from_pickle(VALID_ECG_REGIONS_PATH)

    # detect rpeaks in the valid regions of the ECG data
    print("Detecting R peaks in the ECG data in %i files using %s:" % (total_files, rpeak_function_name))
    for file in valid_files:
        print_percent_done(progressed_files, total_files)
        try:
            detection_intervals = valid_ecg_regions[file]
            progressed_files += 1
        except KeyError:
            print("Valid regions for the ECG data in " + file + " are missing. Skipping this file.")
            continue
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)
        this_rpeaks = np.array([], dtype = int)
        for interval in detection_intervals:
            this_result = rpeak_function(
                sigbufs, 
                sigfreqs, 
                ecg_key,
                interval,
                )
            this_rpeaks = np.append(this_rpeaks, this_result)
        
        all_rpeaks[file] = this_rpeaks
    
    print_percent_done(progressed_files, total_files)
    
    save_to_pickle(all_rpeaks, pickle_path)


def evaluate_rpeak_detection_accuracy(
        data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        accurate_peaks_directory: str,
        accurate_peaks_name: str,
        valid_accuracy_file_types: list,
        compare_functions_names: list,
        rpeak_distance_threshold_seconds: float,
    ):
    """
    Evaluate the accuracy of the R peak detection methods.

    Accurate R peaks available for GIF data: 
    They were also detected automatically but later corrected manually, so they can be 
    used as a reference.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    ecg_key: str
        key for the ECG data in the data dictionary
    accurate_peaks_directory: str
        directory where the accurate R peaks are stored
    accurate_peaks_name: str
        name the accurate R peaks are associated with
    valid_accuracy_file_types: list
        valid file types in the accurate peaks directory
    compare_functions_names: list
        names of the functions used for R peak detection
        ATTENTION: The names must be the same as the ones used during the R peak detection
    rpeak_distance_threshold_seconds: float
        time period in seconds over which two different R peaks are still considered the same
    
    RETURNS:
    --------------------------------
    None, but the Accuracy values are saved as dictionary to a pickle file in following format:
    {
        "file_name": [ [function_1 values], [function_2 values], ... ],
        ...
    }
    with function values being: rmse_without_same, rmse_with_same, number_of_same_values, number_of_values_considered_as_same, len_function_rpeaks, length_accurate_rpeaks
    for rmse_without_same and rmse_with_same see rpeak_detection.compare_rpeak_detection_methods()
    """
    user_answer = ask_for_permission_to_override(file_path = RPEAK_ACCURACY_EVALUATION_PATH,
                                                message = "R peak accuracy evaluation")
    
    if user_answer == "n":
        return

    all_data_files = os.listdir(data_directory)
    valid_data_files = [file for file in all_data_files if get_file_type(file) in valid_file_types]

    all_accurate_files = os.listdir(accurate_peaks_directory)
    valid_accurate_files = [file for file in all_accurate_files if get_file_type(file) in valid_accuracy_file_types]

    total_data_files = len(valid_data_files)
    progressed_data_files = 0

    # calculate rmse for all files
    all_files_rpeak_accuracy = dict()
    for file in valid_data_files:
        this_file_rpeak_accuracy = []

        this_file_name = os.path.splitext(file)[0]
        for acc_file in valid_accurate_files:
            if this_file_name in acc_file:
                this_accurate_file = acc_file
        try:
            accurate_rpeaks = rpeak_detection.get_rpeaks_from_rri_file(accurate_peaks_directory + this_accurate_file)
        except ValueError:
            print("Accurate R peaks are missing for %s. Skipping this file." % file)
            continue
        
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)
        frequency = sigfreqs[ecg_key]
        
        length_accurate = len(accurate_rpeaks["N"])
        
        for function_name in compare_functions_names:
            compare_function_pickle_path = create_rpeaks_pickle_path(function_name)
            compare_rpeaks_all_files = load_from_pickle(compare_function_pickle_path)

            compare_rpeaks = compare_rpeaks_all_files[file]

            len_compare_rpeaks = len(compare_rpeaks)

            rmse_without_same, rmse_with_same, len_same_values, len_analog_values = rpeak_detection.compare_rpeak_detection_methods(
                accurate_rpeaks["N"], 
                compare_rpeaks,
                accurate_peaks_name,
                function_name,
                frequency,
                rpeak_distance_threshold_seconds,
                False
                )
            
            this_file_rpeak_accuracy.append([rmse_without_same, rmse_with_same, len_same_values, len_analog_values, len_compare_rpeaks, length_accurate])
        
        all_files_rpeak_accuracy[file] = this_file_rpeak_accuracy
    
    save_to_pickle(all_files_rpeak_accuracy, RPEAK_ACCURACY_EVALUATION_PATH)


def print_in_middle(string, length):
    """
    Print the string in the middle of the total length.
    """
    len_string = len(string)
    undersize = int((length - len_string) // 2)
    return " " * undersize + string + " " * (length - len_string - undersize)


def print_rpeak_accuracy_results(compare_functions_names: list,  accurate_peaks_name: str, round_rmse_values: int):
    """
    """
    user_answer = ask_for_permission_to_override(file_path = RPEAK_ACCURACY_PRINT_PATH,
                                                    message = "R peak accuracy file")

    if user_answer == "n":
        return

    accuracy_file = open(RPEAK_ACCURACY_PRINT_PATH, "w")

    # write the file header
    message = "R PEAK ACCURACY EVALUATION"
    accuracy_file.write(message + "\n")
    accuracy_file.write("=" * len(message) + "\n\n\n")

    RMSE_EX_CAPTION = "RMSE_exc"
    RMSE_INC_CAPTION = "RMSE_inc"
    FILE_CAPTION = "File"
    TOTAL_LENGTH_CAPTION = "R peaks"
    SAME_VALUES_CAPTION = "Same Values"
    ANALOG_VALUES_CAPTION = "Analog Values"

    # load the data
    all_files_rpeak_accuracy = load_from_pickle(RPEAK_ACCURACY_EVALUATION_PATH)

    collect_rmse_exc = []
    collect_rmse_inc = []

    # round rmse values and collect them to print the mean
    for file in all_files_rpeak_accuracy:
        this_rmse_exc = []
        this_rmse_inc = []
        for func in range(len(compare_functions_names)):
            all_files_rpeak_accuracy[file][func][0] = round(all_files_rpeak_accuracy[file][func][0], round_rmse_values)
            all_files_rpeak_accuracy[file][func][1] = round(all_files_rpeak_accuracy[file][func][1], round_rmse_values)

            this_rmse_exc.append(all_files_rpeak_accuracy[file][func][0])
            this_rmse_inc.append(all_files_rpeak_accuracy[file][func][1])
        
        collect_rmse_exc.append(this_rmse_exc)
        collect_rmse_inc.append(this_rmse_inc)
    
    # calculate mean rmse values
    mean_rmse_exc = np.mean(collect_rmse_exc, axis = 0)
    mean_rmse_inc = np.mean(collect_rmse_inc, axis = 0)

    # calculate mean distance of number of detected R peaks to accurate R peaks
    collect_rpeaks_distance = []

    for file in all_files_rpeak_accuracy:
        this_rpeaks_distance = []
        for func in range(len(compare_functions_names)):
            this_rpeaks_distance.append(abs(all_files_rpeak_accuracy[file][func][4] - all_files_rpeak_accuracy[file][func][5]))
        collect_rpeaks_distance.append(this_rpeaks_distance)
    
    mean_rpeaks_distance = np.mean(collect_rpeaks_distance, axis = 0)

    # calculate ratio of analog values to accurate R peaks
    collect_analogue_values_ratio = []

    for file in all_files_rpeak_accuracy:
        this_same_values_ratio = []
        for func in range(len(compare_functions_names)):
            this_same_values_ratio.append(all_files_rpeak_accuracy[file][func][3] / all_files_rpeak_accuracy[file][func][5])
        collect_analogue_values_ratio.append(this_same_values_ratio)

    mean_same_values_ratio = np.mean(collect_analogue_values_ratio, axis = 0)

    # write the mean values to file
    message = "Mean values to compare used functions:"
    accuracy_file.write(message + "\n")
    accuracy_file.write("-" * len(message) + "\n\n")
    captions = ["Mean RMSE_exc", "Mean RMSE_inc", "Mean R peaks difference", "Mean Analogue Values"]
    caption_values = [mean_rmse_exc, mean_rmse_inc, mean_rpeaks_distance, mean_same_values_ratio]
    for i in range(len(captions)):
        message = captions[i] + " | "
        first = True
        for func in range(len(compare_functions_names)):
            if first:
                accuracy_file.write(message)
                first = False
            else:
                accuracy_file.write(" " * (len(message)-2) + "| ")
            accuracy_file.write(compare_functions_names[func] + ": " + str(caption_values[i][func]))
            accuracy_file.write("\n")
        accuracy_file.write("\n")
    
    accuracy_file.write("\n")
            
    # calcualte max lengths of table columns
    max_func_name = max([len(name) for name in compare_functions_names])

    all_file_lengths = [len(key) for key in all_files_rpeak_accuracy]
    max_file_length = max(len(FILE_CAPTION), max(all_file_lengths)) + 3

    all_rmse_ex_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(compare_functions_names)):
            all_rmse_ex_lengths.append(len(str(all_files_rpeak_accuracy[file][func][0])))
    all_rmse_ex_lengths = np.array(all_rmse_ex_lengths)
    all_rmse_ex_lengths += max_func_name
    max_rmse_ex_length = max(len(RMSE_EX_CAPTION), max(all_rmse_ex_lengths)) + 3

    all_rmse_inc_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(compare_functions_names)):
            all_rmse_inc_lengths.append(len(str(all_files_rpeak_accuracy[file][func][1])))
    all_rmse_inc_lengths = np.array(all_rmse_inc_lengths)
    all_rmse_inc_lengths += max_func_name
    max_rmse_inc_length = max(len(RMSE_INC_CAPTION), max(all_rmse_inc_lengths)) + 3

    all_same_values_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(compare_functions_names)):
            all_same_values_lengths.append(len(str(all_files_rpeak_accuracy[file][func][2])))
    all_same_values_lengths = np.array(all_same_values_lengths)
    all_same_values_lengths += max_func_name
    max_same_values_length = max(len(SAME_VALUES_CAPTION), max(all_same_values_lengths)) + 3

    all_analog_values_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(compare_functions_names)):
            all_analog_values_lengths.append(len(str(all_files_rpeak_accuracy[file][func][3])))
    all_analog_values_lengths = np.array(all_analog_values_lengths)
    all_analog_values_lengths += max_func_name
    max_analog_values_length = max(len(ANALOG_VALUES_CAPTION), max(all_analog_values_lengths)) + 3

    max_func_name = max(max_func_name, len(accurate_peaks_name)) + 3

    all_rpeaks_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(compare_functions_names)):
            all_rpeaks_lengths.append(len(str(all_files_rpeak_accuracy[file][func][4])))

    for file in all_files_rpeak_accuracy:
        for func in range(len(compare_functions_names)):
            all_rpeaks_lengths.append(len(str(all_files_rpeak_accuracy[file][func][5])))
    all_rpeaks_lengths = np.array(all_rpeaks_lengths)

    all_rpeaks_lengths += max_func_name
    max_rpeaks_length = max(len(TOTAL_LENGTH_CAPTION), max(all_rpeaks_lengths)) + 3

    # write the legend for the table
    message = "Legend:"
    accuracy_file.write(message + "\n")
    accuracy_file.write("-" * len(message) + "\n\n")
    accuracy_file.write("RMSE_exc... RMSE excluding same R peaks\n")
    accuracy_file.write("RMSE_inc... RMSE including same R peaks\n")
    accuracy_file.write("R peaks... Total number of R peaks\n")
    accuracy_file.write("Same Values... Number of R peaks that are the same\n")
    accuracy_file.write("Analogue Values... Number of R peaks that are considered as the same (difference < threshold)\n\n\n")

    message = "Table with Accuracy Values for each file:"
    accuracy_file.write(message + "\n")
    accuracy_file.write("-" * len(message) + "\n\n")

    # create table header
    accuracy_file.write(print_in_middle(FILE_CAPTION, max_file_length) + " | ")
    accuracy_file.write(print_in_middle(RMSE_EX_CAPTION, max_rmse_ex_length) + " | ")
    accuracy_file.write(print_in_middle(RMSE_INC_CAPTION, max_rmse_inc_length) + " | ")
    accuracy_file.write(print_in_middle(TOTAL_LENGTH_CAPTION, max_rpeaks_length) + " | ")
    accuracy_file.write(print_in_middle(SAME_VALUES_CAPTION, max_same_values_length) + " | ")
    accuracy_file.write(print_in_middle(ANALOG_VALUES_CAPTION, max_analog_values_length) + " | ")
    accuracy_file.write("\n")
    accuracy_file.write("-" * (max_file_length + max_rmse_ex_length + max_rmse_inc_length + max_rpeaks_length + max_same_values_length + max_analog_values_length + 17) + "\n")

    # write the data
    for file in all_files_rpeak_accuracy:
        accuracy_file.write(print_in_middle(file, max_file_length) + " | ")
        first = True
        for func in range(len(compare_functions_names)):
            if first:
                first = False
            else:
                accuracy_file.write(print_in_middle("", max_file_length) + " | ")
            accuracy_file.write(print_in_middle(compare_functions_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][0]), max_rmse_ex_length) + " | ")
            accuracy_file.write(print_in_middle(compare_functions_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][1]), max_rmse_inc_length) + " | ")
            accuracy_file.write(print_in_middle(compare_functions_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][4]), max_rpeaks_length) + " | ")
            accuracy_file.write(print_in_middle(compare_functions_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][2]), max_same_values_length) + " | ")
            accuracy_file.write(print_in_middle(compare_functions_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][3]), max_analog_values_length) + " | ")
            accuracy_file.write("\n")
        accuracy_file.write(print_in_middle("", max_file_length) + " | ")
        accuracy_file.write(print_in_middle("", max_rmse_ex_length) + " | ")
        accuracy_file.write(print_in_middle("", max_rmse_inc_length) + " | ")
        accuracy_file.write(print_in_middle(accurate_peaks_name + ": " + str(all_files_rpeak_accuracy[file][func][5]), max_rpeaks_length) + " | ")
        accuracy_file.write(print_in_middle("", max_same_values_length) + " | ")
        accuracy_file.write(print_in_middle("", max_analog_values_length) + " | ")
        accuracy_file.write("\n")
        accuracy_file.write("-" * (max_file_length + max_rmse_ex_length + max_rmse_inc_length + max_rpeaks_length + max_same_values_length + max_analog_values_length + 17) + "\n")

    accuracy_file.close()


def detect_rpeaks_in_ecg_data(
        data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        rpeak_primary_function,
        rpeak_secondary_function,
        rpeak_distance_threshold_seconds: float,
    ):
    """
    Detect R peaks in the ECG data and compare the results of two different functions.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    others: see rpeak_detection.compare_rpeak_detection_methods()

    RETURNS:
    --------------------------------
    None, but the valid regions are saved to a pickle file
    """
    user_answer = ask_for_permission_to_override(file_path = CERTAIN_RPEAKS_PATH,
                                                message = "Detected R peaks")
    
    if user_answer == "n":
        return
    
    os.remove(UNCERTAIN_PRIMARY_RPEAKS_PATH)
    os.remove(UNCERTAIN_SECONDARY_RPEAKS_PATH)

    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    total_files = len(valid_files)
    progressed_files = 0

    certain_rpeaks = dict()
    uncertain_primary_rpeaks = dict()
    uncertain_secondary_rpeaks = dict()

    # load valid ecg regions
    valid_ecg_regions = load_from_pickle(VALID_ECG_REGIONS_PATH)

    # detect rpeaks in the valid regions of the ECG data
    print("Detecting R peaks in the ECG data in %i files:" % total_files)
    for file in valid_files:
        print_percent_done(progressed_files, total_files)
        try:
            detection_intervals = valid_ecg_regions[file]
            progressed_files += 1
        except KeyError:
            print("Valid regions for the ECG data in " + file + " are missing. Skipping this file.")
            continue
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)
        this_certain_rpeaks = np.array([], dtype = int)
        this_uncertain_primary_rpeaks = np.array([], dtype = int)
        this_uncertain_secondary_rpeaks = np.array([], dtype = int)
        for interval in detection_intervals:
            this_result = rpeak_detection.combined_rpeak_detection_methods(
                sigbufs, 
                sigfreqs, 
                ecg_key,
                interval, 
                rpeak_primary_function,
                rpeak_secondary_function,
                rpeak_distance_threshold_seconds,
                )
            this_certain_rpeaks = np.append(this_certain_rpeaks, this_result[0])
            this_uncertain_primary_rpeaks = np.append(this_uncertain_primary_rpeaks, this_result[1])
            this_uncertain_secondary_rpeaks = np.append(this_uncertain_secondary_rpeaks, this_result[2])
        
        certain_rpeaks[file] = this_certain_rpeaks
        uncertain_primary_rpeaks[file] = this_uncertain_primary_rpeaks
        uncertain_secondary_rpeaks[file] = this_uncertain_secondary_rpeaks
    
    print_percent_done(progressed_files, total_files)
    
    save_to_pickle(certain_rpeaks, CERTAIN_RPEAKS_PATH)
    save_to_pickle(uncertain_primary_rpeaks, UNCERTAIN_PRIMARY_RPEAKS_PATH)
    save_to_pickle(uncertain_secondary_rpeaks, UNCERTAIN_SECONDARY_RPEAKS_PATH)


def calculate_MAD_in_acceleration_data(
        data_directory: str,
        valid_file_types: list,
        wrist_acceleration_keys: list, 
        mad_time_period_seconds: int,
    ):
    """
    Calculate the MAD value from the wrist acceleration data.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    wrist_acceleration_keys: list
        keys for the wrist acceleration data in the data dictionary
    mad_time_period_seconds: int
        time period in seconds over which the MAD will be calculated

    RETURNS:
    --------------------------------
    None, but the MAD values are saved to a pickle file
    """
    user_answer = ask_for_permission_to_override(file_path = MAD_VALUES_PATH,
                                    message = "MAD Values for the wrist acceleration data")
    
    if user_answer == "n":
        return

    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    total_files = len(valid_files)
    progressed_files = 0

    MAD_values = dict()

    # calculate MAD in the wrist acceleration data
    print("Calculating MAD in the wrist acceleration data in %i files:" % total_files)
    for file in valid_files:
        print_percent_done(progressed_files, total_files)
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)

        MAD_values[file] = MAD.calc_mad(
            sigbufs, 
            sigfreqs, 
            mad_time_period_seconds, 
            wrist_acceleration_keys
            )
        progressed_files += 1
    
    print_percent_done(progressed_files, total_files)

    save_to_pickle(MAD_values, MAD_VALUES_PATH)
        

"""
--------------------------------
PREPARATION SECTION
--------------------------------

In this section we will make preparations for the main part of the project. Depending on
the parameters set in the kwargs dictionary, we will calculate the thresholds needed for
various functions, evaluate the valid regions for the ECG data or just load these
informations, if this was already done before.
"""

def preparation_section():
            
    # make sure temporary directories are empty
    clear_directory(TEMPORARY_PICKLE_DIRECTORY_NAME)
    clear_directory(TEMPORARY_FIGURE_DIRECTORY_PATH)

    # calculate the thresholds or show how calibration data needed for this should look like
    calculate_thresholds_args = create_sub_dict(
        parameters, ["file_path", "ecg_threshold_multiplier", "check_ecg_threshold_dezimal_places",
                    "show_calibration_data", "ecg_key"]
        )

    if parameters["show_calibration_data"]:
        calculate_thresholds(**calculate_thresholds_args)
        raise SystemExit(0)

    if parameters["calculate_thresholds"]:
        calculate_thresholds(**calculate_thresholds_args)

    del calculate_thresholds_args

    # load the thresholds to the parameters dictionary
    check_ecg_thresholds_dict = load_from_pickle(CHECK_ECG_DATA_THRESHOLDS_PATH)
    parameters.update(check_ecg_thresholds_dict)
    del check_ecg_thresholds_dict

    # evaluate valid regions for the ECG data
    if parameters["determine_valid_ecg_regions"]:
        determine_ecg_region_args = create_sub_dict(
            parameters, ["data_directory", "valid_file_types", "check_ecg_std_min_threshold", 
                        "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold", 
                        "check_ecg_time_interval_seconds", "min_valid_length_minutes", 
                        "allowed_invalid_region_length_seconds", "ecg_key"]
            )
        determine_valid_ecg_regions(**determine_ecg_region_args)
        del determine_ecg_region_args

    # detect R peaks in the valid regions of the ECG data
    if parameters["detect_rpeaks"]:
        detect_rpeaks_args = create_sub_dict(
            parameters, ["data_directory", "valid_file_types", "ecg_key", "rpeak_primary_function",
                        "rpeak_secondary_function", "rpeak_distance_threshold_seconds"]
            )
        detect_rpeaks_in_ecg_data(**detect_rpeaks_args)
        del detect_rpeaks_args
    
    # calculate MAD in the wrist acceleration data
    if parameters["calculate_MAD"]:
        calculate_MAD_args = create_sub_dict(
            parameters, ["data_directory", "valid_file_types", "wrist_acceleration_keys", 
                        "mad_time_period_seconds"]
            )
        calculate_MAD_in_acceleration_data(**calculate_MAD_args)
        del calculate_MAD_args


"""
--------------------------------
MAIN SECTION
--------------------------------

In this section we will run the functions we have created until now.
"""

def main():
    # preparation_section()

    # detect_rpeaks_args = create_sub_dict(
    #         parameters, ["data_directory", "valid_file_types", "ecg_key", "rpeak_primary_function",
    #                     "rpeak_name_primary"]
    #         )
    # check_ecg_thresholds_dict = load_from_pickle(CHECK_ECG_DATA_THRESHOLDS_PATH)
    # parameters.update(check_ecg_thresholds_dict)
    # del check_ecg_thresholds_dict
    # determine_ecg_region_args = create_sub_dict(
    #         parameters, ["valid_file_types", "check_ecg_std_min_threshold", 
    #                     "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold", 
    #                     "check_ecg_time_interval_seconds", "min_valid_length_minutes", 
    #                     "allowed_invalid_region_length_seconds", "ecg_key"]
    #         )
    # determine_valid_ecg_regions(data_directory=GIF_DATA_DIRECTORY, **determine_ecg_region_args)
    
    # detect_rpeaks(GIF_DATA_DIRECTORY, parameters["valid_file_types"], parameters["ecg_key"], parameters["rpeak_primary_function"], parameters["rpeak_name_primary"])
    # detect_rpeaks(GIF_DATA_DIRECTORY, parameters["valid_file_types"], parameters["ecg_key"], parameters["rpeak_secondary_function"], parameters["rpeak_name_secondary"])
    # evaluate_rpeak_detection_accuracy(GIF_DATA_DIRECTORY, parameters["valid_file_types"], parameters["ecg_key"], GIF_RPEAKS_DIRECTORY, "Accurate", [".rri"], [parameters["rpeak_name_primary"], parameters["rpeak_name_secondary"]], parameters["rpeak_distance_threshold_seconds"])
    print_rpeak_accuracy_results([parameters["rpeak_name_primary"], parameters["rpeak_name_secondary"]], "Accurate", 4)



    # rpeaks = load_from_pickle(PREPARATION_DIRECTORY + "RPeaks_wfdb.pkl")
    # print(rpeaks)

if __name__ == "__main__":
    main()