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
from side_functions import *


"""
--------------------------------
PARAMETERS AND FILE SECTION
--------------------------------

In this section we set the parameters for the project and define the file/directory names.
"""

# define directory and file names (will always be written in capital letters)
DATA_DIRECTORY = "Data/"

TEMPORARY_PICKLE_DIRECTORY = "Temporary_Pickles/"
TEMPORARY_FIGURE_DIRECTORY = "Temporary_Figures/"

# PREPARATION SECTION
# -------------------
PREPARATION_DIRECTORY = "Preparation/"

# ECG Validation
ECG_CALIBRATION_DATA_PATH = "Calibration_Data/Somnowatch_Messung.edf"
ECG_VALIDATION_THRESHOLDS_PATH = PREPARATION_DIRECTORY + "ECG_Validation_Thresholds.pkl"
VALID_ECG_REGIONS_PATH = PREPARATION_DIRECTORY + "Valid_ECG_Regions.pkl"

# R peak detection
CERTAIN_RPEAKS_PATH = PREPARATION_DIRECTORY + "Certain_Rpeaks.pkl"
UNCERTAIN_PRIMARY_RPEAKS_PATH = PREPARATION_DIRECTORY + "Uncertain_Primary_Rpeaks.pkl"
UNCERTAIN_SECONDARY_RPEAKS_PATH = PREPARATION_DIRECTORY + "Uncertain_Secondary_Rpeaks.pkl"

# MAD calculation
MAD_VALUES_PATH = PREPARATION_DIRECTORY + "MAD_Values.pkl"

# ADDITIONALS SECTION
# -------------------
ADDITIONALS_DIRECTORY = "Additions/"

# R peak accuracy evaluation
RPEAK_ACCURACY_DIRECTORY = ADDITIONALS_DIRECTORY + "RPeak_Accuracy/"
RPEAK_ACCURACY_EVALUATION_PATH = RPEAK_ACCURACY_DIRECTORY + "RPeak_Accuracy_Evaluation.pkl"
RPEAK_ACCURACY_REPORT_PATH = RPEAK_ACCURACY_DIRECTORY + "RPeak_Accuracy.txt"
GIF_RPEAKS_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/RRI/"
GIF_DATA_DIRECTORY = "Data/GIF/SOMNOwatch/"


# create directories if they do not exist
if not os.path.isdir(TEMPORARY_PICKLE_DIRECTORY):
    os.mkdir(TEMPORARY_PICKLE_DIRECTORY)
if not os.path.isdir(TEMPORARY_FIGURE_DIRECTORY):
    os.mkdir(TEMPORARY_FIGURE_DIRECTORY)
if not os.path.isdir(PREPARATION_DIRECTORY):
    os.mkdir(PREPARATION_DIRECTORY)
if not os.path.isdir(ADDITIONALS_DIRECTORY):
    os.mkdir(ADDITIONALS_DIRECTORY)
if not os.path.isdir(RPEAK_ACCURACY_DIRECTORY):
    os.mkdir(RPEAK_ACCURACY_DIRECTORY)

# set the parameters for the project
parameters = dict()

settings_params = {
    # set what sections should be executed
    "run_additionals_section": False, # if True, the ADDITIONALS SECTION will be executed
    "run_preparation_section": True, # if True, the PREPARATION SECTION will be executed
    # set what parts of the ADDITIONALS SECTION should be executed
    "show_calibration_data": True, # if True, the calibration data in the manually chosen intervals will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
    "determine_rpeak_accuracy": True, # if True, the accuracy of the R peak detection functions will be evaluated
    # set what parts of the PREPARATION SECTION should be executed
    "calculate_ecg_thresholds": True, # if True, you will have the option to recalculate the thresholds for the ecg validation
    "determine_valid_ecg_regions": True, # if True, you will have the option to recalculate the valid regions for the ECG data
    "detect_rpeaks": True, # if True, the R peaks will be detected in the ECG data
    "calculate_MAD": True, # if True, the MAD will be calculated for the wrist acceleration data
}

file_params = {
    "data_directory": DATA_DIRECTORY, # directory where the data is stored
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_key": "ECG", # key for the ECG data in the data dictionary
    "wrist_acceleration_keys": ["X", "Y", "Z"], # keys for the wrist acceleration data in the data dictionary
}

valid_ecg_regions_params = {
    "ecg_calibration_file_path": ECG_CALIBRATION_DATA_PATH, # path to the EDF file for threshold calibration
    "ecg_thresholds_multiplier": 0.5, # multiplier for the thresholds in check_data.check_ecg() (between 0 and 1)
    "ecg_thresholds_dezimal_places": 2, # number of dezimal places for the check ecg thresholds in the pickle files
    "ecg_thresholds_save_path": ECG_VALIDATION_THRESHOLDS_PATH, # path to the pickle file where the thresholds are saved
    "check_ecg_time_interval_seconds": 10, # time interval considered when determining the valid regions for the ECG data
    "check_ecg_min_valid_length_minutes": 5, # minimum length of valid data in minutes
    "check_ecg_allowed_invalid_region_length_seconds": 30, # data region (see above) still considered valid if the invalid part is shorter than this
    "valid_ecg_regions_path": VALID_ECG_REGIONS_PATH, # path to the pickle file where the valid regions for the ECG data are saved
}

rpeak_accuracy_params = {
    "rpeak_accuracy_functions": [rpeak_detection.get_rpeaks_wfdb, rpeak_detection.get_rpeaks_old], # names of the R peak detection functions
    "rpeak_accuracy_function_names": ["wfdb", "ecgdetectors"], # names of the R peak detection functions
    "accurate_peaks_name": "Accurate", # name of the accurate R peaks
    "accurate_rpeaks_raw_data_directory": GIF_DATA_DIRECTORY, # directory where the raw data of which we know the accurate R peaks are stored
    "accurate_rpeaks_values_directory": GIF_RPEAKS_DIRECTORY, # directory where the accurate R peak values are stored
    "valid_accurate_rpeak_file_types": [".rri"], # file types that store the accurate R peak data
    "rpeak_accuracy_evaluation_path": RPEAK_ACCURACY_EVALUATION_PATH, # path to the pickle file where the evaluation results are saved
    "rpeak_accuracy_rmse_dezimal_places": 4, # number of dezimal places for the RMSE values in the report
    "rpeak_accuracy_report_path": RPEAK_ACCURACY_REPORT_PATH, # path to the text file where the evaluation results are printed
}

detect_rpeaks_params = {
    "rpeak_primary_function": rpeak_detection.get_rpeaks_wfdb, # primary R peak detection function
    "rpeak_secondary_function": rpeak_detection.get_rpeaks_old, # secondary R peak detection function
    "rpeak_name_primary": "wfdb", # name of the primary R peak detection function
    "rpeak_name_secondary": "ecgdetectors", # name of the secondary R peak detection function
    "rpeak_distance_threshold_seconds": 0.05, # max 50ms
    "certain_rpeaks_path": CERTAIN_RPEAKS_PATH, # path to the pickle file where the certain R peaks are saved (detected by both methods)
    "uncertain_primary_rpeaks_path": UNCERTAIN_PRIMARY_RPEAKS_PATH, # path to the pickle file where the uncertain primary R peaks are saved (remaining R peaks from the primary method)
    "uncertain_secondary_rpeaks_path": UNCERTAIN_SECONDARY_RPEAKS_PATH, # path to the pickle file where the uncertain secondary R peaks are saved (remaining R peaks from the secondary method)
}

calculate_MAD_params = {
    "mad_time_period_seconds": 10, # time period in seconds over which the MAD will be calculated
    "mad_values_path": MAD_VALUES_PATH, # path to the pickle file where the MAD values are saved
}

parameters.update(settings_params)
parameters.update(file_params)
parameters.update(valid_ecg_regions_params)
if settings_params["determine_rpeak_accuracy"] and settings_params["run_additionals_section"]:
    parameters.update(rpeak_accuracy_params)
parameters.update(detect_rpeaks_params)
parameters.update(calculate_MAD_params)

del settings_params
del file_params
del valid_ecg_regions_params
del rpeak_accuracy_params
del detect_rpeaks_params
del calculate_MAD_params

# following parameters are calculated in the PREPARATION section. They are written here for better overview and explanation
params_to_be_calculated = {
    "check_ecg_std_min_threshold": 97.84, # if the standard deviation of the ECG data is below this threshold, the data is considered invalid
    "check_ecg_std_max_threshold": 530.62, # if the standard deviation of the ECG data is above this threshold, the data is considered invalid
    "check_ecg_distance_std_ratio_threshold": 1.99, # if the ratio of the distance between two peaks and twice the standard deviation of the ECG data is above this threshold, the data is considered invalid
    "valid_ecg_regions": dict() # dictionary containing the valid regions for the ECG data
}

# check the parameters
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
if not isinstance(parameters["ecg_thresholds_multiplier"], (int, float)):
    raise ValueError("'ecg_thresholds_multiplier' parameter must be an integer or a float.")
if parameters["ecg_thresholds_multiplier"] <= 0 or parameters["ecg_thresholds_multiplier"] > 1:
    raise ValueError("'ecg_thresholds_multiplier' parameter must be between 0 and 1.")
if not isinstance(parameters["check_ecg_time_interval_seconds"], int):
    raise ValueError("'check_ecg_time_interval_seconds' parameter must be an integer.")

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

ecg_thresholds_variables = ["ecg_calibration_file_path", "ecg_thresholds_multiplier", 
                            "ecg_thresholds_dezimal_places", "ecg_key", "ecg_thresholds_save_path"]

determine_ecg_region_variables = ["data_directory", "valid_file_types", "check_ecg_std_min_threshold", 
            "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold", 
            "check_ecg_time_interval_seconds", "check_ecg_min_valid_length_minutes", 
            "check_ecg_allowed_invalid_region_length_seconds", "ecg_key", "valid_ecg_regions_path"]

detect_rpeaks_variables = ["data_directory", "valid_file_types", "ecg_key", "valid_ecg_regions_path"]

combine_detected_rpeaks_variables = ["data_directory", "valid_file_types", "ecg_key",
                        "rpeak_distance_threshold_seconds", "certain_rpeaks_path", 
                        "uncertain_primary_rpeaks_path", "uncertain_secondary_rpeaks_path"]

calculate_MAD_variables = ["data_directory", "valid_file_types", "wrist_acceleration_keys", 
                        "mad_time_period_seconds", "mad_values_path"]

evaluate_rpeak_detection_accuracy_variables = ["accurate_rpeaks_raw_data_directory", "valid_file_types",
                    "accurate_rpeaks_values_directory", "valid_accurate_rpeak_file_types",
                    "rpeak_distance_threshold_seconds","rpeak_accuracy_evaluation_path", "ecg_key"]

rpeak_accuracy_report_variables = ["rpeak_accuracy_function_names", "accurate_peaks_name", 
                        "rpeak_accuracy_rmse_dezimal_places", "rpeak_accuracy_report_path",
                        "rpeak_accuracy_evaluation_path"]


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

"""
--------------------------------
CALIBRATION DATA
--------------------------------
"""

def ecg_threshold_calibration_intervals():
    """
    Manually chosen calibration intervals for the ECG Validation.
    """
    manual_interval_size = 2560 # 10 seconds for 256 Hz
    manual_lower_borders = [
        2091000, # 2h 17min 10sec for 256 Hz
        6292992, # 6h 49min 41sec for 256 Hz
        2156544, # 2h 20min 24sec for 256 Hz
        1781760 # 1h 56min 0sec for 256 Hz
        ]
    return [(border, border + manual_interval_size) for border in manual_lower_borders], manual_interval_size


"""
--------------------------------
ADDITIONALS SECTION
--------------------------------

In this section we will provide additional calculations that are not relevant for the
main part of the project.
"""

def additional_section(run_section: bool):
    if not run_section:
        return

    """
    --------------------------------
    SHOW CALIBRATION DATA
    --------------------------------
    """
    # manually chosen calibration intervals for the ECG Validation:
    manual_calibration_intervals, manual_interval_size = ecg_threshold_calibration_intervals()

    # show calibration data if user requested it, terminate the script afterwards
    if parameters["show_calibration_data"]:
        names = ["perfect_ecg", "fluctuating_ecg", "noisy_ecg", "negative_peaks"]
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(ECG_CALIBRATION_DATA_PATH)
        ecg_data = sigbufs[parameters["ecg_key"]]
        for interval in manual_calibration_intervals:
            plot_helper.simple_plot(
                ecg_data[interval[0]:interval[1]], 
                np.arange(manual_interval_size),
                TEMPORARY_FIGURE_DIRECTORY + names[manual_calibration_intervals.index(interval)] + ".png"
                )
    
    """
    --------------------------------
    DETERMINE R PEAK ACCURACY
    --------------------------------
    """

    if parameters["determine_rpeak_accuracy"]:
        # change the paths where the data is usually stored
        parameters["data_directory"] = parameters["accurate_rpeaks_raw_data_directory"]
        parameters["ecg_thresholds_save_path"] = RPEAK_ACCURACY_DIRECTORY + get_file_name_from_path(parameters["ecg_thresholds_save_path"])
        parameters["valid_ecg_regions_path"] = RPEAK_ACCURACY_DIRECTORY + get_file_name_from_path(parameters["valid_ecg_regions_path"])

        ecg_thresholds_args = create_sub_dict(parameters, ecg_thresholds_variables)
        ecg_thresholds_args["ecg_calibration_intervals"] = manual_calibration_intervals
        check_data.create_ecg_thresholds(**ecg_thresholds_args)
        del ecg_thresholds_args
        del manual_calibration_intervals

        # load the ecg thresholds to the parameters dictionary
        ecg_validation_thresholds_dict = load_from_pickle(parameters["ecg_thresholds_save_path"])
        parameters.update(ecg_validation_thresholds_dict)
        del ecg_validation_thresholds_dict

        determine_ecg_region_args = create_sub_dict(parameters, determine_ecg_region_variables)
        check_data.determine_valid_ecg_regions(**determine_ecg_region_args)
        del determine_ecg_region_args

        detect_rpeaks_args = create_sub_dict(parameters, detect_rpeaks_variables)

        compare_rpeaks_paths = []
        for i in range(len(parameters["rpeak_accuracy_function_names"])):
            compare_rpeaks_paths.append(create_rpeaks_pickle_path(RPEAK_ACCURACY_DIRECTORY, parameters["rpeak_accuracy_function_names"][i]))

        for i in range(len(parameters["rpeak_accuracy_functions"])):
            detect_rpeaks_args["rpeak_function"] = parameters["rpeak_accuracy_functions"][i]
            detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_accuracy_function_names"][i]
            detect_rpeaks_args["rpeak_path"] = compare_rpeaks_paths[i]
            rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

        del detect_rpeaks_args

        evaluate_rpeak_detection_accuracy_args = create_sub_dict(parameters, evaluate_rpeak_detection_accuracy_variables)
        evaluate_rpeak_detection_accuracy_args["compare_rpeaks_paths"] = compare_rpeaks_paths
        rpeak_detection.evaluate_rpeak_detection_accuracy(**evaluate_rpeak_detection_accuracy_args)
        del evaluate_rpeak_detection_accuracy_args

        rpeak_accuracy_report_args = create_sub_dict(parameters, rpeak_accuracy_report_variables)
        rpeak_detection.print_rpeak_accuracy_results(**rpeak_accuracy_report_args)
    
    raise SystemExit("\nIt is not intended to run the ADDTIONAL SECTION and afterwards the MAIN project. As a matter of assuring the correct execution of the script, the script will be TERMINATED. If you want to execute the MAIN project, please set the 'run_additionals_section' parameter to False in the settings section of the script\n")
        

"""
--------------------------------
PREPARATION SECTION
--------------------------------

In this section we will make preparations for the main part of the project. Depending on
the parameters set in the kwargs dictionary, we will calculate the thresholds needed for
various functions, evaluate the valid regions for the ECG data or just load these
informations, if this was already done before.
"""

def preparation_section(run_section: bool):
    if not run_section:
        return
            
    # make sure temporary directories are empty
    clear_directory(TEMPORARY_PICKLE_DIRECTORY)
    clear_directory(TEMPORARY_FIGURE_DIRECTORY)

    """
    --------------------------------
    ECG REGION VALIDATION
    --------------------------------
    """
    if parameters["calculate_ecg_thresholds"]:
        # manually chosen calibration intervals for the ECG Validation:
        manual_calibration_intervals, manual_interval_size = ecg_threshold_calibration_intervals()

        # calculate the thresholds for the ECG Validation
        ecg_thresholds_args = create_sub_dict(parameters, ecg_thresholds_variables)
        ecg_thresholds_args["ecg_calibration_intervals"] = manual_calibration_intervals
        check_data.create_ecg_thresholds(**ecg_thresholds_args)
        del ecg_thresholds_args
        del manual_calibration_intervals

    # load the ecg thresholds to the parameters dictionary
    ecg_validation_thresholds_dict = load_from_pickle(parameters["ecg_thresholds_save_path"])
    parameters.update(ecg_validation_thresholds_dict)
    del ecg_validation_thresholds_dict

    # evaluate valid regions for the ECG data
    if parameters["determine_valid_ecg_regions"]:
        determine_ecg_region_args = create_sub_dict(parameters, determine_ecg_region_variables)
        check_data.determine_valid_ecg_regions(**determine_ecg_region_args)
        del determine_ecg_region_args
    
    """
    --------------------------------
    R PEAK DETECTION
    --------------------------------
    """

    # detect R peaks in the valid regions of the ECG data
    if parameters["detect_rpeaks"]:
        detect_rpeaks_args = create_sub_dict(parameters, detect_rpeaks_variables)
        
        detect_rpeaks_args["rpeak_function"] = parameters["rpeak_primary_function"]
        detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_name_primary"]
        rpeak_primary_path = create_rpeaks_pickle_path(PREPARATION_DIRECTORY, parameters["rpeak_name_primary"])
        detect_rpeaks_args["rpeak_path"] = rpeak_primary_path
        rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

        detect_rpeaks_args["rpeak_function"] = parameters["rpeak_secondary_function"]
        detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_name_secondary"]
        rpeak_secondary_path = create_rpeaks_pickle_path(PREPARATION_DIRECTORY, parameters["rpeak_name_secondary"])
        detect_rpeaks_args["rpeak_path"] = rpeak_secondary_path
        rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

        combine_detected_rpeaks_args = create_sub_dict(parameters, combine_detected_rpeaks_variables)
        combine_detected_rpeaks_args["rpeak_primary_path"] = rpeak_primary_path
        combine_detected_rpeaks_args["rpeak_secondary_path"] = rpeak_secondary_path
        rpeak_detection.combine_detected_rpeaks(**combine_detected_rpeaks_args)

        del detect_rpeaks_args, combine_detected_rpeaks_args, rpeak_primary_path, rpeak_secondary_path
    
    """
    --------------------------------
    MAD CALCULATION
    --------------------------------
    """
    # calculate MAD in the wrist acceleration data
    if parameters["calculate_MAD"]:
        calculate_MAD_args = create_sub_dict(parameters, calculate_MAD_variables)
        MAD.calculate_MAD_in_acceleration_data(**calculate_MAD_args)
        del calculate_MAD_args


"""
--------------------------------
MAIN SECTION
--------------------------------

In this section we will run the functions we have created until now.
"""

def main():
    additional_section(parameters["run_additionals_section"])
    preparation_section(parameters["run_preparation_section"])

    # rpeaks = load_from_pickle(PREPARATION_DIRECTORY + "RPeaks_wfdb.pkl")
    # print(rpeaks)

if __name__ == "__main__":
    main()