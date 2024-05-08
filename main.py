"""
Author: Johannes Peter Knoll

Main python file for the neural network project.
"""

# import libraries
import copy
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
FILE SECTION
--------------------------------

In this section we define the file/directory names.
"""

# define directory and file names (will always be written in capital letters)
DATA_DIRECTORY = "Data/"

TEMPORARY_PICKLE_DIRECTORY = "Temporary_Pickles/"
TEMPORARY_FIGURE_DIRECTORY = "Temporary_Figures/"

# paths for PREPARATION SECTION
# ------------------------------
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

# paths for ADDITIONALS SECTION
# ------------------------------
ADDITIONALS_DIRECTORY = "Additions/"

# Show Calibration Data
SHOW_CALIBRATION_DATA_DIRECTORY = ADDITIONALS_DIRECTORY + "Show_Calibration_Data/"

# R peak detection comparison
RPEAK_COMPARISON_DIRECTORY = ADDITIONALS_DIRECTORY + "RPeak_Comparison/"
RPEAK_COMPARISON_EVALUATION_PATH = RPEAK_COMPARISON_DIRECTORY + "RPeak_Comparison_Evaluation.pkl"
RPEAK_COMPARISON_REPORT_PATH = RPEAK_COMPARISON_DIRECTORY + "RPeak_Comparison_Report.txt"
GIF_RPEAKS_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/RRI/"
GIF_DATA_DIRECTORY = "Data/GIF/SOMNOwatch/"

# ECG Validation comparison
ECG_VALIDATION_COMPARISON_DIRECTORY = ADDITIONALS_DIRECTORY + "ECG_Validation_Comparison/"
ECG_VALIDATION_COMPARISON_EVALUATION_PATH = ECG_VALIDATION_COMPARISON_DIRECTORY + "ECG_Validation_Comparison_Evaluation.pkl"
ECG_VALIDATION_COMPARISON_REPORT_PATH = ECG_VALIDATION_COMPARISON_DIRECTORY + "ECG_Validation_Comparison_Report.txt"
GIF_ECG_VALIDATION_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/Noise/"


# create directories if they do not exist
if not os.path.isdir(TEMPORARY_PICKLE_DIRECTORY):
    os.mkdir(TEMPORARY_PICKLE_DIRECTORY)
if not os.path.isdir(TEMPORARY_FIGURE_DIRECTORY):
    os.mkdir(TEMPORARY_FIGURE_DIRECTORY)
if not os.path.isdir(PREPARATION_DIRECTORY):
    os.mkdir(PREPARATION_DIRECTORY)


"""
--------------------------------
PARAMETERS SECTION
--------------------------------

In this section we set the parameters for the project.

ATTENTION: 
If you choose to run the ADDITIONALS SECTION, nothing else will be executed.
This is because it should usually be run only once, while you might want to run other 
sections multiple times.
"""

# dictionary that will store all parameters that are used within the project
parameters = dict()

# main parameters: control which sections of the project will be executed
settings_params = {
    # set what sections should be executed
    "run_additionals_section": True, # if True, the ADDITIONALS SECTION will be executed
    "run_preparation_section": True, # if True, the PREPARATION SECTION will be executed
    # set what parts of the ADDITIONALS SECTION should be executed
    "show_calibration_data": True, # if True, the calibration data in the manually chosen intervals will be plotted and saved to the SHOW_CALIBRATION_DATA_DIRECTORY
    "perform_rpeak_comparison": True, # if True, the R peak detection functions will be compared
    "perform_ecg_validation_comparison": True, # if True, the ECG validations will be compared
    # set what parts of the PREPARATION SECTION should be executed
    "calculate_ecg_thresholds": True, # if True, you will have the option to recalculate the thresholds for the ecg validation
    "determine_valid_ecg_regions": True, # if True, you will have the option to recalculate the valid regions for the ECG data
    "detect_rpeaks": True, # if True, the R peaks will be detected in the ECG data
    "calculate_MAD": True, # if True, the MAD will be calculated for the wrist acceleration data
}

# file parameters:
file_params = {
    "data_directory": DATA_DIRECTORY, # directory where the data is stored
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_key": "ECG", # key for the ECG data in the data dictionary
    "wrist_acceleration_keys": ["X", "Y", "Z"], # keys for the wrist acceleration data in the data dictionary
}

# parameters for the PREPARATION SECTION
# --------------------------------------

# parameters for the ECG Validation
valid_ecg_regions_params = {
    "ecg_calibration_file_path": ECG_CALIBRATION_DATA_PATH, # path to the EDF file for threshold calibration
    "ecg_thresholds_multiplier": 0.5, # multiplier for the thresholds in check_data.check_ecg() (between 0 and 1, the higher the more strict the thresholds are)
    "ecg_thresholds_dezimal_places": 2, # number of dezimal places for the check ecg thresholds in the pickle files
    "ecg_thresholds_save_path": ECG_VALIDATION_THRESHOLDS_PATH, # path to the pickle file where the thresholds are saved
    "check_ecg_time_interval_seconds": 10, # time interval considered when determining the valid regions for the ECG data
    "check_ecg_overlapping_interval_steps": 5, # number of times the interval needs to be shifted to the right until the next check_ecg_time_interval_seconds is reached
    "check_ecg_min_valid_length_minutes": 5, # minimum length of valid data in minutes
    "check_ecg_allowed_invalid_region_length_seconds": 30, # data region (see directly above) still considered valid if the invalid part is shorter than this
    "valid_ecg_regions_path": VALID_ECG_REGIONS_PATH, # path to the pickle file where the valid regions for the ECG data are saved
}

# parameters for the R peak detection
detect_rpeaks_params = {
    "rpeak_primary_function": rpeak_detection.get_rpeaks_wfdb, # primary R peak detection function
    "rpeak_secondary_function": rpeak_detection.get_rpeaks_old, # secondary R peak detection function
    "rpeak_name_primary": "wfdb", # name of the primary R peak detection function
    "rpeak_name_secondary": "ecgdetectors", # name of the secondary R peak detection function
    "rpeak_distance_threshold_seconds": 0.05, # If R peaks in the two functions differ by this value, they are still considered the same (max 50ms)
    "certain_rpeaks_path": CERTAIN_RPEAKS_PATH, # path to the pickle file where the certain R peaks are saved (detected by both methods)
    "uncertain_primary_rpeaks_path": UNCERTAIN_PRIMARY_RPEAKS_PATH, # path to the pickle file where the uncertain primary R peaks are saved (remaining R peaks from the primary method)
    "uncertain_secondary_rpeaks_path": UNCERTAIN_SECONDARY_RPEAKS_PATH, # path to the pickle file where the uncertain secondary R peaks are saved (remaining R peaks from the secondary method)
}

# parameters for the MAD calculation
calculate_MAD_params = {
    "mad_time_period_seconds": 10, # time period in seconds over which the MAD will be calculated
    "mad_values_path": MAD_VALUES_PATH, # path to the pickle file where the MAD values are saved
}

# parameters for the ADDITIONALS SECTION
# --------------------------------------

# parameters for the R peak detection comparison
rpeak_comparison_params = {
    "rpeaks_values_directory": GIF_RPEAKS_DIRECTORY, # directory where the given R peak location and classification is stored
    "rpeaks_classification_raw_data_directory": GIF_DATA_DIRECTORY, # directory where the raw data of which we have the rpeak classifications are stored
    "valid_rpeak_values_file_types": [".rri"], # file types that store the R peak classification data
    "include_rpeak_value_classifications": ["N"], # classifications that should be included in the evaluation
    #
    "rpeak_comparison_functions": [rpeak_detection.get_rpeaks_wfdb, rpeak_detection.get_rpeaks_old], # R peak detection functions
    "rpeak_classification_functions": [rpeak_detection.read_rpeaks_from_rri_files], # functions to read the R peak classifications
    "rpeak_comparison_evaluation_path": RPEAK_COMPARISON_EVALUATION_PATH, # path to the pickle file where the evaluation results are saved
    #
    "rpeak_comparison_function_names": ["wfdb", "ecgdetectors", "gif_classification"], # names of all used R peak functions
    "rpeak_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
    "rpeak_comparison_report_path": RPEAK_COMPARISON_REPORT_PATH, # path to the text file that stores the comparison report
}

# parameters for the ECG Validation comparison
ecg_validation_comparison_params = {
    "ecg_validation_comparison_raw_data_directory": GIF_DATA_DIRECTORY, # directory that stores the raw data of which we have the ECG classification
    "ecg_classification_values_directory": GIF_ECG_VALIDATION_DIRECTORY, # directory that stores the ECG classification values
    "ecg_classification_file_types": [".txt"], # file types that store the ECG clasification data
    "ecg_validation_comparison_evaluation_path": ECG_VALIDATION_COMPARISON_EVALUATION_PATH, # path to the pickle file where the comparison results are saved
    "ecg_validation_comparison_report_path": ECG_VALIDATION_COMPARISON_REPORT_PATH, # path to the text file that stores the comparison report
    "ecg_validation_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
}

# add all parameters to the parameters dictionary, so we can access them later more easily
if settings_params["perform_rpeak_comparison"] and settings_params["run_additionals_section"]:
    parameters.update(rpeak_comparison_params)
if settings_params["perform_ecg_validation_comparison"] and settings_params["run_additionals_section"]:
    parameters.update(ecg_validation_comparison_params)
parameters.update(settings_params)
parameters.update(file_params)
parameters.update(valid_ecg_regions_params)
parameters.update(detect_rpeaks_params)
parameters.update(calculate_MAD_params)

# delete the dictionaries as they are saved in the parameters dictionary now
del settings_params, file_params, valid_ecg_regions_params, detect_rpeaks_params, calculate_MAD_params, rpeak_comparison_params, ecg_validation_comparison_params

# following parameters are calculated in the PREPARATION section. They are written here for explanation
params_to_be_calculated = {
    "check_ecg_std_min_threshold": 97.84, # if the standard deviation of the ECG data is below this threshold, the data is considered invalid
    "check_ecg_distance_std_ratio_threshold": 1.99, # if the ratio of the distance between two peaks and twice the standard deviation of the ECG data is above this threshold, the data is considered invalid
}
del params_to_be_calculated

# check the parameters:
# =====================

validate_parameter_settings(parameters)

# create lists of parameters relevant for the following functions (to make the code more readable)

# list for the PREPARATION SECTION
# --------------------------------

ecg_thresholds_variables = ["ecg_calibration_file_path", "ecg_thresholds_multiplier", 
                            "ecg_thresholds_dezimal_places", "ecg_key", "ecg_thresholds_save_path"]

determine_ecg_region_variables = ["data_directory", "valid_file_types", "check_ecg_std_min_threshold", 
            "check_ecg_distance_std_ratio_threshold", "check_ecg_time_interval_seconds",
            "check_ecg_overlapping_interval_steps", "check_ecg_min_valid_length_minutes", 
            "check_ecg_allowed_invalid_region_length_seconds", "ecg_key", "valid_ecg_regions_path"]

detect_rpeaks_variables = ["data_directory", "valid_file_types", "ecg_key", "valid_ecg_regions_path"]

combine_detected_rpeaks_variables = ["data_directory", "valid_file_types", "ecg_key",
                        "rpeak_distance_threshold_seconds", "certain_rpeaks_path", 
                        "uncertain_primary_rpeaks_path", "uncertain_secondary_rpeaks_path"]

calculate_MAD_variables = ["data_directory", "valid_file_types", "wrist_acceleration_keys", 
                        "mad_time_period_seconds", "mad_values_path"]


# lists for the ADDITIONALS SECTION
# ---------------------------------

read_rpeak_classification_variables = ["data_directory", "valid_file_types", "rpeaks_values_directory", 
        "valid_rpeak_values_file_types", "include_rpeak_value_classifications"]

rpeak_detection_comparison_variables = ["data_directory", "valid_file_types", "ecg_key",
                    "rpeak_distance_threshold_seconds", "rpeak_comparison_evaluation_path"]

rpeak_detection_comparison_report_variables = ["rpeak_comparison_function_names", "rpeak_comparison_report_dezimal_places", 
                                   "rpeak_comparison_report_path", "rpeak_comparison_evaluation_path"]

ecg_validation_comparison_variables = ["ecg_classification_values_directory", 
        "ecg_classification_file_types", "ecg_validation_comparison_evaluation_path",
        "valid_ecg_regions_path"]

ecg_validation_comparison_report_variables = ["ecg_validation_comparison_evaluation_path", 
            "ecg_validation_comparison_report_path", "ecg_validation_comparison_report_dezimal_places"]


"""
--------------------------------
CALIBRATION DATA
--------------------------------

In this section we will provide manually chosen calibration intervals for the ECG Validation.

ATTENTION:
Check that the test data and the intervals in which it is used align with the purpose.
Also examine whether the test data used is suitable for the actual data, e.g. the physical
units match, etc.
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
    """
    Section that is not relevant for the main part of the project. It shows calibration data
    and compares different R peak detections and ECG Validations.
    """

    # check if the section should be run
    if not run_section:
        return
    
    # create needed directory if it does not exist
    if not os.path.isdir(ADDITIONALS_DIRECTORY):
        os.mkdir(ADDITIONALS_DIRECTORY)

    """
    --------------------------------
    SHOW CALIBRATION DATA
    --------------------------------
    """
    # get manually chosen calibration intervals for the ECG validation:
    manual_calibration_intervals, manual_interval_size = ecg_threshold_calibration_intervals()

    # show calibration data if user requested it
    if parameters["show_calibration_data"]:
        # create directory to save plots if it does not exist and make sure it is empty
        if not os.path.isdir(SHOW_CALIBRATION_DATA_DIRECTORY):
            os.mkdir(SHOW_CALIBRATION_DATA_DIRECTORY)
        clear_directory(SHOW_CALIBRATION_DATA_DIRECTORY)

        # plot the calibration data in the manually chosen calibration intervals
        names = ["perfect_ecg", "fluctuating_ecg", "noisy_ecg", "negative_peaks"]
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(ECG_CALIBRATION_DATA_PATH)
        ecg_data = sigbufs[parameters["ecg_key"]]
        for interval in manual_calibration_intervals:
            plot_helper.plot_calibration_data(
                ecg_data[interval[0]:interval[1]], 
                np.arange(manual_interval_size),
                SHOW_CALIBRATION_DATA_DIRECTORY + names[manual_calibration_intervals.index(interval)] + ".png"
                )
    
    # perform ecg validation if needed
    if parameters["perform_rpeak_comparison"] or parameters["perform_ecg_validation_comparison"]:
        parameters["ecg_thresholds_save_path"] = ADDITIONALS_DIRECTORY + get_file_name_from_path(parameters["ecg_thresholds_save_path"])
        # create arguments for ecg thresholds evaluation and calculate them
        ecg_thresholds_args = create_sub_dict(parameters, ecg_thresholds_variables)
        ecg_thresholds_args["ecg_calibration_intervals"] = manual_calibration_intervals
        check_data.create_ecg_thresholds(**ecg_thresholds_args)
        del ecg_thresholds_args

        # load the ecg thresholds to the parameters dictionary
        ecg_validation_thresholds_dict = load_from_pickle(parameters["ecg_thresholds_save_path"])
        parameters.update(ecg_validation_thresholds_dict)
        del ecg_validation_thresholds_dict
        del manual_calibration_intervals

        # change the data paths to where the ECG classification is stored
        store_old_data_directory = copy.deepcopy(parameters["data_directory"])
        parameters["data_directory"] = parameters["ecg_validation_comparison_raw_data_directory"]
        parameters["valid_ecg_regions_path"] = ADDITIONALS_DIRECTORY + get_file_name_from_path(parameters["valid_ecg_regions_path"])

        # create arguments for the valid ecg regions evaluation and calculate them
        determine_ecg_region_args = create_sub_dict(parameters, determine_ecg_region_variables)
        check_data.determine_valid_ecg_regions(**determine_ecg_region_args)
        del determine_ecg_region_args
    
    """
    --------------------------------
    COMPARE ECG VALIDATIONS
    --------------------------------
    """
    # compare ECG validations if user requested it
    if parameters["perform_ecg_validation_comparison"]:
        # create directory to save comparison results if it does not exist
        if not os.path.isdir(ECG_VALIDATION_COMPARISON_DIRECTORY):
            os.mkdir(ECG_VALIDATION_COMPARISON_DIRECTORY)
        # create arguments for the ECG validation comparison and perform it
        ecg_validation_comparison_args = create_sub_dict(parameters, ecg_validation_comparison_variables)
        check_data.ecg_validation_comparison(**ecg_validation_comparison_args)
        del ecg_validation_comparison_args

        # create arguments for printing the ECG validation comparison report
        ecg_validation_report_args = create_sub_dict(parameters, ecg_validation_comparison_report_variables)
        check_data.ecg_validation_comparison_report(**ecg_validation_report_args)
    
    """
    --------------------------------
    COMPARE R PEAK DETECTIONS
    --------------------------------
    """

    # compare the R peak detection functions if user requested it
    if parameters["perform_rpeak_comparison"]:
        # create directory to save comparison results if it does not exist
        if not os.path.isdir(RPEAK_COMPARISON_DIRECTORY):
            os.mkdir(RPEAK_COMPARISON_DIRECTORY)
        
        # change the data paths to where the R peak classifications are stored
        if len(parameters["rpeak_classification_functions"]) > 0:
            parameters["data_directory"] = parameters["rpeaks_classification_raw_data_directory"]
        else:
            parameters["data_directory"] = store_old_data_directory

        # create arguments for the R peak detection evaluation
        detect_rpeaks_args = create_sub_dict(parameters, detect_rpeaks_variables)

        # create paths to where the detected R peaks will be saved
        compare_rpeaks_paths = []
        for func_name in parameters["rpeak_comparison_function_names"]:
            compare_rpeaks_paths.append(create_rpeaks_pickle_path(RPEAK_COMPARISON_DIRECTORY, func_name))

        # detect R peaks in the valid regions of the ECG data
        classification_index_offset = 0
        for i in range(len(parameters["rpeak_comparison_functions"])):
            classification_index_offset += 1
            detect_rpeaks_args["rpeak_function"] = parameters["rpeak_comparison_functions"][i]
            detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_comparison_function_names"][i]
            detect_rpeaks_args["rpeak_path"] = compare_rpeaks_paths[i]
            rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

        del detect_rpeaks_args

        # read r peaks from the classification files if they are needed
        read_rpeak_classification_args = create_sub_dict(parameters, read_rpeak_classification_variables)
        for i in range(len(parameters["rpeak_classification_functions"])):
            read_rpeak_classification_args["rpeak_path"] = compare_rpeaks_paths[classification_index_offset + i]
            rpeak_detection.read_rpeaks_from_rri_files(**read_rpeak_classification_args)
        del read_rpeak_classification_args

        # create arguments for the R peak comparison evaluation and perform it
        rpeak_detection_comparison_args = create_sub_dict(parameters, rpeak_detection_comparison_variables)
        rpeak_detection_comparison_args["compare_rpeaks_paths"] = compare_rpeaks_paths
        rpeak_detection.rpeak_detection_comparison(**rpeak_detection_comparison_args)
        del rpeak_detection_comparison_args

        # create arguments for printing the R peak comparison report and print it
        rpeak_comparison_report_args = create_sub_dict(parameters, rpeak_detection_comparison_report_variables)
        rpeak_detection.rpeak_detection_comparison_report(**rpeak_comparison_report_args)
    
    # terminate the script after the ADDITIONALS SECTION
    raise SystemExit("\nIt is not intended to run the ADDTIONAL SECTION and afterwards the MAIN project. Therefore, the script will be TERMINATED. If you want to execute the MAIN project, please set the 'run_additionals_section' parameter to False in the settings section of the script\n")
        

"""
--------------------------------
PREPARATION SECTION
--------------------------------

In this section we will make preparations for the main part of the project. Depending on
the parameters set in the parameters dictionary, we will calculate the thresholds needed for
various functions, evaluate the valid regions for the ECG data, perform R peak detection
and calculate the MAD in the wrist acceleration data.
"""

def preparation_section(run_section: bool):

    # check if the section should be run
    if not run_section:
        return

    """
    --------------------------------
    ECG REGION VALIDATION
    --------------------------------
    """
    if parameters["calculate_ecg_thresholds"]:
        # get manually chosen calibration intervals for the ECG Validation:
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
        # create arguments for the R peak detection
        detect_rpeaks_args = create_sub_dict(parameters, detect_rpeaks_variables)
        
        # detect R peaks using the primary function
        detect_rpeaks_args["rpeak_function"] = parameters["rpeak_primary_function"]
        detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_name_primary"]
        rpeak_primary_path = create_rpeaks_pickle_path(PREPARATION_DIRECTORY, parameters["rpeak_name_primary"])
        detect_rpeaks_args["rpeak_path"] = rpeak_primary_path
        rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

        # detect R peaks using the secondary function
        detect_rpeaks_args["rpeak_function"] = parameters["rpeak_secondary_function"]
        detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_name_secondary"]
        rpeak_secondary_path = create_rpeaks_pickle_path(PREPARATION_DIRECTORY, parameters["rpeak_name_secondary"])
        detect_rpeaks_args["rpeak_path"] = rpeak_secondary_path
        rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

        # combine the detected R peaks into certain and uncertain R peaks
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
    # create arguments for printing the R peak comparison report and print it
    # rpeak_comparison_report_args = create_sub_dict(parameters, rpeak_detection_comparison_report_variables)
    # rpeak_detection.rpeak_detection_comparison_report(**rpeak_comparison_report_args)

    #  # create arguments for printing the ECG validation comparison report
    # ecg_validation_report_args = create_sub_dict(parameters, ecg_validation_comparison_report_variables)
    # check_data.ecg_validation_comparison_report(**ecg_validation_report_args)

    # load the ecg thresholds to the parameters dictionary
    # ecg_validation_thresholds_dict = load_from_pickle(parameters["ecg_thresholds_save_path"])
    # parameters.update(ecg_validation_thresholds_dict)
    # del ecg_validation_thresholds_dict

    # determine_ecg_region_args = create_sub_dict(parameters, determine_ecg_region_variables)
    # check_data.determine_valid_ecg_regions(**determine_ecg_region_args)
    # del determine_ecg_region_args

    # file_name = "SL154_SL154_(1).edf"
    # file_name = "SL214_SL214_(1).edf"

    # valid_regions_dict = load_from_pickle(ADDITIONALS_DIRECTORY + "Valid_ECG_Regions.pkl")
    # sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(GIF_DATA_DIRECTORY + file_name)

    # valid_regions_ratio = check_data.valid_total_ratio(sigbufs, valid_regions_dict[file_name], parameters["ecg_key"])
    # print("(Valid / Total) Regions Ratio: %f %%" % (round(valid_regions_ratio, 4)*100))

    # total_length = len(sigbufs[parameters["ecg_key"]])
    # x_lim = [int(0*total_length), int(0.5*total_length)]

    # plot_helper.plot_valid_regions(
    #     sigbufs, 
    #     valid_regions_dict[file_name], 
    #     parameters["ecg_key"], 
    #     xlim = x_lim
    #     )

    additional_section(parameters["run_additionals_section"])
    # preparation_section(parameters["run_preparation_section"])

    # rpeaks = load_from_pickle(PREPARATION_DIRECTORY + "RPeaks_wfdb.pkl")
    # print(rpeaks)

if __name__ == "__main__":
    main()