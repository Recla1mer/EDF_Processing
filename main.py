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
# ----------------------------------------------------------------------------

HEAD_DATA_DIRECTORY = "Data/" # head directory which should be searched for every subdirectory containing valid data files, only needed if automatic search for DATA_DIRECTORIES is used
DATA_DIRECTORIES = ["Data/", "Data/GIF/SOMNOwatch/"] # manually chosen directories that contain the data files

# TEMPORARY_PICKLE_DIRECTORY = "Temporary_Pickles/"
# TEMPORARY_FIGURE_DIRECTORY = "Temporary_Figures/"

# paths for PREPARATION SECTION
# ------------------------------
PREPARATION_DIRECTORY = "Preparation/"
PREPARATION_RESULTS_NAME = "Preparation_Results.pkl"

# paths for ADDITIONALS SECTION
# ------------------------------
ADDITIONALS_DIRECTORY = "Additions/"
ADDITIONS_RESULTS_PATH = ADDITIONALS_DIRECTORY + "Additions_Results.pkl"

ADDITIONS_RAW_DATA_DIRECTORY = "Data/GIF/SOMNOwatch/"

# Show Calibration Data
SHOW_CALIBRATION_DATA_DIRECTORY = ADDITIONALS_DIRECTORY + "Show_Calibration_Data/"

# R-peak detection comparison
RPEAK_COMPARISON_DIRECTORY = ADDITIONALS_DIRECTORY + "RPeak_Comparison/"
RPEAK_COMPARISON_REPORT_PATH = RPEAK_COMPARISON_DIRECTORY + "RPeak_Comparison_Report.txt"
RPEAK_CLASSIFICATION_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/RRI/"

# ECG Validation comparison
ECG_VALIDATION_COMPARISON_DIRECTORY = ADDITIONALS_DIRECTORY + "ECG_Validation_Comparison/"
ECG_VALIDATION_COMPARISON_REPORT_PATH = ECG_VALIDATION_COMPARISON_DIRECTORY + "ECG_Validation_Comparison_Report.txt"
ECG_CLASSIFICATION_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/Noise/"

"""
--------------------------------
PHYSICAL DIMENSION SECTION
--------------------------------

Of course computers only operate with numbers, so we need to make sure that the physical
dimensions of the data is consistent. This is especially important when we are working with
different data sources. In this section we will define a dictionary that is used to correct
the physical dimensions of the data.

We will store every possible label of the signals as keys in the dictionary. The values will
also be dictionaries in the following format:
    {
        "possible_dimensions": [list of possible physical dimensions],
        "dimension_correction": [list of values that will be multiplied to the data if it has the corresponding physical dimension]
    }
"""

voltage_dimensions = ["uV", "mV"]
voltage_correction = [1, 1e3]

force_dimensions = ["mg"]
force_correction = [1]

physical_dimension_correction_dictionary = {
    "ECG": {"possible_dimensions": voltage_dimensions, "dimension_correction": voltage_correction},
    "X": {"possible_dimensions": force_dimensions, "dimension_correction": force_correction},
    "Y": {"possible_dimensions": force_dimensions, "dimension_correction": force_correction},
    "Z": {"possible_dimensions": force_dimensions, "dimension_correction": force_correction}
}

# delete variables not needed anymore
del voltage_dimensions, voltage_correction, force_dimensions, force_correction


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

# data source settings
data_source_settings = {
    # set if DATA_DIRECTORIES should be searched automatically or if the manually chosen ones are used
    "use_manually_chosen_data_directories": True, # if True, the manually chosen directories will be used, if False, the DATA_DIRECTORIES will be evaluated automatically from the HEAD_DATA_DIRECTORY
}

# dictionary that will store all parameters that are used within the project
parameters = dict()

# main parameters: control which sections of the project will be executed
settings_params = {
    # set what sections should be executed
    "run_additionals_section": True, # if True, the ADDITIONALS SECTION will be executed
    "run_preparation_section": True, # if True, the PREPARATION SECTION will be executed
    # set what parts of the ADDITIONALS SECTION should be executed
    "show_calibration_data": False, # if True, the calibration data in the manually chosen intervals will be plotted and saved to the SHOW_CALIBRATION_DATA_DIRECTORY
    "perform_rpeak_comparison": True, # if True, the r-peak detection functions will be compared
    "perform_ecg_validation_comparison": True, # if True, the ECG validations will be compared
    # set what parts of the PREPARATION SECTION should be executed
    "determine_valid_ecg_regions": True, # if True, you will have the option to recalculate the valid regions for the ECG data
    "detect_rpeaks": True, # if True, the r-peaks will be detected in the ECG data
    "calculate_MAD": True, # if True, the MAD will be calculated for the wrist acceleration data
}

# file parameters:
file_params = {
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_keys": ["ECG"], # possible labels for the ECG data in the data files
    "wrist_acceleration_keys": [["X"], ["Y"], ["Z"]], # possible labels for the wrist acceleration data in the data files
    "physical_dimension_correction_dictionary": physical_dimension_correction_dictionary, # dictionary to correct the physical dimensions of the data
}

# parameters for the PREPARATION SECTION
# --------------------------------------

preparation_results_dictionary_key_params = {
    "file_name_dictionary_key": "file_name", # key that accesses the file name in the dictionaries
    "valid_ecg_regions_dictionary_key": "valid_ecg_regions", # key that accesses the valid ecg regions in the dictionaries
    # dictionary key that accesses r-peaks of certain method: r-peak function name
    "certain_rpeaks_dictionary_key": "certain_rpeaks", # key that accesses the certain r-peaks
    "uncertain_primary_rpeaks_dictionary_key": "uncertain_primary_rpeaks", # key that accesses the uncertain primary r-peaks
    "uncertain_secondary_rpeaks_dictionary_key": "uncertain_secondary_rpeaks", # key that accesses the uncertain secondary r-peaks
    "MAD_dictionary_key": "MAD", # key that accesses the MAD values
}

# parameters for the ECG Validation
valid_ecg_regions_params = {
    "straighten_ecg_signal": True, # if True, the ECG signal will be straightened before the validation (see check_data.straighten_ecg_signal() for more information)
    "check_ecg_time_interval_seconds": 5, # time interval considered when determining the valid regions for the ECG data (as small as possible, but should contain at least two R-peaks)
    "check_ecg_overlapping_interval_steps": 1, # number of times the interval needs to be shifted to the right until the next check_ecg_time_interval_seconds is reached (only useful to increase if check_ecg_time_interval_seconds is small)
    "check_ecg_validation_strictness": [round(strict_val, 2) for strict_val in np.arange(0.0, 1.05, 0.05)], # strictness in relation to mean values (0.0: very unstrict, 1.0: very strict)
    "check_ecg_removed_peak_difference_threshold": 0.3, # difference between the values of std-max-min-difference before and after removing the highest peak must be below this value (difference usually around 0.03)
    "check_ecg_std_min_threshold": 20.0, # if the standard deviation of the ECG data is below this threshold, the data is considered invalid (this is a manual threshold, it is used if the ratio between valid and total regions is below 0.5 after trying it with validation_strictness and the mean values)
    "check_ecg_std_max_threshold": 800.0, # if the standard deviation of the ECG data is above this threshold, the data is considered invalid (this is a manual threshold, see above)
    "check_ecg_distance_std_ratio_threshold": 5.0, # if the ratio of the max-min difference and the standard deviation of the ECG data is below this threshold, the data is considered invalid
    "check_ecg_min_valid_length_minutes": 5, # minimum length of valid data in minutes
    "check_ecg_allowed_invalid_region_length_seconds": 30, # data region (see directly above) still considered valid if the invalid part is shorter than this
}

# parameters for the r-peak detection
detect_rpeaks_params = {
    "rpeak_functions": [rpeak_detection.get_rpeaks_wfdb, rpeak_detection.get_rpeaks_ecgdetectors, rpeak_detection.get_rpeaks_hamilton, rpeak_detection.get_rpeaks_christov], # r-peak detection functions
    "rpeak_function_names": ["wfdb", "ecgdetectors", "hamilton", "christov"], # names of all used r-peak functions
    "rpeak_primary_function_name": "wfdb", # name of the primary r-peak detection function
    "rpeak_secondary_function_name": "ecgdetectors", # name of the secondary r-peak detection function
    "rpeak_distance_threshold_seconds": 0.05, # If r-peaks in the two functions differ by this value, they are still considered the same (max 50ms)
}

# parameters for the r-peak correction
correct_rpeaks_params = {
    "before_correction_rpeak_function_name_addition": "_raw", # addition to the r-peak function name before the correction
}

# parameters for the MAD calculation
calculate_MAD_params = {
    "mad_time_period_seconds": 10, # time period in seconds over which the MAD will be calculated
}

# parameters for the ADDITIONALS SECTION
# --------------------------------------

additions_results_dictionary_key_params = {
    "additions_results_path": ADDITIONS_RESULTS_PATH, # path to pickle file that stores the results for every file as individual dictionary
    "ecg_validation_comparison_dictionary_key": "ecg_validation_comparison", # key that accesses the ECG validation comparison in the dictionaries
    "rpeak_comparison_dictionary_key": "rpeak_comparison", # key that accesses the r-peak comparison values
}

# parameters for the ECG Validation comparison
ecg_validation_comparison_params = {
    "ecg_classification_values_directory": ECG_CLASSIFICATION_DIRECTORY, # directory that stores the ECG classification values
    "ecg_classification_file_types": [".txt"], # file types that store the ECG clasification data
    "ecg_validation_comparison_report_path": ECG_VALIDATION_COMPARISON_REPORT_PATH, # path to the text file that stores the comparison report
    "ecg_validation_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
}

# parameters for the r-peak detection comparison
rpeak_comparison_params = {
    "rpeaks_values_directory": RPEAK_CLASSIFICATION_DIRECTORY, # directory where the given r-peak location and classification is stored
    "valid_rpeak_values_file_types": [".rri"], # file types that store the r-peak classification data
    "include_rpeak_value_classifications": ["N"], # classifications that should be included in the evaluation
    #
    "rpeak_comparison_functions": [rpeak_detection.get_rpeaks_wfdb, rpeak_detection.get_rpeaks_ecgdetectors, rpeak_detection.get_rpeaks_hamilton, rpeak_detection.get_rpeaks_christov], # r-peak detection functions
    "rpeak_classification_function": rpeak_detection.read_rpeaks_from_rri_files, # functions to read the r-peak classifications
    "add_offset_to_classification": -1, #  offset that should be added to the r-peaks (classifications are slightly shifted for some reason)
    #
    "rpeak_comparison_function_names": ["wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification"], # names of all used r-peak functions
    "rpeak_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
    "rpeak_comparison_report_path": RPEAK_COMPARISON_REPORT_PATH, # path to the text file that stores the comparison report
}

# automatically find data directories if user requested it
if not data_source_settings["use_manually_chosen_data_directories"]:
    DATA_DIRECTORIES = retrieve_all_subdirectories_containing_valid_files(
        directory = HEAD_DATA_DIRECTORY, 
        valid_file_types = file_params["valid_file_types"]
    )

# delete variables not needed anymore
del data_source_settings, HEAD_DATA_DIRECTORY, RPEAK_COMPARISON_REPORT_PATH, RPEAK_CLASSIFICATION_DIRECTORY, ECG_VALIDATION_COMPARISON_REPORT_PATH, ECG_CLASSIFICATION_DIRECTORY, physical_dimension_correction_dictionary

# add all parameters to the parameters dictionary, so we can access them later more easily
parameters.update(settings_params)
parameters.update(file_params)
parameters.update(preparation_results_dictionary_key_params)
parameters.update(valid_ecg_regions_params)
parameters.update(detect_rpeaks_params)
parameters.update(correct_rpeaks_params)
parameters.update(calculate_MAD_params)

if not __name__ == "__main__" or settings_params["run_additionals_section"]:
    parameters.update(additions_results_dictionary_key_params)
    parameters.update(ecg_validation_comparison_params)
    parameters.update(rpeak_comparison_params)

# delete the dictionaries as they are saved in the parameters dictionary now
del settings_params, file_params, preparation_results_dictionary_key_params, valid_ecg_regions_params, detect_rpeaks_params, correct_rpeaks_params, calculate_MAD_params, additions_results_dictionary_key_params, ecg_validation_comparison_params, rpeak_comparison_params

# check the parameters:
# =====================

validate_parameter_settings(parameters)

# create lists of parameters relevant for the following functions (to make the code more readable)

# list for the PREPARATION SECTION
# --------------------------------

determine_ecg_region_variables = ["data_directory", "valid_file_types", "ecg_keys", 
    "physical_dimension_correction_dictionary",
    "preparation_results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key", 
    "straighten_ecg_signal", "check_ecg_time_interval_seconds", "check_ecg_overlapping_interval_steps",
    "check_ecg_validation_strictness", "check_ecg_removed_peak_difference_threshold",
    "check_ecg_std_min_threshold", "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold",
    "check_ecg_min_valid_length_minutes", "check_ecg_allowed_invalid_region_length_seconds"]

choose_valid_ecg_regions_for_further_computation_variables = ["data_directory", "ecg_keys", 
    "preparation_results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key"]

detect_rpeaks_variables = ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary",
    "preparation_results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key"]

correct_rpeaks_variables = ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary",
    "before_correction_rpeak_function_name_addition", "preparation_results_path", "file_name_dictionary_key"]

combine_detected_rpeaks_variables = ["data_directory", "ecg_keys", "rpeak_distance_threshold_seconds",
    "rpeak_primary_function_name", "rpeak_secondary_function_name",
    "preparation_results_path", "file_name_dictionary_key", "certain_rpeaks_dictionary_key",
    "uncertain_primary_rpeaks_dictionary_key", "uncertain_secondary_rpeaks_dictionary_key"]

calculate_MAD_variables = ["data_directory", "valid_file_types", "wrist_acceleration_keys", 
    "physical_dimension_correction_dictionary", "mad_time_period_seconds",
    "preparation_results_path", "file_name_dictionary_key", "MAD_dictionary_key"]


# lists for the ADDITIONALS SECTION
# ---------------------------------

ecg_validation_comparison_variables = ["ecg_classification_values_directory", "ecg_classification_file_types", 
    "check_ecg_validation_strictness", "additions_results_path", "file_name_dictionary_key", 
    "valid_ecg_regions_dictionary_key", "ecg_validation_comparison_dictionary_key"]

ecg_validation_comparison_report_variables = ["ecg_validation_comparison_report_path", 
    "ecg_validation_comparison_report_dezimal_places", "check_ecg_validation_strictness",
    "additions_results_path", "file_name_dictionary_key", "ecg_validation_comparison_dictionary_key"]

read_rpeak_classification_variables = ["data_directory", "valid_file_types", "rpeaks_values_directory", 
    "valid_rpeak_values_file_types", "include_rpeak_value_classifications", "add_offset_to_classification",
    "additions_results_path", "file_name_dictionary_key"]

rpeak_detection_comparison_variables = ["data_directory", "ecg_keys", "rpeak_distance_threshold_seconds", 
    "additions_results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key",
    "rpeak_comparison_function_names", "rpeak_comparison_dictionary_key"]

rpeak_detection_comparison_report_variables = ["rpeak_comparison_report_dezimal_places", 
    "rpeak_comparison_report_path", "additions_results_path", "file_name_dictionary_key",
    "rpeak_comparison_function_names", "rpeak_comparison_dictionary_key"]

"""
--------------------------------
CALIBRATION DATA SECTION
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
    and compares different r-peak detections and ECG Validations.
    """

    # check if the section should be run
    if not run_section:
        return
    
    """
    --------------------------------
    SET DATA AND STORAGE PATHS
    --------------------------------
    """
    # create needed directory if it does not exist
    if not os.path.isdir(ADDITIONALS_DIRECTORY):
        os.mkdir(ADDITIONALS_DIRECTORY)

    # set path to where ECG is stored
    parameters["data_directory"] = ADDITIONS_RAW_DATA_DIRECTORY

    # set path to pickle file that saves the results from the additions
    parameters["preparation_results_path"] = ADDITIONS_RESULTS_PATH
    
    # perform ecg validation if needed
    if parameters["perform_rpeak_comparison"] or parameters["perform_ecg_validation_comparison"]:

        # create arguments for the valid ecg regions evaluation and calculate them
        determine_ecg_region_args = create_sub_dict(parameters, determine_ecg_region_variables)

        check_data.determine_valid_ecg_regions(**determine_ecg_region_args)
        del determine_ecg_region_args

        # create arguments for choosing the valid ecg regions for further computation
        choose_valid_ecg_regions_for_further_computation_args = create_sub_dict(parameters, choose_valid_ecg_regions_for_further_computation_variables)
        check_data.choose_valid_ecg_regions_for_further_computation(**choose_valid_ecg_regions_for_further_computation_args)

        del choose_valid_ecg_regions_for_further_computation_args
    
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
    COMPARE R-PEAK DETECTIONS
    --------------------------------
    """

    # compare the r-peak detection functions if user requested it
    if parameters["perform_rpeak_comparison"]:
        # create directory to save comparison results if it does not exist
        if not os.path.isdir(RPEAK_COMPARISON_DIRECTORY):
            os.mkdir(RPEAK_COMPARISON_DIRECTORY)

        # create arguments for the r-peak detection and correction
        detect_rpeaks_args = create_sub_dict(parameters, detect_rpeaks_variables)
        correct_rpeaks_args = create_sub_dict(parameters, correct_rpeaks_variables)

        # detect and correct r-peaks in the valid regions of the ECG data
        classification_index_offset = 0
        for i in range(len(parameters["rpeak_comparison_functions"])):
            classification_index_offset += 1
            detect_rpeaks_args["rpeak_function"] = parameters["rpeak_comparison_functions"][i]
            detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_comparison_function_names"][i]
            rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

            correct_rpeaks_args["rpeak_function_name"] = parameters["rpeak_comparison_function_names"][i]
            rpeak_detection.correct_rpeak_locations(**correct_rpeaks_args)

        del detect_rpeaks_args, correct_rpeaks_args

        # read r-peaks from the classification files if they are needed
        read_rpeak_classification_args = create_sub_dict(parameters, read_rpeak_classification_variables)
        read_rpeak_classification_args["rpeak_classification_dictionary_key"] = parameters["rpeak_comparison_function_names"][classification_index_offset]
        rpeak_detection.read_rpeaks_from_rri_files(**read_rpeak_classification_args)
        del read_rpeak_classification_args

        # create arguments for the r-peak comparison evaluation and perform it
        rpeak_detection_comparison_args = create_sub_dict(parameters, rpeak_detection_comparison_variables)
        rpeak_detection.rpeak_detection_comparison(**rpeak_detection_comparison_args)
        del rpeak_detection_comparison_args

        # create arguments for printing the r-peak comparison report and print it
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
various functions, evaluate the valid regions for the ECG data, perform r-peak detection
and calculate the MAD in the wrist acceleration data.
"""

def preparation_section(run_section: bool):

    # check if the section should be run
    if not run_section:
        return
    
    # create directory if it does not exist
    if not os.path.isdir(PREPARATION_DIRECTORY):
        os.mkdir(PREPARATION_DIRECTORY)

    for DATA_DIRECTORY in DATA_DIRECTORIES:
        """
        --------------------------------
        SET DATA AND STORAGE PATHS
        --------------------------------
        """

        # set path to where ECG is stored
        parameters["data_directory"] = DATA_DIRECTORY

        # set path to pickle file that saves the results from the preparations
        SAVE_DIRECTORY = PREPARATION_DIRECTORY + create_save_path_from_directory_name(DATA_DIRECTORY)
        if not os.path.isdir(SAVE_DIRECTORY):
            os.mkdir(SAVE_DIRECTORY)
        parameters["preparation_results_path"] = SAVE_DIRECTORY + PREPARATION_RESULTS_NAME

        """
        --------------------------------
        ECG REGION VALIDATION
        --------------------------------
        """

        # evaluate valid regions for the ECG data
        if parameters["determine_valid_ecg_regions"]:
            determine_ecg_region_args = create_sub_dict(parameters, determine_ecg_region_variables)
            check_data.determine_valid_ecg_regions(**determine_ecg_region_args)
            del determine_ecg_region_args

            # create arguments for choosing the valid ecg regions for further computation
            choose_valid_ecg_regions_for_further_computation_args = create_sub_dict(parameters, choose_valid_ecg_regions_for_further_computation_variables)
            check_data.choose_valid_ecg_regions_for_further_computation(**choose_valid_ecg_regions_for_further_computation_args)
            del choose_valid_ecg_regions_for_further_computation_args
    
        """
        --------------------------------
        R-PEAK DETECTION
        --------------------------------
        """

        # detect r-peaks in the valid regions of the ECG data
        if parameters["detect_rpeaks"]:
            # create arguments for the r-peak detection and correction
            detect_rpeaks_args = create_sub_dict(parameters, detect_rpeaks_variables)
            correct_rpeaks_args = create_sub_dict(parameters, correct_rpeaks_variables)

            # detect and correct r-peaks in the valid regions of the ECG data
            for i in range(len(parameters["rpeak_functions"])):
                detect_rpeaks_args["rpeak_function"] = parameters["rpeak_functions"][i]
                detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_function_names"][i]
                rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

                correct_rpeaks_args["rpeak_function_name"] = parameters["rpeak_function_names"][i]
                rpeak_detection.correct_rpeak_locations(**correct_rpeaks_args)

            # combine the detected r-peaks into certain and uncertain r-peaks
            # combine_detected_rpeaks_args = create_sub_dict(parameters, combine_detected_rpeaks_variables)
            # rpeak_detection.combine_detected_rpeaks(**combine_detected_rpeaks_args)

            del detect_rpeaks_args, correct_rpeaks_args, # combine_detected_rpeaks_args
    
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

    # run additional section
    additional_section(parameters["run_additionals_section"])
    # delete variables not needed anymore
    global ADDITIONALS_DIRECTORY, ADDITIONS_RESULTS_PATH, ADDITIONS_RAW_DATA_DIRECTORY, SHOW_CALIBRATION_DATA_DIRECTORY, RPEAK_COMPARISON_DIRECTORY, ECG_VALIDATION_COMPARISON_DIRECTORY 
    del ADDITIONALS_DIRECTORY, ADDITIONS_RESULTS_PATH, ADDITIONS_RAW_DATA_DIRECTORY, SHOW_CALIBRATION_DATA_DIRECTORY, RPEAK_COMPARISON_DIRECTORY, ECG_VALIDATION_COMPARISON_DIRECTORY
    
    # run preparation section
    preparation_section(parameters["run_preparation_section"])
    # delete variables not needed anymore
    global PREPARATION_DIRECTORY, PREPARATION_RESULTS_NAME, ECG_VALIDATION_THRESHOLDS_PATH
    del PREPARATION_DIRECTORY, PREPARATION_RESULTS_NAME

if __name__ == "__main__":
    # preparation_results_generator = load_from_pickle(ADDITIONS_RESULTS_PATH)
    # for generator in preparation_results_generator:
    #     print(generator.keys())
    #     print("\n\n")
    main()