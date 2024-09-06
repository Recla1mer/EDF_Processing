"""
Author: Johannes Peter Knoll

Main python file for Processing EDF Data.
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
import rri_from_rpeak
import plot_helper
from side_functions import *


"""
--------------------------------
PHYSICAL DIMENSION CORRECTION
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
SETTING UNIFORM PARAMETERS
--------------------------------

In this section we set the parameters for the project. This seems clumsy, but we want to keep the parameters
uniform throughout the computation. And because multiple functions share the same parameters, we will store
them in a dictionary that is passed to the functions.

Also, because we will call some functions multiple times below, this makes the code a little more readable.

I suggest not paying too much attention to these parameters anyway. They are good as they are.
"""

# dictionary that will store all parameters that are used within the project
parameters = dict()

# file parameters:
file_params = {
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_keys": ["ECG"], # possible labels for the ECG data in the data files
    "wrist_acceleration_keys": [["X"], ["Y"], ["Z"]], # possible labels for the wrist acceleration data in the data files
    "sleep_stage_keys": ["SleepStage"], # possible labels for the sleep stages in the data files
    "physical_dimension_correction_dictionary": physical_dimension_correction_dictionary, # dictionary to correct the physical dimensions of the data
}

# parameters for the PREPARATION SECTION
# --------------------------------------

results_dictionary_key_params = {
    "file_name_dictionary_key": "file_name", # key that accesses the file name in the dictionaries
    "valid_ecg_regions_dictionary_key": "valid_ecg_regions", # key that accesses the valid ecg regions in the dictionaries
    # dictionary key that accesses r-peaks of certain method: r-peak function name
    "certain_rpeaks_dictionary_key": "certain_rpeaks", # key that accesses the certain r-peaks
    "uncertain_primary_rpeaks_dictionary_key": "uncertain_primary_rpeaks", # key that accesses the uncertain primary r-peaks
    "uncertain_secondary_rpeaks_dictionary_key": "uncertain_secondary_rpeaks", # key that accesses the uncertain secondary r-peaks
    "MAD_dictionary_key": "MAD", # key that accesses the MAD values
    "RRI_dictionary_key": "RRI", # key that accesses the RR-intervals
    "SLP_dictionary_key": "SLP", # key that accesses the sleep stages
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
    "mad_time_period_seconds": 1, # time period in seconds over which the MAD will be calculated
}

# parameters for calculating the RRI from the r-peaks
calculate_rri_from_peaks_params = {
    "RRI_sampling_frequency": 4, # target sampling frequency of the RR-intervals
}

only_gif_results_dictionary_key_params = {
    "ecg_validation_comparison_dictionary_key": "ecg_validation_comparison", # key that accesses the ECG validation comparison in the dictionaries
    "ecg_classification_valid_intervals_dictionary_key": "valid_intervals_from_ecg_classification", # key that accesses the valid intervals from the ECG classification in the dictionaries, needed for r-peak detection comparison
    "rpeak_comparison_dictionary_key": "rpeak_comparison", # key that accesses the r-peak comparison values
}

# parameters for the ECG Validation comparison
ecg_validation_comparison_params = {
    "ecg_classification_file_types": [".txt"], # file types that store the ECG clasification data
    "ecg_validation_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
}

# parameters for the r-peak detection comparison
rpeak_comparison_params = {
    "valid_rpeak_values_file_types": [".rri"], # file types that store the r-peak classification data
    "include_rpeak_value_classifications": ["N"], # classifications that should be included in the evaluation
    #
    "rpeak_comparison_functions": [rpeak_detection.get_rpeaks_wfdb, rpeak_detection.get_rpeaks_ecgdetectors, rpeak_detection.get_rpeaks_hamilton, rpeak_detection.get_rpeaks_christov], # r-peak detection functions
    "add_offset_to_classification": -1, #  offset that should be added to the r-peaks from gif classification (classifications are slightly shifted for some reason)
    #
    "rpeak_comparison_function_names": ["wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification"], # names of all used r-peak functions (last one is for classification, if included)
    "rpeak_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
    "remove_peaks_outside_ecg_classification": True, # if True, r-peaks that are not in the valid ECG regions will be removed from the comparison
}

# delete variables not needed anymore
del physical_dimension_correction_dictionary

# add all parameters to the parameters dictionary, so we can access them later more easily
parameters.update(file_params)
parameters.update(results_dictionary_key_params)
parameters.update(valid_ecg_regions_params)
parameters.update(detect_rpeaks_params)
parameters.update(correct_rpeaks_params)
parameters.update(calculate_MAD_params)
parameters.update(calculate_rri_from_peaks_params)


parameters.update(only_gif_results_dictionary_key_params)
parameters.update(ecg_validation_comparison_params)
parameters.update(rpeak_comparison_params)

# delete the dictionaries as they are saved in the parameters dictionary now
del file_params, results_dictionary_key_params, valid_ecg_regions_params, detect_rpeaks_params, correct_rpeaks_params, calculate_MAD_params, calculate_rri_from_peaks_params, only_gif_results_dictionary_key_params, ecg_validation_comparison_params, rpeak_comparison_params

# check the parameters:
# =====================

validate_parameter_settings(parameters)

# create lists of parameters relevant for the following functions (to make the code more readable)

determine_ecg_region_variables = ["data_directory", "valid_file_types", "ecg_keys", 
    "physical_dimension_correction_dictionary",
    "results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key", 
    "straighten_ecg_signal", "check_ecg_time_interval_seconds", "check_ecg_overlapping_interval_steps",
    "check_ecg_validation_strictness", "check_ecg_removed_peak_difference_threshold",
    "check_ecg_std_min_threshold", "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold",
    "check_ecg_min_valid_length_minutes", "check_ecg_allowed_invalid_region_length_seconds"]

choose_valid_ecg_regions_for_further_computation_variables = ["data_directory", "ecg_keys", 
    "results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key"]

detect_rpeaks_variables = ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary",
    "results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key"]

correct_rpeaks_variables = ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary",
    "before_correction_rpeak_function_name_addition", "results_path", "file_name_dictionary_key"]

combine_detected_rpeaks_variables = ["data_directory", "ecg_keys", "rpeak_distance_threshold_seconds",
    "rpeak_primary_function_name", "rpeak_secondary_function_name",
    "results_path", "file_name_dictionary_key", "certain_rpeaks_dictionary_key",
    "uncertain_primary_rpeaks_dictionary_key", "uncertain_secondary_rpeaks_dictionary_key"]

calculate_MAD_variables = ["data_directory", "valid_file_types", "wrist_acceleration_keys", 
    "physical_dimension_correction_dictionary", "mad_time_period_seconds",
    "results_path", "file_name_dictionary_key", "MAD_dictionary_key"]

ecg_validation_comparison_variables = ["ecg_classification_values_directory", "ecg_classification_file_types", 
    "check_ecg_validation_strictness", "results_path", "file_name_dictionary_key", 
    "valid_ecg_regions_dictionary_key", "ecg_validation_comparison_dictionary_key",
    "ecg_classification_valid_intervals_dictionary_key"]

ecg_validation_comparison_report_variables = ["ecg_validation_comparison_report_path", 
    "ecg_validation_comparison_report_dezimal_places", "check_ecg_validation_strictness",
    "results_path", "file_name_dictionary_key", "ecg_validation_comparison_dictionary_key"]

read_rpeak_classification_variables = ["data_directory", "valid_file_types", "rpeaks_values_directory", 
    "valid_rpeak_values_file_types", "include_rpeak_value_classifications", "add_offset_to_classification",
    "results_path", "file_name_dictionary_key"]

rpeak_detection_comparison_variables = ["data_directory", "ecg_keys", "rpeak_distance_threshold_seconds", 
    "results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key",
    "rpeak_comparison_function_names", "rpeak_comparison_dictionary_key",
    "ecg_classification_valid_intervals_dictionary_key", "remove_peaks_outside_ecg_classification"]

rpeak_detection_comparison_report_variables = ["rpeak_comparison_report_dezimal_places", 
    "rpeak_comparison_report_path", "results_path", "file_name_dictionary_key",
    "rpeak_comparison_function_names", "rpeak_comparison_dictionary_key"]

read_out_channel_variables = ["data_directory", "valid_file_types", "channel_key_to_read_out",
    "physical_dimension_correction_dictionary", "results_path", "file_name_dictionary_key", "new_dictionary_key"]

calculate_rri_from_peaks_variables = ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary",
    "rpeak_function_name", "RRI_sampling_frequency", "results_path", "file_name_dictionary_key", "RRI_dictionary_key"]


"""
--------------------------------
PROCESSING DATA FUNCTIONS
--------------------------------

The following functions will call all functions within this project in the right order.
"""


def Processing_GIF(
        GIF_DATA_DIRECTORY: str,
        GIF_RPEAK_DIRECTORY: str,
        GIF_ECG_CLASSIFICATION_DIRECTORY: str,
        GIF_RESULTS_DIRECTORY: str,
        GIF_RESULTS_FILE_NAME: str,
        ECG_COMPARISON_FILE_NAME: str,
        RPEAK_COMPARISON_FILE_NAME: str,
):
    """
    This function is supposed to run all processing functions in the right order on the GIF data.

    It will:
        - evaluate the valid regions for the ECG data
        - perform r-peak detection
        - calculate the MAD in the wrist acceleration data.
        - read out sleep stages and its sampling frequency
        - read out already provided r-peak locations
        - compare already provided ECG classification with the calculated ECG validation
        - compare all differently obtained r-peak locations
        - calculate RRI from already provided r-peak locations

    Here we compare different r-peak detections and ECG Validations, as for the GIF data we already
    have the "ground truth" data available.

    We will also use GIF data to train our neural network, so we will additionally append the sleep stages.
    """

    """
    --------------------------------
    SET DATA AND STORAGE PATHS
    --------------------------------
    """
    # create needed directory if it does not exist
    if not os.path.isdir(GIF_RESULTS_DIRECTORY):
        os.mkdir(GIF_RESULTS_DIRECTORY)

    # set path to where ECG is stored
    parameters["data_directory"] = GIF_DATA_DIRECTORY

    # set path to pickle file that saves the results from the additions
    parameters["results_path"] = GIF_RESULTS_DIRECTORY + GIF_RESULTS_FILE_NAME
    
    # create arguments for the valid ecg regions evaluation and calculate them
    determine_ecg_region_args = create_sub_dict(parameters, determine_ecg_region_variables)

    # perform ecg validation
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
    parameters["ecg_classification_values_directory"] = GIF_ECG_CLASSIFICATION_DIRECTORY
    parameters["ecg_validation_comparison_report_path"] = GIF_RESULTS_DIRECTORY + ECG_COMPARISON_FILE_NAME

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

    # create arguments for the r-peak detection and correction
    detect_rpeaks_args = create_sub_dict(parameters, detect_rpeaks_variables)
    correct_rpeaks_args = create_sub_dict(parameters, correct_rpeaks_variables)

    # detect and correct r-peaks in the valid regions of the ECG data
    for i in range(len(parameters["rpeak_comparison_functions"])):
        detect_rpeaks_args["rpeak_function"] = parameters["rpeak_comparison_functions"][i]
        detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_comparison_function_names"][i]
        rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

        correct_rpeaks_args["rpeak_function_name"] = parameters["rpeak_comparison_function_names"][i]
        rpeak_detection.correct_rpeak_locations(**correct_rpeaks_args)

    del detect_rpeaks_args, correct_rpeaks_args

    # create arguments for the r-peak comparison
    rpeak_detection_comparison_args = create_sub_dict(parameters, rpeak_detection_comparison_variables)

    # read r-peaks from the classification files
    parameters["rpeaks_values_directory"] = GIF_RPEAK_DIRECTORY
    read_rpeak_classification_args = create_sub_dict(parameters, read_rpeak_classification_variables)
    read_rpeak_classification_args["rpeak_classification_dictionary_key"] = parameters["rpeak_comparison_function_names"][-1]
    rpeak_detection.read_rpeaks_from_rri_files(**read_rpeak_classification_args)
    del read_rpeak_classification_args

    # perform r-peak comparison evaluation
    rpeak_detection.rpeak_detection_comparison(**rpeak_detection_comparison_args)
    del rpeak_detection_comparison_args

    # create arguments for printing the r-peak comparison report and print it
    parameters["rpeak_comparison_report_path"] = GIF_RESULTS_DIRECTORY + RPEAK_COMPARISON_FILE_NAME
    rpeak_comparison_report_args = create_sub_dict(parameters, rpeak_detection_comparison_report_variables)
    rpeak_detection.rpeak_detection_comparison_report(**rpeak_comparison_report_args)

    """
    --------------------------------
    CALCULATE RRI FROM R-PEAKS
    --------------------------------
    """
    parameters["rpeak_function_name"] = parameters["rpeak_comparison_function_names"][-1]
    calculate_rri_from_peaks_args = create_sub_dict(parameters, calculate_rri_from_peaks_variables)
    rri_from_rpeak.determine_rri_from_rpeaks(**calculate_rri_from_peaks_args)

    """
    --------------------------------
    MAD CALCULATION
    --------------------------------
    """

    # calculate MAD in the wrist acceleration data
    calculate_MAD_args = create_sub_dict(parameters, calculate_MAD_variables)
    MAD.calculate_MAD_in_acceleration_data(**calculate_MAD_args)
    del calculate_MAD_args

    """
    --------------------------------
    Obtain Sleep Stages
    --------------------------------
    """
    parameters["new_dictionary_key"] = parameters["SLP_dictionary_key"]
    parameters["channel_key_to_read_out"] = parameters["sleep_stage_keys"]
    read_out_channel_args = create_sub_dict(parameters, read_out_channel_variables)
    read_edf.read_out_channel(**read_out_channel_args)
        

def Processing_NAKO(
        NAKO_DATA_DIRECTORIES: list,
        NAKO_RESULTS_DIRECTORY: str,
        NAKO_RESULTS_FILE_NAME: str,
):
    """
    This function is supposed to run the processing functions in the right order on the NAKO data.
    We can not apply the same functions as for the GIF data, because the GIF data additionally 
    provides r-peak locations, ECG classifications and sleep stages.

    This function will:
        - evaluate the valid regions for the ECG data
        - perform r-peak detection
        - calculate the MAD in the wrist acceleration data.
    """
    
    # create directory if it does not exist
    if not os.path.isdir(NAKO_RESULTS_DIRECTORY):
        os.mkdir(NAKO_RESULTS_DIRECTORY)

    for DATA_DIRECTORY in NAKO_DATA_DIRECTORIES:
        """
        --------------------------------
        SET DATA AND STORAGE PATHS
        --------------------------------
        """

        # set path to where ECG is stored
        parameters["data_directory"] = DATA_DIRECTORY

        # set path to pickle file that saves the processing results
        SAVE_DIRECTORY = NAKO_RESULTS_DIRECTORY + create_save_path_from_directory_name(DATA_DIRECTORY)
        if not os.path.isdir(SAVE_DIRECTORY):
            os.mkdir(SAVE_DIRECTORY)
        parameters["results_path"] = SAVE_DIRECTORY + NAKO_RESULTS_FILE_NAME

        """
        --------------------------------
        ECG REGION VALIDATION
        --------------------------------
        """

        # evaluate valid regions for the ECG data
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
        CALCULATE RRI FROM R-PEAKS
        --------------------------------
        """
        parameters["rpeak_function_name"] = parameters["rpeak_function_names"][0]
        calculate_rri_from_peaks_args = create_sub_dict(parameters, calculate_rri_from_peaks_variables)
        rri_from_rpeak.determine_rri_from_rpeaks(**calculate_rri_from_peaks_args)
    
        """
        --------------------------------
        MAD CALCULATION
        --------------------------------
        """

        # calculate MAD in the wrist acceleration data
        calculate_MAD_args = create_sub_dict(parameters, calculate_MAD_variables)
        MAD.calculate_MAD_in_acceleration_data(**calculate_MAD_args)
        del calculate_MAD_args


"""
--------------------------------
MAIN SECTION
--------------------------------

In this section we will run the functions we have created until now.
"""
relevant_keys = ["file_name", "RRI", "RRI_frequency", "MAD", "MAD_frequency", "SLP"]
relevant_keys = ["gif_classification", "RRI"]

results_generator = load_from_pickle("Processed_GIF/GIF_Results.pkl")
for generator_entry in results_generator:
    print(generator_entry["gif_classification"][:20])
    print(generator_entry["RRI"][11248:11248+20])
    break

raise SystemExit

if __name__ == "__main__":
    
    # process GIF data
    Processing_GIF(
        GIF_DATA_DIRECTORY = "Data/GIF/SOMNOwatch/",
        GIF_RPEAK_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/RRI/",
        GIF_ECG_CLASSIFICATION_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/Noise/",
        GIF_RESULTS_DIRECTORY = "Processed_GIF/",
        GIF_RESULTS_FILE_NAME = "GIF_Results.pkl",
        RPEAK_COMPARISON_FILE_NAME = "RPeak_Comparison_Report.txt",
        ECG_COMPARISON_FILE_NAME = "ECG_Validation_Comparison_Report.txt"
    )

    # if you want to retrieve all subdirectories containing valid files, you can use the following function
    """
    DATA_DIRECTORIES = retrieve_all_subdirectories_containing_valid_files(
        directory = "Data/", 
        valid_file_types = [".edf"]
    )
    """

    # process NAKO data
    """
    Processing_NAKO(
        NAKO_DATA_DIRECTORIES = ["Data/", "Data/GIF/SOMNOwatch/"],
        NAKO_RESULTS_DIRECTORY = "Processed_NAKO/",
        NAKO_RESULTS_FILE_NAME = "NAKO_Results.pkl"
    )
    """