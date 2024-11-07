"""
Author: Johannes Peter Knoll

Main python file for Processing EDF Data.
"""

# IMPORTS
import numpy as np
import os

# LOCAL IMPORTS
import read_edf
import MAD
import rpeak_detection
import check_data
import rri_from_rpeak
from side_functions import *

"""
-------------------
PROJECT PARAMETERS
-------------------

Most of the parameters are stored in the project_parameters.py file. They don't change any major functionality
of the code and can be ignored. 

The parameters below are the important ones and are therefore stored in this file.
"""

from project_parameters import *

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
    "use_strictness": 0.6 # (PARAMETER FOR DIFFERENT FUNCTION - FITS TO THIS CATEGORY): If None, the ecg regions corresponding to the strictness must be chosen manually by user input. If a float, this strictness will be used.
}

# parameters for the r-peak detection
detect_rpeaks_params = {
    "rpeak_functions": [rpeak_detection.get_rpeaks_ecgdetectors, rpeak_detection.get_rpeaks_hamilton], # r-peak detection functions
    "rpeak_function_names": ["ecgdetectors", "hamilton"], # names of all used r-peak functions
    # "rpeak_functions": [rpeak_detection.get_rpeaks_wfdb, rpeak_detection.get_rpeaks_ecgdetectors, rpeak_detection.get_rpeaks_hamilton, rpeak_detection.get_rpeaks_christov], # r-peak detection functions
    # "rpeak_function_names": ["wfdb", "ecgdetectors", "hamilton", "christov"], # names of all used r-peak functions
    "rpeak_distance_threshold_seconds": 0.05, # If r-peaks in the two functions differ by this value, they are still considered the same (max 50ms)
}

# parameters for the MAD calculation
calculate_MAD_params = {
    "mad_time_period_seconds": 1, # time period in seconds over which the MAD will be calculated
}

# parameters for calculating the RRI from the r-peaks
calculate_rri_from_peaks_params = {
    "RRI_sampling_frequency": 4, # target sampling frequency of the RR-intervals
    "pad_with": 0, # value to add if RRI at required time point is not calculatable
}

# add all parameters to the parameters dictionary, so we can access them later more easily
parameters.update(valid_ecg_regions_params)
parameters.update(detect_rpeaks_params)
parameters.update(calculate_MAD_params)
parameters.update(calculate_rri_from_peaks_params)

# delete the dictionaries as they are saved in the parameters dictionary now
del valid_ecg_regions_params, detect_rpeaks_params, calculate_MAD_params, calculate_rri_from_peaks_params


"""
--------------------------
PROCESSING DATA FUNCTIONS
--------------------------

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
    ---------------------------
    SET DATA AND STORAGE PATHS
    ---------------------------
    """

    # create needed directory if it does not exist
    create_directories_along_path(GIF_RESULTS_DIRECTORY)

    # set path to where ECG is stored
    parameters["data_directory"] = GIF_DATA_DIRECTORY

    # set path to pickle file that saves the results
    parameters["results_path"] = GIF_RESULTS_DIRECTORY + GIF_RESULTS_FILE_NAME

    # check if previous computation was interrupted:

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(GIF_RESULTS_DIRECTORY + GIF_RESULTS_FILE_NAME) + "computation_in_progress.pkl"

    # ask the user if the results should be overwritten or recovered
    if os.path.isfile(temporary_file_path):
        recover_results_after_error(
            all_results_path = GIF_RESULTS_DIRECTORY + GIF_RESULTS_FILE_NAME, 
            some_results_with_updated_keys_path = temporary_file_path, 
            file_name_dictionary_key = parameters["file_name_dictionary_key"],
        )
    
    del temporary_file_path

    """
    ---------------
    ECG VALIDATION
    ---------------
    """
    
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
    ------------------------
    COMPARE ECG VALIDATIONS
    ------------------------
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
    -----------------
    R-PEAK DETECTION
    -----------------
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

    """
    --------------------------
    COMPARE R-PEAK DETECTIONS
    --------------------------
    """

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
    ---------------------------
    CALCULATE RRI FROM R-PEAKS
    ---------------------------
    """

    parameters["rpeak_function_name"] = parameters["rpeak_comparison_function_names"][-1]
    calculate_rri_from_peaks_args = create_sub_dict(parameters, calculate_rri_from_peaks_variables)
    rri_from_rpeak.determine_rri_from_rpeaks(**calculate_rri_from_peaks_args)

    """
    ----------------
    MAD CALCULATION
    ----------------
    """

    # calculate MAD in the wrist acceleration data
    calculate_MAD_args = create_sub_dict(parameters, calculate_MAD_variables)
    MAD.calculate_MAD_in_acceleration_data(**calculate_MAD_args)
    del calculate_MAD_args
        

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
        - perform r-peak detection and calculate the RRI from the r-peaks
        - calculate the MAD in the wrist acceleration data.
    """
    
    # create directory if it does not exist
    create_directories_along_path(NAKO_RESULTS_DIRECTORY)

    for DATA_DIRECTORY in NAKO_DATA_DIRECTORIES:
        """
        ---------------------------
        SET DATA AND STORAGE PATHS
        ---------------------------
        """

        # set path to where ECG is stored
        parameters["data_directory"] = DATA_DIRECTORY

        # set path to pickle file that saves the processing results
        SAVE_DIRECTORY = NAKO_RESULTS_DIRECTORY + create_save_path_from_directory_name(DATA_DIRECTORY)
        create_directories_along_path(SAVE_DIRECTORY)
        parameters["results_path"] = SAVE_DIRECTORY + NAKO_RESULTS_FILE_NAME

        # check if previous computation was interrupted:

        # path to pickle file which will store results
        temporary_file_path = get_path_without_filename(SAVE_DIRECTORY + NAKO_RESULTS_FILE_NAME) + "computation_in_progress.pkl"

        # ask the user if the results should be overwritten or recovered
        if os.path.isfile(temporary_file_path):
            recover_results_after_error(
                all_results_path = SAVE_DIRECTORY + NAKO_RESULTS_FILE_NAME, 
                some_results_with_updated_keys_path = temporary_file_path, 
                file_name_dictionary_key = parameters["file_name_dictionary_key"],
            )
        
        del temporary_file_path

        """
        ----------------------
        ECG REGION VALIDATION
        ----------------------
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
        -----------------
        R-PEAK DETECTION
        -----------------
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
        ---------------------------
        CALCULATE RRI FROM R-PEAKS
        ---------------------------
        """
        
        parameters["rpeak_function_name"] = "hamilton"
        calculate_rri_from_peaks_args = create_sub_dict(parameters, calculate_rri_from_peaks_variables)
        rri_from_rpeak.determine_rri_from_rpeaks(**calculate_rri_from_peaks_args)
    
        """
        ----------------
        MAD CALCULATION
        ----------------
        """

        # calculate MAD in the wrist acceleration data
        calculate_MAD_args = create_sub_dict(parameters, calculate_MAD_variables)
        MAD.calculate_MAD_in_acceleration_data(**calculate_MAD_args)
        del calculate_MAD_args


"""
-------------
MAIN SECTION
-------------

In this section we will run the functions we have created until now.
"""

if __name__ == "__main__":
    
    # process GIF data
    Processing_GIF(
        GIF_DATA_DIRECTORY = "Data/GIF/SOMNOwatch/",
        GIF_RPEAK_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/RRI/",
        GIF_ECG_CLASSIFICATION_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/Noise/",
        GIF_RESULTS_DIRECTORY = "Processed_GIF_1/",
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
    Processing_NAKO(
        NAKO_DATA_DIRECTORIES = ["Data/GIF/SOMNOwatch/"],
        NAKO_RESULTS_DIRECTORY = "Processed_NAKO_1/",
        NAKO_RESULTS_FILE_NAME = "NAKO_Results.pkl"
    )