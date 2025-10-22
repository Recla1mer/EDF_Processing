"""
Author: Johannes Peter Knoll

Main python file for Processing EDF Data.
"""

# IMPORTS
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import random

# LOCAL IMPORTS
import read_edf
import MAD
import rpeak_detection
import check_data
import rri_from_rpeak
import data_retrieval
from side_functions import *

"""
-------------------
PROJECT PARAMETERS
-------------------

Most of the parameters are stored in the project_parameters.py file. They don't change any major functionality
of the code and can be ignored. 

However, it is important to now what key (string) accesses the ECG and wrist accelerometry data in your 
.edf files. These need to be added to the project_parameters.py file, see file_params and 
physical_dimension_correction_dictionary dictionaries. Just add them analogously to the existing ones, if they
differ.

The parameters below are the important ones and are therefore stored in this file.
"""

from project_parameters import *

# parameters for the ECG Validation
valid_ecg_regions_params = {
    "straighten_ecg_signal": True, # if True, the ECG signal will be straightened before the validation (see check_data.straighten_ecg_signal() for more information)
    "use_ecg_validation_strictness": 0.6, # If None, the ecg regions corresponding to the strictness must be chosen manually by user input. If a float, this strictness will be used.
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
    "realistic_rri_value_range": [0.24, 2.4], # realistic range of RRI values, equals 25-250 bpm
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

The following functions will call all functions within this project in the designed order to ensure
a smooth processing of the data.
"""


def Data_Processing(
        DATA_DIRECTORIES: list,
        RESULTS_DIRECTORY: str,
    ):
    """
    This function is supposed to run all processing and comparing functions in the designed order.

    Order of Execution:
        - ECG Validation: Evaluate where ECG data was recorded correctly to determine evaluatable segments
        - R-peak Detection: Detect R-peak locations the valid segments of the ECG data using specified detectors
        - R-peak Height Retrieval: Calculate a list of values for every R-peak needed to determine the height of the detected R-peaks
        - RRI Calculation: Calculate RR-Intervals from detected R-peak locations
        - MAD Calculation: Calculate Mean Amplitude Deviation (values characterizing motion activity) using wrist accelerometry data
        - Header Information Extraction: Extract essential metadata stored in the .edf file header.
    
    ATTENTION:
    --------------------------------
    For every path to a data directory in DATA_DIRECTORIES, the function will create a new file in the 
    RESULTS_DIRECTORY. It will name this file like the last directory in the path (to the data directory). 
    SO MAKE SURE THEY ARE UNIQUE!
    The algorithm won't be able to distinguish if you made a mistake here or if you want to reprocess the data.
    Example:    DATA_DIRECTORIES = ["Data/Directory_1/", "Data/Directory_2/"] is valid, but
                DATA_DIRECTORIES = ["Data_1/Directory/", "Data_2/Directory/"] is not valid
    
    Before running this function check the file parameters in the 'SETTING UNIFORM PARAMETERS' section in the 
    project_parameters.py file. There you must set what keys access the ECG and wrist accelerometry data
    in your .edf files. If your data uses different keys across files, add them all to ensure they can be accessed.
    Also check if the dimension correction contains all strings to physical dimensions that are used in your 
    .edf files and provide a correction value, that transforms these signals into the physical dimension that 
    was used by us (dimension correction factor = 1).
    
    ARGUMENTS:
    --------------------------------
    DATA_DIRECTORIES: list
        List of paths to the directories where the .edf-files containing the data are located.
    RESULTS_DIRECTORY: str
        Path to the directory where the results should be stored.

    RETURNS:
    --------------------------------
    None, but the results will be stored in the specified directory.

    RESULTS:
    --------------------------------
    Every results (.pkl) file will contain multiple dictionaries. Each dictionary is structured as follows:
    {
        "file_name":     
                Name of the (.edf) file the results are calculated for,

        "valid_ecg_regions_strictness-value":   
                List of valid regions ([[start_index_1, end_index_1], [start_index_2, end_index_2], ...]) in 
                the ECG data for the specified strictness-value. You will have multiple of these entries for 
                every value in parameters["check_ecg_validation_strictness"].

        "valid_ecg_regions": 
                List of valid regions in the ECG data, that is used during R-peak detection,
        
        "rpeak-function-name" + "_raw":
                List of R-peak locations detected by the rpeak-function-name function. You will have multiple 
                of these entries for every R-peak detection function in parameters["rpeak_function_names"].
        
        "rpeak-function-name":
                List of R-peak locations detected by the rpeak-function-name function AFTER CORRECTION. You 
                will have multiple of these entries for every R-peak detection function in parameters["rpeak_function_names"].
        
        "rpeak-function-name" + "_ecg_values":
                List of lists for every R-peak containing values needed to calculate the heights for the
                rpeak locations detected by the rpeak-function-name function. You will have multiple of these 
                entries for every R-peak detection function in parameters["rpeak_function_names"].
        
        "RRI":
                List of RR-intervals calculated from the R-peak locations.
        
        "RRI_frequency":
                Sampling frequency of the RR-intervals.
        
        "MAD":
                List of Mean Amplitude Deviation values calculated from the wrist acceleration data.
        
        "MAD_frequency":
                Sampling frequency of the MAD values. Corresponds to 1 / parameters["mad_time_period_seconds"].
        
        "ECG_frequency":
                Sampling frequency of the ECG data.
        
        "start_date":
                The start date of the EDF file, formatted as "YYYY-MM-DD."
        
        "start_time":
                The start time of the EDF file, formatted as "HH:MM:SS"
    }

    Note: In the project_parameters.py file you can alter the names of the keys in the dictionaries.
    """
    
    # create directory if it does not exist
    create_directories_along_path(RESULTS_DIRECTORY)

    for DATA_DIRECTORY in DATA_DIRECTORIES:
        """
        ---------------------------
        SET DATA AND STORAGE PATHS
        ---------------------------
        """

        # set path to where ECG is stored
        parameters["data_directory"] = DATA_DIRECTORY

        for i in range(len(DATA_DIRECTORY)-2, -1, -1):
            if DATA_DIRECTORY[i] == "/":
                i += 1
                break

        # set path to pickle file that saves the processing results
        RESULTS_PATH = RESULTS_DIRECTORY + DATA_DIRECTORY[i:-1] + "_Results.pkl"
        parameters["results_path"] = RESULTS_PATH

        """
        ----------------------------
        RECOVER RESULTS AFTER ERROR
        ----------------------------
        """

        # path to pickle file which will store results
        temporary_file_path = get_path_without_filename(RESULTS_PATH) + "computation_in_progress.pkl"

        # ask the user if the results should be overwritten or recovered
        if os.path.isfile(temporary_file_path):
            recover_results_after_error(
                all_results_path = RESULTS_PATH, 
                some_results_with_updated_keys_path = temporary_file_path,
            )
        
        del temporary_file_path

        """
        ----------------------
        ECG REGION VALIDATION
        ----------------------
        """

        # evaluate valid regions for the ECG data
        determine_ecg_region_args = create_sub_dict(parameters, ["data_directory", "valid_file_types", "ecg_keys", "physical_dimension_correction_dictionary", "results_path", "straighten_ecg_signal", "use_ecg_validation_strictness", "check_ecg_time_interval_seconds", "check_ecg_overlapping_interval_steps", "check_ecg_validation_strictness", "check_ecg_removed_peak_difference_threshold", "check_ecg_std_min_threshold", "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold", "check_ecg_min_valid_length_minutes", "check_ecg_allowed_invalid_region_length_seconds"])
        check_data.determine_valid_ecg_regions(**determine_ecg_region_args)

        # create arguments for choosing the valid ecg regions for further computation
        if determine_ecg_region_args["use_ecg_validation_strictness"] is None:
            choose_valid_ecg_regions_for_further_computation_args = create_sub_dict(parameters, ["data_directory", "ecg_keys", "results_path", "rpeak_function_names"])
            check_data.choose_valid_ecg_regions_for_further_computation(**choose_valid_ecg_regions_for_further_computation_args)
            del choose_valid_ecg_regions_for_further_computation_args
        
        del determine_ecg_region_args
    
        """
        ---------------------------------------------
        R-PEAK DETECTION and R-PEAK HEIGHT RETRIEVAL
        ---------------------------------------------
        """

        # create arguments for the r-peak detection, correction and height retrieval
        detect_rpeaks_args = create_sub_dict(parameters, ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary", "results_path"])
        correct_rpeaks_args = create_sub_dict(parameters, ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary", "results_path"])
        retrieve_rpeak_heights_args = create_sub_dict(parameters, ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary", "results_path"])

        # detect and correct r-peaks in the valid regions of the ECG data
        for i in range(len(parameters["rpeak_functions"])):
            detect_rpeaks_args["rpeak_function"] = parameters["rpeak_functions"][i]
            detect_rpeaks_args["rpeak_function_name"] = parameters["rpeak_function_names"][i]
            rpeak_detection.detect_rpeaks(**detect_rpeaks_args)

            correct_rpeaks_args["rpeak_function_name"] = parameters["rpeak_function_names"][i]
            rpeak_detection.correct_rpeak_locations(**correct_rpeaks_args)

            retrieve_rpeak_heights_args["rpeak_function_name"] = parameters["rpeak_function_names"][i]
            # rpeak_detection.determine_rpeak_heights(**retrieve_rpeak_heights_args)

        del detect_rpeaks_args, correct_rpeaks_args, retrieve_rpeak_heights_args

        """
        ---------------------------
        CALCULATE RRI FROM R-PEAKS
        ---------------------------
        """
        
        parameters["rpeak_function_name"] = "hamilton"
        calculate_rri_from_peaks_args = create_sub_dict(parameters, ["data_directory", "rpeak_function_name", "RRI_sampling_frequency", "pad_with", "results_path", "realistic_rri_value_range", "mad_time_period_seconds"])
        rri_from_rpeak.determine_rri_from_rpeaks(**calculate_rri_from_peaks_args)
        del calculate_rri_from_peaks_args
    
        """
        ----------------
        MAD CALCULATION
        ----------------
        """

        # calculate MAD in the wrist acceleration data
        calculate_MAD_args = create_sub_dict(parameters, ["data_directory", "valid_file_types", "wrist_acceleration_keys", "physical_dimension_correction_dictionary", "mad_time_period_seconds", "results_path"])
        MAD.calculate_MAD_in_acceleration_data(**calculate_MAD_args)
        del calculate_MAD_args

        """
        ----------------------------
        RETRIEVE HEADER INFORMATION
        ----------------------------
        """

        # retrieve header information for existing data in the results
        retrieve_header_information_args = create_sub_dict(parameters, ["data_directory", "results_path"])
        read_edf.retrieve_file_header_information(**retrieve_header_information_args)
        del retrieve_header_information_args


def Data_Processing_and_Comparing(
        DATA_DIRECTORY: str,
        ECG_CLASSIFICATION_DIRECTORY: str,
        RPEAK_DIRECTORY: str,
        AVAILABLE_MAD_RRI_PATH: str,
        RESULTS_DIRECTORY: str,
        RESULTS_FILE_NAME: str,
        ECG_COMPARISON_FILE_NAME: str,
        RPEAK_COMPARISON_FILE_NAME: str,
        RRI_COMPARISON_FILE_NAME: str,
        MAD_COMPARISON_FILE_NAME: str
    ):
    """
    This function is supposed to run all processing and comparing functions in the designed order.

    Order of Execution:
        - ECG Validation: Evaluate where ECG data was recorded correctly to determine evaluatable segments
        - ECG Comparison: Compare already provided ECG classification with the calculated ECG Validation
        - R-peak Detection: Detect R-peak locations the valid segments of the ECG data using specified detectors
        - R-peak Comparison: Read out already provided r-peak locations and compare them and those of the specified detection functions with each other
        - RRI Calculation: Calculate RR-Intervals from detected R-peak locations
        - MAD Calculation: Calculate Mean Amplitude Deviation (values characterizing motion activity) using wrist accelerometry data

    I designed this function to process the data from the GIF study as it provides ECG classifications
    and r-peak locations. I think they checked these values manually, so we can consider them as "ground truth".

    ATTENTION:
    --------------------------------
    The individual functions were designed to be used on the data provided by the GIF study. If your data
    is not structured in the same way, you will have to adjust the functions accordingly. I suggest to
    test 'Data_Processing' first.

    ARGUMENTS:
    --------------------------------
    DATA_DIRECTORY: str
        Path to the directory where the ECG data is stored.
    ECG_CLASSIFICATION_DIRECTORY: str
        Path to the directory where the ECG classifications are stored.
    RPEAK_DIRECTORY: str
        Path to the directory where the r-peak locations are stored.
    AVAILABLE_MAD_RRI_PATH: str
        Path to the directory where the available MAD and RRI values are stored.
    RESULTS_DIRECTORY: str
        Path to the directory where the results should be stored.
    RESULTS_FILE_NAME: str
        Name of the file where the results should be stored.
    ECG_COMPARISON_FILE_NAME: str
        Name of the file where the results of the ECG comparison should be shown.
    RPEAK_COMPARISON_FILE_NAME: str
        Name of the file where the results of the r-peak comparison should be shown.
    RRI_COMPARISON_FILE_NAME: str
        Name of the file where the results of the RRI comparison should be shown.
    MAD_COMPARISON_FILE_NAME: str
        Name of the file where the results of the MAD comparison should be shown.
    
    RETURNS:
    --------------------------------
    None, but the results will be stored in the specified directory.

    RESULTS:
    --------------------------------
    Same as in 'Data_Processing', but with additional entries for the ECG and R-peak comparison.
    """

    """
    ---------------------------
    SET DATA AND STORAGE PATHS
    ---------------------------
    """

    # create needed directory if it does not exist
    create_directories_along_path(RESULTS_DIRECTORY)

    # set path to where ECG is stored
    parameters["data_directory"] = DATA_DIRECTORY

    # set path to pickle file that saves the results
    parameters["results_path"] = RESULTS_DIRECTORY + RESULTS_FILE_NAME

    """
    ----------------------------
    RECOVER RESULTS AFTER ERROR
    ----------------------------
    """

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(RESULTS_DIRECTORY + RESULTS_FILE_NAME) + "computation_in_progress.pkl"

    # ask the user if the results should be overwritten or recovered
    if os.path.isfile(temporary_file_path):
        recover_results_after_error(
            all_results_path = RESULTS_DIRECTORY + RESULTS_FILE_NAME, 
            some_results_with_updated_keys_path = temporary_file_path, 
        )
    
    del temporary_file_path

    """
    ---------------
    ECG VALIDATION
    ---------------
    """
    
    # create arguments for the valid ecg regions evaluation and calculate them
    determine_ecg_region_args = create_sub_dict(parameters, ["data_directory", "valid_file_types", "ecg_keys", "physical_dimension_correction_dictionary", "results_path", "straighten_ecg_signal", "use_ecg_validation_strictness", "check_ecg_time_interval_seconds", "check_ecg_overlapping_interval_steps", "check_ecg_validation_strictness", "check_ecg_removed_peak_difference_threshold", "check_ecg_std_min_threshold", "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold", "check_ecg_min_valid_length_minutes", "check_ecg_allowed_invalid_region_length_seconds"])

    # perform ecg validation
    check_data.determine_valid_ecg_regions(**determine_ecg_region_args)

    # create arguments for choosing the valid ecg regions for further computation
    if determine_ecg_region_args["use_ecg_validation_strictness"] is None:
        choose_valid_ecg_regions_for_further_computation_args = create_sub_dict(parameters, ["data_directory", "ecg_keys", "results_path", "rpeak_function_names"])
        check_data.choose_valid_ecg_regions_for_further_computation(**choose_valid_ecg_regions_for_further_computation_args)
        del choose_valid_ecg_regions_for_further_computation_args
    
    del determine_ecg_region_args

    """
    ------------------------
    COMPARE ECG VALIDATIONS
    ------------------------
    """
    parameters["ecg_classification_values_directory"] = ECG_CLASSIFICATION_DIRECTORY
    parameters["ecg_validation_comparison_report_path"] = RESULTS_DIRECTORY + ECG_COMPARISON_FILE_NAME

    # create arguments for the ECG validation comparison and perform it
    ecg_validation_comparison_args = create_sub_dict(parameters, ["ecg_classification_values_directory", "ecg_classification_file_types", "check_ecg_validation_strictness", "results_path"])
    check_data.ecg_validation_comparison(**ecg_validation_comparison_args)
    del ecg_validation_comparison_args

    # create arguments for printing the ECG validation comparison report
    ecg_validation_report_args = create_sub_dict(parameters, ["ecg_validation_comparison_report_path", "ecg_validation_comparison_report_dezimal_places", "check_ecg_validation_strictness", "results_path"])
    check_data.ecg_validation_comparison_report(**ecg_validation_report_args)
    del ecg_validation_report_args

    """
    -----------------
    R-PEAK DETECTION
    -----------------
    """

    # create arguments for the r-peak detection and correction
    detect_rpeaks_args = create_sub_dict(parameters, ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary", "results_path"])
    correct_rpeaks_args = create_sub_dict(parameters, ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary", "results_path"])

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
    rpeak_detection_comparison_args = create_sub_dict(parameters, ["data_directory", "rpeak_distance_threshold_seconds", "results_path", "rpeak_comparison_function_names", "remove_peaks_outside_ecg_classification"])

    # read r-peaks from the classification files
    parameters["rpeaks_values_directory"] = RPEAK_DIRECTORY
    read_rpeak_classification_args = create_sub_dict(parameters, ["data_directory", "valid_file_types", "rpeaks_values_directory", "valid_rpeak_values_file_types", "include_rpeak_value_classifications", "add_offset_to_classification", "results_path"])
    read_rpeak_classification_args["rpeak_classification_dictionary_key"] = parameters["rpeak_comparison_function_names"][-1]
    rpeak_detection.read_rpeaks_from_rri_files(**read_rpeak_classification_args)
    del read_rpeak_classification_args

    # perform r-peak comparison evaluation
    rpeak_detection.rpeak_detection_comparison(**rpeak_detection_comparison_args)
    del rpeak_detection_comparison_args

    # create arguments for printing the r-peak comparison report and print it
    parameters["rpeak_comparison_report_path"] = RESULTS_DIRECTORY + RPEAK_COMPARISON_FILE_NAME
    rpeak_comparison_report_args = create_sub_dict(parameters, ["rpeak_comparison_report_dezimal_places", "rpeak_comparison_report_path", "results_path", "rpeak_comparison_function_names"])
    rpeak_detection.rpeak_detection_comparison_report(**rpeak_comparison_report_args)
    del rpeak_comparison_report_args

    """
    ---------------------------
    CALCULATE RRI FROM R-PEAKS
    ---------------------------
    """

    parameters["rpeak_function_name"] = "hamilton"
    calculate_rri_from_peaks_args = create_sub_dict(parameters, ["data_directory", "rpeak_function_name", "RRI_sampling_frequency", "pad_with", "results_path", "realistic_rri_value_range", "mad_time_period_seconds"])
    rri_from_rpeak.determine_rri_from_rpeaks(**calculate_rri_from_peaks_args)
    del calculate_rri_from_peaks_args

    """
    ---------------------
    COMPARING RRI VALUES
    ---------------------
    """

    parameters["path_to_h5file"] = AVAILABLE_MAD_RRI_PATH
    parameters["rri_comparison_report_path"] = RESULTS_DIRECTORY + RRI_COMPARISON_FILE_NAME
    rri_comparison_args = create_sub_dict(parameters, ["path_to_h5file", "results_path", "rri_comparison_report_dezimal_places", "rri_comparison_report_path", "mad_time_period_seconds"])
    rri_from_rpeak.rri_comparison(**rri_comparison_args)
    del rri_comparison_args

    """
    ----------------
    MAD CALCULATION
    ----------------
    """

    # calculate MAD in the wrist acceleration data
    calculate_MAD_args = create_sub_dict(parameters, ["data_directory", "valid_file_types", "wrist_acceleration_keys", "physical_dimension_correction_dictionary", "mad_time_period_seconds", "results_path"])
    MAD.calculate_MAD_in_acceleration_data(**calculate_MAD_args)
    del calculate_MAD_args

    """
    ---------------------
    COMPARING MAD VALUES
    ---------------------
    """

    parameters["mad_comparison_report_path"] = RESULTS_DIRECTORY + MAD_COMPARISON_FILE_NAME
    mad_comparison_args = create_sub_dict(parameters, ["path_to_h5file", "results_path", "mad_comparison_report_dezimal_places", "mad_comparison_report_path"])
    MAD.mad_comparison(**mad_comparison_args)
    del mad_comparison_args


"""
--------------------------
RETRIEVING IMPORTANT DATA
--------------------------

During Data Processing, a lot of data is calculated. For the main project: 'Sleep Stage Classification' we 
only need the RRI and MAD values within the same time period. After Processing, this is not guaranteed, because
the RRI values are only calculated for the valid ECG regions. The following function will extract the 
corresponding MAD values to every time period. If multiple time periods (valid ecg regions) are present in 
one file, the values will be saved to different dictionaries.
"""

def Extract_RRI_MAD(
        DATA_DIRECTORIES: list,
        RESULTS_DIRECTORY: str,
        EXTRACTED_DATA_DIRECTORY: str,
    ):
    """
    This function will extract the RRI and MAD values from the results files and save them to a new location,
    as described above.

    ARGUMENTS:
    --------------------------------
    DATA_DIRECTORIES: list
        List of paths to the directories where the ECG data is stored.
    RESULTS_DIRECTORY: str
        Path to the directory where the results were stored.
    EXTRACTED_DATA_DIRECTORY: str
        Path to the directory where the extracted data should be stored.
    
    RETURNS:
    --------------------------------
    None, but the extracted data will be stored in the specified directory.

    RESULTS:
    --------------------------------
    Every extracted data (.pkl) file will contain multiple dictionaries. Each dictionary will not correspond 
    to one file (like above) but to a specific time period within a file (time period of one valid ecg region). 
    If this results in multiple dictionaries for each file (more than 1 valid ecg region in this file) the
    corresponding ID will be the file name with the added position of the valid ecg region 
    (e.g.: file name + "_0" / + "_1", ...). 
    
    Each dictionary is structured as follows:
    {
        "ID":     
                Variation of the (.edf) file name the results were calculated for, 
                (number appended if multiple valid ecg regions).

        "start_date":
                The start date of the EDF file, formatted as "YYYY-MM-DD".
        
        "start_time":
                The start time of the EDF file, formatted as "HH:MM:SS".
        
        "time_interval":
                List of the start and end time points (in seconds after "start_time") of this dictionaries 
                time period.
        
        "RRI":
                List of RR-intervals calculated from the r-peak locations within this time period.
        
        "RRI_frequency":
                Sampling frequency of the RR-intervals.
        
        "MAD":
                List of Mean Amplitude Deviation values calculated from the wrist acceleration data within 
                this time period.
        
        "MAD_frequency":
                Sampling frequency of the MAD values. Corresponds to 1 / parameters["mad_time_period_seconds"].
    }
    """

    # create directory if it does not exist
    create_directories_along_path(EXTRACTED_DATA_DIRECTORY)

    for DATA_DIRECTORY in DATA_DIRECTORIES:
        """
        ---------------------------
        SET DATA AND STORAGE PATHS
        ---------------------------
        """

        # set path to where ECG is stored
        parameters["data_directory"] = DATA_DIRECTORY

        for i in range(len(DATA_DIRECTORY)-2, -1, -1):
            if DATA_DIRECTORY[i] == "/":
                i += 1
                break

        # set path to pickle file that saves the processing results
        RESULTS_PATH = RESULTS_DIRECTORY + DATA_DIRECTORY[i:-1] + "_Results.pkl"
        parameters["results_path"] = RESULTS_PATH

        # set path to pickle file that will only contain the RRI and MAD values in the same time period
        EXTRACTED_DATA_PATH = EXTRACTED_DATA_DIRECTORY + DATA_DIRECTORY[i:-1] + ".pkl"
        parameters["rri_mad_data_path"] = EXTRACTED_DATA_PATH

        """
        ---------------------------
        EXTRACT RRI AND MAD VALUES
        ---------------------------
        """

        # create arguments for extracting the RRI and MAD values
        retrieve_rri_mad_data_args = create_sub_dict(parameters, ["data_directory", "rri_mad_data_path", "results_path"])
        data_retrieval.retrieve_rri_mad_data_in_same_time_period(**retrieve_rri_mad_data_args)


def better_int(value):
    if int(value) != value:
        print(value)
        raise SystemError("Value was supposed to be an integer but turned out as float.")
    return int(value)


def ADD_RRI_MAD_SLP(
        new_save_file_path: str,
        results_path: str,
        gif_data_directory: str,
        slp_files_directory: str,
        lights_and_time_shift_csv_path: str,
        min_data_length_seconds: int = 5*3600,
        fill_ecg_gaps_threshold_seconds: int = 0,
        RRI_frequency = 4,
        MAD_frequency = 1
    ):
    """
    """

    # load time shift and times of lights off and on
    lights_and_time_shift = pd.read_csv(lights_and_time_shift_csv_path, sep=" ")

    # access preprocessed GIF data
    results_generator = load_from_pickle(results_path)

    # get all edf file names
    all_gif_edf_files = os.listdir(gif_data_directory)

    sleep_signal_distances = list()

    for data_dict in results_generator:
        # add Rpeak from Munich for sections where original were missing
        patient_id = data_dict["ID"]
        
        if "start_date" in data_dict:
            somno_rec_date = data_dict["start_date"]
        else:
            somno_rec_date = data_dict["????-??-??"]

        if "start_time" not in data_dict:
            print("Synchronization not possible. No start time found for patient:", patient_id)
            continue

        rec_time = data_dict["start_time"]
        rec_time_numbers = rec_time.split(":")
        if len(rec_time_numbers) == 3:
            somno_start_time_seconds_somno = int(rec_time_numbers[0]) * 3600 + int(rec_time_numbers[1]) * 60 + int(rec_time_numbers[2])
        elif len(rec_time_numbers) == 2:
            somno_start_time_seconds_somno = int(rec_time_numbers[0]) * 60 + int(rec_time_numbers[1])

        # load SLP data
        slp_file_path = slp_files_directory + patient_id + ".slp"
        if not os.path.isfile(slp_file_path):
            print("SLP file not found for patient:", patient_id)
            continue

        slp_file = open(slp_file_path, "rb")
        slp_file_lines = slp_file.readlines()
        slp_file.close()

        numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

        first_slp_line = slp_file_lines[0].decode("utf-8").strip()
        final_index = len(first_slp_line)
        for char_pos in range(len(first_slp_line)):
            if first_slp_line[char_pos] not in numbers:
                final_index = char_pos
                break
        slp_start_time_psg = first_slp_line[0:final_index]
        slp_start_time_seconds_psg = float(slp_start_time_psg) * 3600

        last_slp_line = slp_file_lines[-1].decode("utf-8").strip()
        final_index = len(last_slp_line)
        for char_pos in range(len(last_slp_line)):
            if last_slp_line[char_pos] not in numbers:
                final_index = char_pos
                break
        slp_end_time_psg = last_slp_line[0:final_index]
        slp_end_time_seconds_psg = float(slp_end_time_psg) * 3600

        del numbers, first_slp_line, last_slp_line, final_index

        slp = list()
        for slp_line in slp_file_lines:
            slp_line = slp_line.decode("utf-8").strip()
            if len(slp_line) > 1:
                continue
            slp.append(int(slp_line))

        # access time shift of this patient
        time_shift = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["time_shift"].values
        lights_off = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["lights_off"].values
        lights_on = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["lights_on"].values
        slope = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["slope"].values
        if len(time_shift) == 0:
            time_shift, lights_off, lights_on, slope = 0, 0, 0, 0
        else:
            time_shift = float(time_shift[0])
            lights_off = float(lights_off[0])
            lights_on = float(lights_on[0])
            slope = float(slope[0])
        
        SLP_frequency = 1/30

        # access file containing x, y and z wrist acceleration data
        corresponding_file_patient = None
        for file_name in all_gif_edf_files:
            if file_name.startswith(patient_id) and file_name.endswith(".edf"):
                corresponding_file_patient = file_name
                break
        if corresponding_file_patient is None:
            print("No .edf file found for patient:", patient_id)
            continue

        # load the data
        f = read_edf.pyedflib.EdfReader(gif_data_directory + corresponding_file_patient)
        signal_labels = f.getSignalLabels()
        x_acceleration = f.readSignal(signal_labels.index("X"))
        y_acceleration = f.readSignal(signal_labels.index("Y"))
        z_acceleration = f.readSignal(signal_labels.index("Z"))
        acceleration_sample_frequency = f.getSampleFrequency(signal_labels.index("X"))
        f._close()
        
        # synchronize ECG data with PSG data (psg time = somno time + shift + slope * (somno time - 2h))
        # somno time = (psg time + slope * 2h - shift) / (1 + slope)

        synchronized_SLP = list()
        synchronized_RRI = list()
        synchronized_MAD = list()
        start_end_valid_region_psg_times = list()

        ecg_sampling_frequency = data_dict["ECG_frequency"]
        
        valid_ecg_regions_somno = data_dict["valid_ecg_regions"]

        # somno_start_time_seconds_psg = somno_start_time_seconds_somno + time_shift + slope * (somno_start_time_seconds_somno - 7200)
        
        hamilton_rpeaks_somno = np.unique(data_dict["hamilton"])
        hamilton_rpeaks_seconds_psg = [(somno_start_time_seconds_somno + peak/ecg_sampling_frequency) + time_shift + slope * ((somno_start_time_seconds_somno + peak/ecg_sampling_frequency) - 7200) for peak in hamilton_rpeaks_somno]

        # print(patient_id, time_shift, somno_start_time_seconds_psg, slp_start_time_seconds_psg)
        # print(valid_ecg_regions_somno)
        # print(np.array(valid_ecg_regions_somno) / ecg_sampling_frequency)

        for valid_region in valid_ecg_regions_somno:
            # transform valid ecg region into psg times
            valid_region_psg = [(somno_start_time_seconds_somno + valid_region[0]/ecg_sampling_frequency) + time_shift + slope * ((somno_start_time_seconds_somno + valid_region[0]/ecg_sampling_frequency) - 7200),
                                (somno_start_time_seconds_somno + valid_region[1]/ecg_sampling_frequency) + time_shift + slope * ((somno_start_time_seconds_somno + valid_region[1]/ecg_sampling_frequency) - 7200)]

            # print(np.array(valid_region_psg))

            # access first time point in valid ecg region that is in sync with the SLP data
            if slp_start_time_seconds_psg > valid_region_psg[0]:
                number_slp_values = int((slp_start_time_seconds_psg - valid_region_psg[0]) * SLP_frequency)
                first_slp_value_in_valid_region = slp_start_time_seconds_psg - number_slp_values / SLP_frequency
            else:
                number_slp_values = np.ceil((valid_region_psg[0] - slp_start_time_seconds_psg) * SLP_frequency)
                first_slp_value_in_valid_region = slp_start_time_seconds_psg + number_slp_values / SLP_frequency
                
            # access last time point in valid ecg region that is in sync with the SLP data
            number_slp_values_in_valid_region = int((valid_region_psg[1] - first_slp_value_in_valid_region) * SLP_frequency)
            last_slp_value_in_valid_region = first_slp_value_in_valid_region + number_slp_values_in_valid_region / SLP_frequency

            # store start and end time of valid region in psg time
            start_end_valid_region_psg_times.append([first_slp_value_in_valid_region, last_slp_value_in_valid_region])
            valid_interval_size_seconds_psg = better_int(last_slp_value_in_valid_region - first_slp_value_in_valid_region)

            # check results
            if (first_slp_value_in_valid_region - slp_start_time_seconds_psg) % (1/SLP_frequency) != 0:
                raise ValueError("Time point of first SLP value in valid region is not in sync with SLP data.")
            if (last_slp_value_in_valid_region - slp_start_time_seconds_psg) % (1/SLP_frequency) != 0:
                raise ValueError("Time point of last SLP value in valid region is not in sync with SLP data.")

            # print(first_slp_value_in_valid_region, last_slp_value_in_valid_region, valid_interval_size_seconds_psg, first_slp_value_in_valid_region - slp_start_time_seconds_psg, last_slp_value_in_valid_region - slp_start_time_seconds_psg)

            # retrieve slp values for this valid ecg region
            slp_values_in_valid_region = list()
            start_index_slp = int(round((first_slp_value_in_valid_region - slp_start_time_seconds_psg) * SLP_frequency))
            
            for i in range(start_index_slp, number_slp_values_in_valid_region+start_index_slp):
                if i < 0 or i >= len(slp):
                    slp_values_in_valid_region.append(0)
                else:
                    slp_values_in_valid_region.append(slp[i])

            synchronized_SLP.append(slp_values_in_valid_region)

            # print(start_index_slp, len(slp_values_in_valid_region))

            # calculate RRI from R-peaks in valid ecg regions
            rri_values_in_valid_region = list()
            number_rri_entries = int(valid_interval_size_seconds_psg * RRI_frequency)
            start_looking_at = 1

            # iterate over all rri entries, find rri datapoint between two rpeaks (in seconds) and return their difference
            for i in range(number_rri_entries):
                rri_datapoint_second = i / RRI_frequency + first_slp_value_in_valid_region

                this_rri = 0
                for j in range(start_looking_at, len(hamilton_rpeaks_seconds_psg)):
                    start_looking_at = j

                    if hamilton_rpeaks_seconds_psg[j] == rri_datapoint_second and j not in [0, len(hamilton_rpeaks_seconds_psg)-1]:
                        this_rri = (hamilton_rpeaks_seconds_psg[j+1] - hamilton_rpeaks_seconds_psg[j-1]) / 2
                        break
                    if hamilton_rpeaks_seconds_psg[j-1] <= rri_datapoint_second and rri_datapoint_second <= hamilton_rpeaks_seconds_psg[j]:
                        this_rri = hamilton_rpeaks_seconds_psg[j] - hamilton_rpeaks_seconds_psg[j-1]
                        break
                    if hamilton_rpeaks_seconds_psg[j-1] > rri_datapoint_second:
                        break
                
                rri_values_in_valid_region.append(this_rri)
            
            synchronized_RRI.append(rri_values_in_valid_region)

            # calculate MAD from wrist acceleration data in valid ecg regions
            # calculate MAD from wrist acceleration data in valid ecg regions
            number_mad_values_in_valid_region = better_int(valid_interval_size_seconds_psg * MAD_frequency)
            mad_acc_borders_psg = [[first_slp_value_in_valid_region + i/MAD_frequency, first_slp_value_in_valid_region + (i+1)/MAD_frequency] for i in range(number_mad_values_in_valid_region)] # type: ignore
            mad_acc_borders_somno = [[(border[0] + slope * 7200 - time_shift) / (1 + slope), (border[1] + slope * 7200 - time_shift) / (1 + slope)] for border in mad_acc_borders_psg] # type: ignore
            mad_acc_borders_num = [[int((border[0] - somno_start_time_seconds_somno) * acceleration_sample_frequency), int(np.ceil((border[1] - somno_start_time_seconds_somno) * acceleration_sample_frequency))] for border in mad_acc_borders_somno] # type: ignore
            if mad_acc_borders_num[0][0] < 0:
                mad_acc_borders_num[0][0] = 0
                if mad_acc_borders_num[0][1] < 0:
                    raise ValueError("Something went wrong.")
            if mad_acc_borders_num[-1][1] > len(x_acceleration):
                raise ValueError("Something went wrong.")
            
            # calculate MAD for each segment
            mad_values_in_valid_region = list()
            for segment in mad_acc_borders_num:    
                mad_values_in_valid_region.append(MAD.calc_mad_in_interval(
                    acceleration_data_lists=[x_acceleration, y_acceleration, z_acceleration],
                    start_position=segment[0],
                    end_position=segment[1],
                ))
            
            synchronized_MAD.append(mad_values_in_valid_region)

            if len(slp_values_in_valid_region)/SLP_frequency != len(rri_values_in_valid_region)/RRI_frequency or len(slp_values_in_valid_region)/SLP_frequency != len(mad_values_in_valid_region)/MAD_frequency:
                raise ValueError("Number of SLP, RRI and MAD values in valid region does not match.")
        
        if fill_ecg_gaps_threshold_seconds > 0:
            ecg_gaps = []
            for i in range(len(start_end_valid_region_psg_times)-1):
                ecg_gaps.append((start_end_valid_region_psg_times[i+1][0] - start_end_valid_region_psg_times[i][1]))
            
            for i in range(len(ecg_gaps)):
                ecg_gap = better_int(ecg_gaps[i])
                if ecg_gap % (1/RRI_frequency) != 0 or ecg_gap % (1/MAD_frequency) != 0 or ecg_gap % (1/SLP_frequency) != 0:
                    raise ValueError(f"Gap length {ecg_gap} seconds is not a multiple of RRI, MAD or SLP sampling frequency for patient: {patient_id} at index {i}")
                ecg_gaps[i] = ecg_gap

            filled_gaps_RRI = list([] for _ in range(len(synchronized_RRI)))
            filled_gaps_RRI[0] = list(synchronized_RRI[0])
            filled_gaps_MAD = list([] for _ in range(len(synchronized_MAD)))
            filled_gaps_MAD[0] = list(synchronized_MAD[0])
            filled_gaps_SLP = list([] for _ in range(len(synchronized_SLP)))
            filled_gaps_SLP[0] = list(synchronized_SLP[0])
            filled_gaps_times_psg = [list(start_end_valid_region_psg_times[0])]

            append_to = 0
            for i in range(1, len(synchronized_RRI)):
                # fill gaps in RRI, MAD and SLP data
                if ecg_gaps[i-1] > fill_ecg_gaps_threshold_seconds:
                    append_to += 1
                    filled_gaps_times_psg[-1][1] = start_end_valid_region_psg_times[i-1][1]
                    filled_gaps_times_psg.append(list(start_end_valid_region_psg_times[i]))
                    # fill gaps with artifact values
                else:
                    filled_gaps_RRI[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * RRI_frequency))])
                    filled_gaps_MAD[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * MAD_frequency))])
                    filled_gaps_SLP[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * SLP_frequency))])

                filled_gaps_RRI[append_to].extend(synchronized_RRI[i])
                filled_gaps_MAD[append_to].extend(synchronized_MAD[i])
                filled_gaps_SLP[append_to].extend(synchronized_SLP[i])
            
            filled_gaps_times_psg[append_to][1] = start_end_valid_region_psg_times[-1][1]
            
            for i in range(len(filled_gaps_RRI)-1, -1, -1):
                if len(filled_gaps_RRI[i]) == 0:
                    filled_gaps_RRI.pop(i)
                if len(filled_gaps_MAD[i]) == 0:
                    filled_gaps_MAD.pop(i)
                if len(filled_gaps_SLP[i]) == 0:
                    filled_gaps_SLP.pop(i)
            
            for i in range(len(filled_gaps_RRI)):
                if len(filled_gaps_RRI[i]) / RRI_frequency != len(filled_gaps_MAD[i]) / MAD_frequency or len(filled_gaps_RRI[i]) / RRI_frequency != len(filled_gaps_SLP[i]) / SLP_frequency:
                    print(len(filled_gaps_RRI[i]) / RRI_frequency, len(filled_gaps_MAD[i]) / MAD_frequency, len(filled_gaps_SLP[i]) / SLP_frequency)
                    raise ValueError(f"RRI, MAD and SLP data lengths do not match for patient: {patient_id} at index {i}")
        else:
            filled_gaps_RRI = synchronized_RRI
            filled_gaps_MAD = synchronized_MAD
            filled_gaps_SLP = synchronized_SLP
            filled_gaps_times_psg = start_end_valid_region_psg_times

        with open(new_save_file_path, "ab") as f:
            # create new datapoints and save them
            count_removed = 0
            for i in range(len(filled_gaps_RRI)):

                new_data_dict = dict()
                if i-count_removed >= 1:
                    new_data_dict["ID"] = patient_id + "_" + str(i-count_removed)
                else:
                    new_data_dict["ID"] = patient_id

                if filled_gaps_times_psg[i][1] - filled_gaps_times_psg[i][0] < min_data_length_seconds * ecg_sampling_frequency:
                    count_removed += 1
                    continue
                
                # somno time = (psg time + slope * 2h - shift) / (1 + slope)
                this_start_time_somno = (filled_gaps_times_psg[i][0] + slope * 7200 - time_shift) / (1 + slope)
                this_end_time_somno = (filled_gaps_times_psg[i][1] + slope * 7200 - time_shift) / (1 + slope)

                new_data_dict["start_date"] = somno_rec_date
                new_data_dict["start_time_somno"] = this_start_time_somno
                new_data_dict["end_time_somno"] = this_end_time_somno
                new_data_dict["start_time_psg"] = filled_gaps_times_psg[i][0]
                new_data_dict["end_time_psg"] = filled_gaps_times_psg[i][1]

                new_data_dict["RRI"] = filled_gaps_RRI[i]
                new_data_dict["RRI_frequency"] = RRI_frequency
                new_data_dict["MAD"] = filled_gaps_MAD[i]
                new_data_dict["MAD_frequency"] = MAD_frequency
                new_data_dict["SLP"] = filled_gaps_SLP[i]
                new_data_dict["SLP_frequency"] = SLP_frequency

                pickle.dump(new_data_dict, f)


def round_start_end_apnea_time(event_start_string, event_end_string):
    # if there are milliseconds in the start time string round down to full second
    event_start_string = event_start_string.replace(",", ".")
    event_end_string = event_end_string.replace(",", ".")

    cropped = 0
    if "." in event_start_string:
        for i in range(len(event_start_string)-1, -1, -1):
            if event_start_string[i] == ".":
                cropped = event_end_string[i:]
                event_start_string = event_start_string[0:i]
                break
    event_start = datetime.strptime(event_start_string, "%Y-%m-%d %H:%M:%S")
    if float(cropped) >= 0.5:
        event_start += timedelta(seconds=1)

    # if there are milliseconds in the end time string round up to next full second
    if "." in event_end_string:
        cropped = 0
        for i in range(len(event_end_string)-1, -1, -1):
            if event_end_string[i] == ".":
                cropped = event_end_string[i:]
                event_end_string = event_end_string[0:i]
                break
        
        event_end = datetime.strptime(event_end_string, "%Y-%m-%d %H:%M:%S")
        if float(cropped) >= 0.5:
            event_end += timedelta(seconds=1)
    else:
        event_end = datetime.strptime(event_end_string, "%Y-%m-%d %H:%M:%S")
    
    return event_start, event_end


def ADD_RRI_MAD_APNEA(
        new_save_file_path: str,
        results_path: str,
        gif_data_directory: str,
        lights_and_time_shift_csv_path: str,
        apnea_events_csv_path: str,
        min_data_length_seconds: int = 0*3600,
        fill_ecg_gaps_threshold_seconds: int = 0,
        RRI_frequency = 4,
        MAD_frequency = 1,
        SAE_frequency = 1,
    ):
    """
    """

    # load time shift and times of lights off and on
    lights_and_time_shift = pd.read_csv(lights_and_time_shift_csv_path, sep=" ")

    # access preprocessed GIF data
    results_generator = load_from_pickle(results_path)

    # get all edf file names
    all_gif_edf_files = os.listdir(gif_data_directory)

    for data_dict in results_generator:
        patient_id = data_dict["ID"]

        if "start_date" in data_dict:
            somno_rec_date = data_dict["start_date"]
            somno_start_date = datetime.strptime(somno_rec_date, "%Y-%m-%d")
            rec_date_available = True
        else:
            rec_date_available = False

        if "start_time" not in data_dict:
            print("Synchronization not possible. No start time found for patient:", patient_id)
            continue

        rec_time = data_dict["start_time"]
        rec_time_numbers = rec_time.split(":")
        if len(rec_time_numbers) == 3:
            somno_start_time_seconds_somno = int(rec_time_numbers[0]) * 3600 + int(rec_time_numbers[1]) * 60 + int(rec_time_numbers[2])
        elif len(rec_time_numbers) == 2:
            somno_start_time_seconds_somno = int(rec_time_numbers[0]) * 60 + int(rec_time_numbers[1])

        # load SAE data
        sae_file_path = apnea_events_csv_path + patient_id + ".csv"
        if not os.path.isfile(sae_file_path):
            print("SAE file not found for patient:", patient_id)
            continue

        # create SAE data
        sae_data = pd.read_csv(sae_file_path, sep=",")
        all_apnea_classes = ['Apnea', 'Obstructive Apnea', 'Central Apnea', 'Mixed Apnea', 'Hypopnea', 'Obstructive Hypopnea', 'Central Hypopnea']

        no_apnea_events = True
        if len(sae_data.values) > 0:
            no_apnea_events = False

            sae_start_time_string = sae_data.values[0][1]
            sae_end_time_string = sae_data.values[-1][2]
            sae_start_time, sae_end_time = round_start_end_apnea_time(sae_start_time_string, sae_end_time_string)

            for sae_line in sae_data.values:
                event_start_string = sae_line[1]
                event_end_string = sae_line[2]
                event_start, event_end = round_start_end_apnea_time(event_start_string, event_end_string)

                if event_start < sae_start_time:
                    sae_start_time = event_start
                if event_end > sae_end_time:
                    sae_end_time = event_end
            
            if not rec_date_available:
                if sae_start_time.hour < 16:
                    print("Recording date of ECG and ACC data not available and can't also be deduced certainly for patient:", patient_id)
                    continue
                somno_start_date = sae_start_time
            
            sae_duration = better_int((sae_end_time-sae_start_time).total_seconds())
            sae_array = [0 for _ in range(sae_duration)]

            sae_start_time_seconds = sae_start_time.hour * 3600 + sae_start_time.minute * 60 + sae_start_time.second
            start_date_difference = (sae_start_time.date() - somno_start_date.date()).days
            sae_start_time_seconds = sae_start_time_seconds + start_date_difference * 24 * 3600

            count_events = 0
            count_conflict = 0
            for sae_line in sae_data.values:
                apnea_event = all_apnea_classes.index(str(sae_line[0])) + 1
                event_start_string = sae_line[1]
                event_end_string = sae_line[2]
                event_start, event_end = round_start_end_apnea_time(event_start_string, event_end_string)

                event_start_index = better_int((event_start-sae_start_time).total_seconds())
                event_end_index = better_int((event_end-sae_start_time).total_seconds())
                for i in range(event_start_index, event_end_index):
                    if sae_array[i] != 0 and sae_array[i] != apnea_event:
                        count_conflict += 1
                        sae_array[i] = max(sae_array[i], apnea_event)
                    else:
                        sae_array[i] = apnea_event

                    count_events += 1
            
            print(f"Patient {patient_id} has {count_events} apnea events with {count_conflict} conflicts. (Relative conflicts: {count_conflict/count_events:.2%})")

        # access time shift of this patient
        time_shift = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["time_shift"].values
        lights_off = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["lights_off"].values
        lights_on = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["lights_on"].values
        slope = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["slope"].values
        if len(time_shift) == 0:
            time_shift, lights_off, lights_on, slope = 0, 0, 0, 0
        else:
            time_shift = float(time_shift[0])
            lights_off = float(lights_off[0])
            lights_on = float(lights_on[0])
            slope = float(slope[0])

        # access file containing x, y and z wrist acceleration data
        corresponding_file_patient = None
        for file_name in all_gif_edf_files:
            if file_name.startswith(patient_id) and file_name.endswith(".edf"):
                corresponding_file_patient = file_name
                break
        if corresponding_file_patient is None:
            print("No .edf file found for patient:", patient_id)
            continue

        # load the data
        f = read_edf.pyedflib.EdfReader(gif_data_directory + corresponding_file_patient)
        signal_labels = f.getSignalLabels()
        x_acceleration = f.readSignal(signal_labels.index("X"))
        y_acceleration = f.readSignal(signal_labels.index("Y"))
        z_acceleration = f.readSignal(signal_labels.index("Z"))
        acceleration_sample_frequency = f.getSampleFrequency(signal_labels.index("X"))
        f._close()
        
        # synchronize ECG data with PSG data (psg time = somno time + shift + slope * (somno time - 2h))
        # somno time = (psg time + slope * 2h - shift) / (1 + slope)

        synchronized_SAE = list()
        synchronized_RRI = list()
        synchronized_MAD = list()
        start_end_valid_region_psg_times = list()

        ecg_sampling_frequency = data_dict["ECG_frequency"]
        
        valid_ecg_regions_somno = data_dict["valid_ecg_regions"]

        # somno_start_time_seconds_psg = somno_start_time_seconds_somno + time_shift + slope * (somno_start_time_seconds_somno - 7200)
        
        hamilton_rpeaks_somno = np.unique(data_dict["hamilton"])
        hamilton_rpeaks_seconds_psg = [(somno_start_time_seconds_somno + peak/ecg_sampling_frequency) + time_shift + slope * ((somno_start_time_seconds_somno + peak/ecg_sampling_frequency) - 7200) for peak in hamilton_rpeaks_somno]

        # print(patient_id, time_shift, somno_start_time_seconds_psg, slp_start_time_seconds_psg)
        # print(valid_ecg_regions_somno)
        # print(np.array(valid_ecg_regions_somno) / ecg_sampling_frequency)

        for valid_region in valid_ecg_regions_somno:
            # transform valid ecg region into psg times
            valid_region_psg = [(somno_start_time_seconds_somno + valid_region[0]/ecg_sampling_frequency) + time_shift + slope * ((somno_start_time_seconds_somno + valid_region[0]/ecg_sampling_frequency) - 7200),
                                (somno_start_time_seconds_somno + valid_region[1]/ecg_sampling_frequency) + time_shift + slope * ((somno_start_time_seconds_somno + valid_region[1]/ecg_sampling_frequency) - 7200)]

            # print(np.array(valid_region_psg))

            # access first time point in valid ecg region that is in sync with the SLP data
            if no_apnea_events:
                first_sae_value_in_valid_region = valid_region_psg[0]
                sae_start_time_seconds = valid_region_psg[0]
                sae_array = []
            else:
                if sae_start_time_seconds > valid_region_psg[0]: # type: ignore
                    number_sae_values = int((sae_start_time_seconds - valid_region_psg[0]) * SAE_frequency) # type: ignore
                    first_sae_value_in_valid_region = sae_start_time_seconds - number_sae_values / SAE_frequency # type: ignore
                else:
                    number_sae_values = np.ceil((valid_region_psg[0] - sae_start_time_seconds) * SAE_frequency) # type: ignore
                    first_sae_value_in_valid_region = sae_start_time_seconds + number_sae_values / SAE_frequency # type: ignore

            # access last time point in valid ecg region that is in sync with the SLP data
            number_sae_values_in_valid_region = int((valid_region_psg[1] - first_sae_value_in_valid_region) * SAE_frequency)
            last_sae_value_in_valid_region = first_sae_value_in_valid_region + number_sae_values_in_valid_region / SAE_frequency

            # store start and end time of valid region in psg time
            start_end_valid_region_psg_times.append([first_sae_value_in_valid_region, last_sae_value_in_valid_region])
            valid_interval_size_seconds_psg = better_int(last_sae_value_in_valid_region - first_sae_value_in_valid_region)

            # check results
            if (first_sae_value_in_valid_region - sae_start_time_seconds) % (1/SAE_frequency) != 0: # type: ignore
                raise ValueError("Time point of first SLP value in valid region is not in sync with SLP data.")
            if (last_sae_value_in_valid_region - sae_start_time_seconds) % (1/SAE_frequency) != 0: # type: ignore
                raise ValueError("Time point of last SLP value in valid region is not in sync with SLP data.")

            # print(first_sae_value_in_valid_region, last_sae_value_in_valid_region, valid_interval_size_seconds_psg, first_sae_value_in_valid_region - sae_start_time_seconds, last_sae_value_in_valid_region - sae_start_time_seconds)

            # retrieve sae values for this valid ecg region
            sae_values_in_valid_region = list()
            start_index_sae = int(round((first_sae_value_in_valid_region - sae_start_time_seconds) * SAE_frequency)) # type: ignore

            for i in range(start_index_sae, number_sae_values_in_valid_region+start_index_sae):
                if i < 0 or i >= len(sae_array): # type: ignore
                    sae_values_in_valid_region.append(0)
                else:
                    sae_values_in_valid_region.append(sae_array[i])

            synchronized_SAE.append(sae_values_in_valid_region)

            # print(start_index_sae, len(sae_values_in_valid_region))

            # calculate RRI from R-peaks in valid ecg regions
            rri_values_in_valid_region = list()
            number_rri_entries = int(valid_interval_size_seconds_psg * RRI_frequency)
            start_looking_at = 1

            # iterate over all rri entries, find rri datapoint between two rpeaks (in seconds) and return their difference
            for i in range(number_rri_entries):
                rri_datapoint_second = i / RRI_frequency + first_sae_value_in_valid_region

                this_rri = 0
                for j in range(start_looking_at, len(hamilton_rpeaks_seconds_psg)):
                    start_looking_at = j

                    if hamilton_rpeaks_seconds_psg[j] == rri_datapoint_second and j not in [0, len(hamilton_rpeaks_seconds_psg)-1]:
                        this_rri = (hamilton_rpeaks_seconds_psg[j+1] - hamilton_rpeaks_seconds_psg[j-1]) / 2
                        break
                    if hamilton_rpeaks_seconds_psg[j-1] <= rri_datapoint_second and rri_datapoint_second <= hamilton_rpeaks_seconds_psg[j]:
                        this_rri = hamilton_rpeaks_seconds_psg[j] - hamilton_rpeaks_seconds_psg[j-1]
                        break
                    if hamilton_rpeaks_seconds_psg[j-1] > rri_datapoint_second:
                        break
                
                rri_values_in_valid_region.append(this_rri)
            
            synchronized_RRI.append(rri_values_in_valid_region)

            # calculate MAD from wrist acceleration data in valid ecg regions
            number_mad_values_in_valid_region = better_int(valid_interval_size_seconds_psg * MAD_frequency)
            mad_acc_borders_psg = [[first_sae_value_in_valid_region + i/MAD_frequency, first_sae_value_in_valid_region + (i+1)/MAD_frequency] for i in range(number_mad_values_in_valid_region)] # type: ignore
            mad_acc_borders_somno = [[(border[0] + slope * 7200 - time_shift) / (1 + slope), (border[1] + slope * 7200 - time_shift) / (1 + slope)] for border in mad_acc_borders_psg] # type: ignore
            mad_acc_borders_num = [[int((border[0] - somno_start_time_seconds_somno) * acceleration_sample_frequency), int(np.ceil((border[1] - somno_start_time_seconds_somno) * acceleration_sample_frequency))] for border in mad_acc_borders_somno] # type: ignore
            if mad_acc_borders_num[0][0] < 0:
                mad_acc_borders_num[0][0] = 0
                if mad_acc_borders_num[0][1] < 0:
                    raise ValueError("Something went wrong.")
            if mad_acc_borders_num[-1][1] > len(x_acceleration):
                raise ValueError("Something went wrong.")

            # calculate MAD for each segment
            mad_values_in_valid_region = list()
            for segment in mad_acc_borders_num:    
                mad_values_in_valid_region.append(MAD.calc_mad_in_interval(
                    acceleration_data_lists=[x_acceleration, y_acceleration, z_acceleration],
                    start_position=segment[0],
                    end_position=segment[1],
                ))
            
            synchronized_MAD.append(mad_values_in_valid_region)

            if len(sae_values_in_valid_region)/SAE_frequency != len(rri_values_in_valid_region)/RRI_frequency or len(sae_values_in_valid_region)/SAE_frequency != len(mad_values_in_valid_region)/MAD_frequency:
                raise ValueError("Number of SAE, RRI and MAD values in valid region does not match.")

        if fill_ecg_gaps_threshold_seconds > 0 and not no_apnea_events:
            ecg_gaps = []
            for i in range(len(start_end_valid_region_psg_times)-1):
                ecg_gaps.append((start_end_valid_region_psg_times[i+1][0] - start_end_valid_region_psg_times[i][1]))
            
            for i in range(len(ecg_gaps)):
                ecg_gap = better_int(ecg_gaps[i])
                if ecg_gap % (1/RRI_frequency) != 0 or ecg_gap % (1/MAD_frequency) != 0 or ecg_gap % (1/SAE_frequency) != 0:
                    raise ValueError(f"Gap length {ecg_gap} seconds is not a multiple of RRI, MAD or SAE sampling frequency for patient: {patient_id} at index {i}")
                ecg_gaps[i] = ecg_gap

            filled_gaps_RRI = list([] for _ in range(len(synchronized_RRI)))
            filled_gaps_RRI[0] = list(synchronized_RRI[0])
            filled_gaps_MAD = list([] for _ in range(len(synchronized_MAD)))
            filled_gaps_MAD[0] = list(synchronized_MAD[0])
            filled_gaps_SAE = list([] for _ in range(len(synchronized_SAE)))
            filled_gaps_SAE[0] = list(synchronized_SAE[0])
            filled_gaps_times_psg = [list(start_end_valid_region_psg_times[0])]

            append_to = 0
            for i in range(1, len(synchronized_RRI)):
                # fill gaps in RRI, MAD and SAE data
                if ecg_gaps[i-1] > fill_ecg_gaps_threshold_seconds:
                    append_to += 1
                    filled_gaps_times_psg[-1][1] = start_end_valid_region_psg_times[i-1][1]
                    filled_gaps_times_psg.append(list(start_end_valid_region_psg_times[i]))
                    # fill gaps with artifact values
                else:
                    filled_gaps_RRI[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * RRI_frequency))])
                    filled_gaps_MAD[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * MAD_frequency))])
                    filled_gaps_SAE[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * SAE_frequency))])

                filled_gaps_RRI[append_to].extend(synchronized_RRI[i])
                filled_gaps_MAD[append_to].extend(synchronized_MAD[i])
                filled_gaps_SAE[append_to].extend(synchronized_SAE[i])

            filled_gaps_times_psg[append_to][1] = start_end_valid_region_psg_times[-1][1]
            
            for i in range(len(filled_gaps_RRI)-1, -1, -1):
                if len(filled_gaps_RRI[i]) == 0:
                    filled_gaps_RRI.pop(i)
                if len(filled_gaps_MAD[i]) == 0:
                    filled_gaps_MAD.pop(i)
                if len(filled_gaps_SAE[i]) == 0:
                    filled_gaps_SAE.pop(i)

            for i in range(len(filled_gaps_RRI)):
                if len(filled_gaps_RRI[i]) / RRI_frequency != len(filled_gaps_MAD[i]) / MAD_frequency or len(filled_gaps_RRI[i]) / RRI_frequency != len(filled_gaps_SAE[i]) / SAE_frequency:
                    print(len(filled_gaps_RRI[i]) / RRI_frequency, len(filled_gaps_MAD[i]) / MAD_frequency, len(filled_gaps_SAE[i]) / SAE_frequency)
                    raise ValueError(f"RRI, MAD and SAE data lengths do not match for patient: {patient_id} at index {i}")
        else:
            filled_gaps_RRI = synchronized_RRI
            filled_gaps_MAD = synchronized_MAD
            filled_gaps_SAE = synchronized_SAE
            filled_gaps_times_psg = start_end_valid_region_psg_times

        with open(new_save_file_path, "ab") as f:
            # create new datapoints and save them
            count_removed = 0
            for i in range(len(filled_gaps_RRI)):

                new_data_dict = dict()
                if i-count_removed >= 1:
                    new_data_dict["ID"] = patient_id + "_" + str(i-count_removed)
                else:
                    new_data_dict["ID"] = patient_id

                if filled_gaps_times_psg[i][1] - filled_gaps_times_psg[i][0] < min_data_length_seconds * ecg_sampling_frequency:
                    count_removed += 1
                    continue
                
                # somno time = (psg time + slope * 2h - shift) / (1 + slope)
                this_start_time_somno = (filled_gaps_times_psg[i][0] + slope * 7200 - time_shift) / (1 + slope)
                this_end_time_somno = (filled_gaps_times_psg[i][1] + slope * 7200 - time_shift) / (1 + slope)

                new_data_dict["start_date"] = somno_rec_date
                new_data_dict["start_time_somno"] = this_start_time_somno
                new_data_dict["end_time_somno"] = this_end_time_somno
                new_data_dict["start_time_psg"] = filled_gaps_times_psg[i][0]
                new_data_dict["end_time_psg"] = filled_gaps_times_psg[i][1]

                new_data_dict["RRI"] = filled_gaps_RRI[i]
                new_data_dict["RRI_frequency"] = RRI_frequency
                new_data_dict["MAD"] = filled_gaps_MAD[i]
                new_data_dict["MAD_frequency"] = MAD_frequency
                new_data_dict["SAE"] = filled_gaps_SAE[i]
                new_data_dict["SAE_frequency"] = SAE_frequency

                pickle.dump(new_data_dict, f)


def ADD_SLP_TO_GIF(
        gif_path: str,
        slp_files_directory: str,
        lights_and_time_shift_csv_path: str,
        munich_rri_directory: str,
        check_rpeak_similarity: bool = False,
        min_data_length_seconds: int = 5*3600,
        fill_ecg_gaps_threshold_seconds: int = 3*60,
    ):
    """
    Adds SLP data to a GIF file.

    PARAMETERS:
    --------------------------------
    gif_path: str
        Path to the input GIF file.

    slp_path: str
        Path to the SLP data file.

    output_path: str
        Path to the output GIF file with SLP data added.

    RETURNS:
    --------------------------------
    None, but the output GIF file will be created.
    """

    # load time shift and times of lights off and on
    lights_and_time_shift = pd.read_csv(lights_and_time_shift_csv_path, sep=" ")

    # access preprocessed GIF data
    results_generator = load_from_pickle(gif_path)

    sleep_signal_distances = list()

    for data_dict in results_generator:
        # add Rpeak from Munich for sections where original were missing
        patient_id = data_dict["ID"]
        if patient_id != "SL228":
            continue
        
        valid_ecg_regions = data_dict["valid_ecg_regions"]
        hamilton_rpeaks = np.unique(data_dict["hamilton"])
        rec_date = data_dict["start_date"]
        rec_time = data_dict["start_time"]
        ecg_sampling_frequency = data_dict["ECG_frequency"]
        # print(rec_date, rec_time, patient_id)
        
        munich_rri_path = munich_rri_directory + patient_id
        
        if not os.path.isfile(munich_rri_path):
            continue

        open_munich_file = open(munich_rri_path, "rb")
        rri_file_lines = open_munich_file.readlines()
        open_munich_file.close()

        munich_rpeaks = list()

        header_finished = False
        for line in rri_file_lines:
            line = line.decode("utf-8").strip()

            if line[0:9] == "rec-date=":
                munich_rec_date = line[9:]
                munich_rec_date = str(int(munich_rec_date[-4:])) + "-" + str(int(munich_rec_date[3:5])) + "-" + str(int(munich_rec_date[0:2]))
            elif line[0:9] == "rec-time=":
                munich_rec_time = line[9:]
                munich_rec_time = str(int(munich_rec_time[0:2])) + ":" + str(int(munich_rec_time[3:5])) + ":" + str(int(munich_rec_time[6:8]))
            elif line[0] == "-":
                header_finished = True
                continue

            if header_finished:
                for string_position in range(0, len(line)):
                    if not line[string_position].isdigit():
                        munich_rpeaks.append(int(line[0:string_position]))
                        break
        
        munich_rpeaks = np.unique(munich_rpeaks)

        if munich_rec_date != rec_date or munich_rec_time != rec_time:
            print("Munich RRI file date/time does not match GIF data:", munich_rec_date, munich_rec_time, "vs", rec_date, rec_time)
            continue

        rec_time_numbers = rec_time.split(":")
        if len(rec_time_numbers) == 3:
            somno_start_time_seconds = int(rec_time_numbers[0]) * 3600 + int(rec_time_numbers[1]) * 60 + int(rec_time_numbers[2])
        elif len(rec_time_numbers) == 2:
            somno_start_time_seconds = int(rec_time_numbers[0]) * 60 + int(rec_time_numbers[1])
        end_somno_time_seconds = somno_start_time_seconds + hamilton_rpeaks[-1]/ecg_sampling_frequency
            
        if check_rpeak_similarity:
            # check rpeak differences
            total_difference = 0
            number_of_differences = 0
            upper_bound_hamilton = len(hamilton_rpeaks) - 1
            upper_bound_munich = len(munich_rpeaks) - 1

            for ham_rpeak_pos in range(len(hamilton_rpeaks)):
                print(ham_rpeak_pos, end="\r")

                min_difference = hamilton_rpeaks[-1]
                last_position = 0
                for mu_rpeak_pos in range(last_position, len(munich_rpeaks)):
                    this_difference = abs(munich_rpeaks[mu_rpeak_pos] - hamilton_rpeaks[ham_rpeak_pos])
                    if this_difference < min_difference:
                        min_difference = this_difference
                    else:
                        total_difference += min_difference
                        number_of_differences += 1
                        last_position = mu_rpeak_pos + 1
                        if mu_rpeak_pos + 1 >= upper_bound_munich or ham_rpeak_pos + 1 >= upper_bound_hamilton:
                            break
                        min_difference = abs(munich_rpeaks[mu_rpeak_pos] - hamilton_rpeaks[ham_rpeak_pos + 1])
                        break
            
            if number_of_differences > 0:
                print(patient_id, total_difference / number_of_differences)
            else:
                print(patient_id, "No differences found")
        
            # check munich rpeaks outside of valid ecg regions
            mu_outside_valid_regions = [[] for _ in range(len(valid_ecg_regions)+1)]
            for mu_rpeak in munich_rpeaks:
                if mu_rpeak < valid_ecg_regions[0][0]:
                    mu_outside_valid_regions[0].append(mu_rpeak)
                else:
                    break
            for mu_rpeak_position in range(len(munich_rpeaks)):
                mu_rpeak = munich_rpeaks[mu_rpeak_position]
                # check if the rpeak is within a valid region
                for valid_region_position in range(len(valid_ecg_regions)):
                    if valid_ecg_regions[valid_region_position-1][1] < mu_rpeak < valid_ecg_regions[valid_region_position][0]:
                        mu_outside_valid_regions[valid_region_position+1].append(mu_rpeak)
                        break
                if mu_rpeak > valid_ecg_regions[-1][1]:
                    break
            for mu_position in range(mu_rpeak_position, len(munich_rpeaks)):
                mu_outside_valid_regions[-1].append(munich_rpeaks[mu_position])

            mu_outside_valid_regions_distances = [[mu_outside_valid_regions[j][i+1] - mu_outside_valid_regions[j][i] for i in range(len(mu_outside_valid_regions[j])-1)] for j in range(len(mu_outside_valid_regions))]
            for distance_array in mu_outside_valid_regions_distances:
                if len(distance_array) == 0:
                    distance_array.append(-1)
            mu_outside_valid_regions_mean_distance = [np.mean(mu_outside_valid_regions_distances[j]) for j in range(len(mu_outside_valid_regions_distances)) if len(mu_outside_valid_regions_distances[j]) > 0]

            print(patient_id)
            print(valid_ecg_regions)
            print(mu_outside_valid_regions_mean_distance)
        
        # load SLP data
        slp_file_path = slp_files_directory + patient_id + ".slp"
        if not os.path.isfile(slp_file_path):
            print("SLP file not found for patient:", patient_id)
            continue

        slp_file = open(slp_file_path, "rb")
        slp_file_lines = slp_file.readlines()
        slp_file.close()

        numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

        first_slp_line = slp_file_lines[0].decode("utf-8").strip()
        final_index = len(first_slp_line)
        for char_pos in range(len(first_slp_line)):
            if first_slp_line[char_pos] not in numbers:
                final_index = char_pos
                break
        slp_start_time_psg = first_slp_line[0:final_index]
        slp_start_time_seconds_psg = float(slp_start_time_psg) * 3600

        last_slp_line = slp_file_lines[-1].decode("utf-8").strip()
        final_index = len(last_slp_line)
        for char_pos in range(len(last_slp_line)):
            if last_slp_line[char_pos] not in numbers:
                final_index = char_pos
                break
        slp_end_time_psg = last_slp_line[0:final_index]
        slp_end_time_seconds_psg = float(slp_end_time_psg) * 3600

        del numbers, first_slp_line, last_slp_line, final_index

        slp = list()
        for slp_line in slp_file_lines:
            slp_line = slp_line.decode("utf-8").strip()
            if len(slp_line) > 1:
                continue
            slp.append(int(slp_line))

        # access time shift of this patient
        time_shift = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["time_shift"].values
        lights_off = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["lights_off"].values
        lights_on = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["lights_on"].values
        slope = lights_and_time_shift[lights_and_time_shift["subject"] == patient_id]["slope"].values
        if len(time_shift) == 0:
            time_shift, lights_off, lights_on, slope = 0, 0, 0, 0
        else:
            time_shift = float(time_shift[0])
            lights_off = float(lights_off[0])
            lights_on = float(lights_on[0])
            slope = float(slope[0])
        
        # synchronize ECG data with PSG data (psg time = somno time + shift + slope * (somno time - 2h))
        # somno time = (psg time + slope * 2h - shift) / (1 + slope)
        # shift and slope constants => psg1 - psg2 = (somno1 - somno2) (1 + slope)
        # => RRI = rpeak2 - rpeak1 => change rpeak locations (time in somno) according formula: rpeak = rpeak (1 + slope)
        #COMMENTS ABOVE NOT IMPORTANT ANYMORE, COMPLETE SYNCHRONIZATION NOT POSSIBLE AS MAD ALREADY CALCULATED

        # synchronize ECG data with PSG data (psg time = somno time + shift + slope * (somno time - 2h))

        # MAIN PROBLEM: MAD Calculated before synchronization, interpolation not possible due to MAD formula
        # RRI already synchronized on MAD time points
        # SLP values must be shifted in time to be in sync (low error: MAD every 1s -> Maximum distance = 0.5s = 1/60 SLP stage 'area')
        
        # access MAD and RRI (already synchronized, but not in same shape)
        basic_RRI = data_dict["RRI"]
        RRI_frequency = data_dict["RRI_frequency"]
        basic_MAD = data_dict["MAD"]
        MAD_frequency = data_dict["MAD_frequency"]

        synchronized_RRI = list()
        synchronized_MAD = list()
        synchronized_SLP = list()
        synchronized_signals_start_time_seconds_somno = list()

        # psg time = somno time + shift + slope * (somno time - 2h)
        # somno time = (psg time + slope * 2h - shift) / (1 + slope)
        slp_start_time_seconds_somno = (slp_start_time_seconds_psg + slope * 7200 - time_shift) / (1 + slope)
        somno_psg_start_time_distance_somno = somno_start_time_seconds - slp_start_time_seconds_somno

        print(valid_ecg_regions)
        print((np.array(valid_ecg_regions)/ecg_sampling_frequency+somno_start_time_seconds)/3600)

        data_time_region = list()

        for i in range(len(valid_ecg_regions)):
            valid_interval = valid_ecg_regions[i]

            # find first signal position within valid ecg region that is shared by all signals (time point where each signal writes the next value)
            # look in rri calculation: rri start is equal to first_shared_signal_position, not valid_interval[0]
            first_shared_signal_position = find_time_point_shared_by_signals(
                signal_position = valid_interval[0],
                signal_sampling_frequency = ecg_sampling_frequency,
                other_sampling_frequencies = [RRI_frequency, MAD_frequency],
                update_position_by = int(1)
            )

            # find slp time point that comes after time point of first shared signal position
            first_shared_signal_position_time_seconds = first_shared_signal_position / ecg_sampling_frequency
            upper_slp_time = slp_start_time_seconds_somno - somno_start_time_seconds
            if slp_start_time_seconds_somno > first_shared_signal_position_time_seconds:
                while True:
                    if upper_slp_time < first_shared_signal_position_time_seconds:
                        upper_slp_time += 30
                        break
                    upper_slp_time -= 30
            else:
                while True:
                    if upper_slp_time > first_shared_signal_position_time_seconds:
                        break
                    upper_slp_time += 30
            
            # now (as step size is lower) find signal position closest to the upper_slp_time
            upper_slp_time_in_ecg = upper_slp_time * ecg_sampling_frequency
            last_distance = abs(upper_slp_time_in_ecg - first_shared_signal_position)
            update_signal_position = first_shared_signal_position
            while True:
                current_distance = abs(upper_slp_time_in_ecg - update_signal_position)
                if current_distance > last_distance:
                    update_signal_position -= ecg_sampling_frequency
                    break
                
                last_distance = current_distance
                update_signal_position += ecg_sampling_frequency

            sleep_signal_distances.append(last_distance)
            skip_datapoints = update_signal_position - first_shared_signal_position
            first_shared_signal_position = update_signal_position

            # find amount of datapoints so that a whole number of RRI, MAD and SLP values fit into interval without exceeding valid_interval[1]
            for j in range(valid_interval[1], -1, -1):
                this_distance_seconds = (j - first_shared_signal_position) / ecg_sampling_frequency
                all_frequencies_fit = True
                for frequency in [RRI_frequency, MAD_frequency, 1/30]:
                    if (this_distance_seconds * frequency) % 1 != 0:
                        all_frequencies_fit = False
                        break
                if all_frequencies_fit:
                    last_shared_signal_position = j
                    break
            
            data_time_region.append((first_shared_signal_position, last_shared_signal_position))

            synchronized_RRI.append(basic_RRI[i][int(skip_datapoints*RRI_frequency/ecg_sampling_frequency) : int((last_shared_signal_position-first_shared_signal_position+skip_datapoints)*RRI_frequency/ecg_sampling_frequency)])
            synchronized_MAD.append(basic_MAD[int(first_shared_signal_position*MAD_frequency/ecg_sampling_frequency) : int(last_shared_signal_position*MAD_frequency/ecg_sampling_frequency)])
            synchronized_signals_start_time_seconds_somno.append(int(first_shared_signal_position/ecg_sampling_frequency))

            # retreive SLP values in the same time period
            number_required_slp_values = int((last_shared_signal_position-first_shared_signal_position)/30/ecg_sampling_frequency)
            start_index_slp = int(round((first_shared_signal_position/ecg_sampling_frequency + somno_psg_start_time_distance_somno) / 30))
            this_slp_values = list()
            
            for j in range(start_index_slp, number_required_slp_values+start_index_slp):
                if j < 0 or j >= len(slp):
                    this_slp_values.append(0)
                else:
                    this_slp_values.append(slp[j])
            
            synchronized_SLP.append(this_slp_values)
        
        for i in range(len(synchronized_RRI)):
            print(len(synchronized_RRI[i])/RRI_frequency, len(synchronized_MAD[i])/MAD_frequency, len(synchronized_SLP[i])*30)
            if len(synchronized_RRI[i])/RRI_frequency != len(synchronized_MAD[i])/MAD_frequency or len(synchronized_RRI[i])/RRI_frequency != len(synchronized_SLP[i])*30:
                print(len(synchronized_RRI[i])/RRI_frequency, len(synchronized_MAD[i])/MAD_frequency, len(synchronized_SLP[i])*30)
                raise SystemError(f"RRI, MAD and SLP data lengths do not match for patient: {patient_id} at index {i}")
        
        if fill_ecg_gaps_threshold_seconds > 0:
            ecg_gaps = []
            for i in range(len(data_time_region)-1):
                ecg_gaps.append((data_time_region[i+1][0] - data_time_region[i][1])/ecg_sampling_frequency)
            
            print(ecg_gaps)

            filled_gaps_RRI = list([] for _ in range(len(synchronized_RRI)))
            filled_gaps_RRI[0] = list(synchronized_RRI[0])
            filled_gaps_MAD = list([] for _ in range(len(synchronized_MAD)))
            filled_gaps_MAD[0] = list(synchronized_MAD[0])
            filled_gaps_SLP = list([] for _ in range(len(synchronized_SLP)))
            filled_gaps_SLP[0] = list(synchronized_SLP[0])
            filled_gaps_data_times = [list(data_time_region[0])]

            append_to = 0
            for i in range(1, len(synchronized_RRI)):
                # fill gaps in RRI, MAD and SLP data
                if ecg_gaps[i-1] > fill_ecg_gaps_threshold_seconds:
                    append_to += 1
                    filled_gaps_data_times[-1][1] = data_time_region[i-1][1]
                    filled_gaps_data_times.append(list(data_time_region[i]))
                    # fill gaps with artifact values
                else:
                    filled_gaps_RRI[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * RRI_frequency))])
                    filled_gaps_MAD[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * MAD_frequency))])
                    filled_gaps_SLP[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] / 30))])

                filled_gaps_RRI[append_to].extend(synchronized_RRI[i])
                filled_gaps_MAD[append_to].extend(synchronized_MAD[i])
                filled_gaps_SLP[append_to].extend(synchronized_SLP[i])
            
            filled_gaps_data_times[append_to][1] = data_time_region[-1][1]
            
            for i in range(len(filled_gaps_RRI)-1, -1, -1):
                if len(filled_gaps_RRI[i]) == 0:
                    filled_gaps_RRI.pop(i)
                if len(filled_gaps_MAD[i]) == 0:
                    filled_gaps_MAD.pop(i)
                if len(filled_gaps_SLP[i]) == 0:
                    filled_gaps_SLP.pop(i)
        else:
            filled_gaps_RRI = synchronized_RRI
            filled_gaps_MAD = synchronized_MAD
            filled_gaps_SLP = synchronized_SLP
        
        for i in range(len(filled_gaps_RRI)):
            print(len(filled_gaps_RRI[i])/RRI_frequency, len(filled_gaps_MAD[i])/MAD_frequency, len(filled_gaps_SLP[i])*30)
            # raise SystemError(f"RRI, MAD and SLP data lengths do not match for patient: {patient_id} at index {i}")
        
        print(np.array(data_time_region) / ecg_sampling_frequency)
        print(filled_gaps_data_times)

        # create new datapoints and save them
        count_removed = 0
        for i in range(len(filled_gaps_RRI)):

            new_data_dict = dict()
            if i-count_removed >= 1:
                new_data_dict["ID"] = patient_id + "_" + str(i-count_removed)
            else:
                new_data_dict["ID"] = patient_id
            
            if filled_gaps_data_times[i][1] - filled_gaps_data_times[i][0] < min_data_length_seconds * ecg_sampling_frequency:
                count_removed += 1
                continue

            this_start_time_somno = filled_gaps_data_times[i][0] / ecg_sampling_frequency + somno_start_time_seconds
            this_end_time_somno = filled_gaps_data_times[i][1] / ecg_sampling_frequency + somno_start_time_seconds

            new_data_dict["start_date"] = rec_date
            new_data_dict["start_time_somno"] = this_start_time_somno
            new_data_dict["start_time_psg"] = this_start_time_somno + time_shift + slope * (this_start_time_somno - 7200)
            new_data_dict["end_time_somno"] = this_end_time_somno
            new_data_dict["end_time_psg"] = this_end_time_somno + time_shift + slope * (this_end_time_somno - 7200)

            new_data_dict["RRI"] = filled_gaps_RRI[i]
            new_data_dict["RRI_frequency"] = RRI_frequency
            new_data_dict["MAD"] = filled_gaps_MAD[i]
            new_data_dict["MAD_frequency"] = MAD_frequency
            new_data_dict["SLP"] = filled_gaps_SLP[i]
            new_data_dict["SLP_frequency"] = 1/30  # SLP is sampled every 30 seconds


def apnea_info(path = "Data/GIF/sleep_apnea_events/"):

    embla_ids = [file for file in os.listdir("Data/GIF/SAE_Embla/") if file.endswith(".csv")]
    somnoscreen_ids = [file for file in os.listdir("Data/GIF/SAE_Somnoscreen/") if file.endswith(".csv")]
    alice_ids = [file for file in os.listdir("Data/GIF/SAE_Alice/") if file.endswith(".csv")]

    count_alice_overlap = [0, 0]
    count_somnoscreen_overlap = [0, 0]
    count_embla_overlap = [0, 0]

    files_with_overlap = 0
    
    durations = list()
    apnea_classes = list()
    classes_frequency = list()
    files = os.listdir(path)

    for file in files:
        file_already_overlapped = False
        if not file.endswith(".csv"):
            continue

        source = "Unknown"
        if file in alice_ids:
            source = "Alice"
        elif file in somnoscreen_ids:
            source = "SomnoScreen"
        elif file in embla_ids:
            source = "Embla"
    
        file_path = os.path.join(path, file)
        data = pd.read_csv(file_path, sep=",")
        for line in data.values:
            # print(line[0], line[1], line[2], line[3])
            duration = float(line[3])
            apnea = str(line[0])
            if apnea not in apnea_classes:
                apnea_classes.append(apnea)
                classes_frequency.append(1)
                durations.append([duration])
            else:
                classes_frequency[apnea_classes.index(apnea)] += 1
                durations[apnea_classes.index(apnea)].append(duration)
            # start_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
            # end_time = datetime.strptime(line[2], "%Y-%m-%d %H:%M:%S")
        
        # check time overlap
        start_times = data["Start Time (YYYY-MM-DD HH:MM:SS)"].values
        end_times = data["End Time (YYYY-MM-DD HH:MM:SS)"].values
        all_durations = data["Duration (seconds)"].values
        all_events = data["Event"].values
        
        file_events_duration = sum(all_durations)
        total_overlap = 0
        event_overlap = []

        for i in range(len(start_times)):
            check_this_start = datetime.strptime(start_times[i], "%Y-%m-%d %H:%M:%S")
            check_this_end = datetime.strptime(end_times[i], "%Y-%m-%d %H:%M:%S")

            for j in range(i+1, len(start_times)):
                if all_events[i] == all_events[j]:
                    continue

                compare_start = datetime.strptime(start_times[j], "%Y-%m-%d %H:%M:%S")
                compare_end = datetime.strptime(end_times[j], "%Y-%m-%d %H:%M:%S")

                if check_this_end <= compare_start or compare_end <= check_this_start:
                    total_overlap += 0
                else:
                    if not file_already_overlapped:
                        files_with_overlap += 1
                    file_already_overlapped = True
                    latest_start = max(check_this_start, compare_start)
                    earliest_end = min(check_this_end, compare_end)
                    this_overlap = (earliest_end - latest_start).total_seconds()
                    total_overlap += this_overlap
                    if [all_events[i], all_events[j]] not in event_overlap and [all_events[j], all_events[i]] not in event_overlap:
                        event_overlap.append([all_events[i], all_events[j]])
                    
                    if file in alice_ids:
                        count_alice_overlap[0] += 1
                        count_alice_overlap[1] += this_overlap # type: ignore
                    elif file in somnoscreen_ids:
                        count_somnoscreen_overlap[0] += 1
                        count_somnoscreen_overlap[1] += this_overlap # type: ignore
                    elif file in embla_ids:
                        count_embla_overlap[0] += 1
                        count_embla_overlap[1] += this_overlap # type: ignore

                    # print("File:", file, "Event", i, "and", j, "overlap by", this_overlap, "seconds, start1:", check_this_start, "end1:", check_this_end, "start2:", compare_start, "end2:", compare_end)

        if file_already_overlapped:
            print("File:", file," | Source:", source, " | Total events duration:", file_events_duration, " | Total overlap duration:", total_overlap, " | Overlap percentage:", total_overlap/file_events_duration*100 if file_events_duration > 0 else 0)
            print("Overlapping event types:", event_overlap)
    
    print()
    print("Total files with overlaps:", files_with_overlap)
    print()
    print("Embla Overlapping Instances / Duration:", count_embla_overlap)
    print("SomnoScreen Overlapping Instances / Duration:", count_somnoscreen_overlap)
    print("Alice Overlapping Instances / Duration:", count_alice_overlap)

    print()
    print(apnea_classes)
    print(classes_frequency)

    save_dict = {
        "apnea_classes": apnea_classes,
        "durations": durations,
        "frequency": classes_frequency
    }

    with open("gif_apnea_info.pkl", "wb") as f:
        pickle.dump(save_dict, f)


def check_apnea_dataset():
    data = load_from_pickle("gif_apnea_test.pkl")
    show_time = datetime.strptime("2018-07-23 22:49:46", "%Y-%m-%d %H:%M:%S")

    for entry in data:
        print(entry["ID"], len(entry["RRI"]), len(entry["MAD"]), len(entry["SAE"]))
        print(entry["RRI"][:10], entry["MAD"][:10], entry["SAE"][:10])
        print()
        show_index = (show_time.date() - datetime.strptime(entry["start_date"], "%Y-%m-%d").date()).days * 24 * 3600 + show_time.hour * 3600 + show_time.minute * 60 + show_time.second - entry["start_time_psg"]
        show_index = better_int(show_index)
        print(show_time, show_index)
        print(entry["SAE"][show_index:show_index+30])
        print()


def map_string_to_apnea_event(apnea_string: str):
    central_keywords = ["central", "zentral"]
    mixed_keywords = ["mixed", "gemischt"]
    obstructive_keywords = ["obstructive", "obstruktiv"]
    hypopnea_keywords = ["hypopnea", "hypopnoe", "hypopneo", "hypopnae"]
    apnea_keywords = ["apnea", "apnoe", "apneo", "apnae"]

    new_string = ""

    apnea_string = apnea_string.lower()
    for keyword in central_keywords:
        if keyword in apnea_string:
            new_string += "Central"
            break
    for keyword in mixed_keywords:
        if keyword in apnea_string:
            new_string += "Mixed"
            break
    for keyword in obstructive_keywords:
        if keyword in apnea_string:
            new_string += "Obstructive"
            break
    for keyword in hypopnea_keywords:
        if keyword in apnea_string:
            if len(new_string) > 0:
                new_string += " "
            new_string += "Hypopnea"
            break
    for keyword in apnea_keywords:
        if keyword in apnea_string:
            if len(new_string) > 0:
                new_string += " "
            new_string += "Apnea"
            break
    
    if "Apnea" not in new_string and "Hypopnea" not in new_string:
        raise ValueError("Could not map apnea string:", apnea_string)
    
    # if " " not in new_string:
    #     print("Warning: Mapped apnea string has no space, original string:", apnea_string, "mapped to:", new_string)

    return new_string


def map_string_to_date(date_string: str):
    numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    date_symbols = ["/", "\\", "-", "."]

    for char_index in range(len(date_string)-9):
        if date_string[char_index] in numbers and date_string[char_index+1] in numbers and date_string[char_index+2] in numbers and date_string[char_index+3] in numbers and date_string[char_index+4] in date_symbols and date_string[char_index+5] in numbers and date_string[char_index+6] in numbers and date_string[char_index+7] in date_symbols and date_string[char_index+8] in numbers and date_string[char_index+9] in numbers:
            year = date_string[char_index:char_index+4]
            month = date_string[char_index+5:char_index+7]
            day = date_string[char_index+8:char_index+10]
            return f"{year}-{month}-{day}"
        elif date_string[char_index] in numbers and date_string[char_index+1] in numbers and date_string[char_index+2] in date_symbols and date_string[char_index+3] in numbers and date_string[char_index+4] in numbers and date_string[char_index+5] in date_symbols and date_string[char_index+6] in numbers and date_string[char_index+7] in numbers and date_string[char_index+8] in numbers and date_string[char_index+9] in numbers:
            day = date_string[char_index:char_index+2]
            month = date_string[char_index+3:char_index+5]
            year = date_string[char_index+6:char_index+10]
            return f"{year}-{month}-{day}"
    
    raise ValueError("Could not map date string:", date_string)


def map_string_to_duration(duration_string: str):
    numbers_and_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
    duration_string = duration_string.replace(",", ".")

    within_number = False
    for char_index in range(len(duration_string)):
        if duration_string[char_index] in numbers_and_symbols and not within_number:
            start_index = char_index
            within_number = True
        elif duration_string[char_index] not in numbers_and_symbols and within_number:
            return float(duration_string[start_index:char_index])
    
    return float(duration_string[start_index:])


def uniform_apnea_files(
        path = "Data/GIF/apnea_original/",
        new_directory = "Data/GIF/sleep_apnea_events/"
    ):
    """
    """

    numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    numbers_and_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","]
    date_symbols = ["/", "\\", "-", "."]

    all_files = os.listdir(path)
    for file in all_files:
        
        if file.endswith("Ereignisse.txt"):
            file_path = os.path.join(path, file)
            file_lines = open(file_path, "rb").readlines()
            file_lines = [line.decode("utf-8", errors="ignore").strip() for line in file_lines]

            suggested_patient_id = file[0:5]
            suggested_date = None
            suggested_start_time = None
            suggested_table_start = 0

            check_max_lines = min(50, len(file_lines))

            for line_index in range(0, check_max_lines):
                line = file_lines[line_index]
                if suggested_date is None:
                    for char_index in range(len(line)-9):
                        if line[char_index] in numbers and line[char_index+1] in numbers and line[char_index+2] in numbers and line[char_index+3] in numbers and line[char_index+4] in date_symbols and line[char_index+5] in numbers and line[char_index+6] in numbers and line[char_index+7] in date_symbols and line[char_index+8] in numbers and line[char_index+9] in numbers:
                            year = line[char_index:char_index+4]
                            month = line[char_index+5:char_index+7]
                            day = line[char_index+8:char_index+10]
                            suggested_date = f"{year}-{month}-{day}"
                            break
                        elif line[char_index] in numbers and line[char_index+1] in numbers and line[char_index+2] in date_symbols and line[char_index+3] in numbers and line[char_index+4] in numbers and line[char_index+5] in date_symbols and line[char_index+6] in numbers and line[char_index+7] in numbers and line[char_index+8] in numbers and line[char_index+9] in numbers:
                            day = line[char_index:char_index+2]
                            month = line[char_index+3:char_index+5]
                            year = line[char_index+6:char_index+10]
                            suggested_date = f"{year}-{month}-{day}"
                            break
                
                if len(line) < 2:
                    suggested_table_start = line_index + 2
            
            for line_index in range(suggested_table_start, check_max_lines):
                line = file_lines[line_index]
                if suggested_start_time is None:
                    for char_index in range(len(line)-7):
                        if line[char_index] in numbers and line[char_index+1] in numbers and line[char_index+2] == ":" and line[char_index+3] in numbers and line[char_index+4] in numbers and line[char_index+5] == ":" and line[char_index+6] in numbers and line[char_index+7] in numbers:
                            suggested_start_time = line[char_index:char_index+8]
                            break
                else:
                    break

            for line_index in range(0, check_max_lines):
                print(line_index, file_lines[line_index])

            print()
            print("="*80)
            print("Suggested patient ID:", suggested_patient_id)
            print("Suggested recording date:", suggested_date)
            print("Suggested recording start time:", suggested_start_time)
            print("Suggested table start line number:", suggested_table_start)
            print("="*80)

            answer = "y"
            # answer = input("\nDo you want to use the suggested values? (y/n): ")
            if answer.lower() == "y":
                rec_date = suggested_date
                rec_start_time = suggested_start_time
                patient_id = suggested_patient_id
                table_starts = suggested_table_start
            else:    
                rec_date = input("\nEnter recording date (YYYY-MM-DD): ")
                rec_start_time = input("Enter recording start time (HH:MM:SS): ")
                patient_id = input("Enter patient ID: ")
                table_starts = input("Enter line number where table starts: ")
            
            rec_start_time_seconds = datetime.strptime(rec_start_time, "%H:%M:%S") # type: ignore
            rec_start_time_seconds = rec_start_time_seconds.hour * 3600 + rec_start_time_seconds.minute * 60 + rec_start_time_seconds.second
            print()

            new_apnea_file = open(f"{new_directory}{patient_id}.csv", "w")
            new_apnea_file.write("Event,Start Time (YYYY-MM-DD HH:MM:SS),End Time (YYYY-MM-DD HH:MM:SS),Duration (seconds)\n")
            
            skipped_lines = []

            for line_index in range(int(table_starts), len(file_lines)):
                line = file_lines[line_index]
                lowercase_line = line.lower()
                
                if "pnea" not in lowercase_line and "pnae" not in lowercase_line and "pnoe" not in lowercase_line and "pneo" not in lowercase_line:
                    skipped_lines.append(line)
                    continue

                this_apnea_event = map_string_to_apnea_event(lowercase_line)

                # retrieve event start time
                start_time_string = None
                for line_index in range(len(line)-7):
                    if line[line_index] in numbers and line[line_index+1] in numbers and line[line_index+2] == ":" and line[line_index+3] in numbers and line[line_index+4] in numbers and line[line_index+5] == ":" and line[line_index+6] in numbers and line[line_index+7] in numbers:
                        for end_index in range(line_index+8, len(line)):
                            if line[end_index] not in numbers_and_symbols:
                                break
                        start_time_string = line[line_index:end_index]
                        break

                if start_time_string is None:
                    print("Could not find start time in line:", line)
                    start_time_string = input("Enter start time (HH:MM:SS): ")
                    continue
                    
                # check if start time is smaller than rec time, if yes, add one day
                start_time_seconds = datetime.strptime(start_time_string, "%H:%M:%S")
                start_time_seconds = start_time_seconds.hour * 3600 + start_time_seconds.minute * 60 + start_time_seconds.second
                if start_time_seconds < rec_start_time_seconds:
                    rec_date_dt = datetime.strptime(rec_date, "%Y-%m-%d") # type: ignore
                    rec_date_dt += timedelta(days=1)
                    insert_rec_date = rec_date_dt.strftime("%Y-%m-%d")
                else:
                    insert_rec_date = rec_date

                start_time_string = f"{insert_rec_date} {start_time_string}"
                start_time = round_start_end_apnea_time(start_time_string, start_time_string)[0]
            
                # retrieve event duration
                within_number = False
                for line_index in range(len(line)-1, -1, -1):
                    if line[line_index] in numbers and not within_number:
                        end_index = line_index + 1
                        within_number = True
                    if line[line_index] not in numbers_and_symbols and within_number:
                        start_index = line_index + 1
                        break

                duration_string = line[start_index:end_index]
                for char_index in range(len(duration_string)):
                    if duration_string[char_index] == ",":
                        duration_string = duration_string[0:char_index] + "." + duration_string[char_index+1:]
                        break
                
                duration = float(duration_string)
                if int(duration) != duration:
                    duration = int(duration) + 1
                else:
                    duration = int(duration)
                
                end_time = start_time + timedelta(seconds=duration)

                start_time_string = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_time_string = end_time.strftime("%Y-%m-%d %H:%M:%S")

                new_line = f"{this_apnea_event},{start_time_string},{end_time_string},{duration}\n"

                print("-"*80)
                print(line)
                print("-"*80)
                print(new_line)
                print()

                new_apnea_file.write(new_line)

            new_apnea_file.close()

            # input("\nFinished file. Press Enter to continue...")

            print("\nSkipped lines:")
            for line in skipped_lines:
                print(line)

            # input("\nFinished Skipped Lines. Press Enter to continue...")

        elif file.endswith("- events.csv"):
            file_path = os.path.join(path, file)

            # fix file encoding
            file_lines = open(file_path, "rb").readlines()
            file_lines = [line.decode("utf-8", errors="ignore").strip() for line in file_lines]

            temporary_path = path + "temporary.csv"
            temporary_file = open(temporary_path, "wb")
            for line in file_lines:
                new_line = ""
                for char in line:
                    if char in ["\""]:
                        continue
                    new_line += char
                new_line += "\n"
                temporary_file.write(new_line.encode("utf-8"))
            temporary_file.close()

            dataframe = pd.read_csv(temporary_path, sep=";")

            patient_id = file[0:5]
            new_apnea_file_path = f"{new_directory}{patient_id}.csv"

            # Validierung, Datum, Zeit, Dauer, Ereignistyp

            # drop rows with Validierung == "-"
            # dataframe = dataframe[dataframe["Validierung"] != "-"]

            # collect rows to drop
            events = dataframe["Ereignistyp"].values
            validation = dataframe["Validierung"].values
            drop_indices = []

            for row_index in range(len(events)):
                current_event = str(events[row_index]).lower()
                current_validation = validation[row_index]

                if current_validation == "-":
                    drop_indices.append(row_index)
                    continue

                if "pnea" not in current_event and "pnae" not in current_event and "pnoe" not in current_event and "pneo" not in current_event:
                    drop_indices.append(row_index)
                    continue
            
            # collect dropped rows
            dropped_events = dataframe.iloc[drop_indices]

            # drop rows
            dataframe = dataframe.drop(index=drop_indices)

            # reacces all required columns
            required_events = [map_string_to_apnea_event(str(event).lower()) for event in dataframe["Ereignistyp"].values]

            required_dates = [map_string_to_date(str(date)) for date in dataframe["Datum"].values]
            required_start_times = dataframe["Zeit"].values

            start_time_strings = [f"{required_dates[i]} {required_start_times[i]}" for i in range(len(required_dates))]
            start_times = [round_start_end_apnea_time(start_time_strings[i], start_time_strings[i])[0] for i in range(len(start_time_strings))]

            required_durations = [int(np.round(map_string_to_duration(str(duration)))) for duration in dataframe["Dauer"].values] # type: ignore
            end_times = [start_times[i] + timedelta(seconds=required_durations[i]) for i in range(len(start_times))]
            
            end_time_strings = [end_times[i].strftime("%Y-%m-%d %H:%M:%S") for i in range(len(end_times))]
            start_time_strings = [start_times[i].strftime("%Y-%m-%d %H:%M:%S") for i in range(len(start_times))]
            
            # new dataframe
            new_dataframe = pd.DataFrame(columns=["Event", "Start Time (YYYY-MM-DD HH:MM:SS)", "End Time (YYYY-MM-DD HH:MM:SS)", "Duration (seconds)"])

            # append all columns
            new_dataframe["Event"] = required_events
            new_dataframe["Start Time (YYYY-MM-DD HH:MM:SS)"] = start_time_strings
            new_dataframe["End Time (YYYY-MM-DD HH:MM:SS)"] = end_time_strings
            new_dataframe["Duration (seconds)"] = required_durations

            # print skipped lines
            print("\nSkipped lines:")
            print("-"*80)
            pd.set_option('display.max_rows', 10000)
            print(dropped_events)
            print("-"*80)

            # save new dataframe
            new_dataframe.to_csv(new_apnea_file_path, index=False)

            # delete temporary file
            os.remove(temporary_path)

            # input("\nFinished file. Press Enter to continue...")

        elif file.endswith("Flow Events.txt"):
            file_path = os.path.join(path, file)
            file_lines = open(file_path, "rb").readlines()
            file_lines = [line.decode("utf-8", errors="ignore").strip() for line in file_lines]

            suggested_patient_id = file[0:5]
            suggested_date = None
            suggested_start_time = None
            suggested_table_start = 0

            check_max_lines = min(10, len(file_lines))

            for line_index in range(0, check_max_lines):
                line = file_lines[line_index]
                
                if suggested_date is None:
                    for char_index in range(len(line)-9):
                        if line[char_index] in numbers and line[char_index+1] in numbers and line[char_index+2] in numbers and line[char_index+3] in numbers and line[char_index+4] in date_symbols and line[char_index+5] in numbers and line[char_index+6] in numbers and line[char_index+7] in date_symbols and line[char_index+8] in numbers and line[char_index+9] in numbers:
                            year = line[char_index:char_index+4]
                            month = line[char_index+5:char_index+7]
                            day = line[char_index+8:char_index+10]
                            suggested_date = f"{year}-{month}-{day}"
                            break
                        elif line[char_index] in numbers and line[char_index+1] in numbers and line[char_index+2] in date_symbols and line[char_index+3] in numbers and line[char_index+4] in numbers and line[char_index+5] in date_symbols and line[char_index+6] in numbers and line[char_index+7] in numbers and line[char_index+8] in numbers and line[char_index+9] in numbers:
                            day = line[char_index:char_index+2]
                            month = line[char_index+3:char_index+5]
                            year = line[char_index+6:char_index+10]
                            suggested_date = f"{year}-{month}-{day}"
                            break
                
                if suggested_start_time is None:
                    for char_index in range(len(line)-7):
                        if line[char_index] in numbers and line[char_index+1] in numbers and line[char_index+2] == ":" and line[char_index+3] in numbers and line[char_index+4] in numbers and line[char_index+5] == ":" and line[char_index+6] in numbers and line[char_index+7] in numbers:
                            suggested_start_time = line[char_index:char_index+8]
                            break
                
                if len(line) < 2:
                    suggested_table_start = line_index + 1

            for line_index in range(0,check_max_lines):
                print(line_index, file_lines[line_index])

            print()
            print("="*80)
            print("Suggested patient ID:", suggested_patient_id)
            print("Suggested recording date:", suggested_date)
            print("Suggested recording start time:", suggested_start_time)
            print("Suggested table start line number:", suggested_table_start)
            print("="*80)

            answer = "y"
            # answer = input("\nDo you want to use the suggested values? (y/n): ")
            if answer.lower() == "y":
                rec_date = suggested_date
                rec_start_time = suggested_start_time
                patient_id = suggested_patient_id
                table_starts = suggested_table_start
            else:    
                rec_date = input("\nEnter recording date (YYYY-MM-DD): ")
                rec_start_time = input("Enter recording start time (HH:MM:SS): ")
                patient_id = input("Enter patient ID: ")
                table_starts = input("Enter line number where table starts: ")
            
            rec_start_time_seconds = datetime.strptime(rec_start_time, "%H:%M:%S") # type: ignore
            rec_start_time_seconds = rec_start_time_seconds.hour * 3600 + rec_start_time_seconds.minute * 60 + rec_start_time_seconds.second
            print()

            new_apnea_file = open(f"{new_directory}{patient_id}.csv", "w")
            new_apnea_file.write("Event,Start Time (YYYY-MM-DD HH:MM:SS),End Time (YYYY-MM-DD HH:MM:SS),Duration (seconds)\n")
            
            skipped_lines = []

            for line_index in range(int(table_starts), len(file_lines)):
                line = file_lines[line_index]
                lowercase_line = line.lower()
                
                if "pnea" not in lowercase_line and "pnae" not in lowercase_line and "pnoe" not in lowercase_line and "pneo" not in lowercase_line:
                    skipped_lines.append(line)
                    continue

                this_apnea_event = map_string_to_apnea_event(lowercase_line)

                # retrieve event start time
                first_semicolon_index = None
                for line_index in range(len(line)):
                    if line[line_index] == "-":
                        divis_index = line_index
                    if line[line_index] == ";" and first_semicolon_index is None:
                        first_semicolon_index = line_index
                    if line[line_index] == ";" and first_semicolon_index is not None:
                        second_semicolon_index = line_index
                        break
                
                start_time_string = line[:divis_index]
                end_time_string = line[divis_index+1:first_semicolon_index]
                    
                # check if start time is smaller than rec time, if yes, add one day
                start_time_seconds = datetime.strptime(start_time_string[:8], "%H:%M:%S")
                start_time_seconds = start_time_seconds.hour * 3600 + start_time_seconds.minute * 60 + start_time_seconds.second
                if start_time_seconds < rec_start_time_seconds:
                    rec_date_dt = datetime.strptime(rec_date, "%Y-%m-%d") # type: ignore
                    rec_date_dt += timedelta(days=1)
                    insert_rec_date = rec_date_dt.strftime("%Y-%m-%d")
                else:
                    insert_rec_date = rec_date
                start_time_string = f"{insert_rec_date} {start_time_string}"

                # check if end time is smaller than rec time, if yes, add one day
                end_time_seconds = datetime.strptime(end_time_string[:8], "%H:%M:%S")
                end_time_seconds = end_time_seconds.hour * 3600 + end_time_seconds.minute * 60 + end_time_seconds.second
                if end_time_seconds < rec_start_time_seconds:
                    rec_date_dt = datetime.strptime(rec_date, "%Y-%m-%d") # type: ignore
                    rec_date_dt += timedelta(days=1)
                    insert_rec_date = rec_date_dt.strftime("%Y-%m-%d")
                else:
                    insert_rec_date = rec_date
                end_time_string = f"{insert_rec_date} {end_time_string}"

                start_time, end_time = round_start_end_apnea_time(start_time_string, end_time_string)
            
                duration = better_int((end_time - start_time).total_seconds())

                start_time_string = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_time_string = end_time.strftime("%Y-%m-%d %H:%M:%S")

                new_line = f"{this_apnea_event},{start_time_string},{end_time_string},{duration}\n"

                print("-"*80)
                print(line)
                print("-"*80)
                print(new_line)
                print()

                new_apnea_file.write(new_line)

            new_apnea_file.close()

            # input("\nFinished file. Press Enter to continue...")

            print("\nSkipped lines:")
            for line in skipped_lines:
                print(line)

            # input("\nFinished Skipped Lines. Press Enter to continue...")
        else:
            print("Unknown file type:", file)


"""
-------------
MAIN SECTION
-------------
"""

if __name__ == "__main__":

    # uniform_apnea_files(
    #     path = "Data/GIF/Alice/",
    #     new_directory = "Data/GIF/SAE_Alice/"
    # )
    # uniform_apnea_files(
    #     path = "Data/GIF/Embla/",
    #     new_directory = "Data/GIF/SAE_Embla/"
    # )
    # uniform_apnea_files(
    #     path = "Data/GIF/Somnoscreen/",
    #     new_directory = "Data/GIF/SAE_Somnoscreen/"
    # )
    
    # apnea_info()

    # ADD_RRI_MAD_APNEA(
    #     new_save_file_path = "gif_sleep_apnea_events.pkl",
    #     results_path = "Data/GIF/GIF.pkl",
    #     gif_data_directory = "Data/GIF/SOMNOwatch/",
    #     lights_and_time_shift_csv_path = "Data/GIF/GIF-lights.csv",
    #     apnea_events_csv_path = "Data/GIF/sleep_apnea_events/",
    #     min_data_length_seconds = 0*3600,
    #     fill_ecg_gaps_threshold_seconds = 3*60,
    #     RRI_frequency = 4,
    #     MAD_frequency = 1,
    #     SAE_frequency = 1,
    # )

    # check_apnea_dataset()

    # ADD_RRI_MAD_SLP(
    #     new_save_file_path = "gif_test.pkl",
    #     results_path = "Data/GIF/GIF.pkl",
    #     gif_data_directory = "Data/GIF/SOMNOwatch/",
    #     slp_files_directory = "Data/GIF/PSG_GIF/",
    #     lights_and_time_shift_csv_path = "Data/GIF/GIF-lights.csv",
    #     min_data_length_seconds = 0*3600,
    #     fill_ecg_gaps_threshold_seconds = 10*3600,
    #     RRI_frequency = 4,
    #     MAD_frequency = 1
    # )

    ADD_RRI_MAD_APNEA(
        new_save_file_path = "gif_sleep_apnea_events.pkl",
        results_path = "Processed_GIF/GIF_Results.pkl",
        gif_data_directory = "Data/GIF/SOMNOwatch/",
        lights_and_time_shift_csv_path = "Data/GIF/GIF-lights.csv",
        apnea_events_csv_path = "Data/GIF/sleep_apnea_events/",
        min_data_length_seconds = 0*3600,
        fill_ecg_gaps_threshold_seconds = 0,
        RRI_frequency = 4,
        MAD_frequency = 1,
        SAE_frequency = 1,
    )

    # check_apnea_dataset()

    ADD_RRI_MAD_SLP(
        new_save_file_path = "gif_sleep_stages.pkl",
        results_path = "Processed_GIF/GIF_Results.pkl",
        gif_data_directory = "Data/GIF/SOMNOwatch/",
        slp_files_directory = "Data/GIF/PSG_GIF/",
        lights_and_time_shift_csv_path = "Data/GIF/GIF-lights.csv",
        min_data_length_seconds = 0*3600,
        fill_ecg_gaps_threshold_seconds = 0,
        RRI_frequency = 4,
        MAD_frequency = 1
    )

    raise SystemExit

    path = "Data/GIF/all_apnea_events/"

    all_apnea_classes = ['Apnea', 'Obstructive Apnea', 'Central Apnea', 'Mixed Apnea', 'Hypopnea', 'Obstructive Hypopnea', 'Central Hypopnea']
    files = os.listdir(path)
    for apnea_file in files:
        if not apnea_file.endswith(".csv"):
            continue

        file_path = os.path.join(path, apnea_file)
        sae_data = pd.read_csv(file_path, sep=",")

        if len(sae_data.values) == 0:
            continue

        sae_start_time_string = sae_data.values[0][1]
        sae_end_time_string = sae_data.values[-1][2]
        sae_start_time, sae_end_time = round_start_end_apnea_time(sae_start_time_string, sae_end_time_string)

        for sae_line in sae_data.values:
            event_start_string = sae_line[1]
            event_end_string = sae_line[2]
            event_start, event_end = round_start_end_apnea_time(event_start_string, event_end_string)

            if event_start < sae_start_time:
                sae_start_time = event_start
            if event_end > sae_end_time:
                sae_end_time = event_end
        
        sae_duration = better_int((sae_end_time-sae_start_time).total_seconds())
        sae_array = [[0] for _ in range(sae_duration)]
        sae_start_time_seconds = sae_start_time.hour * 3600 + sae_start_time.minute * 60 + sae_start_time.second
        print(sae_start_time, sae_start_time_seconds)
        break

        count_conflict = 0
        count_events = 0
        for sae_line in sae_data.values:
            apnea_event = all_apnea_classes.index(str(sae_line[0])) + 1
            event_start_string = sae_line[1]
            event_end_string = sae_line[2]
            event_start, event_end = round_start_end_apnea_time(event_start_string, event_end_string)

            event_start_index = better_int((event_start-sae_start_time).total_seconds())
            event_end_index = better_int((event_end-sae_start_time).total_seconds())
            for i in range(event_start_index, event_end_index):
                if sae_array[i] != [0] and sae_array[i] != [apnea_event]:
                    count_conflict += 1
                    sae_array[i].append(apnea_event)
                else:
                    sae_array[i] = [apnea_event]
                
                count_events += 1


    
    raise SystemExit

    ADD_RRI_MAD_SLP(
        new_save_file_path = "gif_test.pkl",
        results_path = "Data/GIF/GIF.pkl",
        gif_data_directory = "Data/GIF/SOMNOwatch/",
        slp_files_directory = "Data/GIF/PSG_GIF/",
        lights_and_time_shift_csv_path = "Data/GIF/GIF-lights.csv",
        min_data_length_seconds = 0*3600,
        fill_ecg_gaps_threshold_seconds = 10*3600,
        RRI_frequency = 4,
        MAD_frequency = 1
    )

    # ADD_SLP_TO_GIF(
    #     gif_path = "Data/GIF/GIF.pkl",
    #     slp_files_directory = "Data/GIF/PSG_GIF/",
    #     lights_and_time_shift_csv_path = "Data/GIF/GIF-lights.csv",
    #     munich_rri_directory = "Data/GIF/Rpeak_GIF_Munich/",
    #     min_data_length_seconds = 0*3600,
    #     fill_ecg_gaps_threshold_seconds = 100,
    # )

    raise SystemExit

    # process GIF data
    Data_Processing_and_Comparing(
        DATA_DIRECTORY = "Data/GIF/SOMNOwatch/",
        ECG_CLASSIFICATION_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/Noise/",
        RPEAK_DIRECTORY = "Data/GIF/Analyse_Somno_TUM/RRI/",
        AVAILABLE_MAD_RRI_PATH = "Data/GIF_dataset.h5",
        RESULTS_DIRECTORY = "Processed_GIF/",
        RESULTS_FILE_NAME = "GIF_Results.pkl",
        ECG_COMPARISON_FILE_NAME = "ECG_Validation_Comparison_Report.txt",
        RPEAK_COMPARISON_FILE_NAME = "RPeak_Comparison_Report.txt",
        RRI_COMPARISON_FILE_NAME = "RRI_Comparison_Report.txt",
        MAD_COMPARISON_FILE_NAME = "MAD_Comparison_Report.txt"
    )

    # if you want to retrieve all subdirectories containing valid files, you can use the following function
    """
    DATA_DIRECTORIES = retrieve_all_subdirectories_containing_valid_files(
        directory = "Data/", 
        valid_file_types = [".edf"]
    )
    """
        
    EDF_Data_Directories = ["Data/", "Data/GIF/SOMNOwatch/"]
    # EDF_Data_Directories = ["/media/yaopeng/data1/NAKO-33a/", "/media/yaopeng/data1/NAKO-33b/", "/media/yaopeng/data1/NAKO-609/", "/media/yaopeng/data1/NAKO-419/", "/media/yaopeng/data1/NAKO-84/"]
    Processing_Result_Directory = "Processed_NAKO/"

    # process NAKO data
    Data_Processing(
        DATA_DIRECTORIES = EDF_Data_Directories,
        RESULTS_DIRECTORY = Processing_Result_Directory,
    )

    # extract RRI and MAD values
    Extract_RRI_MAD(
        DATA_DIRECTORIES = EDF_Data_Directories,
        RESULTS_DIRECTORY = Processing_Result_Directory,
        EXTRACTED_DATA_DIRECTORY = "RRI_and_MAD/"
    )

    # run following code snippet to remove some dictionary entries (in case you do not want to overwrite them manually)
    # for file in ["Processed_NAKO/SOMNOwatch_Results.pkl"]:
    #     delete_dictionary_entries_from_file(file_path = file, dictionary_keys = ["MAD", "MAD_frequency"])