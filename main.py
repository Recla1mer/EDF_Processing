"""
Author: Johannes Peter Knoll

Main python file for Processing EDF Data.
"""

# IMPORTS
import numpy as np
import os
import pandas as pd

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

    for data_dict in results_generator:
        # add Rpeak from Munich for sections where original were missing
        patient_id = data_dict["ID"]
        
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

            append_to = 0
            for i in range(1, len(synchronized_RRI)):
                # fill gaps in RRI, MAD and SLP data
                if ecg_gaps[i-1] > fill_ecg_gaps_threshold_seconds:
                    append_to += 1
                    # fill gaps with artifact values
                else:
                    filled_gaps_RRI[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * RRI_frequency))])
                    filled_gaps_MAD[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] * MAD_frequency))])
                    filled_gaps_SLP[append_to].extend([0 for _ in range(int(ecg_gaps[i-1] / 30))])

                filled_gaps_RRI[append_to].extend(synchronized_RRI[i])
                filled_gaps_MAD[append_to].extend(synchronized_MAD[i])
                filled_gaps_SLP[append_to].extend(synchronized_SLP[i])
            
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

        # create new datapoints and save them
        for i in range(len(filled_gaps_RRI)):
            new_data_dict = dict()
            if i >= 1:
                new_data_dict["ID"] = patient_id + "_" + str(i)
            else:
                new_data_dict["ID"] = patient_id
            
            new_data_dict["start_time_somno"] = rec_time
            new_data_dict["start_time_psg"] = psg_start_time_seconds
            new_data_dict["end_time_psg"] = psg_end_time_seconds
            new_data_dict["start_time_somno"] = somno_start_time_seconds
            new_data_dict["end_time_somno"] = somno_end_time_seconds

        # whatever weird shit odd ass number results from this formula is our new 0
        # slp_start_time_seconds_somno = (slp_start_time_seconds_psg + slope * 7200 - time_shift) / (1 + slope)
        # print(slp_start_time_seconds_psg, slp_start_time_seconds_somno)

        break

        for valid_interval in valid_ecg_regions:
            # retrieve the next closest time point which synchronizes start times of SOMNOwatch (RPeak) and PSG (SLP)
            # and chooses rri start time so that rri and slp values end up on same time points
            start_slp_time_somno = (psg_start_time_seconds + slope * 7200 - time_shift) / (1 + slope)
            break

            valid_ecg_time_start_somno = valid_interval[0]/256
            valid_ecg_time_end_somno = valid_interval[1]/256

            valid_ecg_time_start_psg = valid_ecg_time_start_somno + time_shift + slope * (valid_ecg_time_start_somno - 7200)

            if somno_start_time_seconds + valid_ecg_time_start_somno + time_shift < lights_on:
                # if the start time of the valid interval is before lights on, we cannot synchronize it
                continue
            this_time_point = find_time_point_shared_by_signals(
                signal_position = valid_interval[0],
                signal_sampling_frequency = 256,
                other_sampling_frequencies = [data_dict["RRI_frequency"], data_dict["MAD_frequency"], 1/30]
            )

            this_length = valid_interval[1] - valid_interval[0]

            # get the rpeaks in the valid interval and shift them to the start of the interval to ensure the rri is calculated correctly
            this_rpeaks = np.array([peak for peak in rpeaks if valid_interval[0] <= peak <= valid_interval[1]])
            this_rpeaks = this_rpeaks - this_time_point

            # calculate the rri
            this_rri = calculate_rri_from_peaks(
                rpeaks = this_rpeaks, # type: ignore
                ecg_sampling_frequency = ecg_sampling_frequency,
                target_sampling_frequency = RRI_sampling_frequency,
                signal_length = this_length,
                pad_with = pad_with
            )

            # correct rri values which are outside of the realistic range
            for i in range(len(this_rri)):
                if this_rri[i] < realistic_rri_value_range[0] or this_rri[i] > realistic_rri_value_range[1]:
                    this_rri[i] = pad_with

            rri.append(this_rri)

        # print(munich_rec_date, munich_rec_time)
        print(patient_id, int(float(psg_end_time)*3600-float(psg_start_time)*3600-len(slp)*30), int(float(lights_off)*3600-float(lights_on+24)*3600-len(slp)*30))
        break

"""
-------------
MAIN SECTION
-------------
"""

if __name__ == "__main__":

    ADD_SLP_TO_GIF(
        gif_path = "Data/GIF/GIF.pkl",
        slp_files_directory = "Data/GIF/PSG_GIF/",
        lights_and_time_shift_csv_path = "Data/GIF/GIF-lights.csv",
        munich_rri_directory = "Data/GIF/Rpeak_GIF_Munich/"
    )

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