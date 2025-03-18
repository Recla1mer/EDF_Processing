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
            rpeak_detection.determine_rpeak_heights(**retrieve_rpeak_heights_args)

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


"""
-------------
MAIN SECTION
-------------
"""

if __name__ == "__main__":

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