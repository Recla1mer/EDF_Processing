"""
Author: Johannes Peter Knoll

Python File containing unimportant parameters for the main.py file. They mostly change the name of the keys 
in the dictionaries that are used to store the results.

They are stored in this file to keep the main.py file clean and readable.

NEVER CHANGE THESE PARAMETERS AFTER THE FIRST RUN OF THE PROGRAM.
"""

# LOCAL IMPORTS
import rpeak_detection

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
    "physical_dimension_correction_dictionary": physical_dimension_correction_dictionary, # dictionary to correct the physical dimensions of the data
}

# parameters for the keys in the results dictionary
results_dictionary_key_params = {
    "file_name_dictionary_key": "file_name", # key that accesses the file name in the dictionaries
    "valid_ecg_regions_dictionary_key": "valid_ecg_regions", # key that accesses the valid ecg regions in the dictionaries
    # dictionary key that accesses r-peaks of certain method: r-peak function name
    "certain_rpeaks_dictionary_key": "certain_rpeaks", # key that accesses the certain r-peaks
    "uncertain_primary_rpeaks_dictionary_key": "uncertain_primary_rpeaks", # key that accesses the uncertain primary r-peaks
    "uncertain_secondary_rpeaks_dictionary_key": "uncertain_secondary_rpeaks", # key that accesses the uncertain secondary r-peaks
    "MAD_dictionary_key": "MAD", # key that accesses the MAD values
    "RRI_dictionary_key": "RRI", # key that accesses the RR-intervals
}

# parameters for the r-peak combination, currently unused
combine_rpeaks_params = {
    "rpeak_primary_function_name": "wfdb", # name of the primary r-peak detection function
    "rpeak_secondary_function_name": "ecgdetectors", # name of the secondary r-peak detection function
}

# parameters for the r-peak correction
correct_rpeaks_params = {
    "before_correction_rpeak_function_name_addition": "_raw", # addition to the r-peak function name before the correction
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

parameters.update(file_params)
parameters.update(results_dictionary_key_params)
parameters.update(correct_rpeaks_params)

parameters.update(only_gif_results_dictionary_key_params)
parameters.update(ecg_validation_comparison_params)
parameters.update(rpeak_comparison_params)

# delete variables not needed anymore
del physical_dimension_correction_dictionary

# delete the dictionaries as they are saved in the parameters dictionary now
del file_params, results_dictionary_key_params, combine_rpeaks_params, correct_rpeaks_params, only_gif_results_dictionary_key_params, ecg_validation_comparison_params, rpeak_comparison_params

# create lists of parameters relevant for the following functions (to make the code more readable)
determine_ecg_region_variables = ["data_directory", "valid_file_types", "ecg_keys", 
    "physical_dimension_correction_dictionary",
    "results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key", 
    "straighten_ecg_signal", "check_ecg_time_interval_seconds", "check_ecg_overlapping_interval_steps",
    "check_ecg_validation_strictness", "check_ecg_removed_peak_difference_threshold",
    "check_ecg_std_min_threshold", "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold",
    "check_ecg_min_valid_length_minutes", "check_ecg_allowed_invalid_region_length_seconds"]

choose_valid_ecg_regions_for_further_computation_variables = ["data_directory", "ecg_keys", 
    "results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key", "rpeak_function_names",
    "before_correction_rpeak_function_name_addition", "use_strictness"]

detect_rpeaks_variables = ["data_directory", "ecg_keys", "physical_dimension_correction_dictionary",
    "results_path", "file_name_dictionary_key", "valid_ecg_regions_dictionary_key", "before_correction_rpeak_function_name_addition"]

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
    "rpeak_function_name", "RRI_sampling_frequency", "pad_with", "results_path", "file_name_dictionary_key", "RRI_dictionary_key"]