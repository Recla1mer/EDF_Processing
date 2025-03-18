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
    "ECG II": {"possible_dimensions": voltage_dimensions, "dimension_correction": voltage_correction},
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
parameters = dict() # type: ignore

# file parameters:
file_params = {
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_keys": ["ECG", "ECG II"], # possible labels for the ECG data in the data files
    "wrist_acceleration_keys": [["X"], ["Y"], ["Z"]], # possible labels for the wrist acceleration data in the data files
    "physical_dimension_correction_dictionary": physical_dimension_correction_dictionary, # dictionary to correct the physical dimensions of the data
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

# parameters for the rri comparison
rri_comparison_params = {
    "rri_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
}

# parameters for the MAD comparison
mad_comparison_params = {
    "mad_comparison_report_dezimal_places": 4, # number of dezimal places in the comparison report
}

parameters.update(file_params)

parameters.update(ecg_validation_comparison_params)
parameters.update(rpeak_comparison_params)
parameters.update(rri_comparison_params)
parameters.update(mad_comparison_params)

# delete variables not needed anymore
del physical_dimension_correction_dictionary

# delete the dictionaries as they are saved in the parameters dictionary now
del file_params, ecg_validation_comparison_params, rpeak_comparison_params, rri_comparison_params, mad_comparison_params