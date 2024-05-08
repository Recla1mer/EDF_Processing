"""
Author: Johannes Peter Knoll

In this file we provide functions that are not just needed in the main file, but also in
other ones. Their purpose is to keep them a little cleaner and more intuitive.
"""

# IMPORTS
import os
import pickle


def validate_parameter_settings(parameters):
    """
    In the main file, we have a dictionary with all the parameters. 
    This function checks if the parameters are valid.

    ARGUMENTS:
    --------------------------------
    parameters: dict
        dictionary containing all parameters
    
    RETURNS:
    --------------------------------
    None, but raises an error if a parameter is invalid
    """
    # file parameters:
    if not isinstance(parameters["data_directory"], str):
        raise ValueError("'data_directory' parameter must be a string.")
    if not isinstance(parameters["valid_file_types"], list):
        raise ValueError("'valid_file_types' parameter must be a list.")
    if not isinstance(parameters["ecg_key"], str):
        raise ValueError("'ecg_key' parameter must be a string.")
    if not isinstance(parameters["wrist_acceleration_keys"], list):
        raise ValueError("'wrist_acceleration_keys' parameter must be a list.")
    
    """
    --------------------------------------
    parameters for the PREPARATION SECTION
    --------------------------------------
    """

    # parameters for the ECG Validation
    if not isinstance(parameters["ecg_calibration_file_path"], str):
        raise ValueError("'ecg_calibration_file_path' parameter must be a string.")
    if not isinstance(parameters["ecg_thresholds_multiplier"], (int, float)):
        raise ValueError("'ecg_thresholds_multiplier' parameter must be an integer or a float.")
    if parameters["ecg_thresholds_multiplier"] <= 0 or parameters["ecg_thresholds_multiplier"] > 1:
        raise ValueError("'ecg_thresholds_multiplier' parameter must be within (0,1].")
    if not isinstance(parameters["ecg_thresholds_dezimal_places"], int):
        raise ValueError("'ecg_thresholds_dezimal_places' parameter must be an integer.")
    if not isinstance(parameters["ecg_thresholds_save_path"], str):
        raise ValueError("'ecg_thresholds_save_path' parameter must be a string.")
    if not isinstance(parameters["check_ecg_time_interval_seconds"], int):
        raise ValueError("'check_ecg_time_interval_seconds' parameter must be an integer.")
    if not isinstance(parameters["check_ecg_overlapping_interval_steps"], int):
        raise ValueError("'check_ecg_overlapping_interval_steps' parameter must be an integer.")
    if not isinstance(parameters["check_ecg_min_valid_length_minutes"], int):
        raise ValueError("'check_ecg_min_valid_length_minutes' parameter must be an integer.")
    if not isinstance(parameters["check_ecg_allowed_invalid_region_length_seconds"], int):
        raise ValueError("'check_ecg_allowed_invalid_region_length_seconds' parameter must be an integer.")
    if not isinstance(parameters["valid_ecg_regions_path"], str):
        raise ValueError("'valid_ecg_regions_path' parameter must be a string.")

    # parameters for the R peak detection
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
    if not isinstance(parameters["certain_rpeaks_path"], str):
        raise ValueError("'certain_rpeaks_path' parameter must be a string.")
    if not isinstance(parameters["uncertain_primary_rpeaks_path"], str):
        raise ValueError("'uncertain_primary_rpeaks_path' parameter must be a string.")
    if not isinstance(parameters["uncertain_secondary_rpeaks_path"], str):
        raise ValueError("'uncertain_secondary_rpeaks_path' parameter must be a string.")

    # parameters for the MAD calculation
    if not isinstance(parameters["mad_time_period_seconds"], int):
        raise ValueError("'mad_time_period_seconds' parameter must be an integer.")
    if not isinstance(parameters["mad_values_path"], str):
        raise ValueError("'mad_values_path' parameter must be a string.")
    
    """
    --------------------------------------
    parameters for the ADDITIONALS SECTION
    --------------------------------------
    """
    
    # parameters for the R peak accuracy evaluation
    if not isinstance(parameters["rpeaks_values_directory"], str):
        raise ValueError("'rpeaks_values_directory' parameter must be a string.")
    if not isinstance(parameters["valid_rpeak_values_file_types"], list):
        raise ValueError("'valid_rpeak_values_file_types' parameter must be a list.")
    if not isinstance(parameters["include_rpeak_value_classifications"], list):
        raise ValueError("'include_rpeak_value_classifications' parameter must be a list.")
    if not isinstance(parameters["rpeak_comparison_functions"], list):
        raise ValueError("'rpeak_comparison_functions' parameter must be a list.")
    if not isinstance(parameters["rpeak_classification_functions"], list):
        raise ValueError("'rpeak_classification_functions' parameter must be a list.")
    if not isinstance(parameters["rpeak_comparison_evaluation_path"], str):
        raise ValueError("'rpeak_comparison_evaluation_path' parameter must be a string.")
    if not isinstance(parameters["rpeak_comparison_function_names"], list):
        raise ValueError("'rpeak_comparison_function_names' parameter must be a list.")
    if not isinstance(parameters["rpeak_comparison_report_dezimal_places"], int):
        raise ValueError("'rpeak_comparison_report_dezimal_places' parameter must be an integer.")
    if not isinstance(parameters["rpeak_comparison_report_path"], str):
        raise ValueError("'rpeak_comparison_report_path' parameter must be a string.")
    if not isinstance(parameters["rpeaks_classification_raw_data_directory"], str):
        raise ValueError("'rpeaks_classification_raw_data_directory' parameter must be a string.")

    # parameters for the ECG Validation comparison
    if not isinstance(parameters["ecg_validation_comparison_raw_data_directory"], str):
        raise ValueError("'ecg_validation_comparison_raw_data_directory' parameter must be a string.")
    if not isinstance(parameters["ecg_classification_values_directory"], str):
        raise ValueError("'ecg_classification_values_directory' parameter must be a string.")
    if not isinstance(parameters["ecg_classification_file_types"], list):
        raise ValueError("'ecg_classification_file_types' parameter must be a list.")
    if not isinstance(parameters["ecg_validation_comparison_evaluation_path"], str):
        raise ValueError("'ecg_validation_comparison_evaluation_path' parameter must be a string.")
    if not isinstance(parameters["ecg_validation_comparison_report_path"], str):
        raise ValueError("'ecg_validation_comparison_report_path' parameter must be a string.")
    if not isinstance(parameters["ecg_validation_comparison_report_dezimal_places"], int):
        raise ValueError("'ecg_validation_comparison_report_dezimal_places' parameter must be an integer.")


def progress_bar(index, total, bar_len=50, title='Please wait'):
    """
    Source: https://stackoverflow.com/questions/6169217/replace-console-output-in-python
    """
    percent_done = index/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')

    if round(percent_done) == 100:
        print('\t✅')


def clear_directory(directory):
    """
    Clear the given directory of all files and subdirectories.

    ARGUMENTS:
    --------------------------------
    directory: str
        path to the directory to be cleared
    
    RETURNS:
    --------------------------------
    None
    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            if os.path.isdir(file_path):
                clear_directory(file_path)
        except Exception as e:
            print(e)


def get_file_type(file_name):
    """
    Get the file type/extension of a file.

    ARGUMENTS:
    --------------------------------
    file_name: str
        name of the file
    
    RETURNS:
    --------------------------------
    str
        file type/extension
    """
    return os.path.splitext(file_name)[1]


def get_file_name_from_path(file_path):
    """
    Separate the file name (including the type/extension) from the file path.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the file
    
    RETURNS:
    --------------------------------
    str
        file name (including the type/extension)
    """
    for i in range(len(file_path)-1, -1, -1):
        if file_path[i] == "/":
            return file_path[i+1:]


def save_to_pickle(data, file_name):
    """
    Save data to a pickle file.

    ARGUMENTS:
    --------------------------------
    data: any
        data to be saved
    file_name: str
        path to the pickle file
    
    RETURNS:
    --------------------------------
    None
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(file_name):
    """
    Load data from a pickle file.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the pickle file
    
    RETURNS:
    --------------------------------
    any
        data from the pickle file
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def ask_for_permission_to_override(file_path: str, message: str):
    """
    If a file already exists, ask the user if they want to overwrite it.
    If the file does not exist, return "y". If the user wants to overwrite the file, delete it.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the file
    message: str
        message to be shown to the user

    RETURNS:
    --------------------------------
    user_answer: str
        "y" if the user wants to overwrite the file, "n" if not
    """
    if os.path.isfile(file_path):
        first_try = True
        while True:
            if first_try:
                user_answer = input(message + " Are you sure you want to overwrite them? (y/n)")
            else:
                user_answer = input("Please answer with 'y' or 'n'.")
            if user_answer == "y":
                os.remove(file_path)
                break
            elif user_answer == "n":
                print("Existing Data was not overwritten. Continuing...")
                break
            else:
                first_try = False
                print("Answer not recognized.")
    else:
        user_answer = "y"
    
    return user_answer


def create_sub_dict(dictionary, keys):
    """
    Create a sub dictionary of the main one with the given keys.

    ARGUMENTS:
    --------------------------------
    dictionary: dict
        main dictionary
    keys: list
        keys for the sub dictionary
    
    RETURNS:
    --------------------------------
    dict
        sub dictionary containing only the given keys
    """
    return {key: dictionary[key] for key in keys}


def create_rpeaks_pickle_path(Directory, rpeak_function_name):
    """
    Create the path for the pickle file where the rpeaks are saved for each method.

    ARGUMENTS:
    --------------------------------
    Directory: str
        directory where the pickle file will be saved
    rpeak_function_name: str
        name of the rpeak detection method
    
    RETURNS:
    --------------------------------
    str
        path to the pickle file
    """
    return Directory + "RPeaks_" + rpeak_function_name + ".pkl"


def print_in_middle(string: str, length: int):
    """
    Function to center a string in a given length. Needed in printing tables.

    ARGUMENTS:
    --------------------------------
    string: str
        string that should be centered
    length: int
        length in which the string should be centered
    
    RETURNS:
    --------------------------------
    centered_string: str
        string centered in the given length
    """
    len_string = len(string)
    undersize = int((length - len_string) // 2)
    return " " * (length - len_string - undersize) + string + " " * undersize


def print_left_aligned(string: str, length: int):
    """
    Function to left align a string in a given length. Needed in printing tables.

    ARGUMENTS:
    --------------------------------
    string: str
        string that should be left aligned
    length: int
        length in which the string should be left aligned
    
    RETURNS:
    --------------------------------
    left_aligned_string: str
        string left aligned in the given length
    """
    len_string = len(string)
    return string + " " * (length - len_string)