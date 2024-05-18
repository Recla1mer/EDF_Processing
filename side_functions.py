"""
Author: Johannes Peter Knoll

In this file we provide functions that are not just needed in the main file, but also in
other ones. Their purpose is to keep them a little cleaner and more intuitive.
"""

# IMPORTS
import os
import pickle
import copy
import numpy as np


def validate_parameter_settings(parameters: dict):
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
    if not isinstance(parameters["valid_file_types"], list):
        raise ValueError("'valid_file_types' parameter must be a list.")
    if not isinstance(parameters["ecg_keys"], list):
        raise ValueError("'ecg_key' parameter must be a list.")
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

    # parameters for the R peak detection
    if not callable(parameters["rpeak_primary_function"]):
        raise ValueError("'rpeak_primary_function' parameter must be a function.")
    if not callable(parameters["rpeak_secondary_function"]):
        raise ValueError("'rpeak_secondary_function' parameter must be a function.")
    if not isinstance(parameters["rpeak_primary_function_name"], str):
        raise ValueError("'rpeak_primary_function_name' parameter must be a string.")
    if not isinstance(parameters["rpeak_secondary_function_name"], str):
        raise ValueError("'rpeak_secondary_function_name' parameter must be a string.")
    if not isinstance(parameters["rpeak_distance_threshold_seconds"], float):
        raise ValueError("'rpeak_distance_threshold_seconds' parameter must be a float.")

    # parameters for the MAD calculation
    if not isinstance(parameters["mad_time_period_seconds"], int):
        raise ValueError("'mad_time_period_seconds' parameter must be an integer.")
    
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
    if not isinstance(parameters["rpeak_comparison_function_names"], list):
        raise ValueError("'rpeak_comparison_function_names' parameter must be a list.")
    if not isinstance(parameters["rpeak_comparison_report_dezimal_places"], int):
        raise ValueError("'rpeak_comparison_report_dezimal_places' parameter must be an integer.")
    if not isinstance(parameters["rpeak_comparison_report_path"], str):
        raise ValueError("'rpeak_comparison_report_path' parameter must be a string.")

    # parameters for the ECG Validation comparison
    if not isinstance(parameters["ecg_classification_values_directory"], str):
        raise ValueError("'ecg_classification_values_directory' parameter must be a string.")
    if not isinstance(parameters["ecg_classification_file_types"], list):
        raise ValueError("'ecg_classification_file_types' parameter must be a list.")
    if not isinstance(parameters["ecg_validation_comparison_report_path"], str):
        raise ValueError("'ecg_validation_comparison_report_path' parameter must be a string.")
    if not isinstance(parameters["ecg_validation_comparison_report_dezimal_places"], int):
        raise ValueError("'ecg_validation_comparison_report_dezimal_places' parameter must be an integer.")


def progress_bar(index: int, total: int, bar_len=50, title="Please wait"):
    """
    Prints a progress bar in the console.

    Idea taken from:
    https://stackoverflow.com/questions/6169217/replace-console-output-in-python

    ARGUMENTS:
    --------------------------------
    index: int
        current index
    total: int
        total number
    bar_len: int
        length of the progress bar
    title: str
        title of the progress bar

    RETURNS:
    --------------------------------
    None, but prints the progress bar to the console
    """
    percent_done = index/total*100
    rounded_percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {rounded_percent_done}% done', end='\r')

    if percent_done == 100:
        print('\t✅')


def retrieve_all_subdirectories_containing_valid_files(directory: str, valid_file_types: list):
    """
    Search given directory and every subdirectory for files with the given file types. If 
    wanted files present in a directory, return the path of the directory.
    
    ARGUMENTS:
    --------------------------------
    directory: str
        path to the head directory
    valid_file_types: list
        list of valid file types
    
    RETURNS:
    --------------------------------
    all_paths: list
        list of paths to directories containing valid files
    """
    all_files = os.listdir(directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    all_paths = []
    if len(valid_files) > 0:
        all_paths.append(directory)

    for file in all_files:
        if os.path.isdir(directory + file):
            these_paths = retrieve_all_subdirectories_containing_valid_files(directory + file + "/", valid_file_types)
            for paths in these_paths:
                all_paths.append(paths)

    return all_paths


def create_save_path_from_directory_name(directory: str):
    """
    We will save the results calculated in main.py in equally named directories the data 
    is stored in. This function creates a directory name that can be associated with the 
    path of the data directory.

    Example:
        data_directory = "data/NAKO/ECG_Data/"
        save_directory = "<direcotry_that_stores_results>/data_NAKO_ECG_Data/"

    ARGUMENTS:
    --------------------------------
    directory: str
        path to the directory
    
    RETURNS:
    --------------------------------
    save_path: str
        path to the save directory
    """
    
    new_directory_name = ""
    for i in range(len(directory)):
        if directory[i] == "/" and i != len(directory)-1:
            new_directory_name += "_"
        else:
            new_directory_name += directory[i]
        
    return new_directory_name


def clear_directory(directory: str):
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


def get_file_type(file_name: str):
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


def get_file_name_from_path(file_path: str):
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


def get_path_without_filename(file_path: str):
    """
    Separate the path from the file name (including the type/extension).

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the file
    
    RETURNS:
    --------------------------------
    str
        path without the file name
    """
    for i in range(len(file_path)-1, -1, -1):
        if file_path[i] == "/":
            return file_path[:i+1]
    return ""


def save_to_pickle(data, file_name):
    """
    Save data to a pickle file, overwriting the file if it already exists.

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


def append_to_pickle(data, file_name):
    """
    Append data to a pickle file, without deleting previous data.

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
    with open(file_name, "ab") as f:
        pickle.dump(data, f)


def load_from_pickle(file_name: str):
    """
    Load data from a pickle file as a generator.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the pickle file
    key: str
        key of the data to be loaded
    
    RETURNS:
    --------------------------------
    any
        data from the pickle file
    """
    # with open(file_name, "rb") as f:
    #     data = pickle.load(f)
    # return data
    with open(file_name, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def get_pickle_length(file_name: str):
    """
    Get the number of items in a pickle file.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the pickle file
    
    RETURNS:
    --------------------------------
    int
        number of items in the pickle file
    """
    with open(file_name, "rb") as f:
        counter = 0
        while True:
            try:
                pickle.load(f)
                counter += 1
            except EOFError:
                break
    return counter


def append_entry_to_dictionary_in_pickle_file(
        file_path: str, 
        append_to_file: str,
        file_name_dictionary_key: str,
        new_dictionary_entry: dict
    ):
    """
    Append a dictionary entry to a certain dictionary in a pickle file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the pickle file
    append_to_file: str
        file name present in dictionary to which the new data should be appended
    file_name_dictionary_key: str
        key of the dictionary storing the file names
    new_dictionary_entry: dict
        new dictionary entry to be appended
    
    RETURNS:
    --------------------------------
    None, but the new dictionary entry is appended to the corresponding dictionary in the pickle file
    """
    temporary_file_path = get_path_without_filename(file_path) + "work_in_progress.pkl"

    try:
        results_directory_generator = load_from_pickle(file_path)
    except:
        results_directory_generator = []
    dictionary_found = False

    for results_directory in results_directory_generator:
        try:
            if append_to_file == results_directory[file_name_dictionary_key]:
                results_directory.update(new_dictionary_entry)
                dictionary_found = True
        except:
            pass

        append_to_pickle(results_directory, temporary_file_path)
    
    if not dictionary_found:
        new_dictionary = {file_name_dictionary_key: append_to_file}
        new_dictionary.update(new_dictionary_entry)
        append_to_pickle(new_dictionary, temporary_file_path)
        
    os.remove(file_path)
    os.rename(temporary_file_path, file_path)


def ask_for_permission_to_override_file(file_path: str, message: str):
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


def ask_for_permission_to_override_dictionary_entry(
        file_path: str, 
        dictionary_entry: str, 
        additionally_remove_entries = []
    ):
    """
    Check if the directory that saves the results already contains dictionary entries with the
    same name. If yes, ask the user if they want to override it. If the user wants to override
    the dictionary entry, delete it.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the pickle file
    dictionary_entry: str
        name of the dictionary entry
    additionally_remove_entries: list
        list of entries that should be removed additionally if user wants to overwrite

    """
    if not os.path.isfile(file_path):
        return "no_file_found"

    ask_to_override = False
    results_directory_generator = load_from_pickle(file_path)
    for results_directory in results_directory_generator:
        if dictionary_entry in results_directory:
            ask_to_override = True
            break

    if ask_to_override:
        first_try = True
        while True:
            if first_try:
                user_answer = input("At least one dictionary in " + file_path + " contains the key: \"" + dictionary_entry + "\". Are you sure you want to overwrite them? (y/n)")
            else:
                user_answer = input("Please answer with 'y' or 'n'.")
            if user_answer == "y":
                temporary_file_path = get_path_without_filename(file_path) + "work_in_progress.pkl"
                results_directory_generator = load_from_pickle(file_path)
                for results_directory in results_directory_generator:
                    if dictionary_entry in results_directory:
                        del results_directory[dictionary_entry]
                    for add_entry in additionally_remove_entries:
                        if add_entry in results_directory:
                            del results_directory[add_entry]
                    append_to_pickle(results_directory, temporary_file_path)
                os.remove(file_path)
                os.rename(temporary_file_path, file_path)
            elif user_answer == "n":
                print("Existing Data was not overwritten. Continuing...")
                break
            else:
                first_try = False
                print("Answer not recognized.")
    else:
        user_answer = "y"
    
    return user_answer


def create_sub_dict(dictionary: dict, keys: list):
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


def create_rpeaks_pickle_path(directory: str, rpeak_function_name: str):
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
    return directory + "RPeaks_" + rpeak_function_name + ".pkl"


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