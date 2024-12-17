"""
Author: Johannes Peter Knoll

In this file we provide functions that are not just needed in the main file, but also in
other ones. Their purpose is to keep them a little cleaner and more intuitive.
"""

# IMPORTS
import os
import pickle
import time


def print_smart_time(time_seconds: float):
    """
    Convert seconds to a time format that is easier to read.

    ARGUMENTS:
    --------------------------------
    time_seconds: int
        time in seconds
    
    RETURNS:
    --------------------------------
    str
        time in a more readable format
    """
    if time_seconds <= 1:
        return str(round(time_seconds, 1)) + "s"
    else:
        time_seconds = round(time_seconds)
        days = time_seconds // 86400
        if days > 0:
            time_seconds = time_seconds % 86400
        hours = time_seconds // 3600
        if hours > 0:
            time_seconds = time_seconds % 3600
        minutes = time_seconds // 60
        seconds = time_seconds % 60

        if days > 0:
            return str(days) + "d " + str(hours) + "h"
        if hours > 0:
            return str(hours) + "h " + str(minutes) + "m"
        elif minutes > 0:
            return str(minutes) + "m " + str(seconds) + "s"
        else:
            return str(seconds) + "s"


def progress_bar(index: int, total: int, start_time: float, bar_len=50, title="Please wait"):
    """
    Prints a progress bar to the console.

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
    if total == 0:
        return

    # estimate time remaining
    if index == 0:
        time_remaining_str = "Calculating..."
    else:
        time_passed = time.time() - start_time
        time_remaining = time_passed/index*(total-index)
        time_remaining_str = print_smart_time(time_remaining)
        time_remaining_str += " "*((len("Calculating...")-len(time_remaining_str)))

    # code from source
    percent_done = index/total*100
    rounded_percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {rounded_percent_done}% done. Time remaining: {time_remaining_str}', end='\r')

    if percent_done == 100:
        print('\t✅')


def retrieve_all_subdirectories_containing_valid_files(directory: str, valid_file_types: list):
    """
    Search given directory and every subdirectory for files with the given file types. If 
    wanted files present in a directory, return the path of the directory.

    Used in main.py: User has the possibility to provide a head directory, which will be
    searched from this function to return all subdirectories that contain relevant data.
    Otherwise the user needs to provide them manually.
    
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
    # list all files in the directory
    all_files = os.listdir(directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    # Check if the directory contains valid files. If yes: append directory to list of all directories
    all_paths = []
    if len(valid_files) > 0:
        all_paths.append(directory)

    # repeat process for every subdirectory
    for file in all_files:
        if os.path.isdir(directory + file):
            these_paths = retrieve_all_subdirectories_containing_valid_files(directory + file + "/", valid_file_types)
            for paths in these_paths:
                all_paths.append(paths)

    return all_paths


def create_directories_along_path(file_path: str):
    """
    Create all directories along a given path that do not exist yet.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    file_path: str
        path to a file
    """

    if "/" in file_path:
        path_parts = file_path.split("/")
        for i in range(1, len(path_parts)):
            path = "/".join(path_parts[:i])
            if not os.path.exists(path):
                os.mkdir(path)


def clear_directory(directory: str):
    """
    Clear the given directory of all files and subdirectories.

    ARGUMENTS:
    --------------------------------
    directory: str
        path to the directory to be cleared
    
    RETURNS:
    --------------------------------
    None, but the directory is cleared from all files and subdirectories
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
            except:
                break
            # except EOFError:
            #     break


def get_pickle_length(file_name: str, dictionary_key: str):
    """
    Get the number of items in a pickle file that do not contain the given dictionary key.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the pickle file
    dictionary_key: str
        key of the dictionary
    
    RETURNS:
    --------------------------------
    int
        number of items in the pickle file
    """
    with open(file_name, "rb") as f:
        counter = 0
        while True:
            try:
                this_dictionary = pickle.load(f)
                if dictionary_key not in this_dictionary:
                    counter += 1
            except:
                break
    return counter


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
                user_answer = input("\nPlease answer with 'y' or 'n'.")
            if user_answer == "y":
                os.remove(file_path)
                break
            elif user_answer == "n":
                print("\nExisting Data was not overwritten. Continuing...")
                break
            else:
                first_try = False
                print("\nAnswer not recognized.")
    else:
        user_answer = "y"
    
    return user_answer


def ask_for_permission_to_override_dictionary_entry(
        file_path: str, 
        dictionary_entry: str, 
        additionally_remove_entries: list = [],
        remove_similar_keys: bool = False
    ):
    """
    Check if the file that saves the results already contains dictionary entries with the
    same name. If yes, ask the user if they want to override them. If the user wants to override
    the dictionary entry, delete it from all dictionaries in the file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the pickle file
    dictionary_entry: str
        name of the dictionary entry
    additionally_remove_entries: list
        list of entries that should be removed additionally if user wants to overwrite
    remove_similar_keys: bool
        if True, remove all keys that contain the dictionary_entry as a substring
    
    RETURNS:
    --------------------------------
    user_answer: str
        "no_file_found" if file does not exist
        "y" if user wants to overwrite the dictionary key or if they are not present
        "n" if dictionary keys exist but user does not want to overwrite
    """

    if not os.path.isfile(file_path):
        return "no_file_found"

    ask_to_override = False
    count_all_entries = 0
    count_entries_with_dictionary_entry = 0
    results_directory_generator = load_from_pickle(file_path)
    for results_directory in results_directory_generator:
        if dictionary_entry in results_directory:
            ask_to_override = True
            count_entries_with_dictionary_entry += 1
        count_all_entries += 1

    if ask_to_override:
        first_try = True
        while True:
            if first_try:
                if len(additionally_remove_entries) == 0:
                    user_answer = input("\n" + str(count_entries_with_dictionary_entry) + " of " + str(count_all_entries) + " dictionaries in " + file_path + " contain the key: \"" + dictionary_entry + "\". Are you sure you want to overwrite all of them? (y/n)")
                else:
                    user_input_message = "\n" + str(count_entries_with_dictionary_entry) + " of " + str(count_all_entries) + " dictionaries in " + file_path + " contain the keys: (\"" + dictionary_entry + "\""
                    for add_rem_entry in additionally_remove_entries:
                        user_input_message += ", \"" + add_rem_entry + "\""
                    user_input_message += "). Are you sure you want to overwrite all of them? (y/n)"
                    user_answer = input(user_input_message)

            else:
                user_answer = input("\nPlease answer with 'y' or 'n'.")

            if user_answer == "y":
                temporary_file_path = get_path_without_filename(file_path) + "computation_in_progress.pkl"
                results_directory_generator = load_from_pickle(file_path)
                for results_directory in results_directory_generator:
                    if dictionary_entry in results_directory:
                        del results_directory[dictionary_entry]    
                    if remove_similar_keys:
                        additionally_remove_entries = [key for key in results_directory if dictionary_entry in key]
                    for add_entry in additionally_remove_entries:
                        if add_entry in results_directory:
                            del results_directory[add_entry]
                    append_to_pickle(results_directory, temporary_file_path)
                os.remove(file_path)
                os.rename(temporary_file_path, file_path)
                break
            elif user_answer == "n":
                print("\nExisting Data was not overwritten. Continuing...")
                break
            else:
                first_try = False
                print("\nAnswer not recognized.")
    else:
        user_answer = "y"
    
    return user_answer


def delete_dictionary_entries_from_file(file_path: str, dictionary_keys: list):
    """
    Delete dictionary keys from all dictionaries in the file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the pickle file
    dictionary_keys: list
        list of dictionary keys to be deleted
    """

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(file_path) + "computation_in_progress.pkl"

    # if the temporary file already exists, something went wrong
    if os.path.isfile(temporary_file_path):
        raise Exception("The file: " + temporary_file_path + " should not exist. Either a previous computation was interrupted or another computation is ongoing.")

    # load existing results
    results_directory_generator = load_from_pickle(file_path)

    for results_directory in results_directory_generator:
        # delete dictionary entries if they exist
        for dict_key in dictionary_keys:
            if dict_key in results_directory:
                del results_directory[dict_key]
        append_to_pickle(results_directory, temporary_file_path)
    
    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        os.remove(file_path)
        os.rename(temporary_file_path, file_path)


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


def print_in_middle(string: str, length: int):
    """
    Function to center a string in a given length. Needed for printing tables.

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
    Function to left align a string in a given length. Needed for printing tables.

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


def print_smart_rounding(number: float, number_decimals: int):
    """
    Rounds number to a given number of decimals while maintaining most digits as possible by transforming it 
    into a number with a power of ten.

    ARGUMENTS:
    --------------------------------
    number: float
        number to be rounded
    number_decimals: int
        number of decimals to round to
    
    RETURNS:
    --------------------------------
    str
        rounded number
    """

    if number < 10:
        power_of_ten = 0
        while True:
            if number >= 1:
                break
            number *= 10
            power_of_ten += 1
        
        if power_of_ten == 0:
            return str(round(number, number_decimals))
        return str(round(number, number_decimals)) + "e-" + str(power_of_ten)
    else:
        power_of_ten = 0
        while True:
            if number < 10:
                break
            number /= 10
            power_of_ten += 1

        if power_of_ten == 0:
            return str(round(number, number_decimals))
        return str(round(number, number_decimals)) + "e" + str(power_of_ten)


def manually_remove_file_from_results(file_name: str, results_path: str, file_name_dictionary_key: str):
    """
    Remove dictionary containing values for file_name from the results file.

    ARGUMENTS:
    --------------------------------
    file_name: str
        name of the file
    results_path: str
        path to the results file
    file_name_dictionary_key: str
        key of the dictionary containing the file name
    
    RETURNS:
    --------------------------------
    None, but the dictionary is removed from the results file
    """

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(results_path) + "computation_in_progress.pkl"
    if os.path.isfile(temporary_file_path):
        os.remove(temporary_file_path)
    
    # load existing results
    results_generator = load_from_pickle(results_path)

    for generator_entry in results_generator:
        try:
            if generator_entry[file_name_dictionary_key] == file_name:
                continue
        except:
            pass
        
        append_to_pickle(generator_entry, temporary_file_path)
    
    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        os.remove(results_path)
        os.rename(temporary_file_path, results_path)


def recover_results_after_error(
        all_results_path: str, 
        some_results_with_updated_keys_path: str, 
        file_name_dictionary_key: str
    ):
    """
    If the program crashes during the calculation (or which is more likely: the computer gets
    disconnected from power), the results are stored in a temporary file, but will be lost if
    the program is restarted. This function recovers the results from the temporary file and 
    stores them in the results file, if the user wants to do so.

    ARGUMENTS:
    --------------------------------
    all_results_path: str
        path to the results file that stores all results
    some_results_with_updated_keys_path: str
        path to the temporary file that stores some of the results with additional keys
    file_name_dictionary_key: str
        key of the dictionary containing the file name
    
    RETURNS:
    --------------------------------
    None, but the results file is recovered
    """
    while True:
        user_answer = input("\nIt seems like there are results left from a previous computation which was interrupted. Do you want to recover the results? Otherwise they will be discarded. (y/n)")
        if user_answer == "y":
            break
        elif user_answer == "n":
            break
        else:
            print("\nAnswer not recognized. Please answer with 'y' or 'n'.")
    
    # path to temporary pickle file which will store results
    temporary_file_path = get_path_without_filename(all_results_path) + "recover_in_progress.pkl"
    if os.path.isfile(temporary_file_path):
        os.remove(temporary_file_path)
    
    if user_answer == "n":
        os.remove(some_results_with_updated_keys_path)
        return

    if user_answer == "y":
        # list to store file names that are included in the file that stores some of the results
        file_names_in_some_results = []
        
        # load pickle file which stores some of the results with additional keys
        some_results_generator = load_from_pickle(some_results_with_updated_keys_path)
        for generator_entry in some_results_generator:
            try:
                file_names_in_some_results.append(generator_entry[file_name_dictionary_key])
            except:
                continue
        
        os.rename(some_results_with_updated_keys_path, temporary_file_path)
        
        # load all existing results
        if os.path.isfile(all_results_path):
            all_results_generator = load_from_pickle(all_results_path)
            for generator_entry in all_results_generator:
                try:
                    if generator_entry[file_name_dictionary_key] in file_names_in_some_results:
                        continue
                except:
                    continue
                append_to_pickle(generator_entry, temporary_file_path)
        
        # rename the file that stores the calculated data
        if os.path.isfile(temporary_file_path):
            if os.path.isfile(all_results_path):
                os.remove(all_results_path)
            os.rename(temporary_file_path, all_results_path)


def find_time_point_shared_by_signals(
        signal_position: int,
        signal_sampling_frequency: int,
        other_sampling_frequencies: list
    ):
    """
    Only in valid regions of the ECG signal the R-Peaks were detected. We will therefore also only calculate
    the RR-intervals in these regions. We want to match the start of the RR-intervals with the start of the
    ECG signal. Therefore we need to find the next closest point to the given ecg_position which is a multiple
    of the rri_sampling_frequency / ecg_sampling_frequency.

    ARGUMENTS:
    --------------------------------
    signal_position: int
        position in the signal
    signal_sampling_frequency: int
        sampling frequency of the signal
    other_sampling_frequencies: list
        list of sampling frequencies of the other signal
    
    RETURNS:
    --------------------------------
    time_point: int
        time position at which you have points in the ECG signal and RR-intervals * ecg_sampling_frequency
    """

    while True:
        success = True
        for freq in other_sampling_frequencies:
            if signal_position % (signal_sampling_frequency / freq) != 0:
                success = False
                break

        if success:
            return int(signal_position)
        
        signal_position += 1