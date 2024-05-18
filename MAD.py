"""
Author: Johannes Peter Knoll

Python implementation of mean amplitude deviation (MAD) calculation for movement acceleration data.
"""

# IMPORTS
import numpy as np

# LOCAL IMPORTS
from side_functions import *
import read_edf


def check_mad_conditions(
        acceleration_data_lists: list, 
        frequencies: list, 
    ):
    """
    Check if the data is valid for MAD calculation (uniform frequency and data points).

    ARGUMENTS:
    --------------------------------
    acceleration_data_lists: list
        list of acceleration data arrays (x, y, z - axis)
    frequencies: list
        list of sampling frequencies of the acceleration data arrays

    NO RETURN VALUE: raises ValueError if conditions are not met
    """

    #check if frequencies are the same
    uniform_frequency = frequencies[0]
    for freq in frequencies:
        if freq != uniform_frequency:
            raise ValueError("Frequencies are not the same. Calculation of MAD requires the same frequency for all data arrays.")

    #check if the data arrays are the same length
    uniform_length = len(acceleration_data_lists[0])
    for data_axis in acceleration_data_lists:
        if len(data_axis) != uniform_length:
            raise ValueError("Data arrays are not the same length. Calculation of MAD requires the same length for all data arrays.")


def calc_mad_in_interval(
        acceleration_data_lists: list, 
        start_position: int, 
        end_position: int, 
    ):
    """
    Calculate MAD in a given time frame.
        current_acceleration = root(x^2 + y^2 + z^2)
        average_acceleration = sum(current_acceleration) / interval_size
        MAD = sum(abs(current_acceleration - average_acceleration)) / interval_size

    ARGUMENTS:
    --------------------------------
    acceleration_data_lists: list
        list of acceleration data arrays (x, y, z - axis)
    start_position: int
        start position of the interval
    end_position: int
        end position of the interval

    RETURNS: 
    --------------------------------
    MAD: float
        mean amplitude deviation of the movement acceleration data in the given interval
    """
    #calculate average acceleration over given interval (start_position to end_position)
    interval_size = end_position - start_position
    average_acceleration = 0
    for i in np.arange(start_position, end_position):
        current_acceleration = 0
        for data_axis in acceleration_data_lists:
            current_acceleration += data_axis[i]**2
        current_acceleration = current_acceleration**0.5
        average_acceleration += current_acceleration
    average_acceleration /= interval_size

    #calculate MAD
    mad = 0
    for i in np.arange(start_position, end_position):
        current_acceleration = 0
        for data_axis in acceleration_data_lists:
            current_acceleration += data_axis[i]**2
        current_acceleration = current_acceleration**0.5
        mad += abs(current_acceleration - average_acceleration)
    mad /= interval_size

    return mad


def calc_mad(
        acceleration_data_lists: list,
        frequencies: list, 
        time_period: int,
    ):
    """
    Calculate mean amplitude deviation (MAD) of movement acceleration data.

    ARGUMENTS:
    --------------------------------
    acceleration_data_lists: list
        list of acceleration data arrays (x, y, z - axis)
    frequencies: list
        list of sampling frequencies of the acceleration data arrays
    time_period: int
        length of the time period in seconds over which the MAD will be calculated
    wrist_acceleration_keys: list
        list of keys of data dictionary that are relevant for MAD calculation

    RETURNS:
    --------------------------------
    MAD: list
        list of MAD values for each interval: MAD[i] = MAD in interval i
    """
    #check if data is valid
    check_mad_conditions(
        acceleration_data_lists = acceleration_data_lists,
        frequencies = frequencies
        )

    #transform time_period to number of samples
    number_of_samples = int(time_period * frequencies[0])

    #calculate MAD in intervals
    MAD = []
    for i in np.arange(0, len(acceleration_data_lists[0]), number_of_samples):
        MAD.append(calc_mad_in_interval(
            acceleration_data_lists = acceleration_data_lists,
            start_position = i,
            end_position = i + time_period)
            )

    return np.array(MAD)


def calculate_MAD_in_acceleration_data(
        data_directory: str,
        valid_file_types: list,
        wrist_acceleration_keys: list,
        physical_dimension_correction_dictionary: dict,
        mad_time_period_seconds: int,
        preparation_results_path: str,
        file_name_dictionary_key: str,
        MAD_dictionary_key: str,
    ):
    """
    Calculate the MAD values for the wrist acceleration data for all valid files in the
    data directory and save them to a pickle file.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    wrist_acceleration_keys: list
        keys for the wrist acceleration data in the data dictionary
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    mad_time_period_seconds: int
        time period in seconds over which the MAD will be calculated
    preparation_results_path: str
        path to the pickle file where the MAD values are saved
    file_name_dictionary_key: str
        dictionary key to access the file name
    MAD_dictionary_key: str
        dictionary key to access the MAD values

    RETURNS:
    --------------------------------
    None, but the MAD values are saved to a pickle file as a dictionary in the following
    format:
        {
            file_name_dictionary_key: name of file 1,
            MAD_dictionary_key: MAD values for file 1,
            ...
        }
        ...
    """

    # check if MAD values already exist and if yes ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = preparation_results_path,
        dictionary_key = MAD_dictionary_key
    )
    
    # cancel if user does not want to override
    if user_answer == "n":
        return
    
    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(preparation_results_path) + "computation_in_progress.pkl"
    
    # create list to store unprocessable files
    unprocessable_files = []

    if user_answer == "y":
        # load existing results
        preparation_results_generator = load_from_pickle(preparation_results_path)

        # create variables to track progress
        total_files = get_pickle_length(preparation_results_path)
        progressed_files = 0

        # calculate MAD in the wrist acceleration data
        print("\nCalculating MAD in the wrist acceleration data in %i files from \"%s\":" % (total_files, data_directory))
        for generator_entry in preparation_results_generator:
            # show progress
            progress_bar(progressed_files, total_files)
            progressed_files += 1

            # try to load the data and correct the physical dimension if needed
            try:
                # create lists to save the acceleration data and frequencies for each axis
                acceleration_data = []
                acceleration_data_frequencies = []

                # get the acceleration data and frequency for each axis
                for possible_axis_keys in wrist_acceleration_keys:
                    this_axis_signal, this_axis_frequency = read_edf.get_data_from_edf_channel(
                        file_path = data_directory + file,
                        possible_channel_labels = possible_axis_keys,
                        physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                    )

                    # append data to corresponding lists
                    acceleration_data.append(this_axis_signal)
                    acceleration_data_frequencies.append(this_axis_frequency)
                
                # calculate MAD values
                this_MAD_values = calc_mad(
                    acceleration_data_lists = acceleration_data,
                    frequencies = acceleration_data_frequencies,
                    time_period = mad_time_period_seconds, 
                    )
            except:
                unprocessable_files.append(data_directory + file)
                continue
            
            # save MAD values
            generator_entry[MAD_dictionary_key] = this_MAD_values
            append_to_pickle(generator_entry, temporary_file_path)
    
    elif user_answer == "":
        # get all valid files
        all_files = os.listdir(data_directory)
        valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

        # create variables to track progress
        total_files = len(valid_files)
        progressed_files = 0

        # calculate MAD in the wrist acceleration data
        print("\nCalculating MAD in the wrist acceleration data in %i files from \"%s\":" % (total_files, data_directory))
        for file in valid_files:
            # show progress
            progress_bar(progressed_files, total_files)
            progressed_files += 1

            # try to load the data and correct the physical dimension if needed
            try:
                # create lists to save the acceleration data and frequencies for each axis
                acceleration_data = []
                acceleration_data_frequencies = []

                # get the acceleration data and frequency for each axis
                for possible_axis_keys in wrist_acceleration_keys:
                    this_axis_signal, this_axis_frequency = read_edf.get_data_from_edf_channel(
                        file_path = data_directory + file,
                        possible_channel_labels = possible_axis_keys,
                        physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                    )

                    # append data to corresponding lists
                    acceleration_data.append(this_axis_signal)
                    acceleration_data_frequencies.append(this_axis_frequency)
                
                # calculate MAD values
                this_MAD_values = calc_mad(
                    acceleration_data_lists = acceleration_data,
                    frequencies = acceleration_data_frequencies,
                    time_period = mad_time_period_seconds, 
                    )
                
            except:
                unprocessable_files.append(data_directory + file)
                continue

            # save MAD values for this file
            this_files_dictionary_entry = {
                file_name_dictionary_key: file,
                MAD_dictionary_key: this_MAD_values
                }
            append_to_pickle(this_files_dictionary_entry, temporary_file_path)
    
    progress_bar(progressed_files, total_files)

    # print unprocessable files
    if len(unprocessable_files) > 0:
        print("\nFor the following files the MAD values could not be calculated:")
        print(unprocessable_files)
        print("Possible reasons:")
        print(" "*5 + "- ECG file contains format errors")
        print(" "*5 + "- No matching label in wrist_acceleration_keys and the files")
        print(" "*5 + "- Physical dimension of label is unknown")