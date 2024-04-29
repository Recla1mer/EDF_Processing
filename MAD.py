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
        data: dict, 
        frequency: dict, 
        wrist_acceleration_keys: list
    ):
    """
    Check if the data is valid for MAD calculation (uniform frequency and data points).

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequencies of the data arrays
    wrist_acceleration_keys: list
        list of keys of data dictionary that are relevant for MAD calculation

    NO RETURN VALUE: raises ValueError if conditions are not met
    """

    #check if frequencies are the same
    compare_key = wrist_acceleration_keys[0]
    uniform_frequency = frequency[compare_key]
    for key in wrist_acceleration_keys:
        if frequency[key] != uniform_frequency:
            raise ValueError("Frequencies are not the same. Calculation of MAD requires the same frequency for all data arrays.")

    #check if the data arrays are the same length
    uniform_length = len(data[compare_key])
    for key in wrist_acceleration_keys:
        if len(data[key]) != uniform_length:
            raise ValueError("Data arrays are not the same length. Calculation of MAD requires the same length for all data arrays.")
    del compare_key


def calc_mad_in_interval(
        data: dict, 
        start_position: int, 
        end_position: int, 
        wrist_acceleration_keys: list
    ):
    """
    Calculate MAD in a given time frame.
        current_acceleration = root(x^2 + y^2 + z^2)
        average_acceleration = sum(current_acceleration) / interval_size
        MAD = sum(abs(current_acceleration - average_acceleration)) / interval_size

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    start_position: int
        start position of the interval
    end_position: int
        end position of the interval
    wrist_acceleration_keys: list
        list of keys of data dictionary that are relevant for MAD calculation

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
        for key in wrist_acceleration_keys:
            current_acceleration += data[key][i]*2
        current_acceleration *= 0.5
        average_acceleration += current_acceleration
    average_acceleration /= interval_size

    #calculate MAD
    mad = 0
    for i in np.arange(start_position, end_position):
        current_acceleration = 0
        for key in wrist_acceleration_keys:
            current_acceleration += data[key][i]*2
        current_acceleration *= 0.5
        mad += abs(current_acceleration - average_acceleration)
    mad /= interval_size

    return mad


def calc_mad(
        data: dict, 
        frequency: dict, 
        time_period: int, 
        wrist_acceleration_keys: list
    ):
    """
    Calculate mean amplitude deviation (MAD) of movement acceleration data.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequencies of the data arrays
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
    check_mad_conditions(data, frequency, wrist_acceleration_keys)

    #transform time_period to number of samples
    number_of_samples = int(time_period * frequency[wrist_acceleration_keys[0]])

    #calculate MAD in intervals
    MAD = []
    for i in np.arange(0, len(data[wrist_acceleration_keys[0]]), number_of_samples):
        MAD.append(calc_mad_in_interval(data, i, i + time_period, wrist_acceleration_keys))

    return np.array(MAD)


def calculate_MAD_in_acceleration_data(
        data_directory: str,
        valid_file_types: list,
        wrist_acceleration_keys: list, 
        mad_time_period_seconds: int,
        mad_values_path: str
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
    mad_time_period_seconds: int
        time period in seconds over which the MAD will be calculated
    mad_values_path: str
        path where the MAD values will be saved

    RETURNS:
    --------------------------------
    None, but the MAD values are saved to a pickle file as a dictionary in the following
    format:
        {
            "file_name_1": MAD_values_1,
            "file_name_2": MAD_values_2,
            ...
        }
    """

    # check if MAD values already exist and if yes ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = mad_values_path,
                    message = "\nMAD Values for the wrist acceleration data already exist in " + mad_values_path + ".")
    
    # cancel if user does not want to override
    if user_answer == "n":
        return

    # get all valid files
    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    # create variables to track progress
    total_files = len(valid_files)
    progressed_files = 0

    # create dictionary to save the MAD values
    MAD_values = dict()

    # calculate MAD in the wrist acceleration data
    print("Calculating MAD in the wrist acceleration data in %i files:" % total_files)
    for file in valid_files:
        # show progress
        progress_bar(progressed_files, total_files)

        # read the data
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)

        # caculate MAD values
        MAD_values[file] = calc_mad(
            data = sigbufs, 
            frequency = sigfreqs, 
            time_period = mad_time_period_seconds, 
            wrist_acceleration_keys = wrist_acceleration_keys
            )
        progressed_files += 1
    
    progress_bar(progressed_files, total_files)

    # save MAD values
    save_to_pickle(MAD_values, mad_values_path)