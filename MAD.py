"""
Python implementation of mean amplitude deviation (MAD) calculation for movement acceleration data.

Main function: calc_mad
"""

import numpy as np

def check_mad_conditions(data: dict, frequency: dict, relevant_keys = ["X", "Y", "Z"]):
    """
    Check if the data is valid for MAD calculation (uniform frequency and data points).

    ARGUMENTS: explained in calc_mad

    NO RETURN VALUE: raises ValueError if conditions are not met
    """

    #check if frequencies are the same
    compare_key = relevant_keys[0]
    uniform_frequency = frequency[compare_key]
    for key in relevant_keys:
        if frequency[key] != uniform_frequency:
            raise ValueError("Frequencies are not the same. Calculation of MAD requires the same frequency for all data arrays.")

    #check if the data arrays are the same length
    uniform_length = len(data[compare_key])
    for key in relevant_keys:
        if len(data[key]) != uniform_length:
            raise ValueError("Data arrays are not the same length. Calculation of MAD requires the same length for all data arrays.")
    del compare_key


def calc_mad_in_interval(data: dict, start_position: int, end_position: int, relevant_keys = ["X", "Y", "Z"]):
    """
    Calculate MAD in a given time frame.
        current_acceleration = root(x^2 + y^2 + z^2)
        average_acceleration = sum(current_acceleration) / interval_size
        MAD = sum(abs(current_acceleration - average_acceleration)) / interval_size

    ARGUMENTS: explained in calc_mad

    RETURNS: single MAD value
    """
    #calculate average acceleration over given interval (start_position to end_position)
    interval_size = end_position - start_position
    average_acceleration = 0
    for i in np.arange(start_position, end_position):
        current_acceleration = 0
        for key in relevant_keys:
            current_acceleration += data[key][i]*2
        current_acceleration *= 0.5
        average_acceleration += current_acceleration
    average_acceleration /= interval_size

    #calculate MAD
    mad = 0
    for i in np.arange(start_position, end_position):
        current_acceleration = 0
        for key in relevant_keys:
            current_acceleration += data[key][i]*2
        current_acceleration *= 0.5
        mad += abs(current_acceleration - average_acceleration)
    mad /= interval_size

    return mad


def calc_mad(data: dict, frequency: dict, time_period: int, relevant_keys = ["X", "Y", "Z"]):
    """
    Calculate mean amplitude deviation (MAD) of movement acceleration data.

    ARGUMENTS:
    data: dictionary of data (sigbufs in read_edf.py)
    frequency: dictionary of frequencies (sigfreqs in read_edf.py)
    time_period: time period in seconds over which MAD value should be calculated
    relevant_keys: keys of data dictionary that are relevant for MAD calculation

    RETURNS: list of MAD values
    """
    #check if data is valid
    check_mad_conditions(data, frequency, relevant_keys)

    #transform time_period to number of samples
    number_of_samples = int(time_period * frequency[relevant_keys[0]])

    #calculate MAD in intervals
    MAD = []
    for i in np.arange(0, len(data[relevant_keys[0]]), number_of_samples):
        MAD.append(calc_mad_in_interval(data, i, i + time_period, relevant_keys))

    return MAD

            