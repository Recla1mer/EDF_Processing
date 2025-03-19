"""
Author: Johannes Peter Knoll

Provides function to retrieve RRI and MAD data in the same time periods from the results file.
"""

# IMPORTS
import numpy as np

# LOCAL IMPORTS
import read_edf
from side_functions import *


def remove_redundant_file_name_part(file_name: str):
    """
    For some reason most file names look like this: 'SL256_SL256_(1).edf'. I do not know why, only the first part
    is necessary. The following function removes the redundant part.

    ARGUMENTS:
    --------------------------------
    file_name: str
        file name that should be processed
    
    RETURNS:
    --------------------------------
    str
        processed file name
    """
    for i in range(len(file_name)):
        skip_first = i
        if file_name[i] not in ["_", "(", ")", " ", "-", ".", ":", ";", ",", "/"]:
            break

    # find patterns that repeat in the file name
    usefull_pattern = file_name
    for i in range(2, len(file_name)):
        if file_name[skip_first:i] in file_name[i:]:
            usefull_pattern = file_name[skip_first:i]
    
    # remove redundant parts
    while True:
        if usefull_pattern[-1] in ["_", "(", ")", " ", "-", ".", ":", ";", ",", "/"]:
            usefull_pattern = usefull_pattern[:-1]
        else:
            break
    
    return usefull_pattern


def retrieve_rri_mad_data_in_same_time_period(
        data_directory: str,
        results_path: str,
        rri_mad_data_path: str,
    ):
    """
    During Data Processing, a lot of data is calculated. For the main project: 'Sleep Stage Classification' we 
    only need the RRI and MAD values within the same time interval. After Processing, this is not guaranteed, because
    the RRI values are only calculated for the valid ECG regions. The following function will extract the 
    corresponding MAD values to every time interval. If multiple time intervals (valid ecg regions) are present in 
    one file, the values will be saved to different dictionaries.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    results_path: str
        path to the pickle file where the valid regions are saved
    rri_mad_data_path: str
        path to the pickle file where the RRI and MAD values are saved
    
    RETURNS:
    --------------------------------
    None, but saves results to a pickle file as explained in 'main.py' file - 'Extract_RRI_MAD' function.
    """

    # check if the file already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_file(file_path = rri_mad_data_path,
            message = "\nFile containing extracted data already exists in: " + rri_mad_data_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # create list to store files that could not be processed
    unprocessable_files = []

    # create variables to track progress
    total_files = get_pickle_length(results_path, " ")
    progressed_files = 0
    start_time = time.time()

    # load the results file
    results_generator = load_from_pickle(results_path)

    if total_files > 0:
        print("\nExtracting RRI and MAD values in same time period from %i files from \"%s\":" % (total_files, data_directory))

    # iterate over all results and create new files
    for generator_entry in results_generator:
        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            file_name = generator_entry["file_name"]

            # get the ecg sampling frequency
            ecg_sampling_frequency = generator_entry["ECG_frequency"]

            # get the RRI sampling frequency
            RRI_sampling_frequency = generator_entry["RRI_frequency"]

            # get the MAD sampling frequency
            MAD_sampling_frequency = generator_entry["MAD_frequency"]

            # get the valid ECG regions
            valid_ecg_regions = generator_entry["valid_ecg_regions"]

            # create new dictionary for every valid ECG region
            for i in range(len(valid_ecg_regions)):
                valid_interval = valid_ecg_regions[i]

                # find first signal position within valid ecg region that is shared by all signals (time point where each signal writes the next value)
                first_shared_signal_position = find_time_point_shared_by_signals(
                    signal_position = valid_interval[0],
                    signal_sampling_frequency = ecg_sampling_frequency,
                    other_sampling_frequencies = [RRI_sampling_frequency, MAD_sampling_frequency],
                    update_position_by = int(1)
                )

                # find last signal position within valid ecg region that is shared by all signals (time point where each signal writes the next value)
                last_shared_signal_position = find_time_point_shared_by_signals(
                    signal_position = valid_interval[1],
                    signal_sampling_frequency = ecg_sampling_frequency,
                    other_sampling_frequencies = [RRI_sampling_frequency, MAD_sampling_frequency],
                    update_position_by = int(-1)
                )

                # create new datapoint identifier
                if len(valid_ecg_regions) == 1:
                    new_file_name_identifier = remove_redundant_file_name_part(file_name)
                else:
                    new_file_name_identifier = remove_redundant_file_name_part(file_name) + "_" + str(i)

                # create new dictionary for the important data
                important_data = {
                    "ID": new_file_name_identifier,
                    "start_date": generator_entry["start_date"],
                    "start_time": generator_entry["start_time"],
                    "time_interval": [int(first_shared_signal_position/ecg_sampling_frequency), int(last_shared_signal_position/ecg_sampling_frequency)],
                    "RRI": generator_entry["RRI"][i][0 : int((last_shared_signal_position-first_shared_signal_position)*RRI_sampling_frequency/ecg_sampling_frequency)], # look in rri calculation: rri start is equal to first_shared_signal_position, not valid_interval[0]
                    "MAD": generator_entry["MAD"][int(first_shared_signal_position*MAD_sampling_frequency/ecg_sampling_frequency) : int(last_shared_signal_position*MAD_sampling_frequency/ecg_sampling_frequency)],
                    "RRI_frequency": RRI_sampling_frequency,
                    "MAD_frequency": MAD_sampling_frequency,
                }

                # save the important data
                append_to_pickle(important_data, rri_mad_data_path)
        except:
            unprocessable_files.append(file_name)
    
    progress_bar(progressed_files, total_files, start_time)

    # print unprocessable files 
    if len(unprocessable_files) > 0:
        print("\nFor the following " + str(len(unprocessable_files)) + " files the RRI and MAD values could not be extracted:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- RRI or MAD values are missing in the results file.")