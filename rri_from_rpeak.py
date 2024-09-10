"""
Author: Johannes Peter Knoll

Python file to calculate the RR-intervals from the detected r-peaks.
"""

# IMPORTS
import numpy as np

# LOCAL IMPORTS
import read_edf
from side_functions import *


def calculate_average_rri_from_peaks(
        rpeaks: list,
        ecg_sampling_frequency: int,
        target_sampling_frequency: float,
        signal_length: int
    ):
    """
    Calculate the RR-intervals from the detected r-peaks. Return with the target sampling frequency.
    
    Designed to be run for low values of target_sampling_frequency. A human should have an RRI between
    1/3 (180 beats per minute -> during sport?) and 1.2 (50 bpm, during sleep?) seconds. Average RRI
    should be around 0.6 - 1 seconds.

    So if target sampling frequency is below 0.25 Hz (4 seconds), the function will calculate the average
    RRI for each datapoint, as in this case you will most likely have more than 3 R-peaks in one datapoint.

    Therefore if 

    ARGUMENTS:
    --------------------------------
    rpeaks: list
        list of detected r-peaks
    ecg_sampling_frequency: int
        sampling frequency of the ECG data
    target_sampling_frequency: int
        sampling frequency of the RR-intervals
    signal_length: int
        length of the ECG signal
    
    RETURNS:
    --------------------------------
    rri: list
        list of RR-intervals
    """
    # check parameters
    if target_sampling_frequency > 0.25:
        raise ValueError("This function is designed to be run for low values (<= 0.25 Hz) of target_sampling_frequency. Please use calculate_momentarily_rri_from_peaks for higher values.")

    # calculate the number of entries in the rri list
    number_rri_entries = int(signal_length / ecg_sampling_frequency * target_sampling_frequency)

    # rewrite rpeaks from (number of sample) to (second)
    rpeaks = np.array(rpeaks) # type: ignore
    rpeak_position_seconds = rpeaks / ecg_sampling_frequency # type: ignore

    collect_rpeaks = []
    start_looking_at = 0

    # collect all rpeaks within the same rri datapoint
    for i in range(1, number_rri_entries+1):
        lower_rri_second = (i-1) / target_sampling_frequency
        upper_rri_second = i / target_sampling_frequency
        these_rpeaks = []

        for j in range(start_looking_at, len(rpeak_position_seconds)):
            if rpeak_position_seconds[j] <= upper_rri_second and lower_rri_second <= rpeak_position_seconds[j]:
                these_rpeaks.append(rpeak_position_seconds[j])
                start_looking_at -= 1
            if rpeak_position_seconds[j] > upper_rri_second:
                start_looking_at = j
                break

        collect_rpeaks.append(these_rpeaks)
    
    # if only one rpeak is found in the rri datapoint, try to add the previous or next rpeak
    # if that is not possible, add 0 and 1000 to the list (to create really high rri, that can be filtered out later)
    for i in range(len(collect_rpeaks)):
        if len(collect_rpeaks[i]) == 1:
            try:
                collect_rpeaks[i].append(collect_rpeaks[i-1][-1])
            except:
                index = i+1
                while True:
                    try:
                        collect_rpeaks[i].append(collect_rpeaks[index][0])
                        break
                    except:
                        index += 1

        elif len(collect_rpeaks[i]) == 0:
            try:
                collect_rpeaks[i].append(collect_rpeaks[i-1][-1])

                index = i+1
                while True:
                    try:
                        collect_rpeaks[i].append(collect_rpeaks[index][0])
                        break
                    except:
                        index += 1
            except:
                collect_rpeaks[i].append(0)
                collect_rpeaks[i].append(1000)
    
    rri = []
    
    # calculate average rri for each rri datapoint
    for i in range(len(collect_rpeaks)):
        if len(collect_rpeaks[i]) >= 2:
            rri.append((collect_rpeaks[i][-1] - collect_rpeaks[i][0]) / (len(collect_rpeaks[i])-1))
        else:
            rri.append(1000)
    
    return rri


def calculate_momentarily_rri_from_peaks(
        rpeaks: list,
        ecg_sampling_frequency: int,
        target_sampling_frequency: float,
        signal_length: int
    ):
    """
    Calculate the RR-intervals from the detected r-peaks. Return with the target sampling frequency.

    ARGUMENTS:
    --------------------------------
    rpeaks: list
        list of detected r-peaks
    ecg_sampling_frequency: int
        sampling frequency of the ECG data
    target_sampling_frequency: int
        sampling frequency of the RR-intervals
    signal_length: int
        length of the ECG signal
    
    RETURNS:
    --------------------------------
    rri: list
        list of RR-intervals
    """
    # check parameters
    if target_sampling_frequency <= 0.25:
        raise ValueError("This function is designed to be run for high values (> 0.25 Hz) of target_sampling_frequency. Please use calculate_average_rri_from_peaks for lower values.")

    # calculate the number of entries in the rri list
    number_rri_entries = int(signal_length / ecg_sampling_frequency * target_sampling_frequency)

    # rewrite rpeaks from (number of sample) to (second)
    rpeaks = np.array(rpeaks) # type: ignore
    rpeak_position_seconds = rpeaks / ecg_sampling_frequency # type: ignore

    rri = []
    start_looking_at = 1

    # iterate over all rri entries, find rri datapoint between two rpeaks (in seconds) and return their difference
    for i in range(number_rri_entries):
        rri_datapoint_second = i / target_sampling_frequency

        this_rri = 0
        for j in range(start_looking_at, len(rpeak_position_seconds)):
            start_looking_at = j
            if rpeak_position_seconds[j-1] <= rri_datapoint_second and rri_datapoint_second <= rpeak_position_seconds[j]:
                this_rri = rpeak_position_seconds[j] - rpeak_position_seconds[j-1]
                break
        
        rri.append(this_rri)
    
    return rri


def calculate_rri_from_peaks(
        rpeaks: list,
        ecg_sampling_frequency: int,
        target_sampling_frequency: float,
        signal_length: int
    ):
    """
    Calculate the RR-intervals from the detected r-peaks. Return with the target sampling frequency.

    ARGUMENTS:
    --------------------------------
    rpeaks: list
        list of detected r-peaks
    ecg_sampling_frequency: int
        sampling frequency of the ECG data
    target_sampling_frequency: int
        sampling frequency of the RR-intervals
    signal_length: int
        length of the ECG signal
    
    RETURNS:
    --------------------------------
    rri: list
        list of RR-intervals
    """
    if target_sampling_frequency <= 0.25:
        return calculate_average_rri_from_peaks(
            rpeaks = rpeaks,
            ecg_sampling_frequency = ecg_sampling_frequency,
            target_sampling_frequency = target_sampling_frequency,
            signal_length = signal_length
        )
    else:
        return calculate_momentarily_rri_from_peaks(
            rpeaks = rpeaks,
            ecg_sampling_frequency = ecg_sampling_frequency,
            target_sampling_frequency = target_sampling_frequency,
            signal_length = signal_length
        )


def determine_rri_from_rpeaks(
        data_directory: str,
        ecg_keys: list,
        physical_dimension_correction_dictionary: dict,
        rpeak_function_name: str,
        RRI_sampling_frequency: int,
        results_path: str,
        file_name_dictionary_key: str,
        RRI_dictionary_key: str,
    ):
    """
    Calculate the RR-intervals from the detected r-peaks and save them to a pickle file.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    ecg_keys: list
        list of possible labels for the ECG data
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    rpeak_function_name: str
        name of the r-peak detection function
    RRI_sampling_frequency: int
        target sampling frequency of the RR-intervals
    results_path: str
        path to the pickle file where the valid regions are saved
    file_name_dictionary_key
        dictionary key to access the file name
    RRI_dictionary_key: str
        dictionary key to access the RR-intervals

    RETURNS:
    --------------------------------
    None, but the rpeaks are saved as dictionaries to a pickle file in the following format:
    {
        file_name_dictionary_key: file_name_1,
        rri_dictionary_key: rri_1,
        ...
    }
        ...
    """
    
    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(results_path) + "computation_in_progress.pkl"

    # if the temporary file already exists, it means a previous computation was interrupted
    # ask the user if the results should be overwritten or recovered
    if os.path.isfile(temporary_file_path):
        recover_results_after_error(
            all_results_path = results_path, 
            some_results_with_updated_keys_path = temporary_file_path, 
            file_name_dictionary_key = file_name_dictionary_key,
        )
    
    # check if correction of r-peaks already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = results_path,
        dictionary_entry = RRI_dictionary_key
    )

    # cancel if needed data is missing
    if user_answer == "no_file_found":
        print("\nFile containing detected r-peaks not found. As they are needed to correct them in the first place, the correction will be skipped.")
        return

    # create lists to store unprocessable files
    unprocessable_files = []

    # load results
    results_generator = load_from_pickle(results_path)
    
    # create variables to track progress
    start_time = time.time()
    total_files = get_pickle_length(results_path, RRI_dictionary_key)
    progressed_files = 0

    if total_files > 0:
        print("\nCalculating RR-Intervals from r-peaks detected by %s in %i files:" % (rpeak_function_name, total_files))
    
    # correct rpeaks
    for generator_entry in results_generator:
        # skip if corrected r-peaks already exist and the user does not want to override
        if user_answer == "n" and RRI_dictionary_key in generator_entry.keys():
            append_to_pickle(generator_entry, temporary_file_path)
            continue

        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            # get the valid regions for the ECG data and file name
            file_name = generator_entry[file_name_dictionary_key]

            # try to load the data and correct the physical dimension if needed
            ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
                file_path = data_directory + file_name,
                possible_channel_labels = ecg_keys,
                physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
            )

            # get the r-peaks
            rpeaks = generator_entry[rpeak_function_name]

            # calculate the rri
            rri = calculate_rri_from_peaks(
                rpeaks = rpeaks,
                ecg_sampling_frequency = ecg_sampling_frequency,
                target_sampling_frequency = RRI_sampling_frequency,
                signal_length = len(ecg_signal)
            )
        
            # add the rri to the dictionary
            generator_entry[RRI_dictionary_key] = rri
            generator_entry[RRI_dictionary_key + "_frequency"] = RRI_sampling_frequency

        except:
            unprocessable_files.append(file_name)
        
        append_to_pickle(generator_entry, temporary_file_path)
    
    progress_bar(progressed_files, total_files, start_time)

    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        os.remove(results_path)
        os.rename(temporary_file_path, results_path)

    # print unprocessable files
    if len(unprocessable_files) > 0:
        print("\nFor the following " + str(len(unprocessable_files)) + " files the rri could not be calculated from the r-peaks:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- Dictionary keys that access the file name and/or r-peaks do not exist in the results. Check keys in file or recalculate them.")
        print(" "*5 + "- .edf file contains format errors")