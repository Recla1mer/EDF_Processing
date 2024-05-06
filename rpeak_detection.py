"""
Author: Johannes Peter Knoll

Python implementation to detect R-peaks in ECG data.
Useful Link: https://www.samproell.io/posts/signal/ecg-library-comparison/
"""

# IMPORTS
import numpy as np
import time

# import libraries for rpeak detection
import neurokit2
import wfdb.processing

# import old code used for rpeak detection
import old_code.rpeak_detection as old_rpeak

# LOCAL IMPORTS
import read_edf
from side_functions import *


def get_rpeaks_old(
        data: dict, 
        frequency: dict, 
        ecg_key: str, 
        detection_interval: tuple
    ):
    """
    Detect R-peaks in ECG data using the code that was previously used by my research group.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequency of the signals
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple
        interval in which the R-peaks should be detected
        if None, the whole ECG data will be used

    RETURNS:
    --------------------------------
    rpeaks_old: 1D numpy array
        list of R-peak locations
    """

    # retrieve sampling rate (or 'frequency')
    sampling_rate = frequency[ecg_key]

    # get the ECG data in the detection interval
    if detection_interval is None:
        ecg_signal = data[ecg_key]
    else:
        ecg_signal = data[ecg_key][detection_interval[0]:detection_interval[1]]

    # detect the R-peaks
    rpeaks_old = old_rpeak.get_rpeaks(ecg_signal, sampling_rate)
    
    # if not the whole ECG data is used, the R-peaks are shifted by the start of the detection interval and need to be corrected
    if detection_interval is not None:
        rpeaks_old += detection_interval[0]
    
    return rpeaks_old


def get_rpeaks_neuro(
        data: dict, 
        frequency: dict, 
        ecg_key: str, 
        detection_interval: tuple
    ):
    """
    Detect R-peaks in ECG data using the neurokit2 library.
    See link mentioned above: very fast, good performance

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequency of the signals
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple
        interval in which the R-peaks should be detected
        if None, the whole ECG data will be used

    RETURNS:
    --------------------------------
    rpeaks_old: 1D numpy array
        list of R-peak locations
    """

    # retrieve sampling rate (or 'frequency')
    sampling_rate = frequency[ecg_key]

    # get the ECG data in the detection interval
    if detection_interval is None:
        ecg_signal = data[ecg_key]
    else:
        ecg_signal = data[ecg_key][detection_interval[0]:detection_interval[1]]

    # detect the R-peaks
    _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    rpeaks = results["ECG_R_Peaks"]

    rpeaks_corrected = wfdb.processing.correct_peaks(
        ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up"
    )
    #wfdb.plot_items(ecg_signal, [rpeaks_corrected])  # styling options omitted

    # if not the whole ECG data is used, the R-peaks are shifted by the start of the detection interval and need to be corrected
    if detection_interval is not None:
        rpeaks_corrected += detection_interval[0]
    
    return rpeaks_corrected


def get_rpeaks_wfdb(
        data: dict, 
        frequency: dict, 
        ecg_key: str, 
        detection_interval: tuple
    ):
    """
    Detect R-peaks in ECG data using the wfdb library.
    See link mentioned above: excellent performance, but slower. 

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequency of the signals
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple
        interval in which the R-peaks should be detected
        if None, the whole ECG data will be used

    RETURNS:
    --------------------------------
    rpeaks_old: 1D numpy array
        list of R-peak locations
    """

    # retrieve sampling rate (or 'frequency')
    sampling_rate = frequency[ecg_key]

    # get the ECG data in the detection interval
    if detection_interval is None:
        ecg_signal = data[ecg_key]
    else:
        ecg_signal = data[ecg_key][detection_interval[0]:detection_interval[1]]

    # detect the R-peaks
    rpeaks = wfdb.processing.xqrs_detect(ecg_signal, fs=sampling_rate, verbose=False)
    rpeaks_corrected = wfdb.processing.correct_peaks(
        ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up"
    )

    # if not the whole ECG data is used, the R-peaks are shifted by the start of the detection interval and need to be corrected
    if detection_interval is not None:
        rpeaks_corrected += detection_interval[0]

    return rpeaks_corrected


def detect_rpeaks(
        data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        rpeak_function,
        rpeak_function_name: str,
        rpeak_path: str,
        valid_ecg_regions_path: str
    ):
    """
    Detect R peaks in the valid ecg regions for all valid file types in the given data
    directory.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    ecg_key: str
        key for the ECG data in the data dictionary
    rpeak_function: function
        function to detect the R peaks
    rpeak_function_name: str
        name of the R peak detection function
    rpeak_path: str
        path where the R peaks should be saved
    valid_ecg_regions_path: str
        path to the valid ECG regions

    RETURNS:
    --------------------------------
    None, but the rpeaks are saved to a pickle file in the following format:
    {
        "file_name_1": rpeaks_1,
        ...
    }
    """

    # check if R peaks already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = rpeak_path,
                            message = "\nDetected R peaks already exist in: " + rpeak_path)
    
     # cancel if user does not want to override
    if user_answer == "n":
        return

    # get all valid files
    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    # create variables to track progress
    total_files = len(valid_files)
    progressed_files = 0

    # create dictionary to save the R peaks
    all_rpeaks = dict()

    # load valid ecg regions
    valid_ecg_regions = load_from_pickle(valid_ecg_regions_path)

    # detect rpeaks in the valid regions of the ECG data
    print("\nDetecting R peaks in the ECG data in %i files using %s:" % (total_files, rpeak_function_name))
    for file in valid_files:
        # show progress
        progress_bar(progressed_files, total_files)

        # get the valid regions for the ECG data, if they do not exist: skip this file
        try:
            detection_intervals = valid_ecg_regions[file]
            progressed_files += 1
        except KeyError:
            print("Valid regions for the ECG data in " + file + " are missing. Skipping this file.")
            continue

        # get the ECG data
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)

        # detect the R peaks in the valid ecg regions
        this_rpeaks = np.array([], dtype = int)
        for interval in detection_intervals:
            this_result = rpeak_function(
                sigbufs, 
                sigfreqs, 
                ecg_key,
                interval,
                )
            this_rpeaks = np.append(this_rpeaks, this_result)
        
        all_rpeaks[file] = this_rpeaks
    
    progress_bar(progressed_files, total_files)
    
    # save the R peaks to a pickle file
    save_to_pickle(all_rpeaks, rpeak_path)


def combine_rpeaks(
        rpeaks_primary: list,
        rpeaks_secondary: list,
        frequency: int,
        rpeak_distance_threshold_seconds: float,
    ):
    """
    This function combines the R-peaks detected by two different methods. If two R-peaks
    are closer than the threshold, they are considered as the same and the R-peak detected
    by the primary method is used.

    You will see that this function collects multiple possible matches (threshold
    condition met). At the end, the closest one is chosen. Of course this might result in
    a wrong decision if there might be a better match later on. 
    However, in this case the R-peaks would be that close, that the resulting heart rate
    would be 1200 bpm or higher (for a meaningful threshold <= 0.05 s), which is not
    realistic and the detection would be wrong anyway.

    At the end, the R-peaks that were detected by both methods, the R-peaks that were only
    detected by the primary method and the R-peaks that were only detected by the secondary
    method are returned as lists.

    Suggested is the wfdb library and the detection function that was previously used by
    my research group (see old_code/rpeak_detection.py).

    ARGUMENTS:
    --------------------------------
    rpeaks_primary: list
        R-peak locations detected by the primary method
    rpeaks_secondary: list
        R-peak locations detected by the secondary method
    frequency: int
        sampling rate / frequency of the ECG data
    rpeak_distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same in seconds

    RETURNS:
    --------------------------------
    rpeaks_intersected: 1D numpy array
        R-peak locations that were detected by both methods
    rpeaks_only_primary: 1D numpy array
        R-peak locations that were only detected by the primary method
    rpeaks_only_secondary: 1D numpy array
        R-peak locations that were only detected by the secondary method
    """

    # convert the threshold from seconds to iterations
    distance_threshold_iterations = int(rpeak_distance_threshold_seconds * frequency)

    # intersects_before = len(np.intersect1d(rpeaks_primary, rpeaks_secondary))

    # if two R-peaks are closer than the threshold, they are considered as the same
    # both will be changed to the same value (primary R-peak)

    last_matching_rpeak = -1 # stores position of the last matching R-peak
    for i in range(len(rpeaks_primary)):
        # only check R-peaks that do not match already
        if rpeaks_primary[i] not in rpeaks_secondary:
            # store possible match possitions and their distance to the primary R-peak
            possible_matches = []
            possible_matches_values = []
            # iterate over the secondary R-peaks starting from the last matching R-peak
            for j in range(last_matching_rpeak + 1, len(rpeaks_secondary)):
                # calculate the distance and store it
                this_distance = rpeaks_secondary[j] - rpeaks_primary[i]
                possible_matches_values.append(abs(this_distance))
                possible_matches.append(j)
                # if the distance is larger than the threshold, stop the iteration
                if this_distance > distance_threshold_iterations:
                    break
            # if there are possible matches, take the closest one
            if len(possible_matches_values) > 0:
                if min(possible_matches_values) < distance_threshold_iterations:
                    last_matching_rpeak = possible_matches[possible_matches_values.index(min(possible_matches_values))]
                    rpeaks_secondary[last_matching_rpeak] = rpeaks_primary[i]
    
    # intersects_after = len(np.intersect1d(rpeaks_primary, rpeaks_secondary))

    # intersect the R-peaks
    rpeaks_intersected = np.intersect1d(rpeaks_primary, rpeaks_secondary)

    # get the R-peaks that are only in one of the two methods
    rpeaks_only_primary = np.setdiff1d(rpeaks_primary, rpeaks_secondary)
    rpeaks_only_secondary = np.setdiff1d(rpeaks_secondary, rpeaks_primary)

    return rpeaks_intersected, rpeaks_only_primary, rpeaks_only_secondary


def combine_detected_rpeaks(
        data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        rpeak_primary_path: str,
        rpeak_secondary_path: str,
        rpeak_distance_threshold_seconds: float,
        certain_rpeaks_path: str,
        uncertain_primary_rpeaks_path: str,
        uncertain_secondary_rpeaks_path: str,
    ):
    """
    Load detected R peaks from two different methods and combine them as described in
    the function combine_rpeaks(). The certain (detected by both methods) and uncertain
    (detected by only one method) R peaks are saved to pickle files.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    ecg_key: str
        key for the ECG data in the data dictionary
    rpeak_primary_path: str
        path to the R peaks detected by the primary method
    rpeak_secondary_path: str
        path to the R peaks detected by the secondary method
    rpeak_distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same
    certain_rpeaks_path: str
        path where the R peaks that were detected by both methods are saved
    uncertain_primary_rpeaks_path: str
        path where the R peaks that were only detected by the primary method are saved
    uncertain_secondary_rpeaks_path: str
        path where the R peaks that were only detected by the secondary method are saved

    RETURNS:
    --------------------------------
    None, but the R peaks are saved as dictionarys to pickle files in the following formats:
    certain_rpeaks: {
                    "file_name_1": certain_rpeaks_1,
                    ...
                    }
    uncertain_primary_rpeaks: {
                    "file_name_1": uncertain_primary_rpeaks_1,
                    ...
                    }
    uncertain_secondary_rpeaks: {
                    "file_name_1": uncertain_secondary_rpeaks_1,
                    ...
                    }
    """

    # check if the R peaks were already combined and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = certain_rpeaks_path,
                                    message = "\nDetected R peaks were already combined.")
    
    # cancel if user does not want to override
    if user_answer == "n":
        return
    
    # delete the old files if they exist
    try:
        os.remove(uncertain_primary_rpeaks_path)
        os.remove(uncertain_secondary_rpeaks_path)
    except FileNotFoundError:
        pass

    # get all valid files
    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    # create variables to track progress
    total_files = len(valid_files)
    progressed_files = 0

    # create dictionaries to save the R peaks
    certain_rpeaks = dict()
    uncertain_primary_rpeaks = dict()
    uncertain_secondary_rpeaks = dict()

    # load detected R peaks
    all_rpeaks_primary = load_from_pickle(rpeak_primary_path)
    all_rpeaks_secondary = load_from_pickle(rpeak_secondary_path)

    # combine detected R peaks
    print("\nCombining detected R peaks for %i files:" % total_files)
    for file in valid_files:
        # show progress
        progress_bar(progressed_files, total_files)
        progressed_files += 1
        
        # get the frequency
        sigfreqs = read_edf.get_edf_data(data_directory + file)[1]

        # combine the R peaks
        these_combined_rpeaks = combine_rpeaks(
            rpeaks_primary = all_rpeaks_primary[file],
            rpeaks_secondary = all_rpeaks_secondary[file],
            frequency = sigfreqs[ecg_key],
            rpeak_distance_threshold_seconds = rpeak_distance_threshold_seconds
            )
        
        # save the R peaks to the dictionaries
        certain_rpeaks[file] = these_combined_rpeaks[0]
        uncertain_primary_rpeaks[file] = these_combined_rpeaks[1]
        uncertain_secondary_rpeaks[file] = these_combined_rpeaks[2]
    
    progress_bar(progressed_files, total_files)
    
    # save the R peaks to pickle files
    save_to_pickle(certain_rpeaks, certain_rpeaks_path)
    save_to_pickle(uncertain_primary_rpeaks, uncertain_primary_rpeaks_path)
    save_to_pickle(uncertain_secondary_rpeaks, uncertain_secondary_rpeaks_path)


"""
Following code won't be used for the final implementation, but is useful for testing and
comparing the results of different R-peak detection methods. R peaks are also already
available for the GIF data. They might or might not have been calculated automatically and
later checked manually.

They are stored in .rri files. Therefore we need to implement functions to compare the
results of different R-peaks and to read the accurate R-peaks from the .rri files.
They are stored in the following format: "integer letter" after a file header containing
various information separated by a line of "-".
"""


def compare_rpeak_detections(
        first_rpeaks: list,
        second_rpeaks: list,
        frequency: int,
        rpeak_distance_threshold_seconds: float,
    ):
    """
    Compare the results of two different R-peak detections.

    ARGUMENTS:
    --------------------------------
    first_rpeaks: list
        R-peak locations detected by the first method
    second_rpeaks: list
        R-peak locations detected by the second method
    frequency: int
        sampling rate (frequency) of the ECG data
    rpeak_distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same

    RETURNS:
    --------------------------------
    rmse_without_same: float
        root mean squared error without the R-peaks that were detected by both methods
    rmse_with_same: float
        root mean squared error with the R-peaks that were detected by both methods
    number_of_same_values: int
        number of R-peaks that were detected by both methods
    number_of_values_considered_as_same: int
        number of R-peaks that were considered as the same
    """

    # convert the threshold from seconds to iterations
    distance_threshold_iterations = int(rpeak_distance_threshold_seconds * frequency)

    # lists to store the r peaks that are considered as the same (distance < threshold, distance != 0)
    analog_value_in_first = []
    analog_value_in_second = []

    # list to store the r peaks that are the same (distance = 0)
    same_values = []

    # if two R-peaks are closer than the threshold, they are considered as the same
    last_matching_rpeak = -1 # stores position of the last matching R-peak
    for i in range(len(first_rpeaks)):
        # check R-peaks that do not match already, otherwise append them to the same_values
        if first_rpeaks[i] not in second_rpeaks:
            # store possible match possitions and their distance to the primary R-peak
            possible_matches = []
            possible_matches_values = []
            # iterate over the second R-peaks starting from the last matching R-peak
            for j in range(last_matching_rpeak + 1, len(second_rpeaks)):
                # calculate the distance and store it
                this_distance = second_rpeaks[j] - first_rpeaks[i]
                possible_matches_values.append(abs(this_distance))
                possible_matches.append(j)
                # if the distance is larger than the threshold, stop the iteration
                if this_distance > distance_threshold_iterations:
                    break
            # if there are possible matches, take the closest one and append the R-peaks to the lists
            if len(possible_matches_values) > 0:
                if min(possible_matches_values) < distance_threshold_iterations:
                    last_matching_rpeak = possible_matches[possible_matches_values.index(min(possible_matches_values))]
                    analog_value_in_first.append(first_rpeaks[i])
                    analog_value_in_second.append(second_rpeaks[last_matching_rpeak])
        else:
            same_values.append(first_rpeaks[i])
    
    # convert the lists to numpy arrays for further calculations
    analog_value_in_first = np.array(analog_value_in_first)
    analog_value_in_second = np.array(analog_value_in_second)
    same_values = np.array(same_values)

    # calculate mean squared error and root mean squared error for R-peaks that were considered as the same
    if len(analog_value_in_first) > 0:
        mse_without_same = np.mean((analog_value_in_first - analog_value_in_second)**2)
    else:
        mse_without_same = 0
    rmse_without_same = np.sqrt(mse_without_same)

    # add the R-peaks that were detected by both methods to the lists of R-peaks that are considered as the same
    analog_value_in_first = np.append(analog_value_in_first, same_values)
    analog_value_in_second = np.append(analog_value_in_second, same_values)

    # calculate mean squared error and root mean squared error for R-peaks that were considered as the same and are the same
    if len(analog_value_in_first) > 0:
        mse_with_same = np.mean((analog_value_in_first - analog_value_in_second)**2)
    else:
        mse_with_same = 0
    rmse_with_same = np.sqrt(mse_with_same)

    # get the R-peaks that were only detected by one method
    # remaining_in_first = np.setdiff1d(first_rpeaks, analog_value_in_first)
    # remaining_in_second = np.setdiff1d(second_rpeaks, analog_value_in_second)
    
    return rmse_without_same, rmse_with_same, len(same_values), len(analog_value_in_second)


def rri_string_evaluation(string):
    """
    Appearance of string entrys in the .rri file: "integer letter".
    The integer shows the R peak position and the letter classifies the R peak.

    This functions returns the first integer and letter in the string. If either the letter
    or the integer does not exist, they are set to " ".

    ARGUMENTS:
    --------------------------------
    string: str
        string entry in the .rri file
    
    RETURNS:
    --------------------------------
    rpeak: int
        R peak position
    letter: str
        classification of the R peak

    """
    # set default values if the integer or the letter do not exist
    rpeak = " "
    letter = " "

    was_number = False
    for i in range(len(string)):
        if string[i].isdigit():
            if not was_number:
                start = i
            was_number = True
        else:
            if was_number:
                rpeak = int(string[start:i])
            was_number = False

            if string[i].isalpha():
                letter = string[i]
                break
    
    return rpeak, letter


def get_rpeaks_classification_from_rri_file(file_path: str):
    """
    Get R-peak classification from an .rri file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the .rri file

    RETURNS:
    --------------------------------
    rpeaks: dict
        dictionary containing the R-peaks depending on their classification in following
        format:
        {
            "classification_letter": np.array of R-peaks of this classification,
        } 
    """

    # read the RRI file
    rri_file = open(file_path, "r")
    rri = rri_file.readlines() #.split("\n")
    rri_file.close()
    
    # start of rpeaks are separated by a line of "-" from the file header in the .rri files
    # retrieve the start of the R-peaks in the file
    for i in range(len(rri)):
        count_dash = 0
        for j in range(len(rri[i])):
            if rri[i][j] == "-":
                count_dash += 1
        if count_dash/len(rri[i]) > 0.9:
            start = i + 1
            break

    # create dictionary to store the R-peaks depending on their classification
    rpeaks = dict()

    # get the R-peaks from the RRI file
    for i in range(start, len(rri)):
        this_rpeak, letter = rri_string_evaluation(rri[i])
        if isinstance(this_rpeak, int) and letter.isalpha():
            if letter in rpeaks:
                rpeaks[letter].append(this_rpeak)
            else:
                rpeaks[letter] = [this_rpeak]
    
    # convert the lists to numpy arrays
    for key in rpeaks:
        rpeaks[key] = np.array(rpeaks[key])

    return rpeaks


def read_rpeaks_from_rri_files(
        data_directory: str, 
        valid_file_types: list, 
        rpeaks_values_directory: str,
        valid_rpeak_values_file_types: list,
        include_rpeak_value_classifications: list,
        rpeak_path: str
    ):
    """
    Read the R peak values from all .rri files in the rpeaks_values_dirextory and save them

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the raw data is stored, to which we know the R peak values
    valid_file_types: list
        valid file types in the data directory
    rpeaks_values_directory: str
        directory where the R peak values are stored
    valid_rpeak_values_file_types: list
        valid file types in the rpeaks_values_directory
    include_rpeak_value_classifications: list
        list of the R peak classifications that should be included in the "R peak detection"
    rpeak_path: str
        path where the R peaks should be saved
    
    RETURNS:
    --------------------------------
    None, but the R peak values are saved as dictionary to a pickle file in following format:
    {
        "file_name": np.array of R-peaks of this file,
        ...
    }
    """

    # get all valid files
    all_data_files = os.listdir(data_directory)
    valid_data_files = [file for file in all_data_files if get_file_type(file) in valid_file_types]

    # get all valid files that contain rpeaks for previous directory
    all_values_files = os.listdir(rpeaks_values_directory)
    valid_values_files = [file for file in all_values_files if get_file_type(file) in valid_rpeak_values_file_types]

    # create variables to track progress
    total_data_files = len(valid_data_files)
    progressed_data_files = 0

    # create dictionary to store the R peak values for all files
    all_rpeaks = dict()
    
    # read the R peaks from the files
    print("\nReading R peak values from %i files:" % total_data_files)
    for file in valid_data_files:
        # show progress
        progress_bar(progressed_data_files, total_data_files)
        progressed_data_files += 1

        # get the file name without the file type
        this_file_name = os.path.splitext(file)[0]

        # get corresponding R peak value file name for this file
        for value_file in valid_values_files:
            if this_file_name in value_file:
                this_value_file = value_file
        try:
            rpeaks_values = get_rpeaks_classification_from_rri_file(rpeaks_values_directory + this_value_file)
        except ValueError:
            print("Accurate R peaks are missing for %s. Skipping this file." % file)
            continue

        # save R peak values with wanted classification to the dictionary
        all_rpeaks[file] = np.array([], dtype = int)
        for classification in include_rpeak_value_classifications:
            try:
                all_rpeaks[file] = np.append(all_rpeaks[file], rpeaks_values[classification])
            except KeyError:
                print("Classification %s is missing in %s. Skipping this classification." % (classification, file))
    
    progress_bar(progressed_data_files, total_data_files)

    # save the R peak values to a pickle file
    save_to_pickle(all_rpeaks, rpeak_path)


def rpeak_detection_comparison(
        data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        compare_rpeaks_paths: list,
        rpeak_distance_threshold_seconds: float,
        rpeak_comparison_evaluation_path: str
    ):
    """
    Evaluate the accuracy of the R peak detection methods.

    Accurate R peaks available for GIF data: 
    They were also detected automatically but later corrected manually, so they can be 
    used as a reference.

    ARGUMENTS:
    --------------------------------
    accurate_rpeaks_raw_data_directory: str
        directory where the raw ECG data is stored to which the accurate R peaks exist
    valid_file_types: list
        valid file types in the accurate_rpeaks_raw_data_directory
    ecg_key: str
        key for the ECG data in the data dictionary
    accurate_rpeaks_values_directory: str
        directory where the accurate R peaks are stored
    valid_accurate_rpeak_file_types: list
        valid file types in the accurate_rpeaks_values_directory
    compare_rpeaks_paths: list
        paths to the R peaks that should be compared to the accurate R peaks
    rpeak_distance_threshold_seconds: float
        time period in seconds over which two different R peaks are still considered the same
    rpeak_accuracy_evaluation_path: str
        path where the R peak accuracy values should be saved
    
    RETURNS:
    --------------------------------
    None, but the Accuracy values are saved as dictionary to a pickle file in following format:
    {
        "file_name": [ [function_1 values], [function_2 values], ... ],
        ...
    }
    with function values being: rmse_without_same, rmse_with_same, number_of_same_values, 
                                number_of_values_considered_as_same, len_function_rpeaks, 
                                length_accurate_rpeaks
    for rmse_without_same and rmse_with_same see rpeak_detection.compare_rpeak_detection_methods()
    """

    # check if the evaluation already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = rpeak_comparison_evaluation_path,
                        message = "\nEvaluation of R peak detection comparison already exists in " + rpeak_comparison_evaluation_path + ".")
    
    # cancel if user does not want to override
    if user_answer == "n":
        return

    # get all valid files
    all_data_files = os.listdir(data_directory)
    valid_data_files = [file for file in all_data_files if get_file_type(file) in valid_file_types]

    # create variables to track progress
    total_data_files = len(valid_data_files)
    progressed_data_files = 0

    # create dictionary to store the R peak accuracy values of all detection methods for all files
    all_files_rpeak_comparison = dict()
    
    # calculate the R peak accuracy values
    print("\nCalculating R peak comparison values for %i files:" % total_data_files)
    for file in valid_data_files:
        # show progress
        progress_bar(progressed_data_files, total_data_files)
        progressed_data_files += 1

        # create list to store the R peak comparison values for all detection methods as list
        this_file_rpeak_comparison = []
        
        # get the frequency of the ECG data
        sigfreqs = read_edf.get_edf_data(data_directory + file)[1]
        frequency = sigfreqs[ecg_key]
        
        # compare the R peaks of the different detection methods
        for path_index in range(len(compare_rpeaks_paths)):
            # load dictionaries with detected R peaks (contains R peaks of all files)
            first_rpeaks_all_files = load_from_pickle(compare_rpeaks_paths[path_index])
            second_rpeaks_all_files = load_from_pickle(compare_rpeaks_paths[path_index-1])

            # get the R peaks of the current file
            first_rpeaks = first_rpeaks_all_files[file]
            second_rpeaks = second_rpeaks_all_files[file]

            # get the number of detected R peaks
            number_first_rpeaks = len(first_rpeaks)
            number_second_rpeaks = len(second_rpeaks)

            # calculate the R peak comparison values
            rmse_without_same, rmse_with_same, len_same_values, len_analog_values = compare_rpeak_detections(
                first_rpeaks = first_rpeaks, 
                second_rpeaks = second_rpeaks,
                frequency = frequency,
                rpeak_distance_threshold_seconds = rpeak_distance_threshold_seconds,
                )
            
            # append list of R peak comparison values for these two detection methods to the list
            this_file_rpeak_comparison.append([rmse_without_same, rmse_with_same, len_same_values, len_analog_values, number_first_rpeaks, number_second_rpeaks])
        
        # save the R peak comparison values for this file to the dictionary
        all_files_rpeak_comparison[file] = this_file_rpeak_comparison
    
    progress_bar(progressed_data_files, total_data_files)
    
    # save the R peak accuracy values to a pickle file
    save_to_pickle(all_files_rpeak_comparison, rpeak_comparison_evaluation_path)


def rpeak_detection_comparison_report(
        rpeak_comparison_function_names: list,  
        rpeak_comparison_report_dezimal_places: int,
        rpeak_comparison_report_path: str,
        rpeak_comparison_evaluation_path: str
    ):
    """
    Save the results of the R peak accuracy evaluation as a report to a text file.

    ARGUMENTS:
    --------------------------------
    rpeak_accuracy_function_names: list
        names of the R peak detection methods
    accurate_peaks_name: str
        name of the accurate R peaks
    rpeak_accuracy_rmse_dezimal_places: int
        number of dezimal places for the RMSE values
    rpeak_accuracy_report_path: str
        path where the R peak accuracy report should be saved
    rpeak_accuracy_evaluation_path: str
        path to the R peak accuracy evaluation values (created by evaluate_rpeak_detection_accuracy())
    
    RETURNS:
    --------------------------------
    None, but the R peak accuracy report is saved to a text file in the given path
    Format of the report: Table showing results for each file
    """

    # check if the report already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = rpeak_comparison_report_path,
            message = "\nR peak comparison report already exists in " + rpeak_comparison_report_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # open the file to write the report to
    comparison_file = open(rpeak_comparison_report_path, "w")

    # write the file header
    message = "R PEAK COMPARISON REPORT"
    comparison_file.write(message + "\n")
    comparison_file.write("=" * len(message) + "\n\n\n")

    # load the data
    all_files_rpeak_comparison = load_from_pickle(rpeak_comparison_evaluation_path)

    # create mean row captions
    MEAN_ROW_CAPTION = "Mean Values:"
    RMSE_EX_MEAN = "RMSE_exc: "
    RMSE_INC_MEAN = "RMSE_inc: "
    SAME_VALUES_RATIO_MEAN = "Same Ratio: "
    ANALOG_VALUES_MEAN = "Analog Ratio: "
    TOTAL_DISTANCE_MEAN = "R peak distance: "
    TOTAL_DISTANCE_RATIO_MEAN = "      |-> Ratio: "
    mean_value_captions = [RMSE_EX_MEAN, RMSE_INC_MEAN, SAME_VALUES_RATIO_MEAN, ANALOG_VALUES_MEAN, TOTAL_DISTANCE_MEAN, TOTAL_DISTANCE_RATIO_MEAN]
    max_mean_value_caption_length = max([len(value) for value in mean_value_captions])

    # create lists to collect the various values to calculate the mean
    collect_rmse_exc = []
    collect_rmse_inc = []
    collect_rpeaks_distance = []
    collect_rpeaks_distance_ratio = []
    collect_analogue_values_ratio = []
    collect_same_values_ratio = []

    # collect various values to calculate the mean
    for file in all_files_rpeak_comparison:
        this_rmse_exc = []
        this_rmse_inc = []
        this_rpeaks_distance = []
        this_rpeaks_distance_ratio = []
        this_analogue_values_ratio = []
        this_same_values_ratio = []

        for funcs_index in range(len(rpeak_comparison_function_names)):
            # round rmse values
            all_files_rpeak_comparison[file][funcs_index][0] = round(all_files_rpeak_comparison[file][funcs_index][0], rpeak_comparison_report_dezimal_places)
            all_files_rpeak_comparison[file][funcs_index][1] = round(all_files_rpeak_comparison[file][funcs_index][1], rpeak_comparison_report_dezimal_places)

            # collect rmse values
            this_rmse_exc.append(all_files_rpeak_comparison[file][funcs_index][0])
            this_rmse_inc.append(all_files_rpeak_comparison[file][funcs_index][1])
            
            # collect distance of number of detected R peaks
            this_rpeaks_distance.append(abs(all_files_rpeak_comparison[file][funcs_index][4] - all_files_rpeak_comparison[file][funcs_index][5]))

            # collect ratio of distance of number of detected R peaks
            this_rpeaks_distance_ratio.append([this_rpeaks_distance[funcs_index] / all_files_rpeak_comparison[file][funcs_index][4], this_rpeaks_distance[funcs_index] / all_files_rpeak_comparison[file][funcs_index][5]])

            # collect ratio of analog values to number of r peaks
            this_analogue_values_ratio.append([all_files_rpeak_comparison[file][funcs_index][3] / all_files_rpeak_comparison[file][funcs_index][4], all_files_rpeak_comparison[file][funcs_index][3] / all_files_rpeak_comparison[file][funcs_index][5]])

            # collect ratio of same values to number of r peaks
            this_same_values_ratio.append([all_files_rpeak_comparison[file][funcs_index][2] / all_files_rpeak_comparison[file][funcs_index][4], all_files_rpeak_comparison[file][funcs_index][2] / all_files_rpeak_comparison[file][funcs_index][5]])
        
        collect_rmse_exc.append(this_rmse_exc)
        collect_rmse_inc.append(this_rmse_inc)
        collect_rpeaks_distance.append(this_rpeaks_distance)
        collect_rpeaks_distance_ratio.append(this_rpeaks_distance_ratio)
        collect_analogue_values_ratio.append(this_analogue_values_ratio)
        collect_same_values_ratio.append(this_same_values_ratio)
    
    # calculate mean values
    mean_rmse_exc = np.mean(collect_rmse_exc, axis = 0)
    mean_rmse_inc = np.mean(collect_rmse_inc, axis = 0)
    mean_rpeaks_distance = np.mean(collect_rpeaks_distance, axis = 0)
    mean_rpeaks_distance_ratio = np.mean(collect_rpeaks_distance_ratio, axis = 0)
    mean_analogue_values_ratio = np.mean(collect_analogue_values_ratio, axis = 0)
    mean_same_values_ratio = np.mean(collect_same_values_ratio, axis = 0)

    mean_row_values = []
    mean_row_lengths = []
    for funcs_index in range(len(rpeak_comparison_function_names)):
        this_column = []
        this_column.append(str(round(mean_rmse_exc[funcs_index], rpeak_comparison_report_dezimal_places)))
        this_column.append(str(round(mean_rmse_inc[funcs_index], rpeak_comparison_report_dezimal_places)))
        this_column.append(str(round(mean_same_values_ratio[funcs_index][0], rpeak_comparison_report_dezimal_places)) + " / " + str(round(mean_same_values_ratio[funcs_index][1], rpeak_comparison_report_dezimal_places)))
        this_column.append(str(round(mean_analogue_values_ratio[funcs_index][0], rpeak_comparison_report_dezimal_places)) + " / " + str(round(mean_analogue_values_ratio[funcs_index][1], rpeak_comparison_report_dezimal_places)))
        this_column.append(str(round(mean_rpeaks_distance[funcs_index], rpeak_comparison_report_dezimal_places)))
        this_column.append(str(round(mean_rpeaks_distance_ratio[funcs_index][0], rpeak_comparison_report_dezimal_places)) + " / " + str(round(mean_rpeaks_distance_ratio[funcs_index][1], rpeak_comparison_report_dezimal_places)))
        mean_row_values.append(this_column)
        mean_row_lengths.append([len(value) for value in this_column])
    
    mean_row_lengths = np.array(mean_row_lengths)

    # create table column captions
    FILE_CAPTION = "File"
    column_captions = [FILE_CAPTION]
    for funcs_index in range(len(rpeak_comparison_function_names)):
        column_captions.append(rpeak_comparison_function_names[funcs_index] + " / " + rpeak_comparison_function_names[funcs_index-1])
    
    # create column value captions
    RMSE_EX_CAPTION = "RMSE_exc: "
    RMSE_INC_CAPTION = "RMSE_inc: "
    SAME_VALUES_CAPTION = "Same Values: "
    SAME_VALUES_RATIO_CAPTION = "  |-> Ratio: "
    ANALOG_VALUES_CAPTION = "Analog Values: "
    ANALOG_VALUES_RATIO_CAPTION = "    |-> Ratio: "
    TOTAL_LENGTH_CAPTION = "R peaks: "
    value_captions = [RMSE_EX_CAPTION, RMSE_INC_CAPTION, TOTAL_LENGTH_CAPTION, SAME_VALUES_CAPTION, SAME_VALUES_RATIO_CAPTION, ANALOG_VALUES_CAPTION, ANALOG_VALUES_RATIO_CAPTION]
    max_value_caption_length = max([len(value) for value in value_captions])
    max_value_caption_length = max(max_value_caption_length, max_mean_value_caption_length)
            
    # calcualte max lengths of table columns
    column_caption_length = [len(name) for name in column_captions]

    all_file_lengths = [len(key) for key in all_files_rpeak_comparison]
    all_file_lengths.append(len(MEAN_ROW_CAPTION))
    max_file_length_column = max(len(FILE_CAPTION), max(all_file_lengths))

    all_columns = []
    all_column_lengths = []

    for funcs_index in range(len(rpeak_comparison_function_names)):
        this_column = []
        for file in all_files_rpeak_comparison:
            # RMSE excluding same R peaks:
            this_column.append(str(all_files_rpeak_comparison[file][funcs_index][0]))
            # RMSE including same R peaks:
            this_column.append(str(all_files_rpeak_comparison[file][funcs_index][1]))
            # Total number of R peaks:
            this_column.append(str(all_files_rpeak_comparison[file][funcs_index][4]) + " / " + str(all_files_rpeak_comparison[file][funcs_index][5]))
            # Number of R peaks that are the same:
            this_column.append(str(all_files_rpeak_comparison[file][funcs_index][2]))
            # Ratio of same values to number of r peaks:
            key_to_index = list(all_files_rpeak_comparison.keys()).index(file)
            same_val_ratio_1 = round(collect_same_values_ratio[key_to_index][funcs_index][0], rpeak_comparison_report_dezimal_places)
            same_val_ratio_2 = round(collect_same_values_ratio[key_to_index][funcs_index][1], rpeak_comparison_report_dezimal_places)
            this_column.append(str(same_val_ratio_1) + " / " + str(same_val_ratio_2))
            # Number of R peaks that are considered as the same (difference < threshold):
            this_column.append(str(all_files_rpeak_comparison[file][funcs_index][3]))
            # Ratio of analog values to number of r peaks:
            analog_val_ratio_1 = round(collect_analogue_values_ratio[key_to_index][funcs_index][0], rpeak_comparison_report_dezimal_places)
            analog_val_ratio_2 = round(collect_analogue_values_ratio[key_to_index][funcs_index][1], rpeak_comparison_report_dezimal_places)
            this_column.append(str(analog_val_ratio_1) + " / " + str(analog_val_ratio_2))

        all_columns.append(this_column)
        all_column_lengths.append([len(value) for value in this_column])
    
    all_column_lengths = np.array(all_column_lengths)
    all_column_lengths = np.append(all_column_lengths, mean_row_lengths, axis = 1)
    max_column_length = np.max(all_column_lengths, axis = 1)
    max_column_length = np.insert(max_column_length, 0, max_file_length_column-max_value_caption_length)

    column_caption_length = np.array(column_caption_length)
    column_caption_length -= max_value_caption_length
    for i in range(1, len(max_column_length)):
        max_column_length[i] = max(max_column_length[i], column_caption_length[i])

    # write the legend for the table
    message = "Legend:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")
    comparison_file.write(RMSE_EX_CAPTION + "RMSE excluding same R peaks\n")
    comparison_file.write(RMSE_INC_CAPTION + "RMSE including same R peaks\n")
    comparison_file.write(SAME_VALUES_CAPTION +  "Number of R peaks that are the same\n")
    comparison_file.write(ANALOG_VALUES_CAPTION + "Number of R peaks that are considered as the same (difference < threshold)\n")
    comparison_file.write(TOTAL_LENGTH_CAPTION + "Total number of R peaks\n\n\n")

    message = "Table with Accuracy Values for each file:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")

    # create table header
    total_length = 0
    for i in range(len(column_captions)):
        comparison_file.write(print_in_middle(column_captions[i], max_column_length[i] + max_value_caption_length) + " | ")
        total_length += max_column_length[i] + max_value_caption_length + 3
    total_length -= 1

    comparison_file.write("\n")
    comparison_file.write("-" * total_length + "\n")

    # write the mean values
    vertical_center_index = len(mean_value_captions) // 2

    for value_index in range(len(mean_value_captions)):
        if value_index == vertical_center_index:
            comparison_file.write(print_in_middle(MEAN_ROW_CAPTION, max_column_length[0] + max_value_caption_length) + " | ")
        else:
            comparison_file.write(print_in_middle("", max_column_length[0] + max_value_caption_length) + " | ")
        for funcs_index in range(len(rpeak_comparison_function_names)):
            comparison_file.write(print_left_aligned(mean_value_captions[value_index], max_value_caption_length))
            comparison_file.write(print_left_aligned(str(mean_row_values[funcs_index][value_index]), max_column_length[funcs_index+1]))
            comparison_file.write(" | ")
        comparison_file.write("\n")
    comparison_file.write("-" * total_length + "\n")

    # write the data
    vertical_center_index = len(value_captions) // 2

    for file_index in range(len(all_files_rpeak_comparison)):
        for value_index in range(len(value_captions)):
            if value_index == vertical_center_index:
                index_to_key = str(list(all_files_rpeak_comparison.keys())[file_index])
                comparison_file.write(print_in_middle(index_to_key, max_column_length[0] + max_value_caption_length) + " | ")
            else:
                comparison_file.write(print_in_middle("", max_column_length[0] + max_value_caption_length) + " | ")
            for funcs_index in range(len(rpeak_comparison_function_names)):
                comparison_file.write(print_left_aligned(value_captions[value_index], max_value_caption_length))
                comparison_file.write(print_left_aligned(str(all_columns[funcs_index][file_index+value_index]), max_column_length[funcs_index+1]))
                comparison_file.write(" | ")
            comparison_file.write("\n")
        comparison_file.write("-" * total_length + "\n")

    comparison_file.close()