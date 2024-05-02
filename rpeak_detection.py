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
comparing the results of different R-peak detection methods, as accurate R peaks
(calculated automatically but later checked manually) are available for the GIF data.

They are stored in .rri files and can be used as a reference for the R-peak detection.
Therefore we need to implement functions to compare the results of different R-peaks
and to read the accurate R-peaks from the .rri files.
They are stored in the following format: "integer letter" after a file header containing
various information separated by a line of "-".
"""


def compare_rpeak_detection_methods(
        first_rpeaks: list,
        second_rpeaks: list,
        frequency: int,
        rpeak_distance_threshold_seconds: float,
        **kwargs
    ):
    """
    Compare the results of two different R-peak detection methods.

    ARGUMENTS:
    --------------------------------
    first_rpeaks: list
        R-peak locations detected by the first method
    second_rpeaks: list
        R-peak locations detected by the second method
    frequency: int
        sampling rate / frequency of the ECG data
    rpeak_distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same

    KEYWORD ARGUMENTS (they are hidden because were only necessary for testing purposes):
    --------------------------------
    first_name: str
        name of the first method
    second_name: str
        name of the second method
    print_results: bool
        if True, the results will be printed to the console

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

    # set default values
    kwargs.setdefault("first_name", "First Method")
    kwargs.setdefault("second_name", "Second Method")
    kwargs.setdefault("print_results", False)

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

    # print the results if desired
    if kwargs["print_results"]:
        print("Number of same values: ", len(same_values))
        print("Number of values considered as equal (distance < %f s): %i" % (rpeak_distance_threshold_seconds, len(analog_value_in_first)))

    # calculate mean squared error and root mean squared error for R-peaks that were considered as the same
    mse_without_same = np.mean((analog_value_in_first - analog_value_in_second)**2)
    rmse_without_same = np.sqrt(mse_without_same)

    # add the R-peaks that were detected by both methods to the lists of R-peaks that are considered as the same
    analog_value_in_first = np.append(analog_value_in_first, same_values)
    analog_value_in_second = np.append(analog_value_in_second, same_values)

    # calculate mean squared error and root mean squared error for R-peaks that were considered as the same and are the same
    mse_with_same = np.mean((analog_value_in_first - analog_value_in_second)**2)
    rmse_with_same = np.sqrt(mse_with_same)

    # get the R-peaks that were only detected by one method
    remaining_in_first = np.setdiff1d(first_rpeaks, analog_value_in_first)
    remaining_in_second = np.setdiff1d(second_rpeaks, analog_value_in_second)

    # print the results if desired
    if kwargs["print_results"]:
        print("")
        print("Number of remaining values in %s: %i" % (kwargs["first_name"], len(remaining_in_first)))
        print("Number of remaining values in %s: %i" % (kwargs["second_name"], len(remaining_in_second)))
        print("Ratio of remaining values in %s: %f %%" % (kwargs["first_name"], round(100*len(remaining_in_first)/len(first_rpeaks), 2)))
        print("Ratio of remaining values in %s: %f %%" % (kwargs["second_name"], round(100*len(remaining_in_second)/len(second_rpeaks), 2)))
        print("")
        #print("Mean squared error without same values: ", mse_without_same)
        print("Root mean squared error without same values: %f = %f s" % (rmse_without_same, rmse_without_same/frequency))
        #print("Mean squared error with same values: ", mse_with_same)
        print("Root mean squared error with same values:%f = %f s" % (rmse_with_same, rmse_with_same/frequency))
    
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


def get_rpeaks_from_rri_file(file_path: str):
    """
    Get R-peaks from an .rri file.

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


def evaluate_rpeak_detection_accuracy(
        accurate_rpeaks_raw_data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        accurate_rpeaks_values_directory: str,
        valid_accurate_rpeak_file_types: list,
        compare_rpeaks_paths: list,
        rpeak_distance_threshold_seconds: float,
        rpeak_accuracy_evaluation_path: str
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
    user_answer = ask_for_permission_to_override(file_path = rpeak_accuracy_evaluation_path,
                        message = "\nEvaluation of R peak detection accuracy already exists in " + rpeak_accuracy_evaluation_path + ".")
    
    # cancel if user does not want to override
    if user_answer == "n":
        return

    # get all valid files
    all_data_files = os.listdir(accurate_rpeaks_raw_data_directory)
    valid_data_files = [file for file in all_data_files if get_file_type(file) in valid_file_types]

    # get all valid accurate R peak files
    all_accurate_files = os.listdir(accurate_rpeaks_values_directory)
    valid_accurate_files = [file for file in all_accurate_files if get_file_type(file) in valid_accurate_rpeak_file_types]

    # create variables to track progress
    total_data_files = len(valid_data_files)
    progressed_data_files = 0

    # create dictionary to store the R peak accuracy values of all detection methods for all files
    all_files_rpeak_accuracy = dict()
    
    # calculate the R peak accuracy values
    print("\nCalculating R peak accuracy values for %i files:" % total_data_files)
    for file in valid_data_files:
        # show progress
        progress_bar(progressed_data_files, total_data_files)
        progressed_data_files += 1

        # create list to store the R peak accuracy values for all detection methods as list
        this_file_rpeak_accuracy = []

        # get the file name without the file type
        this_file_name = os.path.splitext(file)[0]

        # get corresponding accurate R peaks for this file
        for acc_file in valid_accurate_files:
            if this_file_name in acc_file:
                this_accurate_file = acc_file
        try:
            accurate_rpeaks = get_rpeaks_from_rri_file(accurate_rpeaks_values_directory + this_accurate_file)
        except ValueError:
            print("Accurate R peaks are missing for %s. Skipping this file." % file)
            continue
        
        # get the frequency of the ECG data
        sigfreqs = read_edf.get_edf_data(accurate_rpeaks_raw_data_directory + file)[1]
        frequency = sigfreqs[ecg_key]
        
        # get the number of accurate R peaks
        length_accurate = len(accurate_rpeaks["N"])
        
        # compare the R peaks of the different detection methods to the accurate R peaks
        for path in compare_rpeaks_paths:
            # load dictionary with detected R peaks (contains R peaks of all files)
            compare_rpeaks_all_files = load_from_pickle(path)

            # get the R peaks of the current file
            compare_rpeaks = compare_rpeaks_all_files[file]

            # get the number of detected R peaks
            len_compare_rpeaks = len(compare_rpeaks)

            # calculate the R peak accuracy values
            rmse_without_same, rmse_with_same, len_same_values, len_analog_values = compare_rpeak_detection_methods(
                first_rpeaks = accurate_rpeaks["N"], 
                second_rpeaks = compare_rpeaks,
                frequency = frequency,
                rpeak_distance_threshold_seconds = rpeak_distance_threshold_seconds,
                )
            
            # append list of R peak accuracy values for this detection method to the list
            this_file_rpeak_accuracy.append([rmse_without_same, rmse_with_same, len_same_values, len_analog_values, len_compare_rpeaks, length_accurate])
        
        # save the R peak accuracy values for this file to the dictionary
        all_files_rpeak_accuracy[file] = this_file_rpeak_accuracy
    
    progress_bar(progressed_data_files, total_data_files)
    
    # save the R peak accuracy values to a pickle file
    save_to_pickle(all_files_rpeak_accuracy, rpeak_accuracy_evaluation_path)


def print_rpeak_accuracy_results(
        rpeak_accuracy_function_names: list,  
        accurate_peaks_name: str, 
        rpeak_accuracy_rmse_dezimal_places: int,
        rpeak_accuracy_report_path: str,
        rpeak_accuracy_evaluation_path: str
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
    user_answer = ask_for_permission_to_override(file_path = rpeak_accuracy_report_path,
            message = "\nR peak accuracy report already exists in " + rpeak_accuracy_report_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # open the file to write the report to
    accuracy_file = open(rpeak_accuracy_report_path, "w")

    # write the file header
    message = "R PEAK ACCURACY EVALUATION"
    accuracy_file.write(message + "\n")
    accuracy_file.write("=" * len(message) + "\n\n\n")

    # set the table captions
    RMSE_EX_CAPTION = "RMSE_exc"
    RMSE_INC_CAPTION = "RMSE_inc"
    FILE_CAPTION = "File"
    TOTAL_LENGTH_CAPTION = "R peaks"
    SAME_VALUES_CAPTION = "Same Values"
    ANALOG_VALUES_CAPTION = "Analog Values"

    # load the data
    all_files_rpeak_accuracy = load_from_pickle(rpeak_accuracy_evaluation_path)

    # create lists to collect the RMSE values to calculate the mean
    collect_rmse_exc = []
    collect_rmse_inc = []

    # round rmse values and collect them to print the mean
    for file in all_files_rpeak_accuracy:
        this_rmse_exc = []
        this_rmse_inc = []
        for func in range(len(rpeak_accuracy_function_names)):
            all_files_rpeak_accuracy[file][func][0] = round(all_files_rpeak_accuracy[file][func][0], rpeak_accuracy_rmse_dezimal_places)
            all_files_rpeak_accuracy[file][func][1] = round(all_files_rpeak_accuracy[file][func][1], rpeak_accuracy_rmse_dezimal_places)

            this_rmse_exc.append(all_files_rpeak_accuracy[file][func][0])
            this_rmse_inc.append(all_files_rpeak_accuracy[file][func][1])
        
        collect_rmse_exc.append(this_rmse_exc)
        collect_rmse_inc.append(this_rmse_inc)
    
    # calculate mean rmse values
    mean_rmse_exc = np.mean(collect_rmse_exc, axis = 0)
    mean_rmse_inc = np.mean(collect_rmse_inc, axis = 0)

    # calculate mean distance of number of detected R peaks to accurate R peaks
    collect_rpeaks_distance = []

    for file in all_files_rpeak_accuracy:
        this_rpeaks_distance = []
        for func in range(len(rpeak_accuracy_function_names)):
            this_rpeaks_distance.append(abs(all_files_rpeak_accuracy[file][func][4] - all_files_rpeak_accuracy[file][func][5]))
        collect_rpeaks_distance.append(this_rpeaks_distance)
    
    mean_rpeaks_distance = np.mean(collect_rpeaks_distance, axis = 0)

    # calculate ratio of analog values to accurate R peaks
    collect_analogue_values_ratio = []

    for file in all_files_rpeak_accuracy:
        this_same_values_ratio = []
        for func in range(len(rpeak_accuracy_function_names)):
            this_same_values_ratio.append(all_files_rpeak_accuracy[file][func][3] / all_files_rpeak_accuracy[file][func][5])
        collect_analogue_values_ratio.append(this_same_values_ratio)

    mean_same_values_ratio = np.mean(collect_analogue_values_ratio, axis = 0)

    # write the mean values to file
    message = "Mean values to compare used functions:"
    accuracy_file.write(message + "\n")
    accuracy_file.write("-" * len(message) + "\n\n")
    captions = ["Mean RMSE_exc", "Mean RMSE_inc", "Mean R peaks difference", "Mean Analogue Values"]
    caption_values = [mean_rmse_exc, mean_rmse_inc, mean_rpeaks_distance, mean_same_values_ratio]
    for i in range(len(captions)):
        message = captions[i] + " | "
        first = True
        for func in range(len(rpeak_accuracy_function_names)):
            if first:
                accuracy_file.write(message)
                first = False
            else:
                accuracy_file.write(" " * (len(message)-2) + "| ")
            accuracy_file.write(rpeak_accuracy_function_names[func] + ": " + str(caption_values[i][func]))
            accuracy_file.write("\n")
        accuracy_file.write("\n")
    
    accuracy_file.write("\n")
            
    # calcualte max lengths of table columns
    max_func_name = max([len(name) for name in rpeak_accuracy_function_names])

    all_file_lengths = [len(key) for key in all_files_rpeak_accuracy]
    max_file_length = max(len(FILE_CAPTION), max(all_file_lengths)) + 3

    all_rmse_ex_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(rpeak_accuracy_function_names)):
            all_rmse_ex_lengths.append(len(str(all_files_rpeak_accuracy[file][func][0])))
    all_rmse_ex_lengths = np.array(all_rmse_ex_lengths)
    all_rmse_ex_lengths += max_func_name
    max_rmse_ex_length = max(len(RMSE_EX_CAPTION), max(all_rmse_ex_lengths)) + 3

    all_rmse_inc_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(rpeak_accuracy_function_names)):
            all_rmse_inc_lengths.append(len(str(all_files_rpeak_accuracy[file][func][1])))
    all_rmse_inc_lengths = np.array(all_rmse_inc_lengths)
    all_rmse_inc_lengths += max_func_name
    max_rmse_inc_length = max(len(RMSE_INC_CAPTION), max(all_rmse_inc_lengths)) + 3

    all_same_values_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(rpeak_accuracy_function_names)):
            all_same_values_lengths.append(len(str(all_files_rpeak_accuracy[file][func][2])))
    all_same_values_lengths = np.array(all_same_values_lengths)
    all_same_values_lengths += max_func_name
    max_same_values_length = max(len(SAME_VALUES_CAPTION), max(all_same_values_lengths)) + 3

    all_analog_values_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(rpeak_accuracy_function_names)):
            all_analog_values_lengths.append(len(str(all_files_rpeak_accuracy[file][func][3])))
    all_analog_values_lengths = np.array(all_analog_values_lengths)
    all_analog_values_lengths += max_func_name
    max_analog_values_length = max(len(ANALOG_VALUES_CAPTION), max(all_analog_values_lengths)) + 3

    max_func_name = max(max_func_name, len(accurate_peaks_name)) + 3

    all_rpeaks_lengths = []
    for file in all_files_rpeak_accuracy:
        for func in range(len(rpeak_accuracy_function_names)):
            all_rpeaks_lengths.append(len(str(all_files_rpeak_accuracy[file][func][4])))

    for file in all_files_rpeak_accuracy:
        for func in range(len(rpeak_accuracy_function_names)):
            all_rpeaks_lengths.append(len(str(all_files_rpeak_accuracy[file][func][5])))
    all_rpeaks_lengths = np.array(all_rpeaks_lengths)

    all_rpeaks_lengths += max_func_name
    max_rpeaks_length = max(len(TOTAL_LENGTH_CAPTION), max(all_rpeaks_lengths)) + 3

    # write the legend for the table
    message = "Legend:"
    accuracy_file.write(message + "\n")
    accuracy_file.write("-" * len(message) + "\n\n")
    accuracy_file.write("RMSE_exc... RMSE excluding same R peaks\n")
    accuracy_file.write("RMSE_inc... RMSE including same R peaks\n")
    accuracy_file.write("R peaks... Total number of R peaks\n")
    accuracy_file.write("Same Values... Number of R peaks that are the same\n")
    accuracy_file.write("Analogue Values... Number of R peaks that are considered as the same (difference < threshold)\n\n\n")

    message = "Table with Accuracy Values for each file:"
    accuracy_file.write(message + "\n")
    accuracy_file.write("-" * len(message) + "\n\n")

    # create table header
    accuracy_file.write(print_in_middle(FILE_CAPTION, max_file_length) + " | ")
    accuracy_file.write(print_in_middle(RMSE_EX_CAPTION, max_rmse_ex_length) + " | ")
    accuracy_file.write(print_in_middle(RMSE_INC_CAPTION, max_rmse_inc_length) + " | ")
    accuracy_file.write(print_in_middle(TOTAL_LENGTH_CAPTION, max_rpeaks_length) + " | ")
    accuracy_file.write(print_in_middle(SAME_VALUES_CAPTION, max_same_values_length) + " | ")
    accuracy_file.write(print_in_middle(ANALOG_VALUES_CAPTION, max_analog_values_length) + " | ")
    accuracy_file.write("\n")
    accuracy_file.write("-" * (max_file_length + max_rmse_ex_length + max_rmse_inc_length + max_rpeaks_length + max_same_values_length + max_analog_values_length + 17) + "\n")

    # write the data
    for file in all_files_rpeak_accuracy:
        accuracy_file.write(print_in_middle(file, max_file_length) + " | ")
        first = True
        for func in range(len(rpeak_accuracy_function_names)):
            if first:
                first = False
            else:
                accuracy_file.write(print_in_middle("", max_file_length) + " | ")
            accuracy_file.write(print_in_middle(rpeak_accuracy_function_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][0]), max_rmse_ex_length) + " | ")
            accuracy_file.write(print_in_middle(rpeak_accuracy_function_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][1]), max_rmse_inc_length) + " | ")
            accuracy_file.write(print_in_middle(rpeak_accuracy_function_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][4]), max_rpeaks_length) + " | ")
            accuracy_file.write(print_in_middle(rpeak_accuracy_function_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][2]), max_same_values_length) + " | ")
            accuracy_file.write(print_in_middle(rpeak_accuracy_function_names[func] + ": " + str(all_files_rpeak_accuracy[file][func][3]), max_analog_values_length) + " | ")
            accuracy_file.write("\n")
        accuracy_file.write(print_in_middle("", max_file_length) + " | ")
        accuracy_file.write(print_in_middle("", max_rmse_ex_length) + " | ")
        accuracy_file.write(print_in_middle("", max_rmse_inc_length) + " | ")
        accuracy_file.write(print_in_middle(accurate_peaks_name + ": " + str(all_files_rpeak_accuracy[file][func][5]), max_rpeaks_length) + " | ")
        accuracy_file.write(print_in_middle("", max_same_values_length) + " | ")
        accuracy_file.write(print_in_middle("", max_analog_values_length) + " | ")
        accuracy_file.write("\n")
        accuracy_file.write("-" * (max_file_length + max_rmse_ex_length + max_rmse_inc_length + max_rpeaks_length + max_same_values_length + max_analog_values_length + 17) + "\n")

    accuracy_file.close()