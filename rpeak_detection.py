"""
Author: Johannes Peter Knoll

Python implementation to detect R-peaks in ECG data.
Useful Link: https://www.samproell.io/posts/signal/ecg-library-comparison/

Main function: get_rpeaks
"""
import numpy as np
import time

# import libraries for rpeak detection
import neurokit2
import wfdb.processing

# import old code used for rpeak detection
import old_code.rpeak_detection as old_rpeak


def get_rpeaks_old(
        data: dict, 
        frequency: dict, 
        ecg_key: str, 
        detection_interval: tuple
    ):
    """
    Detect R-peaks in ECG data using the old code that was used by the research group before me.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple
        interval in which the R-peaks should be detected

    RETURNS:
    --------------------------------
    rpeaks_old: 1D numpy array
        R-peak locations
    """
    sampling_rate = frequency[ecg_key]
    if detection_interval is None:
        ecg_signal = data[ecg_key]
    else:
        ecg_signal = data[ecg_key][detection_interval[0]:detection_interval[1]]

    rpeaks_old = old_rpeak.get_rpeaks(ecg_signal, sampling_rate)
    
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
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        interval in which the R-peaks should be detected

    RETURNS:
    --------------------------------
    rpeaks_corrected: 1D numpy array
        R-peak locations
    """
    sampling_rate = frequency[ecg_key]
    if detection_interval is None:
        ecg_signal = data[ecg_key]
    else:
        ecg_signal = data[ecg_key][detection_interval[0]:detection_interval[1]]

    _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    rpeaks = results["ECG_R_Peaks"]


    rpeaks_corrected = wfdb.processing.correct_peaks(
        ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up"
    )
    #wfdb.plot_items(ecg_signal, [rpeaks_corrected])  # styling options omitted

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
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        interval in which the R-peaks should be detected

    RETURNS:
    --------------------------------
    rpeaks_corrected: 1D numpy array
        R-peak locations
    """
    sampling_rate = frequency[ecg_key]
    if detection_interval is None:
        ecg_signal = data[ecg_key]
    else:
        ecg_signal = data[ecg_key][detection_interval[0]:detection_interval[1]]

    rpeaks = wfdb.processing.xqrs_detect(ecg_signal, fs=sampling_rate, verbose=False)
    rpeaks_corrected = wfdb.processing.correct_peaks(
        ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up"
    )

    if detection_interval is not None:
        rpeaks_corrected += detection_interval[0]

    return rpeaks_corrected


def combined_rpeak_detection_methods(
        data: dict, 
        frequency: dict, 
        ecg_key: str, 
        detection_interval: tuple,
        rpeak_primary_function,
        rpeak_secondary_function,
        rpeak_distance_threshold_seconds: float,
    ):
    """
    Detect R-peaks in ECG data using two different libraries. 
    This way we can compare the results and categorize the R-peaks as sure or unsure.

    Suggested is the wfdb library and the detection function that was used by the research 
    group before me (see old_code/rpeak_detection.py).

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        interval in which the R-peaks should be detected
    rpeak_primary_function: function, default get_rpeaks_wfdb
        primary R peak detection function
    rpeak_secondary_function: function, default get_rpeaks_old
        secondary R peak detection function
    rpeak_distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same 
        (reasonable to use highest heart rate ever recorded: 480 bpm = 0.125 spb,
        but better results with 300 bpm = 0.2 spb)

    RETURNS:
    --------------------------------
    rpeaks_intersected: 1D numpy array
        R-peak locations that were detected by both methods
    rpeaks_only_primary: 1D numpy array
        R-peak locations that were only detected by the primary method
    rpeaks_only_secondary: 1D numpy array
        R-peak locations that were only detected by the secondary method
    """
    # convert the threshold to iterations
    distance_threshold_iterations = int(rpeak_distance_threshold_seconds * frequency[ecg_key])

    # get R-peaks using both methods
    rpeaks_primary = rpeak_primary_function(data, frequency, ecg_key, detection_interval)
    rpeaks_secondary = rpeak_secondary_function(data, frequency, ecg_key, detection_interval)

    # if two R-peaks are closer than the threshold, they are considered as the same
    # both will be changed to the same value (primary R-peak)

    # intersects_before = len(np.intersect1d(rpeaks_primary, rpeaks_secondary))

    last_matching_rpeak = -1
    for i in range(len(rpeaks_primary)):
        if rpeaks_primary[i] not in rpeaks_secondary:
            possible_matches = []
            possible_matches_values = []
            for j in range(last_matching_rpeak + 1, len(rpeaks_secondary)):
                this_distance = rpeaks_secondary[j] - rpeaks_primary[i]
                possible_matches_values.append(abs(this_distance))
                possible_matches.append(j)
                if this_distance > distance_threshold_iterations:
                    break
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


"""
Following code won't be used for the final implementation, but is useful for testing and
comparing the results of different R-peak detection methods, as accurate R peaks
(calculated automatically but later checked manually) are available for the GIF data.
"""


def compare_rpeak_detection_methods(
        first_rpeaks: list,
        second_rpeaks: list,
        first_name: str,
        second_name: str,
        frequency: int,
        rpeak_distance_threshold_seconds: float,
        print_results: bool
    ):
    """
    Compare the results of two different R-peak detection methods.

    ARGUMENTS:
    --------------------------------
    first_rpeaks: list
        R-peak locations detected by the first method
    second_rpeaks: list
        R-peak locations detected by the second method
    rpeak_distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same
    print_results: bool
        if True, the results will be printed

    RETURNS:
    --------------------------------
    None
    """
    # convert the threshold to iterations
    distance_threshold_iterations = int(rpeak_distance_threshold_seconds * frequency)

    # lists to store the r peaks that are considered as the same (distance < threshold, distance != 0)
    analog_value_in_first = []
    analog_value_in_second = []

    same_values = []

    # if two R-peaks are closer than the threshold, they are considered as the same
    last_matching_rpeak = -1
    for i in range(len(first_rpeaks)):
        if first_rpeaks[i] not in second_rpeaks:
            possible_matches = []
            possible_matches_values = []
            for j in range(last_matching_rpeak + 1, len(second_rpeaks)):
                this_distance = second_rpeaks[j] - first_rpeaks[i]
                possible_matches_values.append(abs(this_distance))
                possible_matches.append(j)
                if this_distance > distance_threshold_iterations:
                    break
            if len(possible_matches_values) > 0:
                if min(possible_matches_values) < distance_threshold_iterations:
                    last_matching_rpeak = possible_matches[possible_matches_values.index(min(possible_matches_values))]
                    analog_value_in_first.append(first_rpeaks[i])
                    analog_value_in_second.append(second_rpeaks[last_matching_rpeak])
        else:
            same_values.append(first_rpeaks[i])
    
    # mean squared error and root mean squared error
    analog_value_in_first = np.array(analog_value_in_first)
    analog_value_in_second = np.array(analog_value_in_second)
    same_values = np.array(same_values)
    if print_results:
        print("Number of same values: ", len(same_values))
        print("Number of values considered as equal (distance < %f s): %i" % (rpeak_distance_threshold_seconds, len(analog_value_in_first)))

    mse_without_same = np.mean((analog_value_in_first - analog_value_in_second)**2)
    rmse_without_same = np.sqrt(mse_without_same)

    analog_value_in_first = np.append(analog_value_in_first, same_values)
    analog_value_in_second = np.append(analog_value_in_second, same_values)

    mse_with_same = np.mean((analog_value_in_first - analog_value_in_second)**2)
    rmse_with_same = np.sqrt(mse_with_same)

    remaining_in_first = np.setdiff1d(first_rpeaks, analog_value_in_first)
    remaining_in_second = np.setdiff1d(second_rpeaks, analog_value_in_second)

    if print_results:
        print("")
        print("Number of remaining values in %s: %i" % (first_name, len(remaining_in_first)))
        print("Number of remaining values in %s: %i" % (second_name, len(remaining_in_second)))
        print("Ratio of remaining values in %s: %f %%" % (first_name, round(100*len(remaining_in_first)/len(first_rpeaks), 2)))
        print("Ratio of remaining values in %s: %f %%" % (second_name, round(100*len(remaining_in_second)/len(second_rpeaks), 2)))
        print("")
        #print("Mean squared error without same values: ", mse_without_same)
        print("Root mean squared error without same values: %f = %f s" % (rmse_without_same, rmse_without_same/frequency))
        #print("Mean squared error with same values: ", mse_with_same)
        print("Root mean squared error with same values:%f = %f s" % (rmse_with_same, rmse_with_same/frequency))
    
    return rmse_without_same, rmse_with_same, len(same_values), len(analog_value_in_second)


def get_value_from_string(string):
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
    try:
        return rpeak, letter
    except:
        return rpeak, "N"


def get_rpeaks_from_rri_file(
    file_path: str    
    ):
    """
    Get R-peaks from an .rri file. Only available for GIF data: 
    They were also detected automatically but later corrected manually, so they should be
    very accurate and can be used as a reference.

    RETURNS:
    --------------------------------
    rpeaks: 1D numpy array
        R-peak locations
    """
    # read the RRI file
    rri_file = open(file_path, "r")
    rri = rri_file.readlines() #.split("\n")
    rri_file.close()
    
    # start of rpeaks are separated by a line of "-" from the other informations in the 
    # .rri files.
    for i in range(len(rri)):
        count_dash = 0
        for j in range(len(rri[i])):
            if rri[i][j] == "-":
                count_dash += 1
        if count_dash/len(rri[i]) > 0.9:
            start = i + 1
            break

    # get the R-peaks from the RRI file
    rpeaks = dict()
    for i in range(start, len(rri)):
        this_rpeak, letter = get_value_from_string(rri[i])
        if isinstance(this_rpeak, int) and letter.isalpha():
            if letter in rpeaks:
                rpeaks[letter].append(this_rpeak)
            else:
                rpeaks[letter] = [this_rpeak]
    for key in rpeaks:
        rpeaks[key] = np.array(rpeaks[key])

    return rpeaks