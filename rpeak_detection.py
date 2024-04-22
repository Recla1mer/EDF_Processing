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