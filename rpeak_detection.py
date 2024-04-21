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


def eval_thresholds_for_wfdb(
        data: dict, 
        frequency: dict,
        detection_intervals: list,
        threshold_multiplier: float,
        threshold_dezimal_places: int,
        ecg_key: str,
    ):
    """
    Calculate the time threshold for optimize_wfdb_detection function.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    detection_intervals: list
        list of detection intervals
    threshold_multiplier: float between 0 and 1
        multiplier that is either Multiplier or Divisor for the threshold values
        (because valid data could also differ slightly from the test intervals used)
    threshold_dezimal_places: int
        number of decimal places for the threshold values
    relevant_key: str
        key of the ECG data in the data dictionary
    
    RETURNS:
    --------------------------------
    time_threshold: float
        threshold for the detection time of R-peaks in the given intervals
    """

    #calculating time for detection (per datapoint) of R-peaks in the given intervals
    detection_times = []
    for interval in detection_intervals:
        start_time = time.time()
        get_rpeaks_wfdb(data, frequency, ecg_key, interval)
        detection_times.append((time.time() - start_time) / (interval[1] - interval[0]))
    
    time_threshold = round(np.max(detection_times) / threshold_multiplier, threshold_dezimal_places)

    return time_threshold


def optimize_wfdb_detection(
        data: dict, 
        frequency: dict, 
        wfdb_time_threshold: float,
        wfdb_time_interval_seconds: int,
        wfdb_check_time_condition: int,
        ecg_key: str, 
        detection_interval: tuple
    ):
    """
    Detect R-peaks in ECG data using the wfdb library.
    
    Detection time with wfdb is usually good (around 500.000 datapoints can be checked per
    second, this is slower compared to other methods but still acceptable).
    But if the data gets faulty, the detection time can increase drastically (20.000 
    datapoints per second). 
    This function is used to optimize the detection time. It will stop to detect R-peaks
    if the detection time exceeds a certain threshold.

    (I thought about changing to a different detection method in this case, but I later
    want to combine the results of different detection methods to get safer results.) 

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    wfdb_time_threshold: float
        threshold for the detection time of R-peaks in the given intervals
        (if the time (in seconds) per datapoint exceeds this threshold, the detection will
        skip the rest of the interval)
    wfdb_time_interval_seconds: int
        The rpeaks for the data will be calculated in intervals of this length
    wfdb_check_time_condition: int
        time can't be measured parallelly to the detection, so the detection interval will
        be splitted further. This parameter defines the number of equally distributed 
        checks during the interval
    ecg_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        data interval in which the R-peaks should be detected

    RETURNS:
    --------------------------------
    rpeaks_corrected: 1D numpy array
        R-peak locations
    """
    if detection_interval is None:
        detection_interval = (0, len(data[ecg_key]))

    sampling_rate = frequency[ecg_key]
    time_interval_iterations = int(wfdb_time_interval_seconds * sampling_rate)
    max_time_per_interval = wfdb_time_threshold * time_interval_iterations
    further_splitting_interval_size = int(time_interval_iterations / wfdb_check_time_condition)

    rpeaks = np.array([], dtype=int)
    count_skipped = 0

    for i in range(detection_interval[0], detection_interval[1], time_interval_iterations):
        # make sure that the upper bound is not exceeding the length of the data
        if i + time_interval_iterations > detection_interval[1]:
            upper_bound = detection_interval[1]
        else:
            upper_bound = i + time_interval_iterations

        # try to get the rpeaks for the current interval, stop if the time exceeds the threshold
        these_rpeaks = np.array([], dtype=int)
        too_slow = False
        # because you can't execute a function and parallelly measure time, the detection interval will be splitted further
        start_time = time.time()
        for j in range(i, upper_bound, further_splitting_interval_size):
            # make sure that the upper bound is not exceeding the length of the data
            if j + further_splitting_interval_size > upper_bound:
                upper_bound_inner = upper_bound
            else:
                upper_bound_inner = j + further_splitting_interval_size

            these_rpeaks = np.append(these_rpeaks, get_rpeaks_wfdb(data, frequency, ecg_key, (j, upper_bound_inner)))

            # check if the time exceeds the threshold, but continue if the detection is almost finished
            if time.time() - start_time > max_time_per_interval and j / (upper_bound-i) < 0.9:
                too_slow = True
                break
        
        if not too_slow:
            rpeaks = np.append(rpeaks, these_rpeaks)
        else:
            count_skipped += upper_bound - i
        
    print(f"Skipped {100 * count_skipped / (detection_interval[1] - detection_interval[0])} % of datapoints because the detection took too long.")
    
    return rpeaks
        

        



def combined_rpeak_detection_methods(
        data: dict, 
        frequency: dict, 
        relevant_key = "ECG", 
        detection_interval = None
    ):
    """
    Detect R-peaks in ECG data using both the wfdb library and the detection function
    that was used by the research group before me (see old_code/rpeak_detection.py).

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    relevant_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        interval in which the R-peaks should be detected

    RETURNS:
    --------------------------------
    rpeaks_corrected: 1D numpy array
        R-peak locations
    """
    #rpeaks_neuro = get_rpeaks_neuro(data, frequency, relevant_key, detection_interval)
    rpeaks_wfdb = get_rpeaks_wfdb(data, frequency, relevant_key, detection_interval)

    if detection_interval is None:
        rpeaks_old = old_rpeak.get_rpeaks(data[relevant_key], frequency[relevant_key])
    else:
        rpeaks_old = old_rpeak.get_rpeaks(data[relevant_key][detection_interval[0]:detection_interval[1]], frequency[relevant_key])

    # get the intersection of the two rpeak detections
    rpeaks_corrected = np.intersect1d(rpeaks_wfdb, rpeaks_old)
    return rpeaks_corrected