"""
Python implementation to detect R-peaks in ECG data.
Useful Link: https://www.samproell.io/posts/signal/ecg-library-comparison/

Main function: get_rpeaks
"""
import numpy as np

# import libraries for rpeak detection
import neurokit2
import wfdb.processing

# import old code used for rpeak detection
import old_code.rpeak_detection as old_rpeak

CALIBRATION_DATA_DIRECTORY = "Calibration_Data/"


def get_rpeaks_neuro(
        data: dict, 
        frequency: dict, 
        relevant_key = "ECG", 
        detection_interval = None
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
    relevant_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        interval in which the R-peaks should be detected

    RETURNS:
    --------------------------------
    rpeaks_corrected: 1D numpy array
        R-peak locations
    """
    sampling_rate = frequency[relevant_key]
    if detection_interval is None:
        ecg_signal = data[relevant_key]
    else:
        ecg_signal = data[relevant_key][detection_interval[0]:detection_interval[1]]

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
        relevant_key = "ECG", 
        detection_interval = None
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
    relevant_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        interval in which the R-peaks should be detected

    RETURNS:
    --------------------------------
    rpeaks_corrected: 1D numpy array
        R-peak locations
    """
    sampling_rate = frequency[relevant_key]
    if detection_interval is None:
        ecg_signal = data[relevant_key]
    else:
        ecg_signal = data[relevant_key][detection_interval[0]:detection_interval[1]]

    rpeaks = wfdb.processing.xqrs_detect(ecg_signal, fs=sampling_rate, verbose=False)
    rpeaks_corrected = wfdb.processing.correct_peaks(
        ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up"
    )

    if detection_interval is not None:
        rpeaks_corrected += detection_interval[0]
    return rpeaks_corrected


def calculate_time_threshold_for_wfdb(calibration_files = ["ecg_1.pkl", "ecg_2.pkl", "ecg_3.pkl"]):
    """
    Calculate the time threshold for optimize_wfdb_detection function.
    """
    return None


def optimize_wfdb_detection(
        data: dict, 
        frequency: dict, 
        time_threshold: float,
        relevant_key = "ECG", 
        detection_interval = None,
    ):
    """
    Detect R-peaks in ECG data using the wfdb library.
    
    Detection time with wfdb is usually good (around 500.000 datapoints can be checked per
    second, this is slower compared to other methods but still  acceptable).
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
    relevant_key: str
        key of the ECG data in the data dictionary
    detection_interval: tuple, default None
        interval in which the R-peaks should be detected
    time_threshold_multiplier: float, default 1.0
        This function 
        

    RETURNS:
    --------------------------------
    rpeaks_corrected: 1D numpy array
        R-peak locations
    """
    return None


def get_rpeaks_old(
        data: dict, 
        frequency: dict, 
        relevant_key = "ECG", 
        detection_interval = None
    ):
    """
    Detect R-peaks in ECG data using the old code that was used by the research group before me.

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
    rpeaks_old: 1D numpy array
        R-peak locations
    """
    sampling_rate = frequency[relevant_key]
    if detection_interval is None:
        ecg_signal = data[relevant_key]
    else:
        ecg_signal = data[relevant_key][detection_interval[0]:detection_interval[1]]

    rpeaks_old = old_rpeak.get_rpeaks(ecg_signal, sampling_rate)
    
    if detection_interval is not None:
        rpeaks_old += detection_interval[0]
    
    return rpeaks_old


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