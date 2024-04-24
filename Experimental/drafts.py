"""
This file contains functions that were discarded during the project.
"""

lower_border = 2091000 # normal
lower_border = 6292992 # normal with fluktuations
lower_border = 2156544 # normal but noisier
lower_border = 1781760 # normal but negative peaks
lower_border = 661504 # hard noise
lower_border = 19059968 # extreme overkill
lower_border = 18344704 # not as extreme overkill
lower_border = 17752064 # hard noise
#lower_border = 10756096 # small but weird spikes
#lower_border = 10788096 # continous flat, one large spike 
#lower_border = 10792704 # continous flat
#lower_border = 15378176 # lots of noise
#lower_border = 15381248 # lots of noise, one spike


def create_calibration_data(
        file_path, 
        show_graphs = False
    ):
    """
    Create calibration data. Please note that the intervals are chosen manually and might 
    need to be adjusted, if you can't use the test data. In this case, you should at least
    plot the data in the given intervals to see what the data should look like 
    (see ARGUMENTS).

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file
    show_graphs: bool, default False
        if True, the data will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
        if False, the data will be saved to pickle files

    RETURNS:
    --------------------------------
    None
    """
    if len(os.listdir(CALIBRATION_DATA_DIRECTORY)) > 0 and not show_graphs:
        first_try = True
        while True:
            if first_try:
                user_answer = input("Calibration data already exists. Are you sure you want to overwrite it? (y/n)")
            else:
                user_answer = input("Please answer with 'y' or 'n'.")
            if user_answer != "y":
                clear_directory(CALIBRATION_DATA_DIRECTORY)
            elif user_answer == "n":
                print("Calibration data was not overwritten. Continuing with the existing data.")
                return
            else:
                first_try = False
                print("Answer not recognized.")

    sigbufs, sigfreqs, duration = read_edf.get_edf_data(file_path)
   
    # Calibration data for check_data
    interval_size = 2560 # 10 seconds for 256 Hz
    
    lower_border = 2091000 # 2h 17min 10sec for 256 Hz  
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "perfect_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], PERFECT_ECG_TEN_SEC)

    lower_border = 6292992 # 6h 49min 41sec for 256 Hz
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "fluctuating_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], FLUCTUATING_ECG_TEN_SEC)

    lower_border = 2156544 # 2h 20min 24sec for 256 Hz
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "noisy_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], NOISY_ECG_TEN_SEC)

    lower_border = 1781760 # 1h 56min 0sec for 256 Hz
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "negative_peaks_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], NEGATIVE_PEAKS_ECG_TEN_SEC)

"""
Following function didnt optimize the detection time as expected. It was way slower in good areas (50% of wfdb)
and deliverd worse results. We are now relying on the detection of valid ecg regions to optimize the detection time.
"""

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


def calculate_thresholds(
        file_path: str, 
        ecg_threshold_multiplier: float,
        wfdb_threshold_multiplier: float,
        check_ecg_threshold_dezimal_places: int,
        wfdb_time_threshold_dezimal_places: int,
        show_calibration_data: bool,
        ecg_key: str,
    ):
    """
    This function calculates the thresholds needed in various functions.
    Please note that the intervals are chosen manually and might need to be adjusted, if 
    you can't use the test data. In this case, you can use this function to plot the data 
    in the given intervals to see what the test data should look like (see ARGUMENTS).

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file for threshold calibration
    ecg_threshold_multiplier: float
        multiplier for the thresholds in check_data.check_ecg()
    wfdb_threshold_multiplier: float
        multiplier for the thresholds in rpeak_detection.optimize_wfdb_detection()
    check_ecg_threshold_dezimal_places: int
        number of dezimal places for the check ecg thresholds in the pickle files
    wfdb_time_threshold_dezimal_places: int
        number of dezimal places for the wfdb time threshold in the pickle files
    show_graphs: bool, default False
        if True, the data will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
        if False, the thresholds will be calculated and saved to THRESHOLD_DIRECTORY

    RETURNS:
    --------------------------------
    None, but the thresholds are saved to a pickle file
    """
    # Load the data
    sigbufs, sigfreqs, duration = read_edf.get_edf_data(file_path)

    # check if ecg thresholds already exist and if yes: ask for permission to override
    if show_calibration_data:
        user_answer = "n"
    else:
        user_answer = ask_for_permission_to_override(file_path = CHECK_ECG_DATA_THRESHOLDS_PATH, 
                                        message = "Thresholds for check_data.check_ecg()")
   
    # Calibration intervals for check_data.check_ecg()
    interval_size = 2560 # 10 seconds for 256 Hz
    lower_borders = [
        2091000, # 2h 17min 10sec for 256 Hz
        6292992, # 6h 49min 41sec for 256 Hz
        2156544, # 2h 20min 24sec for 256 Hz
        1781760 # 1h 56min 0sec for 256 Hz
        ]
    detection_intervals = [(border, border + interval_size) for border in lower_borders]

    # Plot the data if show_graphs is True
    if show_calibration_data:
        names = ["perfect_ecg", "fluctuating_ecg", "noisy_ecg", "negative_peaks"]
        for interval in detection_intervals:
            NNPH.simple_plot(sigbufs[ecg_key][interval[0]:interval[1]], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + names[detection_intervals.index(interval)] + "_ten_sec.png")
    
    # Calculate and save the thresholds for check_data.check_ecg()
    if user_answer == "y":

        threshold_values = check_data.eval_thresholds_for_check_ecg(
            sigbufs, 
            detection_intervals,
            threshold_multiplier = ecg_threshold_multiplier,
            threshold_dezimal_places = check_ecg_threshold_dezimal_places,
            ecg_key = ecg_key,
            )
        
        check_ecg_thresholds = dict()
        check_ecg_thresholds["check_ecg_std_min_threshold"] = threshold_values[0]
        check_ecg_thresholds["check_ecg_std_max_threshold"] = threshold_values[1]
        check_ecg_thresholds["check_ecg_distance_std_ratio_threshold"] = threshold_values[2]
        
        save_to_pickle(check_ecg_thresholds, CHECK_ECG_DATA_THRESHOLDS_PATH)

        del threshold_values
    del detection_intervals
    
    #end for check_ecg threshold
    #start of wfdb threshold
    
    # check if wfdb time threshold already exist and if yes: ask for permission to override
    if not show_calibration_data:
        user_answer = ask_for_permission_to_override(file_path = WFDB_TIME_THRESHOLD_PATH, 
                        message = "Thresholds for rpeak_detection.optimize_wfdb_detection()")
    
    # Calibration intervals for rpeak_detection.optimize_wfdb_detection()
    interval_size = 2560 # 10 seconds for 256 Hz
    lower_borders = [
        2091000, # 2h 17min 10sec for 256 Hz
        6292992, # 6h 49min 41sec for 256 Hz
        2156544, # 2h 20min 24sec for 256 Hz
        1781760 # 1h 56min 0sec for 256 Hz
        ]
    detection_intervals = [(border, border + interval_size) for border in lower_borders]

    # Plot the data if show_graphs is True
    if show_calibration_data:
        names = ["perfect_ecg_wfdb", "fluctuating_ecg_wfdb", "noisy_ecg_wfdb", "negative_peaks_ecg_wfdb"]
        for interval in detection_intervals:
            NNPH.simple_plot(sigbufs[ecg_key][interval[0]:interval[1]], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + names[detection_intervals.index(interval)] + "_ten_sec.png")

    # Calculate and save the thresholds for check_data.check_ecg()
    if user_answer == "y":

        threshold_value = rpeak_detection.eval_thresholds_for_wfdb(
            sigbufs, 
            sigfreqs,
            detection_intervals,
            threshold_multiplier = wfdb_threshold_multiplier,
            threshold_dezimal_places = wfdb_time_threshold_dezimal_places,
            ecg_key = ecg_key,
            )
        
        wfdb_time_threshold = dict()
        wfdb_time_threshold["wfdb_time_threshold"] = threshold_value
        
        save_to_pickle(wfdb_time_threshold, WFDB_TIME_THRESHOLD_PATH)

        del threshold_value
    del detection_intervals


WFDB_TIME_THRESHOLD_PATH = PREPARATION_DIRECTORY + "WFDB_Time_Threshold.pkl"

parameters = {
    "file_path": CALIBRATION_DATA_PATH, # path to the EDF file for threshold calibration
    "data_directory": DATA_DIRECTORY, # directory where the data is stored
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_key": "ECG", # key for the ECG data in the data dictionary
    "wrist_acceleration_keys": ["X", "Y", "Z"], # keys for the wrist acceleration data in the data dictionary
    "ecg_threshold_multiplier": 0.5, # multiplier for the thresholds in check_data.check_ecg() (between 0 and 1)
    "wfdb_threshold_multiplier": 0.5, # multiplier for the threshold in rpeak_detection.optimize_wfdb_detection() (between 0 and 1)
    "show_calibration_data": False, # if True, the calibration data in the manually chosen intervals will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
    "calculate_thresholds": True, # if True, you will have the option to recalculate the thresholds for various functions
    "check_ecg_threshold_dezimal_places": 2, # number of dezimal places for the check ecg thresholds in the pickle files
    "wfdb_time_threshold_dezimal_places": 7, # number of dezimal places for the wfdb time threshold in the pickle files
    "time_interval_seconds": 10, # time interval considered when calculating thresholds
    "min_valid_length_minutes": 5, # minimum length of valid data in minutes
    "allowed_invalid_region_length_seconds": 30, # data region (see above) still considered valid if the invalid part is shorter than this
    "determine_valid_ecg_regions": True, # if True, the valid regions for the ECG data will be determined
    "valid_ecg_regions": dict() # dictionary containing the valid regions for the ECG data, will be overwritten below (so don't bother)
}

if not isinstance(parameters["wfdb_threshold_multiplier"], (int, float)):
    raise ValueError("'wfdb_threshold_multiplier' parameter must be an integer or a float.")
if parameters["wfdb_threshold_multiplier"] <= 0 or parameters["wfdb_threshold_multiplier"] > 1:
    raise ValueError("'wfdb_threshold_multiplier' parameter must be between 0 and 1.")

if not isinstance(parameters["wfdb_time_threshold_dezimal_places"], int):
    raise ValueError("'wfdb_time_threshold_dezimal_places' parameter must be an integer.")


def combined_rpeak_detection_methods(
        data: dict, 
        frequency: dict, 
        ecg_key: str, 
        detection_interval: tuple,
        rpeak_primary_function,
        rpeak_secondary_function,
        rpeak_name_primary: str,
        rpeak_name_secondary: str,
        rpeak_distance_threshold_seconds: float,
        rpeak_failsafe_threshold_min_area_length: int,
        rpeak_failsafe_threshold_max_interruptions: int,
        rpeak_failsafe_threshold_multiplier: float,
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
    primary_function: function, default get_rpeaks_wfdb
        primary R peak detection function
    secondary_function: function, default get_rpeaks_old
        secondary R peak detection function
    name_primary: str, default "wfdb"
        name of the primary R peak detection function
    name_secondary: str, default "ecgdetectors"
        name of the secondary R peak detection function
    distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same 
        (reasonable to use highest heart rate ever recorded: 480 bpm = 0.125 spb,
        but better results with 300 bpm = 0.2 spb)

    ATTENTION:
        if the heart rate is higher than what was assumed by the distance_threshold_seconds,
        the distance_threshold_seconds will be locally adjusted where needed. The new threshold
        will be calculated from the mean of the distances in each area. For this 
        instance the following parameters are needed: 

    failsafe_threshold_min_area_length: int
        minimum length of an area (in R-peaks) where the heart rate is higher than assumed
    failsafe_threshold_max_interruptions: int
        maximum number of interruptions (heartrate lower) in an area where the heart rate is higher than assumed
    failsafe_threshold_multiplier: float
        multiplier for the threshold in areas where the heart rate is higher than assumed


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

    # check if there are areas where the heart rate is higher than assumed by the distance_threshold_seconds
    primary_distance = np.diff(rpeaks_primary)
    secondary_distance = np.diff(rpeaks_secondary)

    threshold_fails_primary = np.where(primary_distance < distance_threshold_iterations)[0]
    threshold_fails_secondary = np.where(secondary_distance < distance_threshold_iterations)[0]

    try:
        higher_heart_rate_area_primary = []
        this_interval = [threshold_fails_primary[0], threshold_fails_primary[0]+1]
        for i in range(1,len(threshold_fails_primary)):
            if (this_interval[1]+rpeak_failsafe_threshold_max_interruptions) >= threshold_fails_primary[i]:
                this_interval[1] = threshold_fails_primary[i]
            else:
                if this_interval[1] - this_interval[0] >= rpeak_failsafe_threshold_min_area_length:
                    higher_heart_rate_area_primary.append(this_interval)
                this_interval = [threshold_fails_primary[i], threshold_fails_primary[i]+1]
        if this_interval[1] - this_interval[0] >= rpeak_failsafe_threshold_min_area_length:
            higher_heart_rate_area_primary.append(this_interval)
    except:
        higher_heart_rate_area_primary = []
    
    try:
        higher_heart_rate_area_secondary = []
        this_interval = [threshold_fails_secondary[0], threshold_fails_secondary[0]+1]
        for i in range(1,len(threshold_fails_secondary)):
            if (this_interval[1]+rpeak_failsafe_threshold_max_interruptions) >= threshold_fails_secondary[i]:
                this_interval[1] = threshold_fails_secondary[i]
            else:
                if this_interval[1] - this_interval[0] >= rpeak_failsafe_threshold_min_area_length:
                    higher_heart_rate_area_secondary.append(this_interval)
                this_interval = [threshold_fails_secondary[i], threshold_fails_secondary[i]+1]
        if this_interval[1] - this_interval[0] >= rpeak_failsafe_threshold_min_area_length:
            higher_heart_rate_area_secondary.append(this_interval)
    except:
        higher_heart_rate_area_secondary = []

    # if two R-peaks are closer than the threshold, they are considered as the same
    # both will be changed to the same value (primary R-peak)

    # intersects_before = len(np.intersect1d(rpeaks_primary, rpeaks_secondary))

    if len(higher_heart_rate_area_primary) > 0 and len(higher_heart_rate_area_secondary) > 0:
        altered_distance_threshold_iterations = distance_threshold_iterations
        last_matching_rpeak = -1
        for i in range(len(rpeaks_primary)):
            for interval in higher_heart_rate_area_primary:
                if interval[0] <= i <= interval[1]:
                    altered_distance_threshold_iterations = int(np.mean(primary_distance[interval[0]:interval[1]]) * rpeak_failsafe_threshold_multiplier)
                    break
            if rpeaks_primary[i] not in rpeaks_secondary:
                possible_matches = []
                possible_matches_values = []
                for j in range(last_matching_rpeak + 1, len(rpeaks_secondary)):
                    for interval in higher_heart_rate_area_secondary:
                        if interval[0] <= j <= interval[1]:
                            altered_distance_threshold_iterations = min(int(np.mean(secondary_distance[interval[0]:interval[1]]) * rpeak_failsafe_threshold_multiplier), altered_distance_threshold_iterations)
                            break
                for j in range(last_matching_rpeak + 1, len(rpeaks_secondary)):
                    this_distance = rpeaks_secondary[j] - rpeaks_primary[i]
                    possible_matches_values.append(abs(this_distance))
                    possible_matches.append(j)
                    if this_distance > altered_distance_threshold_iterations:
                        break
                if min(possible_matches_values) < altered_distance_threshold_iterations:
                    last_matching_rpeak = possible_matches[possible_matches_values.index(min(possible_matches_values))]
                    rpeaks_secondary[last_matching_rpeak] = rpeaks_primary[i]
    else:
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