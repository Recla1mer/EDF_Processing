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


def seperate_plots_from_bib(file_name, save_path):
    """
    Read a pickle file containing a dictionary of plots and save them as individual figures.
    """
    with open(file_name, "rb") as f:
        plots = pickle.load(f)

    for key, value in plots.items():
        plt.figure()
        plt.imshow(value)
        plt.axis("off")
        plt.savefig(save_path + key + ".png", bbox_inches="tight", pad_inches=0)
        plt.close()


def seperate_plots(data, time, signal, save_path, **kwargs):
    """
    Save the plots in the dictionary as individual figures.
    """
    kwargs.setdefault("xlim", [2700,2720])
    
    fig, ax = plt.subplots()

    ax.plot(time, data, label=signal)
    ax.legend(loc="best")
    plt.xlim(kwargs["xlim"])
    plt.savefig(save_path + "_" + signal + ".png")


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


def check_ecg_blocks(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        ecg_key: str
    ):
    """
    This function won't be used in the project. It was a first draft, but the final function:
    check_ecg() is more useful.

    Check where the ECG data is valid.
    (Checks blocks of x minutes for validity)

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the distance to standard deviation ratio
    time_interval_seconds: int
        time interval length to be checked for validity in seconds
    min_valid_length_minutes: int
        minimum length of valid data in minutes
    allowed_invalid_region_length_seconds: int
        allowed length of invalid data in seconds
    ecg_key: str
        key of the ECG data in the data dictionary

    RETURNS:
    --------------------------------
    valid_regions: list
        list of tuples containing the start and end indices of the valid regions
    """

    # calculate the number of iterations from time and frequency
    time_interval_iterations = int(time_interval_seconds * frequency[ecg_key])

    # check condition for given time intervals and add regions (multiple time intervals) to a list if number of invalid intervals is sufficiently low
    valid_regions = []
    current_valid_intervals = 0 # counts valid intervals
    total_intervals = 0 # counts intervals, set to 0 when region is completed (valid or invalid)
    lower_border = 0 # lower border of the region
    skip_interval = 0 # skip intervals if region is valid but total intervals not max (= intervals_per_region)
    min_valid_intervals = int((min_valid_length_minutes * 60 - allowed_invalid_region_length_seconds) / time_interval_seconds) # minimum number of valid intervals in a region
    intervals_per_region = int(min_valid_length_minutes * 60 / time_interval_seconds) # number of intervals in a region
    valid_total_ratio = min_valid_intervals / intervals_per_region # ratio of valid intervals in a region, for the last region that might be too short
    # print("Variables: ", time_interval_iterations, min_valid_intervals, intervals_per_region)

    for i in np.arange(0, len(data[ecg_key]), time_interval_iterations):
        # if region met condition, but there are still intervals left, skip them
        if skip_interval > 0:
            skip_interval -= 1
            continue
        # print("NEW ITERATION: ", i)

        # make sure upper border is not out of bounds
        if i + time_interval_iterations > len(data[ecg_key]):
            upper_border = len(data[ecg_key])
        else:
            upper_border = i + time_interval_iterations
        
        # check if interval is valid
        this_std = np.std(data[ecg_key][i:upper_border])
        this_max = np.max(data[ecg_key][i:upper_border])
        this_min = np.min(data[ecg_key][i:upper_border])
        max_min_distance = this_max - this_min
        std_distance_ratio = 0.5 * max_min_distance / this_std

        if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
            current_valid_intervals += 1
            # print("VALID")
        
        # increase total intervals
        total_intervals += 1
        
        # check if the region is valid
        if current_valid_intervals >= min_valid_intervals:
            # print("VALID REGION: ", lower_border, lower_border + intervals_per_region*time_interval_iterations)
            valid_regions.append((lower_border,lower_border + intervals_per_region*time_interval_iterations))
            lower_border += intervals_per_region*time_interval_iterations
            skip_interval = intervals_per_region - total_intervals
            current_valid_intervals = 0
            total_intervals = 0
            continue
        
        #check if region is invalid
        if total_intervals >= intervals_per_region:
            lower_border += intervals_per_region*time_interval_iterations
            total_intervals = 0
            current_valid_intervals = 0
        
        # check if the last region is valid
        if upper_border == len(data[ecg_key]):
            try:
                if current_valid_intervals / total_intervals >= valid_total_ratio:
                    # print("VALID REGION: ", lower_border, upper_border)
                    valid_regions.append((lower_border,upper_border))
            except:
                continue
    
    print("Valid regions ratio: ", round(len(valid_regions) / (len(data[ecg_key]) / int(min_valid_length_minutes * 60 * frequency[ecg_key])), 5)*100, "%")
    
    return valid_regions


def check_ecg_more_mistakes(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        ecg_key: str
    ):
    """
    This function won't be used in the project. It was the second draft, but the final function:
    check_ecg() is more useful.

    (The function checks in non overlapping intervals. It appends a valid region if its
    long enough with few enough errors. Afterwards it expands the upper border if the following 
    interval is also valid. Problem 1: All errors could be right at the start of a region, without
    excluding them. Problem 2: It won't append an almost long enough region to a valid region, if
    they are separated by a short invalid region, however short it will be.)

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequency of the signals
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the max-min distance to twice the standard deviation ratio
    time_interval_seconds: int
        time interval length in seconds to be checked for validity
    min_valid_length_minutes: int
        minimum length of valid data in minutes
    allowed_invalid_region_length_seconds: int
        length of data in seconds that is allowed to be invalid in a valid region of size min_valid_length_minutes
    ecg_key: str
        key of the ECG data in the data dictionary

    RETURNS:
    --------------------------------
    valid_regions: list
        list of lists containing the start and end indices of the valid regions: valid_regions[i] = [start, end] of region i
    """

    # calculate the number of iterations from time and frequency
    time_interval_iterations = int(time_interval_seconds * frequency[ecg_key])

    # check condition for given time intervals and add regions (multiple time intervals) to a list if number of invalid intervals is sufficiently low
    valid_regions = []
    current_valid_intervals = 0 # counts valid intervals
    total_intervals = 0 # counts intervals, set to 0 when region is completed (valid or invalid)
    lower_border = 0 # lower border of the region
    min_length_reached = False # check if the minimum length is reached
    min_valid_intervals = int((min_valid_length_minutes * 60 - allowed_invalid_region_length_seconds) / time_interval_seconds) # minimum number of valid intervals in a region
    intervals_per_region = int(min_valid_length_minutes * 60 / time_interval_seconds) # number of intervals in a region
    valid_total_ratio = min_valid_intervals / intervals_per_region # ratio of valid intervals in a region, for the last region that might be too short
    # print("Variables: ", time_interval_iterations, min_valid_intervals, intervals_per_region)

    for i in np.arange(0, len(data[ecg_key]), time_interval_iterations):

        # make sure upper border is not out of bounds
        if i + time_interval_iterations > len(data[ecg_key]):
            upper_border = len(data[ecg_key])
        else:
            upper_border = i + time_interval_iterations
        
        # calc std and max-min-distance ratio
        this_std = np.std(data[ecg_key][i:upper_border])
        this_max = np.max(data[ecg_key][i:upper_border])
        this_min = np.min(data[ecg_key][i:upper_border])
        max_min_distance = this_max - this_min
        std_distance_ratio = 0.5 * max_min_distance / this_std

        if min_length_reached:
            # check if interval is valid
            if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
                valid_regions[-1][1] = upper_border
            else:
                min_length_reached = False
                lower_border = upper_border
        else:
            # check if interval is valid
            if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
                current_valid_intervals += 1
            
            # increase total intervals
            total_intervals += 1
            
            # check if the region is valid
            if current_valid_intervals >= min_valid_intervals:
                valid_regions.append([lower_border,lower_border + intervals_per_region*time_interval_iterations])
                lower_border += intervals_per_region*time_interval_iterations
                current_valid_intervals = 0
                total_intervals = 0
                min_length_reached = True
                continue
            
            #check if region is invalid
            if total_intervals >= intervals_per_region:
                lower_border += intervals_per_region*time_interval_iterations
                total_intervals = 0
                current_valid_intervals = 0
            
            # check if the last region is valid
            if upper_border == len(data[ecg_key]):
                try:
                    if current_valid_intervals / total_intervals >= valid_total_ratio:
                        valid_regions.append([lower_border,upper_border])
                except:
                    continue
    
    return valid_regions


def get_edf_data(file_name):
    """
    Reads data from an EDF file.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the EDF file
    
    RETURNS:
    --------------------------------
    sigbufs: dict
        dictionary containing the signals
    sigfreqs: dict
        dictionary containing the frequencies of the signals
    sigdims: dict
        dictionary containing the physical dimensions of the signals
    duration: float
        duration of the EDF file in seconds

    The keys of the dictionaries are the signal labels.

    ATTENTION: 
    --------------------------------
    In the actual EDF file, the signals are shown in blocks over time. This was 
    previously not considered in the pyedflib library. Now it seems to be fixed.
    """

    f = pyedflib.EdfReader(file_name)

    duration = f.file_duration

    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = dict()
    sigfreqs = dict()
    sigdims = dict()

    for i in np.arange(n):
        this_signal = f.readSignal(i)
        sigbufs[signal_labels[i]] = this_signal
        sigfreqs[signal_labels[i]] = f.getSampleFrequency(i)
        sigdims[signal_labels[i]] = f.getPhysicalDimension(i)
    f._close()
    
    return sigbufs, sigfreqs, sigdims, duration


def get_all_dim_signal_labels():
    """
    """
    try_directory = "Data/"
    # print(get_dimensions_and_signal_labels(try_directory))
    txt_file = open("Additions/Dimensions_and_Labels/dim_and_labels.txt", "w")
    all_directories = retrieve_all_subdirectories_with_valid_files(try_directory, [".edf"])
    print(all_directories)

    for directory in all_directories:
        labels, dimensions = get_dimensions_and_signal_labels(directory)
        txt_file.write(directory + "\n")
        txt_file.write("-"*len(directory) + "\n")
        for i in np.arange(len(labels)):
            txt_file.write(labels[i] + ": ")
            for dim in dimensions[i]:
                txt_file.write("\"" + dim + "\", ")
            txt_file.write("\n")
        txt_file.write("\n")

    txt_file.close()


def eval_std_thresholds(
        ECG: list, 
        detection_intervals: list,
        threshold_multiplier: float,
        threshold_dezimal_places: int,
    ):
    """
    Evaluate useful thresholds (check_ecg_std_min_threshold, check_ecg_distance_std_ratio_threshold) 
    for the check_ecg function from given data.

    We will estimate the standard deviation for different valuable signals. From this we
    can create an interval for the standard deviation that is considered good.
    Of course when the signal is bad, and all we have is noise, the standard deviation can
    be similar to that of a good signal. 
    However, the distance between the maximum and minimum value will in this case
    be about the same as twice the standard deviation. Therefore, we will also calculate
    the distance between the maximum and minimum value and divide it by twice the standard
    deviation, to see what this ratio looks like for valuable signals. 
    Anything less will be considered as invalid.

    ARGUMENTS:
    --------------------------------
    ECG: list
        list containing the ECG data
    detection_intervals: list
        list of detection intervals, in which the data is considered valid
    threshold_multiplier: float between 0 and 1
        multiplier that is either Multiplier or Divisor for the threshold values
        (because valid data could also differ slightly from the detection intervals used)
    threshold_dezimal_places: int
        number of decimal places for the threshold values
    
    RETURNS:
    --------------------------------
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the max-min distance to (2 * standard deviation) ratio
    """

    # create lists to save the standard deviation and the max-min distance values
    std_values = []
    max_min_distance_values = []
    
    # calculate the standard deviation and std max-min distance for the detection intervals
    for interval in detection_intervals:
        std_values.append(np.std(ECG[interval[0]:interval[1]]))
        max_min_distance_values.append(np.max(ECG[interval[0]:interval[1]]) - np.min(ECG[interval[0]:interval[1]]))
    
    # calculate the ratios
    std_to_max_min_distance_ratios = 0.5 * np.array(max_min_distance_values) / np.array(std_values)
    
    # calculate the thresholds (take values that will include most datapoints)
    max_std = np.max(std_values)
    min_std = np.min(std_values)
    min_std_distance_ratio = np.min(std_to_max_min_distance_ratios)
    
    # apply the threshold multiplier and round the values
    check_ecg_std_min_threshold = round(min_std*threshold_multiplier,threshold_dezimal_places)
    # check_ecg_std_max_threshold = round(max_std/threshold_multiplier,threshold_dezimal_places)
    check_ecg_distance_std_ratio_threshold = round(min_std_distance_ratio*threshold_multiplier,threshold_dezimal_places)

    return check_ecg_std_min_threshold, check_ecg_distance_std_ratio_threshold


def create_ecg_thresholds(
        ecg_calibration_file_path: str, 
        ecg_keys: list,
        physical_dimension_correction_dictionary: dict,
        ecg_calibration_intervals: list,
        ecg_thresholds_multiplier: float,
        ecg_thresholds_dezimal_places: int,
        ecg_thresholds_save_path: str
    ):
    """
    This function provides the data and calculates the thresholds for ecg data validation.

    Please note that the intervals will be chosen manually and might need to be adjusted,
    if you can't use the calibration data. In this case, you can use the option:
    'show_calibration_data' in the main.py file to show what the calibration data should 
    look like.

    ARGUMENTS:
    --------------------------------
    ecg_calibration_file_path: str
        path to the EDF file for ecg threshold calibration
    ecg_keys: list
        list of possible labels for the ECG data
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    ecg_calibration_intervals: list
        list of tuples containing the start and end indices of the calibration intervals
    ecg_thresholds_multiplier: float
        multiplier for the thresholds (see 'eval_std_thresholds()')
    ecg_thresholds_dezimal_places: int
        number of dezimal places for the ecg thresholds (see 'eval_std_thresholds()')
    ecg_thresholds_save_path: str
        path to the pickle file where the thresholds are saved

    RETURNS:
    --------------------------------
    None, but the thresholds are saved as dictionary to a pickle file with the following
    format:
        {
            "check_ecg_std_min_threshold": check_ecg_std_min_threshold,
            "check_ecg_distance_std_ratio_threshold": check_ecg_distance_std_ratio_threshold
        }
    """

    # check if ecg thresholds already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_file(file_path = ecg_thresholds_save_path, 
        message = "\nThresholds for ECG validation (see check_data.check_ecg()) already exist in " + ecg_thresholds_save_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return
    
    # try to load the data and correct the physical dimension if needed
    try:
        ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
            file_path = ecg_calibration_file_path,
            possible_channel_labels = ecg_keys,
            physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
        )
    except:
        raise SystemExit("No ECG data found in the file " + ecg_calibration_file_path + ". As the ECG thresholds are relevant for further computation, the program was stopped.")

    # Calculate and save the thresholds for check_ecg() function
    threshold_values = eval_std_thresholds(
        ECG = ecg_signal, 
        detection_intervals = ecg_calibration_intervals,
        threshold_multiplier = ecg_thresholds_multiplier,
        threshold_dezimal_places = ecg_thresholds_dezimal_places,
        )
    
    # write the thresholds to a dictionary and save them
    check_ecg_thresholds = dict()
    check_ecg_thresholds["check_ecg_std_min_threshold"] = threshold_values[0]
    check_ecg_thresholds["check_ecg_distance_std_ratio_threshold"] = threshold_values[1]
    
    append_to_pickle(check_ecg_thresholds, ecg_thresholds_save_path)


def old_check_ecg(
        ECG: list, 
        frequency: int,
        check_ecg_std_min_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        overlapping_interval_steps: int,
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
    ):
    """
    Check where the ECG data is valid.
    (valid regions must be x minutes long, invalid regions can be as short as possible)

    Data will be checked in overlapping intervals. Those will be concatenated afterwards.
    Then the gaps between the regions will be checked. If its useful to connect them (regions
    long enough compared to gap, but not too long that they already fulfill the
    min_valid_length_minutes condition), they will be connected.

    ARGUMENTS:
    --------------------------------
    ECG: list
        list containing the ECG data
    frequency: int
        sampling frequency of the ECG data
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the max-min distance to twice the standard deviation ratio
    time_interval_seconds: int
        time interval length in seconds to be checked for validity
    min_valid_length_minutes: int
        minimum length of valid data in minutes
    allowed_invalid_region_length_seconds: int
        length of data in seconds that is allowed to be invalid in a valid region of size min_valid_length_minutes

    RETURNS:
    --------------------------------
    valid_regions: list
        list of lists containing the start and end indices of the valid regions: valid_regions[i] = [start, end] of region i
    """
    # calculate the number of iterations from time and frequency
    time_interval_iterations = int(time_interval_seconds * frequency)
    
    check_ecg_std_min_threshold, check_ecg_distance_std_ratio_threshold = locally_calculate_ecg_thresholds(ECG, time_interval_iterations) # type: ignore
    # print(check_ecg_std_min_threshold, check_ecg_distance_std_ratio_threshold)

    # check condition for given time intervals and add regions (multiple time intervals) to a list if number of invalid intervals is sufficiently low
    overlapping_valid_regions = []
    
    interval_steps = int(time_interval_iterations/overlapping_interval_steps)
    was_valid = False

    for i in np.arange(0, len(ECG), interval_steps):

        # make sure upper border is not out of bounds
        if i + time_interval_iterations > len(ECG):
            upper_border = len(ECG)
        else:
            upper_border = i + time_interval_iterations
        
        # calc std and max-min-distance ratio
        this_std = abs(np.std(ECG[i:upper_border]))
        this_max = np.max(ECG[i:upper_border])
        this_min = np.min(ECG[i:upper_border])
        max_min_distance = abs(this_max - this_min)

        if this_std == 0:
            std_distance_ratio = check_ecg_distance_std_ratio_threshold - 1
        else:
            std_distance_ratio = 0.5 * max_min_distance / this_std

        # check if interval is valid
        if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
            overlapping_valid_regions.append([i,upper_border])
            was_valid = True
        else:
            if was_valid:
                limit = upper_border - time_interval_iterations/overlapping_interval_steps
                for j in range(len(overlapping_valid_regions)-1, -1, -1):
                    if overlapping_valid_regions[j][1] >= limit:
                        del overlapping_valid_regions[j]
                    else:
                        break
            was_valid = False
    
    if len(overlapping_valid_regions) == 0:
        return []

    # concatenate neighbouring intervals
    concatenated_intervals = []
    this_interval = [overlapping_valid_regions[0][0], overlapping_valid_regions[0][1]]
    for i in range(1, len(overlapping_valid_regions)):
        if overlapping_valid_regions[i][0] <= this_interval[1]:
            this_interval[1] = overlapping_valid_regions[i][1]
        else:
            concatenated_intervals.append(this_interval)
            del this_interval
            this_interval = [overlapping_valid_regions[i][0], overlapping_valid_regions[i][1]]
    concatenated_intervals.append(this_interval)

    del overlapping_valid_regions

    # calculate thresholds from other units in iterations
    iterations_per_region = int(min_valid_length_minutes * 60 * frequency)
    allowed_invalid_iterations = int(allowed_invalid_region_length_seconds * frequency)
    allowed_wrong_total_ratio = allowed_invalid_region_length_seconds / min_valid_length_minutes / 60

    # collect possible connections (gap between regions is smaller than allowed_invalid_iterations)
    possible_connections = []
    
    for i in range(1, len(concatenated_intervals)):
        if concatenated_intervals[i][0] - concatenated_intervals[i-1][1] < allowed_invalid_iterations:
            possible_connections.append([i-1, i])
    
    # check which connections are useful to make
    current_invalid_iterations = 0
    total_iterations = 0

    validated_connections = []
    last_connection = -1

    for i in range(len(possible_connections)):
        # calculate the length of the regions and the distance between them
        len_region_1 = concatenated_intervals[possible_connections[i][0]][1] - concatenated_intervals[possible_connections[i][0]][0]
        len_region_2 = concatenated_intervals[possible_connections[i][1]][1] - concatenated_intervals[possible_connections[i][1]][0]
        distance = concatenated_intervals[possible_connections[i][1]][0] - concatenated_intervals[possible_connections[i][0]][1]

        # check if the regions are not too short and not too long
        if len_region_1 > iterations_per_region and len_region_2 > iterations_per_region:
            continue
        
        if len_region_1 < 2*allowed_invalid_iterations or len_region_2 < 2*allowed_invalid_iterations:
            continue
        
        # check if last connection was established and check if you can append to it
        if last_connection == possible_connections[i][0]:
            current_invalid_iterations += distance
            total_iterations += len_region_2 + distance
            if current_invalid_iterations / total_iterations <= allowed_wrong_total_ratio:
                validated_connections[-1].append(possible_connections[i][1])
                # we don't want to add to much errors, should the interval be long enough, therefore we just compare the last two connections (reset values)
                current_invalid_iterations = distance
                total_iterations = len_region_1 + len_region_2 + distance
                last_connection = possible_connections[i][1]
                continue
        
        # check if you can establish a new connection
        current_invalid_iterations = distance
        total_iterations = len_region_1 + len_region_2 + distance
        if current_invalid_iterations / total_iterations <= allowed_wrong_total_ratio:
            validated_connections.append(possible_connections[i])
            last_connection = possible_connections[i][1]
        else:
            last_connection = -1

    # create the intervals for the valid connections
    connected_regions = []
    for con in validated_connections:
        connected_regions.append([concatenated_intervals[con[0]][0], concatenated_intervals[con[-1]][1]])

    # replace the connections in the concatenated intervals
    replaced_connections = []
    for i in range(len(concatenated_intervals)-1, -1, -1):
        for con_index in range(len(validated_connections)):
            if i in validated_connections[con_index]:
                if con_index in replaced_connections:
                    del concatenated_intervals[i]
                    break
                else:
                    concatenated_intervals[i] = connected_regions[con_index]
                    replaced_connections.append(con_index)
                    break
    
    # append long enough regions to the valid regions
    valid_regions = []
    for region in concatenated_intervals:
        if region[1] - region[0] >= iterations_per_region:
            valid_regions.append(region)
    
    return valid_regions