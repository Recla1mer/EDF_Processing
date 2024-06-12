"""
Author: Johannes Peter Knoll

Python implementation of ECG data validation
"""

# IMPORTS
import numpy as np

# LOCAL IMPORTS
import read_edf
from side_functions import *


def straighten_ecg(
        ecg_interval: list,
        frequency: int,
    ):
    """
    Sometimes the ECG signal is overlapped by noise, which increases the standard deviation
    and max-min distance. Both are needed to determine the validity of the ECG signal.
    To not falsify the results, this function tries to straighten the ecg signal.

    To do this, it will first look for the peaks. Then it will vertically center the peaks.
    Afterwards the region inbetween the peaks will be shifted as well, depending on the 
    increase/decrease between the peaks.

    ARGUMENTS:
    --------------------------------
    ecg_interval: list
        list containing the ECG data
    frequency: int
        sampling rate / frequency of the ECG data
    
    RETURNS:
    --------------------------------
    straighten_ecg: list
        list containing the straightened ECG data or the original data
    """
    # goal is to reduce std of the ecg signal. If this is not achieved, the original signal is returned
    # therefore we need to save the original signal and std
    original_ecg_interval = copy.deepcopy(ecg_interval)

    std_before_straightening = np.std(ecg_interval)
    step_iterations = int(0.2*frequency)

    # calculate the max-min differences and minima of the ecg signal in steps (1/5 of a second)
    differences = []
    minima = []
    for i in range(0, len(ecg_interval), step_iterations):
        upper_border = i + step_iterations
        if upper_border > len(ecg_interval):
            upper_border = len(ecg_interval)
        
        this_interval = ecg_interval[i:upper_border]
        this_max = np.max(this_interval)
        this_min = np.min(this_interval)
        this_diff = this_max - this_min

        differences.append(this_diff)
        minima.append(this_min)

    mean_diff = np.mean(differences)

    # straighten the ecg signal as described above
    last_high_difference = 0
    last_offset = 0
    straighten_ecg = [ecg_value for ecg_value in ecg_interval]
    for i in range(0, len(minima)):
        if differences[i] > 1.5*mean_diff:
            offset = minima[i] + 0.5*differences[i]
            upper_border = (i+1)*step_iterations
            if upper_border > len(ecg_interval):
                upper_border = len(ecg_interval)
            for j in range(i*step_iterations, upper_border):
                straighten_ecg[j] -= offset
            if last_high_difference != i and last_high_difference != 0:
                rise = -(offset - last_offset) / ((i-last_high_difference)*step_iterations)
                start_j = last_high_difference*step_iterations
                for j in range(start_j, i*step_iterations):
                    straighten_ecg[j] += rise * (j-start_j) - last_offset
            last_high_difference = i + 1
            last_offset = offset
    
    first_high_difference = 0
    first_offset = 0
    for i in range(0, len(minima)):
        if differences[i] > 1.5*mean_diff:
            first_offset = minima[i] + 0.5*differences[i]
            first_high_difference = i
            break
    
    # also apply shifting to areas before the first and after the last peak
    for j in range(0, first_high_difference*step_iterations):
        straighten_ecg[j] -= first_offset
    
    for j in range(last_high_difference*step_iterations, len(ecg_interval)):
        straighten_ecg[j] -= last_offset
    
    # check if the std is reduced, if not return the original signal
    std_after_straightening = np.std(straighten_ecg)

    if std_after_straightening > 1.2*std_before_straightening:
        origin_min = np.min(original_ecg_interval)
        origin_difference = abs(np.max(original_ecg_interval) - origin_min)
        origin_offset = origin_min + 0.5*origin_difference
        original_ecg_interval -= origin_offset

        return original_ecg_interval
    
    return straighten_ecg


def concatenate_neighbouring_intervals(
        intervals: list,
    ):
    """
    Concatenate overlapping intervals.

    ARGUMENTS:
    --------------------------------
    intervals: list
        list of lists containing the start and end indices of the intervals
    
    RETURNS:
    --------------------------------
    concatenated_intervals: list
        list of lists containing the start and end indices of the concatenated intervals
    """
    if len(intervals) == 0:
        return []
    # concatenate neighbouring intervals
    concatenated_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= concatenated_intervals[-1][1]:
            concatenated_intervals[-1][1] = intervals[i][1]
        else:
            concatenated_intervals.append(intervals[i])

    return concatenated_intervals


def retrieve_unincluded_intervals(
        included_intervals: list,
        total_length: int,
    ):
    """
    Retrieve the unincluded intervals in the given signal length.

    ARGUMENTS:
    --------------------------------
    included_intervals: list
        list of lists containing the start and end indices of the included intervals
    total_length: int
        length of the signal
    
    RETURNS:
    --------------------------------
    unincluded_intervals: list
        list of lists containing the start and end indices of the unincluded intervals
    """
    if len(included_intervals) == 0:
        return [[0, total_length]]
    unincluded_intervals = []
    if included_intervals[0][0] > 0:
        unincluded_intervals.append([0, included_intervals[0][0]])
    for i in range(1, len(included_intervals)):
        unincluded_intervals.append([included_intervals[i-1][1], included_intervals[i][0]])
    if included_intervals[-1][1] < total_length:
        unincluded_intervals.append([included_intervals[-1][1], total_length])
    
    return unincluded_intervals


def valid_total_ratio(ECG: list, valid_regions: list):
    """
    Calculate the ratio of valid to total ecg data.

    ARGUMENTS:
    --------------------------------
    ECG: list
        list containing the ECG data
    valid_regions: list
        list of lists containing the start and end indices of the valid regions

    RETURNS:
    --------------------------------
    valid_ratio: float
        ratio of valid to total ecg data
    """
    valid_data = 0
    for region in valid_regions:
        valid_data += region[1] - region[0]
    valid_ratio = valid_data / len(ECG)
    return valid_ratio


def expand_valid_regions(
        valid_regions: list,
        min_valid_length_iterations: int,
        allowed_invalid_region_length_iterations: int,
    ):
    """
    Expand valid regions to include invalid regions that are too short to be considered as invalid.

    ARGUMENTS:
    --------------------------------
    valid_regions: list
        list of lists containing the start and end indices of the valid regions
    min_valid_length_iterations: int
        minimum length of a valid region in iterations
    allowed_invalid_region_length_iterations: int
        maximum length of invalid datapoints a region can contain to be still considered
        as valid
    
    RETURNS:
    --------------------------------
    connected_intervals: list
        list of lists containing the start and end indices of the expanded intervals
    """
    # if there is only one valid region, you can't expand it
    if len(valid_regions) <= 1:
        return valid_regions
    
    min_num_of_valid_points_in_region = min_valid_length_iterations - allowed_invalid_region_length_iterations

    # create lists to store lower and upper border of the min_valid_length_iterations long regions (long_regions)
    # and the number of valid points in these regions
    number_of_valid_points = []
    long_regions = []
    # sum up number of valid points for every min_valid_length_iterations
    for lower_border in range(valid_regions[0][0], valid_regions[-1][1], min_valid_length_iterations):
        upper_border = lower_border + min_valid_length_iterations
        smaller_region_multiplier = 1
        # make sure upper border is not out of bounds, if it is, adjust the multiplier (smaller region, needs less valid points)
        if upper_border > valid_regions[-1][1]:
            upper_border = valid_regions[-1][1]
            smaller_region_multiplier = min_valid_length_iterations / (upper_border - lower_border)
        
        long_regions.append([lower_border, upper_border])
        
        valid_points_in_this_interval = 0
        
        for i in range(0, len(valid_regions)):
            if valid_regions[i][0] > upper_border:
                break
            if valid_regions[i][1] < lower_border:
                continue
            if valid_regions[i][0] < lower_border:
                if valid_regions[i][1] < upper_border:
                    valid_points_in_this_interval += valid_regions[i][1] - lower_border
                else:
                    valid_points_in_this_interval += upper_border - lower_border
            else:
                if valid_regions[i][1] < upper_border:
                    valid_points_in_this_interval += valid_regions[i][1] - valid_regions[i][0]
                else:
                    valid_points_in_this_interval += upper_border - valid_regions[i][0]
        
        number_of_valid_points.append(valid_points_in_this_interval*smaller_region_multiplier)
    
    # collect invalid region indices
    invalid_long_region_index = []
    for i in range(0, len(number_of_valid_points)):
        if number_of_valid_points[i] < min_num_of_valid_points_in_region:
            invalid_long_region_index.append(i)
    
    # remove invalid regions
    for i in range(len(invalid_long_region_index)-1, -1, -1):
        del long_regions[invalid_long_region_index[i]]

    # concatenate long regions
    long_regions = concatenate_neighbouring_intervals(long_regions)
    # connect invalid gaps that lie inside the long valid regions
    connected_intervals = []
    last_two_intervals_connected = False
    last_connected_region_index = -1
    for i in range(1, len(valid_regions)):
        invalid_left_border = valid_regions[i-1][1]
        invalid_right_border = valid_regions[i][0]
        #print(invalid_left_border, invalid_right_border, invalid_right_border - invalid_left_border)
        for j in range(0, len(long_regions)):
            if long_regions[j][0] > invalid_right_border:
                if not last_two_intervals_connected:
                    connected_intervals.append(valid_regions[i-1])
                    last_connected_region_index = i-1
                last_two_intervals_connected = False
                break
            if long_regions[j][1] <= invalid_right_border:
                continue
            else:
                if long_regions[j][0] < invalid_left_border:
                    if last_two_intervals_connected:
                        connected_intervals[-1][1] = valid_regions[i][1]
                    else:
                        connected_intervals.append([valid_regions[i-1][0], valid_regions[i][1]])
                    last_connected_region_index = i
                    last_two_intervals_connected = True
                    break
                else:
                    if not last_two_intervals_connected:
                        connected_intervals.append(valid_regions[i-1])
                        last_connected_region_index = i-1
                    last_two_intervals_connected = False
                    break

    #print(valid_regions)
    for i in range(last_connected_region_index+1, len(valid_regions)):
        connected_intervals.append(valid_regions[i])
    #print(connected_intervals)
    
    return connected_intervals


def check_ecg(
        ECG: list, 
        frequency: int,
        straighten_ecg_signal: bool,
        check_ecg_time_interval_seconds: int, 
        check_ecg_overlapping_interval_steps: int,
        check_ecg_validation_strictness: list,
        check_ecg_removed_peak_difference_threshold: float,
        check_ecg_std_min_threshold: float, 
        check_ecg_std_max_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        check_ecg_min_valid_length_minutes: int,
        check_ecg_allowed_invalid_region_length_seconds: int,
    ):
    """
    This functions checks where the ECG signal is valid and returns the valid region borders.
    It does this by calculating the standard deviation and the max-min distance of the ECG signal
    in intervals of check_ecg_time_interval_seconds. It then removes the highest peak in this interval
    and recalculates the standard deviation and the max-min distance. 
    
    It collects all of these values and calculates the mean values. From those the thresholds are
    retrieved using the check_ecg_validation_strictness. 
    
    If the ratio of the max-min distance to the standard deviation is lower than the threshold,
    the interval is considered invalid. 
    (in a good ECG signal the peak is much higher than the standard deviation)
    
    If the distance of this ratio after and before removing the highest peak is too high, 
    the interval is considered invalid.
    (if at least two peaks are inside the interval, the values should be similar before 
    and after removing the highest peak)

    If the standard deviation is too high or too low, the interval is considered invalid.
    (if the signal is too noisy or too flat, the values are not reliable)

    ARGUMENTS:
    --------------------------------
    ECG: list
        list containing the ECG data
    frequency: int
        sampling rate / frequency of the ECG data
    straighten_ecg_signal: bool
        if True, the ECG signal will be straightened (see straighten_ecg())
    check_ecg_time_interval_seconds: int
        length of the interval to be checked for validity in seconds
    check_ecg_overlapping_interval_steps: int
        number of steps the interval needs to be shifted to the right until the next check_ecg_time_interval_seconds interval starts
    check_ecg_validation_strictness: float
        strictness of the validation (0: very unstrict, 1: very strict)
    check_ecg_removed_peak_difference_threshold: float
        threshold for the difference of the max-min distance to the standard deviation after and before removing the highest peak
    check_ecg_std_min_threshold: float
        minimum standard deviation to be considered valid 
        (MANUAL THRESHOLD, only used if the ratio of valid to total data is too low. Because then the mean values are off)
    check_ecg_std_max_threshold: float
        maximum standard deviation to be considered valid
        (MANUAL THRESHOLD, see above)
    check_ecg_distance_std_ratio_threshold: float
        minimum ratio of the max-min distance to the standard deviation to be considered valid
        (MANUAL THRESHOLD, see above)
    check_ecg_min_valid_length_minutes: int
        minimum length of a valid region in minutes
    check_ecg_allowed_invalid_region_length_seconds: int
        maximum length of invalid datapoints a region can contain to be still considered as valid
    
    RETURNS:
    --------------------------------
    connected_intervals: list
        list of lists containing the start and end indices of the valid regions
    """
    # calculate the number of iterations from time and frequency
    time_interval_iterations = int(check_ecg_time_interval_seconds * frequency)

    # calculate the step size 
    interval_steps = int(time_interval_iterations/check_ecg_overlapping_interval_steps)

    # create lists to save the standard deviation and the max-min distance values of the intervals
    collect_whole_stds = []
    collect_whole_std_max_min_distance_ratios = []

    # create lists to save the standard deviation and the max-min distance values of the intervals after removing the highest peak
    collect_no_peak_std_distance = []
    collect_no_peak_std_max_min_distance_ratio_distance = []

    for i in np.arange(0, len(ECG), interval_steps):
        # make sure upper border is not out of bounds
        upper_border = i + time_interval_iterations
        if upper_border > len(ECG):
            upper_border = len(ECG)

        # straighten the ecg signal if wanted
        if straighten_ecg_signal:
            ecg_interval = straighten_ecg(ECG[i:upper_border], frequency)
        else:
            ecg_interval = ECG[i:upper_border]

        # calculate the standard deviation and append it to the list
        this_interval_std = np.std(ecg_interval)
        collect_whole_stds.append(this_interval_std)

        # remove area around the peak
        if np.mean(ecg_interval) < 0:
            peak_location = np.argmax(ecg_interval)
        else:
            peak_location = np.argmin(ecg_interval)
        interval_without_peak = []
        for j in range(0, len(ecg_interval)):
            if j < peak_location - 0.1*frequency or j > peak_location + 0.1*frequency:
                interval_without_peak.append(ecg_interval[j])
        
        # recalculate the standard deviation
        no_peak_interval_std = np.std(interval_without_peak)

        # append values to the lists
        if this_interval_std == 0:
            whole_std_max_min_ratio = 0
            collect_whole_std_max_min_distance_ratios.append(0)
            collect_no_peak_std_distance.append(2)
        else:
            whole_std_max_min_ratio = (np.max(ecg_interval) - np.min(ecg_interval))/this_interval_std
            collect_whole_std_max_min_distance_ratios.append(whole_std_max_min_ratio)
            collect_no_peak_std_distance.append(abs(no_peak_interval_std-this_interval_std)/this_interval_std)

        if no_peak_interval_std == 0:
            collect_no_peak_std_max_min_distance_ratio_distance.append(2)
        elif whole_std_max_min_ratio == 0:
            collect_no_peak_std_max_min_distance_ratio_distance.append(2)
        else:
            collect_no_peak_std_max_min_distance_ratio_distance.append(abs((np.max(interval_without_peak) - np.min(interval_without_peak))/no_peak_interval_std-whole_std_max_min_ratio)/whole_std_max_min_ratio)
    
    # now starts the evaluation part of the function
    # - we will first calculate the mean values of whole max-min-std ratio and sort out the invalid intervals (step 1)
    # - from the remaining intervals we will calculate the mean values of the standard deviation and sort out the invalid intervals (step 2)
    # - from the remaining intervals we will sort out invalid intervals from the max-min difference after removing the highest peak (step 3)

    # create lists to store the results for the different strictness values
    store_connected_intervals = []

    mean_whole_std_max_min_distance_ratio = np.mean(collect_whole_std_max_min_distance_ratios)

    # evaluate valid intervals for different strictness values
    for validation_strictness in check_ecg_validation_strictness:
        # step 1:
        # create lists to store intervals that passed the max-min-std ratio comparison
        possibly_valid_stds = []
        passed_max_min_distance = []

        # checking if the std-max-min-distance-ratio is high enough
        for i in range(0, len(collect_whole_std_max_min_distance_ratios)):
            if collect_whole_std_max_min_distance_ratios[i] > validation_strictness*mean_whole_std_max_min_distance_ratio:
                possibly_valid_stds.append(collect_whole_stds[i])
                passed_max_min_distance.append(i)
        
        # skip the next steps if no valid intervals are found
        recalculate_with_manual_thresholds = False
        if len(possibly_valid_stds) == 0:
            recalculate_with_manual_thresholds = True
        
        # step 2:
        # checking if std is neither too high nor too low
        if not recalculate_with_manual_thresholds:
            mean_std = np.mean(possibly_valid_stds)

            lower_limit = (mean_std - np.std(possibly_valid_stds))*validation_strictness
            # prevent if signal is too noisy, that invalid regions are not falsely included:
            if lower_limit < 0.5*check_ecg_std_min_threshold:
                lower_limit = check_ecg_std_min_threshold

            upper_limit = (mean_std + np.std(possibly_valid_stds))*(2-validation_strictness)
            # prevent if signal is too flat (std in average too low), that valid regions are not falsely excluded:
            if upper_limit < 0.5*check_ecg_std_max_threshold:
                upper_limit = check_ecg_std_max_threshold

            possibly_valid_max_min_distance = []
            passed_min_std = []

            for i in range(0, len(possibly_valid_stds)):
                if possibly_valid_stds[i] > lower_limit and possibly_valid_stds[i] < upper_limit:
                    passed_min_std.append(passed_max_min_distance[i])
                    possibly_valid_max_min_distance.append(collect_no_peak_std_max_min_distance_ratio_distance[passed_max_min_distance[i]])
            
            if len(possibly_valid_max_min_distance) == 0:
                recalculate_with_manual_thresholds = True
        
        # step 3:
        # checking if the std-max-min-distance-ratio after peak removal is close enough to the original
        if not recalculate_with_manual_thresholds:
            valid_intervals = []

            for i in range(0, len(possibly_valid_max_min_distance)):
                if not possibly_valid_max_min_distance[i] > check_ecg_removed_peak_difference_threshold:
                    # make sure upper border is not out of bounds
                    lower_border = passed_min_std[i]*interval_steps
                    upper_border = lower_border + time_interval_iterations
                    if upper_border > len(ECG):
                        upper_border = len(ECG)
                    # append to valid intervals
                    valid_intervals.append([lower_border, upper_border])
            
            # check the ratio of valid to total ecg data
            valid_ratio = valid_total_ratio(ECG, valid_intervals)
            
            # if the ratio is too low, the mean values are off and the manual thresholds need to be used
            if valid_ratio < 0.5:
                recalculate_with_manual_thresholds = True

        if recalculate_with_manual_thresholds:
            valid_intervals = []
            for i in range(0, len(collect_whole_stds)):
                if collect_whole_stds[i] >= check_ecg_std_min_threshold:
                    if collect_whole_stds[i] <= check_ecg_std_max_threshold:
                        if collect_whole_std_max_min_distance_ratios[i] >= check_ecg_distance_std_ratio_threshold:
                            if collect_no_peak_std_max_min_distance_ratio_distance[i] <= check_ecg_removed_peak_difference_threshold:
                                # make sure upper border is not out of bounds
                                lower_border = i*interval_steps
                                upper_border = lower_border + time_interval_iterations
                                if upper_border > len(ECG):
                                    upper_border = len(ECG)
                                # append to valid intervals
                                valid_intervals.append([lower_border, upper_border])

        # if recalculate_with_manual_thresholds and ecg_comparison_mode:
        #     store_connected_intervals.append([])
        #     continue
        
        # concatenate neighbouring intervals
        concatenated_intervals = concatenate_neighbouring_intervals(valid_intervals)
        #print(valid_total_ratio(ECG, concatenated_intervals))
        
        # include invalid intervals that are too short to be considered as invalid
        connected_intervals = expand_valid_regions(
            valid_regions = concatenated_intervals, 
            min_valid_length_iterations = int(check_ecg_min_valid_length_minutes*60*frequency), 
            allowed_invalid_region_length_iterations = int(check_ecg_allowed_invalid_region_length_seconds*frequency)
            )
        
        store_connected_intervals.append(connected_intervals)

    return store_connected_intervals


def determine_valid_ecg_regions(
        data_directory: str,
        valid_file_types: list,
        ecg_keys: list,
        physical_dimension_correction_dictionary: dict,
        preparation_results_path: str,
        file_name_dictionary_key: str,
        valid_ecg_regions_dictionary_key: str,
        # check_ecg arguments:
        straighten_ecg_signal: bool,
        check_ecg_time_interval_seconds: int, 
        check_ecg_overlapping_interval_steps: int,
        check_ecg_validation_strictness: list,
        check_ecg_removed_peak_difference_threshold: float,
        check_ecg_std_min_threshold: float, 
        check_ecg_std_max_threshold: float,
        check_ecg_distance_std_ratio_threshold: float,
        check_ecg_min_valid_length_minutes: int,
        check_ecg_allowed_invalid_region_length_seconds: int,
    ):
    """
    Determine the valid ECG regions for all valid file types in the given data directory.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    ecg_keys: list
        list of possible labels for the ECG data
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    preparation_results_path: str
        path to the pickle file where the valid regions are saved
    file_name_dictionary_key
        dictionary key to access the file name
    valid_ecg_regions_dictionary_key: str
        dictionary key to access the valid ecg regions
    others: see check_ecg()

    RETURNS:
    --------------------------------
    None, but the valid regions are saved as dictionaries to a pickle file in the following
    format:
        {
            file_name_dictionary_key: file_name_1,
            valid_ecg_regions_dictionary_key: valid_regions_1
        }
            ...
    See check_ecg() for the format of the valid_regions.
    """
    # load existing results
    prep_results_generator = load_from_pickle(preparation_results_path)

    # additionally remove keys if in ecg_comparison_mode
    additionally_remove_keys = []
    for generator_entry in prep_results_generator:
        for dict_key in generator_entry.keys():
            if valid_ecg_regions_dictionary_key in dict_key:
                additionally_remove_keys.append(dict_key)
    
    if len(additionally_remove_keys) == 0:
        remove_key = valid_ecg_regions_dictionary_key
        additionally_remove_keys = []
    elif len(additionally_remove_keys) == 1:
        remove_key = additionally_remove_keys[0]
        additionally_remove_keys = []
    else:
        remove_key = additionally_remove_keys[0]
        additionally_remove_keys = additionally_remove_keys[1:]



    # check if valid regions already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = preparation_results_path,
        dictionary_entry = remove_key,
        additionally_remove_entries = additionally_remove_keys
        )
    
    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(preparation_results_path) + "computation_in_progress.pkl"
    if os.path.isfile(temporary_file_path):
        os.remove(temporary_file_path)

    # create list to store files that could not be processed
    unprocessable_files = []

    # get all valid files
    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    # create dictionary to store dictionaries that do not contain the needed key
    # (needed to avoid overwriting these entries in the pickle file if user answer is "n")
    store_previous_dictionary_entries = dict()
   
    # skip calculation if user does not want to override
    if user_answer == "n":
        # load existing results
        preparation_results_generator = load_from_pickle(preparation_results_path)

        for generator_entry in preparation_results_generator:
                # check if needed dictionary keys exist
                if file_name_dictionary_key not in generator_entry.keys():
                    continue

                if valid_ecg_regions_dictionary_key not in generator_entry.keys():
                    store_previous_dictionary_entries[generator_entry[file_name_dictionary_key]] = generator_entry
                    continue

                # get current file name
                file_name = generator_entry[file_name_dictionary_key]

                if file_name in valid_files:
                    valid_files.remove(file_name)
                
                append_to_pickle(generator_entry, temporary_file_path)
    
    # create variables to track progress
    start_time = time.time()
    total_files = len(valid_files)
    progressed_files = 0

    if total_files > 0:
        print("\nCalculating valid regions for the ECG data in %i files from \"%s\":" % (total_files, data_directory))

    if user_answer == "y":
        # load existing results
        preparation_results_generator = load_from_pickle(preparation_results_path)

        for generator_entry in preparation_results_generator:
            # show progress
            progress_bar(progressed_files, total_files, start_time)
            progressed_files += 1

            try:
                # get current file name
                file_name = generator_entry[file_name_dictionary_key]

                if file_name in valid_files:
                    valid_files.remove(file_name)

                # try to load the data and correct the physical dimension if needed
                ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
                    file_path = data_directory + file_name,
                    possible_channel_labels = ecg_keys,
                    physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                )

                # calculate the valid regions
                store_valid_intervals_for_strictness = check_ecg(
                    ECG = ecg_signal, 
                    frequency = ecg_sampling_frequency,
                    straighten_ecg_signal = straighten_ecg_signal,
                    check_ecg_time_interval_seconds = check_ecg_time_interval_seconds, 
                    check_ecg_overlapping_interval_steps = check_ecg_overlapping_interval_steps,
                    check_ecg_validation_strictness = check_ecg_validation_strictness,
                    check_ecg_removed_peak_difference_threshold = check_ecg_removed_peak_difference_threshold,
                    check_ecg_std_min_threshold = check_ecg_std_min_threshold, 
                    check_ecg_std_max_threshold = check_ecg_std_max_threshold, 
                    check_ecg_distance_std_ratio_threshold = check_ecg_distance_std_ratio_threshold,
                    check_ecg_min_valid_length_minutes = check_ecg_min_valid_length_minutes,
                    check_ecg_allowed_invalid_region_length_seconds = check_ecg_allowed_invalid_region_length_seconds,
                )
                
                # save the valid regions for this file
                for strictness_index in range(0, len(check_ecg_validation_strictness)):
                    generator_entry[valid_ecg_regions_dictionary_key + "_" + str(check_ecg_validation_strictness[strictness_index])] = store_valid_intervals_for_strictness[strictness_index]

            except:
                unprocessable_files.append(file_name)
            
            append_to_pickle(generator_entry, temporary_file_path)
    
    # calculate the valid regions for the remaining files
    for file_name in valid_files:
        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        if file_name in store_previous_dictionary_entries.keys():
            generator_entry = store_previous_dictionary_entries[file_name]
        else:
            generator_entry = {file_name_dictionary_key: file_name}

        try:
            # try to load the data and correct the physical dimension if needed
            ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
                file_path = data_directory + file_name,
                possible_channel_labels = ecg_keys,
                physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
            )

            # calculate the valid regions
            store_valid_intervals_for_strictness = check_ecg(
                ECG = ecg_signal, 
                frequency = ecg_sampling_frequency,
                straighten_ecg_signal = straighten_ecg_signal,
                check_ecg_time_interval_seconds = check_ecg_time_interval_seconds, 
                check_ecg_overlapping_interval_steps = check_ecg_overlapping_interval_steps,
                check_ecg_validation_strictness = check_ecg_validation_strictness,
                check_ecg_removed_peak_difference_threshold = check_ecg_removed_peak_difference_threshold,
                check_ecg_std_min_threshold = check_ecg_std_min_threshold, 
                check_ecg_std_max_threshold = check_ecg_std_max_threshold, 
                check_ecg_distance_std_ratio_threshold = check_ecg_distance_std_ratio_threshold,
                check_ecg_min_valid_length_minutes = check_ecg_min_valid_length_minutes,
                check_ecg_allowed_invalid_region_length_seconds = check_ecg_allowed_invalid_region_length_seconds,
            )
            
            # save the valid regions for this file to the dictionary
            for strictness_index in range(0, len(check_ecg_validation_strictness)):
                generator_entry[valid_ecg_regions_dictionary_key + "_" + str(check_ecg_validation_strictness[strictness_index])] = store_valid_intervals_for_strictness[strictness_index]

        except:
            unprocessable_files.append(file_name)
        
        # if more than the file name is in the dictionary, save the dictionary to the pickle file
        if len(generator_entry) > 1:
            append_to_pickle(generator_entry, temporary_file_path)
    
    progress_bar(progressed_files, total_files, start_time)

    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        try:
            os.remove(preparation_results_path)
        except:
            pass
        os.rename(temporary_file_path, preparation_results_path)

    # print the files that could not be processed
    if len(unprocessable_files) > 0:
        print("\nThe following files could not be processed for ECG Validation:")
        print(unprocessable_files)
        print("Possible reasons:")
        print(" "*5 + "- ECG file contains format errors")
        print(" "*5 + "- No matching label in ecg_keys and the files")
        print(" "*5 + "- Physical dimension of label is unknown")
        print(" "*5 + "- Dictionary key that accesses the file name does not exist in the results. Check key in file or recalculate them.")


def choose_valid_ecg_regions(
        data_directory: str,
        ecg_keys: list,
        physical_dimension_correction_dictionary: dict,
        preparation_results_path: str,
        file_name_dictionary_key: str,
        valid_ecg_regions_dictionary_key: str,
    ):
    """
    Prints mean valid to total ratios for the evaluation of ecg data for different 
    strictness values and asks the user to choose one, which will be used for further
    computation.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    ecg_keys: list
        list of possible labels for the ECG data
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    preparation_results_path: str
        path to the pickle file where the valid regions are saved
    file_name_dictionary_key
        dictionary key to access the file name
    valid_ecg_regions_dictionary_key: str
        dictionary key to access the valid ecg regions

    RETURNS:
    --------------------------------
    None, but the valid regions for one strictness value are assigned to the valid_ecg_regions_dictionary_key
    """

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(preparation_results_path) + "computation_in_progress.pkl"
    if os.path.isfile(temporary_file_path):
        os.remove(temporary_file_path)
    
    # load existing results
    preparation_results_generator = load_from_pickle(preparation_results_path)

    store_strictness_values = []
    store_valid_total_ratios = []

    for generator_entry in preparation_results_generator:
        # check if needed dictionary keys exist
        if file_name_dictionary_key not in generator_entry.keys():
            continue
        file_name = generator_entry[file_name_dictionary_key]

        for dict_key in generator_entry.keys():
            if valid_ecg_regions_dictionary_key in dict_key:
                # try to load the data and correct the physical dimension if needed
                ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
                    file_path = data_directory + file_name,
                    possible_channel_labels = ecg_keys,
                    physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                )

                valid_total_ratio = valid_total_ratio(
                    ECG = ecg_signal, 
                    valid_regions = generator_entry[dict_key]
                )

                strictness_value = dict_key.split("_")[-1]
                if strictness_value not in store_strictness_values:
                    store_strictness_values.append(strictness_value)
                    store_valid_total_ratios.append([valid_total_ratio])
                else:
                    store_valid_total_ratios[store_strictness_values.index(strictness_value)].append(valid_total_ratio)
    
    # calculate mean valid_total_ratios for strictness values
    mean_valid_total_ratios = np.mean(store_valid_total_ratios, axis=1)

    # print results to console
    print("\nMean valid to total ratios for different strictness values:")
    for i in range(0, len(store_strictness_values)):
        print("Strictness value: %s, Mean valid to total ratio: %.2f" % (store_strictness_values[i], mean_valid_total_ratios[i]))
    
    # ask user for the strictness value
    while True:
        strictness_value = input("\nChoose a strictness value to assign the valid regions to the dictionary key \"%s\": " % valid_ecg_regions_dictionary_key)
        if strictness_value in store_strictness_values:
            break
        else:
            print("Invalid input. Please choose a valid strictness value.")
    
    # assign the valid regions to the dictionary key
    for generator_entry in preparation_results_generator:
        strictness_key = valid_ecg_regions_dictionary_key + "_" + strictness_value
        if strictness_key in generator_entry.keys():
            generator_entry[valid_ecg_regions_dictionary_key] = generator_entry[strictness_key]
        append_to_pickle(generator_entry, temporary_file_path)
    
    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        try:
            os.remove(preparation_results_path)
        except:
            pass
        os.rename(temporary_file_path, preparation_results_path)


"""
Following code won't be used for the final implementation, but is useful for testing and
comparing the results of the ECG validation, as interval classification is available for
the GIF data. They might or might not have been calculated automatically and
later checked manually.

They are stored in .txt files. Therefore we need to implement functions to compare the
results of the validation and to read the intervals from the .txt files.
They are stored in the following format: "integer integer" after a file header containing
various information separated by a line of "-".

The first integer is the index in the ECG data and the second integer is the classification
of the data point (0: valid, 1: invalid).
"""


def ecg_validation_txt_string_evaluation(string: str):
    """
    Appearence of string entries in the .txt files: "integer integer"
    The first integer is the index in the ECG data and the second integer is the classification:
    (0: valid, 1: invalid)

    ARGUMENTS:
    --------------------------------
    string: str
        string to be evaluated
    
    RETURNS:
    --------------------------------
    datapoint: int
        index of the data point
    classification: str
        classification of the data point
    """

    # set default values if the integer or the letter do not exist
    datapoint = " "
    classification = " "

    was_number = False
    for i in range(len(string)):
        if string[i].isdigit():
            if datapoint != " ":
                classification = string[i]
                break
            if not was_number:
                start = i
            was_number = True
        else:
            if was_number:
                datapoint = int(string[start:i])
            was_number = False
    
    return datapoint, classification


def get_ecg_classification_from_txt_file(file_path: str):
    """
    Get the ECG classification from a .txt file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the .txt file containing the ECG classification
    
    RETURNS:
    --------------------------------
    ecg_classification: dict
        dictionary containing the ECG classification in the following format:
        {
            "0": [valid_datapoint_1, valid_datapoint_2, ...],
            "1": [invalid_datapoint_1, invalid_datapoint_2, ...]
        }
    """

    # read the txt file
    with open(file_path, 'r') as file:
        txt_lines = file.readlines()
    
    # start of ecg validation is separated by a line of "-" from the file header in the .txt files
    # retrieve the start of the R-peaks in the file
    for i in range(len(txt_lines)):
        count_dash = 0
        for j in range(len(txt_lines[i])):
            if txt_lines[i][j] == "-":
                count_dash += 1
        if count_dash/len(txt_lines[i]) > 0.9:
            start = i + 1
            break
    
    # create dictionary to save the classification
    ecg_classification = dict()

    # determine valid datapoints from the txt file
    for i in range(start, len(txt_lines)):
        datapoint, classification = ecg_validation_txt_string_evaluation(txt_lines[i])
        
        if isinstance(datapoint, int) and classification.isdigit():
            if classification in ecg_classification:
                ecg_classification[classification].append(datapoint)
            else:
                ecg_classification[classification] = [datapoint]
    
    return ecg_classification


def compare_ecg_validations(
        validated_intervals: list, 
        ecg_classification: dict,
    ):
    """
    Compare the validated intervals with the ECG classification.

    ARGUMENTS:
    --------------------------------
    validated_intervals: list
        list of tuples containing the start and end indices of the intervals considered valid
    ecg_classification: dict
        dictionary containing the ECG classification
    
    RETURNS (list):
    --------------------------------
    correct_valid_ratio: float
        ratio of correctly classified valid points
    correct_invalid_ratio: float
        ratio of correctly classified invalid points
    valid_wrong_ratio: float
        ratio of valid points classified as invalid
    invalid_wrong_ratio: float
        ratio of invalid points classified as valid
    """

    # get points considered valid and invalid by the ECG classification
    try:
        classification_invalid_points = ecg_classification["1"]
    except:
        classification_invalid_points = []
    try:
        classification_valid_points = ecg_classification["0"]
    except:
        classification_valid_points = []

    # create lists to save the intersecting points and the wrong classified points
    intersecting_invalid_points = []
    intersecting_valid_points = []

    invalid_points_wrong = []
    valid_points_wrong = []

    # check if point classified as valid is in the validated intervals
    # append to the corresponding list depending on the result
    for point in classification_valid_points:
        appended = False
        for interval in validated_intervals:
            if point >= interval[0] and point <= interval[1]:
                intersecting_valid_points.append(point)
                appended = True
                break
        if not appended:
            valid_points_wrong.append(point)
    
    # check if point classified as invalid is outside the validated intervals
    # append to the corresponding list depending on the result
    for point in classification_invalid_points:
        appended = False
        for interval in validated_intervals:
            if point >= interval[0] and point <= interval[1]:
                invalid_points_wrong.append(point)
                appended = True
                break
        if not appended:
            intersecting_invalid_points.append(point)

    # calculate the ratios and return them
    try:
        correct_valid_ratio = len(intersecting_valid_points) / len(classification_valid_points)
    except:
        correct_valid_ratio = 1.0
    try:
        correct_invalid_ratio = len(intersecting_invalid_points) / len(classification_invalid_points)
    except:
        correct_invalid_ratio = 1.0

    try:
        valid_wrong_ratio = len(valid_points_wrong) / len(classification_valid_points)
    except:
        valid_wrong_ratio = 1.0
    try:
        invalid_wrong_ratio = len(invalid_points_wrong) / len(classification_invalid_points)
    except:
        invalid_wrong_ratio = 1.0

    return [correct_valid_ratio, correct_invalid_ratio, valid_wrong_ratio, invalid_wrong_ratio]


def ecg_validation_comparison(
        ecg_classification_values_directory: str,
        ecg_classification_file_types: list,
        additions_results_path: str,
        file_name_dictionary_key: str,
        valid_ecg_regions_dictionary_key: str,
        ecg_validation_comparison_dictionary_key: str
    ):
    """
    Compare the ECG validation with the ECG classification values.

    ARGUMENTS:
    --------------------------------
    ecg_classification_values_directory: str
        directory where the ECG classification values are stored
    ecg_classification_file_types: list
        valid file types for the ECG classification values
    ecg_validation_comparison_evaluation_path: str
        path to the pickle file where the evaluation is saved
    additions_results_path: str,
        path to the pickle file where the ecg validation comparison should be saved
    file_name_dictionary_key
        dictionary key to access the file name
    valid_ecg_regions_dictionary_key: str
        dictionary key to access the valid ecg regions
    ecg_validation_comparison_dictionary_key: str
        dictionary key to access the ecg validation comparison
    
    RETURNS:
    --------------------------------
    None, but the evaluation is saved as dictionaries to a pickle file in the following
    format:
        {
            file_name_dictionary_key: file_name_1,
            ecg_validation_comparison_dictionary_key: [correct_valid_ratio, correct_invalid_ratio, valid_wrong_ratio, invalid_wrong_ratio],
            ...
        }
            ...
    """
    
    # check if the evaluation already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = additions_results_path,
        dictionary_entry = ecg_validation_comparison_dictionary_key
        )
    
    # cancel if needed data is missing
    if user_answer == "no_file_found":
        print("File containing valid ecg regions not found. As they are needed for the valid ecg region comparison, it is skipped.")
        return

    # get all determined ECG Validation files
    addition_results_generator = load_from_pickle(additions_results_path)

    # get all ECG classification files
    all_classification_files = os.listdir(ecg_classification_values_directory)
    valid_classification_files = [file for file in all_classification_files if get_file_type(file) in ecg_classification_file_types]

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(additions_results_path) + "computation_in_progress.pkl"
    if os.path.isfile(temporary_file_path):
        os.remove(temporary_file_path)

    # create variables to track progress
    start_time = time.time()
    total_data_files = get_pickle_length(additions_results_path, ecg_validation_comparison_dictionary_key)
    progressed_data_files = 0

    # create lists to store unprocessable files
    unprocessable_files = []
    
    if total_data_files > 0:
        print("\nCalculating ECG validation comparison values for %i files:" % total_data_files)
    
    # calculate the ECG Validation comparison values for all files
    for generator_entry in addition_results_generator:
        # skip if the comparison values already exist and the user does not want to override
        if user_answer == "n" and ecg_validation_comparison_dictionary_key in generator_entry.keys():
            append_to_pickle(generator_entry, temporary_file_path)
            continue

        # show progress
        progress_bar(progressed_data_files, total_data_files, start_time)
        progressed_data_files += 1

        try:
            # get the file key and the validated ECG regions
            this_file = generator_entry[file_name_dictionary_key]

            # get the file name without the file type
            this_file_name = os.path.splitext(this_file)[0]

            # get corresponding ECG classification file name for this file
            file_found = False
            for clfc_file in valid_classification_files:
                if this_file_name in clfc_file:
                    file_found = True
                    this_classification_file = clfc_file
                    break
            if not file_found:
                raise FileNotFoundError

            ecg_classification_dictionary = get_ecg_classification_from_txt_file(ecg_classification_values_directory + this_classification_file)
        
            # compare the differnt ECG validations
            strictness_values = []
            comparison_values_for_strictness = []

            for dict_key in generator_entry:
                if valid_ecg_regions_dictionary_key in dict_key and dict_key != valid_ecg_regions_dictionary_key:
                    strictness_values.append(dict_key.split("_")[-1])
                    comparison_values_for_strictness.append(compare_ecg_validations(
                        validated_intervals = generator_entry[dict_key],
                        ecg_classification = ecg_classification_dictionary
                        ))
        
            # add comparison values for this file
            generator_entry[ecg_validation_comparison_dictionary_key] = [strictness_values, comparison_values_for_strictness]

        except:
            unprocessable_files.append(this_file)
        
        append_to_pickle(generator_entry, temporary_file_path)
    
    progress_bar(progressed_data_files, total_data_files, start_time)

    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        os.remove(additions_results_path)
        os.rename(temporary_file_path, additions_results_path)

    # print the files that could not be processed
    if len(unprocessable_files) > 0:
        print("\nThe following files could not be processed for ECG Validation Comparison:")
        print(unprocessable_files)
        print("Possible reasons:")
        print(" "*5 + "- No corresponding classification file was found")
        print(" "*5 + "- Error during calculating the comparison values")
        print(" "*5 + "- Dictionary key that accesses the file name does not exist in the results. Check keys in file or recalculate them.")


def ecg_validation_comparison_report(
        ecg_validation_comparison_report_path: str,
        additions_results_path: str,
        file_name_dictionary_key: str,
        ecg_validation_comparison_dictionary_key: str,
        ecg_validation_comparison_report_dezimal_places: int,
    ):
    """
    Create a report for the ECG Validation comparison.

    ARGUMENTS:
    --------------------------------
    ecg_validation_comparison_report_path: str
        path to the file where the report is saved
    additions_results_path: str,
        path to the pickle file where the ecg validation comparison should be saved
    file_name_dictionary_key
        dictionary key to access the file name
    ecg_validation_comparison_dictionary_key: str
        dictionary key to access the ecg validation comparison
    ecg_validation_comparison_report_dezimal_places: int
        number of decimal places for the report
    
    RETURNS:
    --------------------------------
    None, but the report is saved to a file as a table
    """

    # check if the report already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_file(
        file_path = ecg_validation_comparison_report_path,
        message = "\nECG validation comparison report already exists in " + ecg_validation_comparison_report_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # open the file to write the report to
    comparison_file = open(ecg_validation_comparison_report_path, "w")

    # load the data
    file_names = []
    strictness_values = []
    ecg_validation_comparison_values = []

    all_files_ecg_validation_generator = load_from_pickle(additions_results_path)
    for generator_entry in all_files_ecg_validation_generator:
        if ecg_validation_comparison_dictionary_key in generator_entry and file_name_dictionary_key in generator_entry:
            file_names.append(generator_entry[file_name_dictionary_key])
            ecg_validation_comparison_values.append(generator_entry[ecg_validation_comparison_dictionary_key][1])
            if len(strictness_values) == 0:
                strictness_values = generator_entry[ecg_validation_comparison_dictionary_key][0]
    
    strictness_max_length = max([len(strict_val) for strict_val in strictness_values])

    # write the file header
    message = "ECG VALIDATION COMPARISON REPORT"
    comparison_file.write(message + "\n")
    comparison_file.write("=" * len(message) + "\n\n")

    comparison_file.write("The following ratios were calculated for different values of validation_strictness.\n")
    comparison_file.write("The cells are written in following format: strictness_value:  ratio_value\n")
    comparison_file.write("This was done so that we can use the best value for our check_ecg_validation_strictness variable.\n\n")

    # set the table captions
    CORRECT_VALID_CAPTION = "Correct Valid"
    CORRECT_INVALID_CAPTION = "Correct Invalid"
    FILE_CAPTION = "File"
    INCORRECT_VALID_CAPTION = "Wrong Valid"
    INCORRECT_INVALID_CAPTION = "Wrong Invalid"

    MEAN_ROW_CAPTION = "Mean values"

    # collect all comparison values
    correct_valid_column = []
    correct_invalid_column = []
    incorrect_valid_column = []
    incorrect_invalid_column = []

    mean_correct_valid_for_strictness = [[] for _ in range(0, len(strictness_values))]
    mean_correct_invalid_for_strictness = [[] for _ in range(0, len(strictness_values))]
    mean_incorrect_valid_for_strictness = [[] for _ in range(0, len(strictness_values))]
    mean_incorrect_invalid_for_strictness = [[] for _ in range(0, len(strictness_values))]

    for strictness_comp_values in ecg_validation_comparison_values:
        for i in range(0, len(strictness_values)):
            length_addition = strictness_max_length - len(strictness_values[i])

            correct_valid_value = strictness_comp_values[i][0]
            mean_correct_valid_for_strictness[i].append(correct_valid_value)
            correct_valid_column.append(strictness_values[i] + " "*length_addition + ":  " + str(round(correct_valid_value, ecg_validation_comparison_report_dezimal_places)))

            correct_invalid_value = strictness_comp_values[i][1]
            mean_correct_invalid_for_strictness[i].append(correct_invalid_value)
            correct_invalid_column.append(strictness_values[i] + " "*length_addition + ":  " + str(round(correct_invalid_value, ecg_validation_comparison_report_dezimal_places)))

            incorrect_valid_value = strictness_comp_values[i][2]
            mean_incorrect_valid_for_strictness[i].append(incorrect_valid_value)
            incorrect_valid_column.append(strictness_values[i] + " "*length_addition + ":  " + str(round(incorrect_valid_value, ecg_validation_comparison_report_dezimal_places)))

            incorrect_invalid_value = strictness_comp_values[i][3]
            mean_incorrect_invalid_for_strictness[i].append(incorrect_invalid_value)
            incorrect_invalid_column.append(strictness_values[i] + " "*length_addition + ":  " + str(round(incorrect_invalid_value, ecg_validation_comparison_report_dezimal_places)))
    
    # calculate mean of them
    mean_correct_valid = np.mean(mean_correct_valid_for_strictness, axis=1)
    mean_correct_invalid = np.mean(mean_correct_invalid_for_strictness, axis=1)
    mean_incorrect_valid = np.mean(mean_incorrect_valid_for_strictness, axis=1)
    mean_incorrect_invalid = np.mean(mean_incorrect_invalid_for_strictness, axis=1)

    for i in range(0, len(strictness_values)):
        length_addition = strictness_max_length - len(strictness_values[i])
        correct_valid_column.insert(i, strictness_values[i] + " "*length_addition + ":  " + str(round(mean_correct_valid[i], ecg_validation_comparison_report_dezimal_places)))
        correct_invalid_column.insert(i, strictness_values[i] + " "*length_addition + ":  " + str(round(mean_correct_invalid[i], ecg_validation_comparison_report_dezimal_places)))
        incorrect_valid_column.insert(i, strictness_values[i] + " "*length_addition + ":  " + str(round(mean_incorrect_valid[i], ecg_validation_comparison_report_dezimal_places)))
        incorrect_invalid_column.insert(i, strictness_values[i] + " "*length_addition + ":  " + str(round(mean_incorrect_invalid[i], ecg_validation_comparison_report_dezimal_places)))
    
    # calculate max column wide
    file_names.insert(0, MEAN_ROW_CAPTION)
    max_file_column_length = max([len(file_name) for file_name in file_names])
    max_file_column_length = max(max_file_column_length, len(FILE_CAPTION))

    max_correct_valid_column_length = max([len(entry) for entry in correct_valid_column])
    max_correct_valid_column_length = max(max_correct_valid_column_length, len(CORRECT_VALID_CAPTION))

    max_correct_invalid_column_length = max([len(entry) for entry in correct_invalid_column])
    max_correct_invalid_column_length = max(max_correct_invalid_column_length, len(CORRECT_INVALID_CAPTION))

    max_incorrect_valid_column_length = max([len(entry) for entry in incorrect_valid_column])
    max_incorrect_valid_column_length = max(max_incorrect_valid_column_length, len(INCORRECT_VALID_CAPTION))

    max_incorrect_invalid_column_length = max([len(entry) for entry in incorrect_invalid_column])
    max_incorrect_invalid_column_length = max(max_incorrect_invalid_column_length, len(INCORRECT_INVALID_CAPTION))

    # write the legend for the table
    message = "Legend:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")
    comparison_file.write(CORRECT_VALID_CAPTION + "... Matching valid regions ratio\n")
    comparison_file.write(CORRECT_INVALID_CAPTION + "... Matching invalid regions ratio\n")
    comparison_file.write(INCORRECT_VALID_CAPTION + "... valid (detected) / invalid (gif) ratio\n")
    comparison_file.write(INCORRECT_INVALID_CAPTION + "... invalid (detected) / valid (gif) ratio\n\n\n")

    message = "Table with comparison values for each file:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")

    # create table header
    comparison_file.write(print_in_middle(FILE_CAPTION, max_file_column_length) + " | ")
    comparison_file.write(print_in_middle(CORRECT_VALID_CAPTION, max_correct_valid_column_length) + " | ")
    comparison_file.write(print_in_middle(CORRECT_INVALID_CAPTION, max_correct_invalid_column_length) + " | ")
    comparison_file.write(print_in_middle(INCORRECT_VALID_CAPTION, max_incorrect_valid_column_length) + " | ")
    comparison_file.write(print_in_middle(INCORRECT_INVALID_CAPTION, max_incorrect_invalid_column_length) + "\n")
    total_length = max_file_column_length + max_correct_valid_column_length + max_correct_invalid_column_length + max_incorrect_valid_column_length + max_incorrect_invalid_column_length + 3*4 + 1
    comparison_file.write("-" * total_length + "\n")

    # write the data
    num_of_strictness_vals = len(strictness_values)
    for file_index in range(0, len(file_names)):
        file = file_names[file_index]
        print_file_name_at = int((file_index + 0.5)*num_of_strictness_vals)

        for i in range(file_index*num_of_strictness_vals, (file_index+1)*num_of_strictness_vals):
            if i == print_file_name_at:
                comparison_file.write(print_in_middle(file, max_file_column_length) + " | ")
            else:
                comparison_file.write(" "*max_file_column_length + " | ")
            comparison_file.write(print_left_aligned(correct_valid_column[i], max_correct_valid_column_length))
            comparison_file.write(" | ")
            comparison_file.write(print_left_aligned(correct_invalid_column[i], max_correct_invalid_column_length))
            comparison_file.write(" | ")
            comparison_file.write(print_left_aligned(incorrect_valid_column[i], max_incorrect_valid_column_length))
            comparison_file.write(" | ")
            comparison_file.write(print_left_aligned(incorrect_invalid_column[i], max_incorrect_invalid_column_length))
            comparison_file.write("\n")
        
        comparison_file.write("-" * total_length + "\n")

    comparison_file.close()