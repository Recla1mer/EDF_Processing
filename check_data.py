"""
Author: Johannes Peter Knoll

Python implementation of ECG data validation
"""

# IMPORTS
import numpy as np

# LOCAL IMPORTS
import read_edf
from side_functions import *


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


def locally_calculate_ecg_thresholds(
        ECG: list,
        time_interval_iterations: int,
    ):
    """
    """
    standard_deviations = []
    std_distance_ratios = []

    max_ecg = np.max(ECG)

    for i in np.arange(0, len(ECG), time_interval_iterations):

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
            std_distance_ratio = max_ecg
        else:
            std_distance_ratio = 0.5 * max_min_distance / this_std
        
        standard_deviations.append(this_std)
        std_distance_ratios.append(std_distance_ratio)
    
    return 0.2*np.mean(standard_deviations), 0.5*np.mean(std_distance_ratios)


def check_ecg(
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

    return concatenated_intervals

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


def determine_valid_ecg_regions(
        data_directory: str,
        valid_file_types: list,
        ecg_keys: list,
        physical_dimension_correction_dictionary: dict,
        preparation_results_path: str,
        file_name_dictionary_key: str,
        valid_ecg_regions_dictionary_key: str,
        check_ecg_std_min_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        check_ecg_time_interval_seconds: int, 
        check_ecg_overlapping_interval_steps: int,
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

    # check if valid regions already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = preparation_results_path,
        dictionary_entry = valid_ecg_regions_dictionary_key
        )
    
    # cancel if user does not want to override
    if user_answer == "n":
        return
    
    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(preparation_results_path) + "computation_in_progress.pkl"
    if os.path.isfile(temporary_file_path):
        os.remove(temporary_file_path)

    # create list to store files that could not be processed
    unprocessable_files = []

    if user_answer == "y":
        # load existing results
        preparation_results_generator = load_from_pickle(preparation_results_path)

        # create variables to track progress
        total_files = get_pickle_length(preparation_results_path)
        progressed_files = 0

        print("\nCalculating valid regions for the ECG data in %i files from \"%s\":" % (total_files, data_directory))

        for generator_entry in preparation_results_generator:
            # show progress
            progress_bar(progressed_files, total_files)
            progressed_files += 1

            try:
                # get current file name
                file_name = generator_entry[file_name_dictionary_key]

                # try to load the data and correct the physical dimension if needed
                ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
                    file_path = data_directory + file_name,
                    possible_channel_labels = ecg_keys,
                    physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                )

                # calculate the valid regions
                this_valid_regions = check_ecg(
                    ECG = ecg_signal, 
                    frequency = ecg_sampling_frequency, 
                    check_ecg_std_min_threshold = check_ecg_std_min_threshold, 
                    check_ecg_distance_std_ratio_threshold = check_ecg_distance_std_ratio_threshold,
                    time_interval_seconds = check_ecg_time_interval_seconds, 
                    overlapping_interval_steps = check_ecg_overlapping_interval_steps,
                    min_valid_length_minutes = check_ecg_min_valid_length_minutes,
                    allowed_invalid_region_length_seconds = check_ecg_allowed_invalid_region_length_seconds,
                    )
                
                # save the valid regions for this file
                generator_entry[valid_ecg_regions_dictionary_key] = this_valid_regions
                append_to_pickle(generator_entry, temporary_file_path)

            except:
                unprocessable_files.append(file_name)
                continue
            
    elif user_answer == "no_file_found":
        # get all valid files
        all_files = os.listdir(data_directory)
        valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

        # create variables to track progress
        total_files = len(valid_files)
        progressed_files = 0

        print("\nCalculating valid regions for the ECG data in %i files from \"%s\":" % (total_files, data_directory))

        for file_name in valid_files:
            # show progress
            progress_bar(progressed_files, total_files)
            progressed_files += 1

            try:
                # try to load the data and correct the physical dimension if needed
                ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
                    file_path = data_directory + file_name,
                    possible_channel_labels = ecg_keys,
                    physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                )

                # calculate the valid regions
                this_valid_regions = check_ecg(
                    ECG = ecg_signal, 
                    frequency = ecg_sampling_frequency, 
                    check_ecg_std_min_threshold = check_ecg_std_min_threshold, 
                    check_ecg_distance_std_ratio_threshold = check_ecg_distance_std_ratio_threshold,
                    time_interval_seconds = check_ecg_time_interval_seconds, 
                    overlapping_interval_steps = check_ecg_overlapping_interval_steps,
                    min_valid_length_minutes = check_ecg_min_valid_length_minutes,
                    allowed_invalid_region_length_seconds = check_ecg_allowed_invalid_region_length_seconds,
                    )
                
                # save the valid regions for this file
                this_files_dictionary_entry = {
                    file_name_dictionary_key: file_name,
                    valid_ecg_regions_dictionary_key: this_valid_regions
                    }
                append_to_pickle(this_files_dictionary_entry, temporary_file_path)

            except:
                unprocessable_files.append(file_name)
                continue

    progress_bar(progressed_files, total_files)

    # rename the file that stores the calculated data
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
    
    # cancel if user does not want to override
    if user_answer == "n":
        return
    
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
    total_data_files = get_pickle_length(additions_results_path)
    progressed_data_files = 0

    # create lists to store unprocessable files
    unprocessable_files = []
    
    # calculate the ECG Validation comparison values for all files
    print("\nCalculating ECG validation comparison values for %i files:" % total_data_files)
    for generator_entry in addition_results_generator:
        # show progress
        progress_bar(progressed_data_files, total_data_files)
        progressed_data_files += 1

        try:
            # get the file key and the validated ECG regions
            this_file = generator_entry[file_name_dictionary_key]

            # get the file name without the file type
            this_file_name = os.path.splitext(this_file)[0]

            # get corresponding ECG classification file name for this file
            for clfc_file in valid_classification_files:
                if this_file_name in clfc_file:
                    this_classification_file = clfc_file

            ecg_classification_dictionary = get_ecg_classification_from_txt_file(ecg_classification_values_directory + this_classification_file)
        
            # compare the differnt ECG validations
            this_file_comparison_values = compare_ecg_validations(
                validated_intervals = generator_entry[valid_ecg_regions_dictionary_key],
                ecg_classification = ecg_classification_dictionary
                )
        
            # save the comparison values for this file
            generator_entry[ecg_validation_comparison_dictionary_key] = this_file_comparison_values
            append_to_pickle(generator_entry, temporary_file_path)

        except:
            unprocessable_files.append(this_file)
            continue
        
    progress_bar(progressed_data_files, total_data_files)

    # rename the file that stores the calculated data
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
    all_files_ecg_validation_generator = load_from_pickle(additions_results_path)
    all_files_ecg_validation_comparison = dict()
    for generator_entry in all_files_ecg_validation_generator:
        if ecg_validation_comparison_dictionary_key in generator_entry and file_name_dictionary_key in generator_entry:
            all_files_ecg_validation_comparison.update({generator_entry[file_name_dictionary_key]: generator_entry[ecg_validation_comparison_dictionary_key]})

    # write the file header
    message = "ECG VALIDATION COMPARISON REPORT"
    comparison_file.write(message + "\n")
    comparison_file.write("=" * len(message) + "\n\n\n")

    # set the table captions
    CORRECT_VALID_CAPTION = "Correct Valid"
    CORRECT_INVALID_CAPTION = "Correct Invalid"
    FILE_CAPTION = "File"
    WRONG_AS_VALID_CAPTION = "Wrong Valid"
    WRONG_AS_INVALID_CAPTION = "Wrong Invalid"

    MEAN_ROW_CAPTION = "Mean values"

    # create lists to collect all acccuracy values and print the mean of them
    correct_valid_values = []
    correct_invalid_values = []
    wrong_as_valid_values = []
    wrong_as_invalid_values = []

    # collect all comparison values
    for file in all_files_ecg_validation_comparison:
        all_files_ecg_validation_comparison[file][0] = round(all_files_ecg_validation_comparison[file][0], ecg_validation_comparison_report_dezimal_places)
        all_files_ecg_validation_comparison[file][1] = round(all_files_ecg_validation_comparison[file][1], ecg_validation_comparison_report_dezimal_places)
        all_files_ecg_validation_comparison[file][2] = round(all_files_ecg_validation_comparison[file][2], ecg_validation_comparison_report_dezimal_places)
        all_files_ecg_validation_comparison[file][3] = round(all_files_ecg_validation_comparison[file][3], ecg_validation_comparison_report_dezimal_places)

        correct_valid_values.append(all_files_ecg_validation_comparison[file][0])
        correct_invalid_values.append(all_files_ecg_validation_comparison[file][1])
        wrong_as_valid_values.append(all_files_ecg_validation_comparison[file][2])
        wrong_as_invalid_values.append(all_files_ecg_validation_comparison[file][3])
    
    # calculate mean of them
    mean_correct_valid = round(np.mean(correct_valid_values), ecg_validation_comparison_report_dezimal_places)
    mean_correct_invalid = round(np.mean(correct_invalid_values), ecg_validation_comparison_report_dezimal_places)
    mean_wrong_as_valid = round(np.mean(wrong_as_valid_values), ecg_validation_comparison_report_dezimal_places)
    mean_wrong_as_invalid = round(np.mean(wrong_as_invalid_values), ecg_validation_comparison_report_dezimal_places)

    quick_items = list(all_files_ecg_validation_comparison.items())
    quick_items.insert(0, (MEAN_ROW_CAPTION, [mean_correct_valid, mean_correct_invalid, mean_wrong_as_valid, mean_wrong_as_invalid]))
    all_files_ecg_validation_comparison = dict(quick_items)

    # calcualte max lengths of table columns
    all_file_lengths = [len(key) for key in all_files_ecg_validation_comparison]
    max_file_length = max(len(FILE_CAPTION), max(all_file_lengths)) + 3

    all_correct_valid_lengths = []
    all_correct_invalid_lengths = []
    all_wrong_as_valid_lengths = []
    all_wrong_as_invalid_lengths = []
    for file in all_files_ecg_validation_comparison:
        all_correct_valid_lengths.append(len(str(all_files_ecg_validation_comparison[file][0])))
        all_correct_invalid_lengths.append(len(str(all_files_ecg_validation_comparison[file][1])))
        all_wrong_as_valid_lengths.append(len(str(all_files_ecg_validation_comparison[file][2])))
        all_wrong_as_invalid_lengths.append(len(str(all_files_ecg_validation_comparison[file][3])))

    all_correct_valid_lengths = np.array(all_correct_valid_lengths)
    all_correct_invalid_lengths = np.array(all_correct_invalid_lengths)
    all_wrong_as_valid_lengths = np.array(all_wrong_as_valid_lengths)
    all_wrong_as_invalid_lengths = np.array(all_wrong_as_invalid_lengths)

    max_correct_valid_length = max(len(CORRECT_VALID_CAPTION), max(all_correct_valid_lengths))
    max_correct_invalid_length = max(len(CORRECT_INVALID_CAPTION), max(all_correct_invalid_lengths))
    max_wrong_as_valid_length = max(len(WRONG_AS_VALID_CAPTION), max(all_wrong_as_valid_lengths))
    max_wrong_as_invalid_length = max(len(WRONG_AS_INVALID_CAPTION), max(all_wrong_as_invalid_lengths))

    # write the legend for the table
    message = "Legend:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")
    comparison_file.write(CORRECT_VALID_CAPTION + "... Matching valid regions ratio\n")
    comparison_file.write(CORRECT_INVALID_CAPTION + "... Matching invalid regions ratio\n")
    comparison_file.write(WRONG_AS_VALID_CAPTION + "... valid (detected) / invalid (gif) ratio\n")
    comparison_file.write(WRONG_AS_INVALID_CAPTION + "... invalid (detected) / valid (gif) ratio\n\n\n")

    message = "Table with comparison values for each file:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")

    # create table header
    comparison_file.write(print_in_middle(FILE_CAPTION, max_file_length) + " | ")
    comparison_file.write(print_in_middle(CORRECT_VALID_CAPTION, max_correct_valid_length) + " | ")
    comparison_file.write(print_in_middle(CORRECT_INVALID_CAPTION, max_correct_invalid_length) + " | ")
    comparison_file.write(print_in_middle(WRONG_AS_VALID_CAPTION, max_wrong_as_valid_length) + " | ")
    comparison_file.write(print_in_middle(WRONG_AS_INVALID_CAPTION, max_wrong_as_invalid_length) + "\n")
    total_length = max_file_length + max_correct_valid_length + max_correct_invalid_length + max_wrong_as_valid_length + max_wrong_as_invalid_length + 3*4 + 1
    comparison_file.write("-" * total_length + "\n")

    # write the data
    for file in all_files_ecg_validation_comparison:
        comparison_file.write(print_in_middle(file, max_file_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][0]), max_correct_valid_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][1]), max_correct_invalid_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][2]), max_wrong_as_valid_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][3]), max_wrong_as_invalid_length) + "\n")
        comparison_file.write("-" * total_length + "\n")

    comparison_file.close()