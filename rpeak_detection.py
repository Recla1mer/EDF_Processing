"""
Author: Johannes Peter Knoll

Python implementation to detect R-peaks in ECG data.
Useful Link: https://www.samproell.io/posts/signal/ecg-library-comparison/

Main function: get_rpeaks
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
    Detect R peaks in the ECG data.

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
    None, but the rpeaks are saved to a pickle file
    """

    # check if R peaks already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = rpeak_path,
                            message = "Detected R peaks already exist in: " + rpeak_path)
    
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
    print("Detecting R peaks in the ECG data in %i files using %s:" % (total_files, rpeak_function_name))
    for file in valid_files:
        # show progress
        progress_bar(progressed_files, total_files)

        # get the valid regions for the ECG data
        try:
            detection_intervals = valid_ecg_regions[file]
            progressed_files += 1
        except KeyError:
            print("Valid regions for the ECG data in " + file + " are missing. Skipping this file.")
            continue

        # get the ECG data
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)

        # detect the R peaks in the valid regions
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
    
    save_to_pickle(all_rpeaks, rpeak_path)


def combine_rpeaks(
        rpeaks_primary: list,
        rpeaks_secondary: list,
        frequency: int,
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
    distance_threshold_iterations = int(rpeak_distance_threshold_seconds * frequency)

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
    Detect R peaks in the ECG data and compare the results of two different functions.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    others: see rpeak_detection.compare_rpeak_detection_methods()

    RETURNS:
    --------------------------------
    None, but the valid regions are saved to a pickle file
    """
    user_answer = ask_for_permission_to_override(file_path = certain_rpeaks_path,
                                    message = "Detected R peaks were already combined.")
    
    if user_answer == "n":
        return
    
    os.remove(uncertain_primary_rpeaks_path)
    os.remove(uncertain_secondary_rpeaks_path)

    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    total_files = len(valid_files)
    progressed_files = 0

    certain_rpeaks = dict()
    uncertain_primary_rpeaks = dict()
    uncertain_secondary_rpeaks = dict()

    # load detected R peaks
    all_rpeaks_primary = load_from_pickle(rpeak_primary_path)
    all_rpeaks_secondary = load_from_pickle(rpeak_secondary_path)

    # combine detected R peaks
    print("Combining detected R peaks for %i files:" % total_files)
    for file in valid_files:
        progress_bar(progressed_files, total_files)
        
        sigfreqs = read_edf.get_edf_data(data_directory + file)[1]

        these_combined_rpeaks = combine_rpeaks(
            rpeaks_primary = all_rpeaks_primary[file],
            rpeaks_secondary = all_rpeaks_secondary[file],
            frequency = sigfreqs[ecg_key],
            rpeak_distance_threshold_seconds = rpeak_distance_threshold_seconds
            )
        
        certain_rpeaks[file] = these_combined_rpeaks[0]
        uncertain_primary_rpeaks[file] = these_combined_rpeaks[1]
        uncertain_secondary_rpeaks[file] = these_combined_rpeaks[2]
    
    progress_bar(progressed_files, total_files)
    
    save_to_pickle(certain_rpeaks, certain_rpeaks_path)
    save_to_pickle(uncertain_primary_rpeaks, uncertain_primary_rpeaks_path)
    save_to_pickle(uncertain_secondary_rpeaks, uncertain_secondary_rpeaks_path)


"""
Following code won't be used for the final implementation, but is useful for testing and
comparing the results of different R-peak detection methods, as accurate R peaks
(calculated automatically but later checked manually) are available for the GIF data.
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
    rpeak_distance_threshold_seconds: float
        threshold for the distance between two R-peaks to be considered as the same
    print_results: bool
        if True, the results will be printed

    RETURNS:
    --------------------------------
    None
    """
    kwargs.setdefault("first_name", "First Method")
    kwargs.setdefault("second_name", "Second Method")
    kwargs.setdefault("print_results", False)

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
    if kwargs["print_results"]:
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
        this_rpeak, letter = rri_string_evaluation(rri[i])
        if isinstance(this_rpeak, int) and letter.isalpha():
            if letter in rpeaks:
                rpeaks[letter].append(this_rpeak)
            else:
                rpeaks[letter] = [this_rpeak]
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
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    ecg_key: str
        key for the ECG data in the data dictionary
    accurate_peaks_directory: str
        directory where the accurate R peaks are stored
    accurate_peaks_name: str
        name the accurate R peaks are associated with
    valid_accuracy_file_types: list
        valid file types in the accurate peaks directory
    compare_rpeaks_paths: list
        paths to the R peaks that should be compared to the accurate R peaks
    rpeak_distance_threshold_seconds: float
        time period in seconds over which two different R peaks are still considered the same
    
    RETURNS:
    --------------------------------
    None, but the Accuracy values are saved as dictionary to a pickle file in following format:
    {
        "file_name": [ [function_1 values], [function_2 values], ... ],
        ...
    }
    with function values being: rmse_without_same, rmse_with_same, number_of_same_values, number_of_values_considered_as_same, len_function_rpeaks, length_accurate_rpeaks
    for rmse_without_same and rmse_with_same see rpeak_detection.compare_rpeak_detection_methods()
    """
    user_answer = ask_for_permission_to_override(file_path = rpeak_accuracy_evaluation_path,
                        message = "Evaluation of R peak detection accuracy already exists.")
    
    if user_answer == "n":
        return

    all_data_files = os.listdir(accurate_rpeaks_raw_data_directory)
    valid_data_files = [file for file in all_data_files if get_file_type(file) in valid_file_types]

    all_accurate_files = os.listdir(accurate_rpeaks_values_directory)
    valid_accurate_files = [file for file in all_accurate_files if get_file_type(file) in valid_accurate_rpeak_file_types]

    total_data_files = len(valid_data_files)
    progressed_data_files = 0

    # calculate rmse for all files
    all_files_rpeak_accuracy = dict()
    for file in valid_data_files:
        this_file_rpeak_accuracy = []

        this_file_name = os.path.splitext(file)[0]
        for acc_file in valid_accurate_files:
            if this_file_name in acc_file:
                this_accurate_file = acc_file
        try:
            accurate_rpeaks = get_rpeaks_from_rri_file(accurate_rpeaks_values_directory + this_accurate_file)
        except ValueError:
            print("Accurate R peaks are missing for %s. Skipping this file." % file)
            continue
        
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(accurate_rpeaks_raw_data_directory + file)
        frequency = sigfreqs[ecg_key]
        
        length_accurate = len(accurate_rpeaks["N"])
        
        for path in compare_rpeaks_paths:
            compare_rpeaks_all_files = load_from_pickle(path)

            compare_rpeaks = compare_rpeaks_all_files[file]

            len_compare_rpeaks = len(compare_rpeaks)

            rmse_without_same, rmse_with_same, len_same_values, len_analog_values = compare_rpeak_detection_methods(
                first_rpeaks = accurate_rpeaks["N"], 
                second_rpeaks = compare_rpeaks,
                frequency = frequency,
                rpeak_distance_threshold_seconds = rpeak_distance_threshold_seconds,
                )
            
            this_file_rpeak_accuracy.append([rmse_without_same, rmse_with_same, len_same_values, len_analog_values, len_compare_rpeaks, length_accurate])
        
        all_files_rpeak_accuracy[file] = this_file_rpeak_accuracy
    
    save_to_pickle(all_files_rpeak_accuracy, rpeak_accuracy_evaluation_path)


def print_in_middle(string, length):
    """
    Print the string in the middle of the total length.
    """
    len_string = len(string)
    undersize = int((length - len_string) // 2)
    return " " * undersize + string + " " * (length - len_string - undersize)


def print_rpeak_accuracy_results(
        rpeak_accuracy_function_names: list,  
        accurate_peaks_name: str, 
        rpeak_accuracy_rmse_dezimal_places: int,
        rpeak_accuracy_report_path: str,
        rpeak_accuracy_evaluation_path: str
    ):
    """
    """
    user_answer = ask_for_permission_to_override(file_path = rpeak_accuracy_report_path,
                                        message = "R peak accuracy report already exists.")

    if user_answer == "n":
        return

    accuracy_file = open(rpeak_accuracy_report_path, "w")

    # write the file header
    message = "R PEAK ACCURACY EVALUATION"
    accuracy_file.write(message + "\n")
    accuracy_file.write("=" * len(message) + "\n\n\n")

    RMSE_EX_CAPTION = "RMSE_exc"
    RMSE_INC_CAPTION = "RMSE_inc"
    FILE_CAPTION = "File"
    TOTAL_LENGTH_CAPTION = "R peaks"
    SAME_VALUES_CAPTION = "Same Values"
    ANALOG_VALUES_CAPTION = "Analog Values"

    # load the data
    all_files_rpeak_accuracy = load_from_pickle(rpeak_accuracy_evaluation_path)

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