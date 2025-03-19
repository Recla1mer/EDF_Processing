"""
Author: Johannes Peter Knoll

Python file to calculate the RR-intervals from the detected r-peaks.
"""

# IMPORTS
import numpy as np
import copy
import h5py

# LOCAL IMPORTS
import read_edf
from side_functions import *


def calculate_average_rri_from_peaks(
        rpeaks: list,
        ecg_sampling_frequency: int,
        target_sampling_frequency: float,
        signal_length: int,
        pad_with: float
    ):
    """
    Calculate the RR-intervals from the detected r-peaks. Return with the target sampling frequency.
    
    Designed to be run for low values of target_sampling_frequency. A human should have an RRI between
    1/3 (180 beats per minute -> during sport?) and 1.2 (50 bpm, during sleep?) seconds. Average RRI
    should be around 0.6 - 1 seconds.

    So if target sampling frequency is below 0.25 Hz (4 seconds), the function will calculate the average
    RRI for each datapoint, as in this case you will most likely have more than 3 R-peaks in the datapoint
    covering the time sequence.

    ARGUMENTS:
    --------------------------------
    rpeaks: list
        list of detected r-peaks
    ecg_sampling_frequency: int
        sampling frequency of the ECG data
    target_sampling_frequency: int
        sampling frequency of the RR-intervals
    signal_length: int
        length of the ECG signal
    pad_with: float
        value to pad the rri list with if no rri value can be calculated
    
    RETURNS:
    --------------------------------
    rri: list
        list of RR-intervals
    """

    # check parameters
    if target_sampling_frequency > 0.25:
        raise ValueError("This function is designed to be run for low values (<= 0.25 Hz) of target_sampling_frequency. Please use calculate_momentarily_rri_from_peaks for higher values.")

    # calculate the number of entries in the rri list
    number_rri_entries = int(np.ceil(signal_length / ecg_sampling_frequency * target_sampling_frequency))

    # rewrite rpeaks from (number of sample) to (second)
    rpeaks = np.array(rpeaks) # type: ignore
    rpeak_position_seconds = rpeaks / ecg_sampling_frequency # type: ignore

    collect_rpeaks = []
    start_looking_at = 0

    # collect all rpeaks within the same rri datapoint
    for i in range(1, number_rri_entries+1):
        lower_rri_second = (i-1) / target_sampling_frequency
        upper_rri_second = i / target_sampling_frequency
        these_rpeaks = []

        for j in range(start_looking_at, len(rpeak_position_seconds)):
            if rpeak_position_seconds[j] <= upper_rri_second and lower_rri_second <= rpeak_position_seconds[j]:
                these_rpeaks.append(rpeak_position_seconds[j])
            if rpeak_position_seconds[j] > upper_rri_second:
                start_looking_at = j
                break

        collect_rpeaks.append(these_rpeaks)
    
    max_index = len(collect_rpeaks) - 1
    # if less than two rpeaks are found in the time sequence, try to add the previous and/or the next rpeak
    for i in range(len(collect_rpeaks)):
        if len(collect_rpeaks[i]) == 1 and i > 0:
            if len(collect_rpeaks[i-1]) > 0:
                # add previous rpeak
                collect_rpeaks[i].insert(0, collect_rpeaks[i-1][-1])
            else:
                # try to add next available rpeak if previous is not available 
                index = i+1
                while True:
                    if index > max_index:
                        # if no rpeak is found, stop looking
                        break
                    if len(collect_rpeaks[index]) > 0:
                        # add next available rpeak
                        collect_rpeaks[i].append(collect_rpeaks[index][0])
                        break
                    index += 1

        elif len(collect_rpeaks[i]) == 0 and i > 0:
            if len(collect_rpeaks[i-1]) > 0:
                # add previous rpeak
                collect_rpeaks[i].insert(0, collect_rpeaks[i-1][-1]) # previous

                # try to add next available rpeak
                index = i+1
                while True:
                    if index > max_index:
                        # if no rpeak is found, stop looking
                        break
                    if len(collect_rpeaks[index]) > 0:
                        collect_rpeaks[i].append(collect_rpeaks[index][0])
                        print(collect_rpeaks[index][0])
                        break
                    index += 1
    
    rri = []
    
    # calculate average rri, if time sequence contains less than 2 rpeaks, meaning
    # there is no other rpeak in wide approximity
    for i in range(len(collect_rpeaks)):
        if len(collect_rpeaks[i]) >= 2:
            rri.append((collect_rpeaks[i][-1] - collect_rpeaks[i][0]) / (len(collect_rpeaks[i])-1))
        else:
            rri.append(pad_with)
    
    return rri


def calculate_momentarily_rri_from_peaks(
        rpeaks: list,
        ecg_sampling_frequency: int,
        target_sampling_frequency: float,
        signal_length: int,
        pad_with: float
    ):
    """
    Calculate the RR-intervals from the detected r-peaks. Return with the target sampling frequency.

    As we expect the sampling frequency to be higher than the heart rate, we look between which two rpeaks
    the datapoint of the RR-interval is located and return the difference of the two rpeaks.

    ARGUMENTS:
    --------------------------------
    rpeaks: list
        list of detected r-peaks
    ecg_sampling_frequency: int
        sampling frequency of the ECG data
    target_sampling_frequency: int
        sampling frequency of the RR-intervals
    signal_length: int
        length of the ECG signal
    pad_with: float
        value to pad the rri list with if no rri value can be calculated
    
    RETURNS:
    --------------------------------
    rri: list
        list of RR-intervals
    """

    # check parameters
    if target_sampling_frequency <= 0.25:
        raise ValueError("This function is designed to be run for high values (> 0.25 Hz) of target_sampling_frequency. Please use calculate_average_rri_from_peaks for lower values.")

    # calculate the number of entries in the rri list
    number_rri_entries = int(np.ceil(signal_length / ecg_sampling_frequency * target_sampling_frequency))

    # rewrite rpeaks from (number of sample) to (second)
    rpeaks = np.array(rpeaks) # type: ignore
    rpeak_position_seconds = rpeaks / ecg_sampling_frequency # type: ignore

    rri = []
    start_looking_at = 1

    # iterate over all rri entries, find rri datapoint between two rpeaks (in seconds) and return their difference
    for i in range(number_rri_entries):
        rri_datapoint_second = i / target_sampling_frequency

        this_rri = pad_with
        for j in range(start_looking_at, len(rpeak_position_seconds)):
            start_looking_at = j

            if rpeak_position_seconds[j] == rri_datapoint_second and j not in [0, len(rpeak_position_seconds)-1]:
                this_rri = (rpeak_position_seconds[j+1] - rpeak_position_seconds[j-1]) / 2
                break
            if rpeak_position_seconds[j-1] <= rri_datapoint_second and rri_datapoint_second <= rpeak_position_seconds[j]:
                this_rri = rpeak_position_seconds[j] - rpeak_position_seconds[j-1]
                break
            if rpeak_position_seconds[j-1] > rri_datapoint_second:
                break
        
        rri.append(this_rri)
    
    return rri


def calculate_rri_from_peaks(
        rpeaks: list,
        ecg_sampling_frequency: int,
        target_sampling_frequency: float,
        signal_length: int,
        pad_with: float
    ):
    """
    Calculate the RR-intervals from the detected r-peaks. Return with the target sampling frequency.

    ARGUMENTS:
    --------------------------------
    rpeaks: list
        list of detected r-peaks
    ecg_sampling_frequency: int
        sampling frequency of the ECG data
    target_sampling_frequency: int
        sampling frequency of the RR-intervals
    signal_length: int
        length of the ECG signal
    pad_with: float
        value to pad the rri list with if no rri value can be calculated
    
    RETURNS:
    --------------------------------
    rri: list
        list of RR-intervals
    """

    if target_sampling_frequency <= 0.25:
        return calculate_average_rri_from_peaks(
            rpeaks = rpeaks,
            ecg_sampling_frequency = ecg_sampling_frequency,
            target_sampling_frequency = target_sampling_frequency,
            signal_length = signal_length,
            pad_with = pad_with
        )
    else:
        return calculate_momentarily_rri_from_peaks(
            rpeaks = rpeaks,
            ecg_sampling_frequency = ecg_sampling_frequency,
            target_sampling_frequency = target_sampling_frequency,
            signal_length = signal_length,
            pad_with = pad_with
        )


def determine_rri_from_rpeaks(
        data_directory: str,
        rpeak_function_name: str,
        RRI_sampling_frequency: int,
        mad_time_period_seconds: int,
        pad_with: float,
        realistic_rri_value_range: list,
        results_path: str,
    ):
    """
    Calculate the RR-intervals from the detected r-peaks and save them to a pickle file.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    rpeak_function_name: str
        name of the r-peak detection function
    RRI_sampling_frequency: int
        target sampling frequency of the RR-intervals
    1 / mad_time_period_seconds: float
        target sampling frequency of the MAD values
    pad_with: float
        value to pad the rri list with if no rri value can be calculated
    realistic_rri_value_range: list
        list of two floats, which represent the lower and upper bound of realistic rri values
        if the rri values are outside of this range, they will be replaced with the pad_with value
    results_path: str
        path to the pickle file where the valid regions are saved

    RETURNS:
    --------------------------------
    None, but the rr intervals are saved as dictionaries to a pickle file in the following format:
    {
        "file_name": file_name_1,
        "RRI": rri_1,
        ...
    }
        ...
    """
    
    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(results_path) + "computation_in_progress.pkl"

    # if the temporary file already exists, something went wrong
    if os.path.isfile(temporary_file_path):
        raise Exception("The file: " + temporary_file_path + " should not exist. Either a previous computation was interrupted or another computation is ongoing.")
    
    # check if correction of r-peaks already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = results_path,
        dictionary_entry = "RRI",
        additionally_remove_entries = ["RRI_frequency"]
    )

    # cancel if needed data is missing
    if user_answer == "no_file_found":
        print("\nFile containing detected r-peaks not found. As they are needed to calcualte the RR-Intervals, the calculation will be skipped.")
        return

    # create lists to store unprocessable files
    unprocessable_files = []

    # create variables to track progress
    total_files = get_pickle_length(results_path, "RRI")
    progressed_files = 0
    start_time = time.time()

    if total_files > 0:
        print("\nCalculating RR-Intervals from r-peaks detected by %s in %i files from \"%s\":" % (rpeak_function_name, total_files, data_directory))
    else:
        return
    
    # load results
    results_generator = load_from_pickle(results_path)
    
    # correct rpeaks
    for generator_entry in results_generator:
        # skip if rri values already exist and the user does not want to override
        if user_answer == "n" and "RRI" in generator_entry.keys():
            append_to_pickle(generator_entry, temporary_file_path)
            continue

        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            # get the file name
            file_name = generator_entry["file_name"]

            # get the ecg sampling frequency
            ecg_sampling_frequency = generator_entry["ECG_frequency"]

            # get the valid regions for the ECG data
            valid_regions = generator_entry["valid_ecg_regions"]

            # get the r-peaks
            rpeaks = generator_entry[rpeak_function_name]

            # create a list to store the rri for the valid regions
            rri = []

            for valid_interval in valid_regions:
                # retrieve the next closest time point which is a multiple of the rri_sampling_frequency / ecg_sampling_frequency
                this_time_point = find_time_point_shared_by_signals(
                    signal_position = copy.deepcopy(valid_interval[0]),
                    signal_sampling_frequency = ecg_sampling_frequency,
                    other_sampling_frequencies = [RRI_sampling_frequency, 1 / mad_time_period_seconds]
                )

                this_length = valid_interval[1] - valid_interval[0]

                # get the rpeaks in the valid interval and shift them to the start of the interval to ensure the rri is calculated correctly
                this_rpeaks = np.array([peak for peak in rpeaks if valid_interval[0] <= peak <= valid_interval[1]])
                this_rpeaks = this_rpeaks - this_time_point

                # calculate the rri
                this_rri = calculate_rri_from_peaks(
                    rpeaks = this_rpeaks, # type: ignore
                    ecg_sampling_frequency = ecg_sampling_frequency,
                    target_sampling_frequency = RRI_sampling_frequency,
                    signal_length = this_length,
                    pad_with = pad_with
                )

                # correct rri values which are outside of the realistic range
                for i in range(len(this_rri)):
                    if this_rri[i] < realistic_rri_value_range[0] or this_rri[i] > realistic_rri_value_range[1]:
                        this_rri[i] = pad_with

                rri.append(this_rri)
        
            # add the rri to the dictionary
            generator_entry["RRI"] = rri
            generator_entry["RRI_frequency"] = RRI_sampling_frequency

        except:
            unprocessable_files.append(file_name)
        
        append_to_pickle(generator_entry, temporary_file_path)
    
    progress_bar(progressed_files, total_files, start_time)

    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        if os.path.isfile(results_path):
            os.remove(results_path)
        os.rename(temporary_file_path, results_path)

    # print unprocessable files
    if len(unprocessable_files) > 0:
        print("\nFor the following " + str(len(unprocessable_files)) + " files the rri could not be calculated from the r-peaks:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- Dictionary keys that access the file name and/or r-peaks do not exist in the results. Check keys in file or recalculate them.")
        print(" "*5 + "- .edf file contains format errors")


def rri_comparison_report(
        rri_comparison_report_dezimal_places: int,
        rri_comparison_report_path: str,
        rri_differences: list, 
        file_names: list
    ):
    """
    Saves results of the RRI comparison to a text file.

    ARGUMENTS:
    --------------------------------
    rri_comparison_report_dezimal_places: int
        number of decimal places to which the RRI differences are rounded
    rri_comparison_report_path: str
        path to the text file where the comparison report is saved
    rri_differences: list
        list of average relative difference between the RRI values for each file
    file_names: list
        list of file names for which the RRI values were compared
    """

    # open the file to write the report to
    comparison_file = open(rri_comparison_report_path, "w")

    # write the header
    message = "Comparison of RRI Calculation"
    comparison_file.write(message + "\n")
    comparison_file.write("="*len(message) + "\n\n")

    rri_difference_column_header = "RRI Difference"
    file_name_column_header = "File Name"

    rri_difference_column = [rri_difference_column_header]
    rri_difference_column.append(print_smart_rounding(np.mean(rri_differences), rri_comparison_report_dezimal_places)) # type: ignore
    for rri_difference in rri_differences:
        rri_difference_column.append(print_smart_rounding(rri_difference, rri_comparison_report_dezimal_places)) # type: ignore

    file_name_column = [file_name_column_header]
    file_name_column.append("Mean")
    for file_name in file_names:
        file_name_column.append(file_name)
    
    rri_difference_column_length = max([len(column) for column in rri_difference_column])
    file_name_column_length = max([len(column) for column in file_name_column])
    
    # write the columns
    for i in range(len(rri_difference_column)):
        message = " " + print_in_middle(file_name_column[i], file_name_column_length) + " | " + print_in_middle(rri_difference_column[i], rri_difference_column_length) + " "
        comparison_file.write(message + "\n")
        if i < len(rri_difference_column)-1:
            comparison_file.write("-"*len(message) + "\n")
    
    comparison_file.close()


def rri_comparison(
        path_to_h5file: str,
        results_path: str,
        mad_time_period_seconds: int,
        rri_comparison_report_dezimal_places: int,
        rri_comparison_report_path: str,
    ):
    """
    Compares RRI values calculated and stored in 'results_path' to the available RRI values accessable in 
    'path_to_h5file'.

    For every file, the average relative difference between the RRI values is calculated and saved to a text 
    file.

    ARGUMENTS:
    --------------------------------
    path_to_h5file: str
        path to the h5 file where the available RRI values are stored
    results_path: str
        path to the pickle file where the RRI values are saved
    1 / mad_time_period_seconds: float
        target sampling frequency of the MAD values
    rri_comparison_report_dezimal_places: int
        number of decimal places to which the RRI differences are rounded
    rri_comparison_report_path: str
        path to the text file where the comparison report is saved
    
    RETURNS:
    --------------------------------
    None, but the comparison report is saved to a text file
    """

    # check if the report already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_file(file_path = rri_comparison_report_path,
            message = "\nRRI comparison report already exists in " + rri_comparison_report_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # access the dataset which provides the valid regions
    h5_dataset = h5py.File(path_to_h5file, 'r')

    # accessing patient ids and rri frequency:
    patients = list(h5_dataset['rri'].keys()) # type: ignore
    available_rri_frequency = h5_dataset["rri"].attrs["freq"]

    # load calculated results
    results_generator = load_from_pickle(results_path)

    rri_difference = list()
    processed_files = list()

    # create variables to track progress
    total_files = get_pickle_length(results_path, " ")
    progressed_files = 0
    start_time = time.time()

    # create lists to store unprocessable files
    unprocessable_files = []
    
    print("\nComparing calculated to available RRI values for %i files:" % (total_files))

    for generator_entry in results_generator:
        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            file_name = generator_entry["file_name"]
            calculated_rri_frequency = generator_entry["RRI_frequency"]
            calculated_mad_frequency = 1 / mad_time_period_seconds

            if calculated_rri_frequency != available_rri_frequency:
                raise ValueError
            
            # get available RRI values
            patient_id = file_name[:5]
            available_rri = np.array(h5_dataset["rri"][patient_id]) # type: ignore

            # load ecg sampling frequency
            ecg_sampling_frequency = generator_entry["ECG_frequency"]

            # get valid regions
            valid_regions = generator_entry["valid_ecg_regions"]

            # get calculated RRI values
            RRI_values = generator_entry["RRI"]

            # the rri values were only calculated in valid regions, so we need to calculate the starting point and length 
            # for each region
            start_of_valid_regions = []
            for i in np.arange(0, len(valid_regions)):
                this_time_point = find_time_point_shared_by_signals(
                    signal_position = copy.deepcopy(valid_regions[i][0]),
                    signal_sampling_frequency = ecg_sampling_frequency,
                    other_sampling_frequencies = [calculated_rri_frequency, calculated_mad_frequency]
                )
                start_of_valid_regions.append(int(this_time_point/ecg_sampling_frequency*calculated_rri_frequency))

            datapoints_in_regions = [len(rri_value_region) for rri_value_region in RRI_values]

            # fuse the RRI values to one list
            calculated_rri = []
            for rri_value_region in RRI_values:
                calculated_rri.extend(rri_value_region)
            calculated_rri = np.array(calculated_rri)

            # for some reason the available ones are longer than the original ECG data, so we will shift them
            differences = list()
            length_difference = 200

            for i in np.arange(0, length_difference):
                collect_available_rri = []
                for j in np.arange(0, len(valid_regions)):
                    collect_available_rri.extend(available_rri[start_of_valid_regions[j]+i:start_of_valid_regions[j]+i+datapoints_in_regions[j]])
                collect_available_rri = np.array(collect_available_rri)

                this_difference = np.mean(np.abs(collect_available_rri - calculated_rri))
                this_max = np.array([max(abs(collect_available_rri[j]), abs(calculated_rri[j]), 0.0000001) for j in np.arange(len(calculated_rri))]) # type: ignore
                differences.append(this_difference / this_max)
            
            rri_difference.append(np.min(differences))
            processed_files.append(file_name)

        except:
            unprocessable_files.append(file_name)
        
    progress_bar(progressed_files, total_files, start_time)

    # print unprocessable files 
    if len(unprocessable_files) > 0:
        print("\nFor the following " + str(len(unprocessable_files)) + " files the RRI values could not be compared:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- Corresponding comparison file not available in the dataset")
        print(" "*5 + "- Error occured during comparing the RRI values")
    
    # write comparison report
    print("\nWriting report for RRI Comparison...")

    rri_comparison_report(
        rri_comparison_report_dezimal_places = rri_comparison_report_dezimal_places,
        rri_comparison_report_path = rri_comparison_report_path,
        rri_differences = rri_difference, 
        file_names = processed_files
    )