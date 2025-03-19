"""
Author: Johannes Peter Knoll

Python implementation of mean amplitude deviation (MAD) calculation for movement acceleration data.
"""

# IMPORTS
import numpy as np

# LOCAL IMPORTS
from side_functions import *
import read_edf


def check_mad_conditions(
        acceleration_data_lists: list, 
        frequencies: list, 
    ):
    """
    Check if the data is valid for MAD calculation (uniform frequency and data points).

    ARGUMENTS:
    --------------------------------
    acceleration_data_lists: list
        list of acceleration data arrays (x, y, z - axis)
    frequencies: list
        list of sampling frequencies of the acceleration data arrays

    NO RETURN VALUE: raises ValueError if conditions are not met
    """

    #check if frequencies are the same
    uniform_frequency = frequencies[0]
    for freq in frequencies:
        if freq != uniform_frequency:
            raise ValueError("Frequencies are not the same. Calculation of MAD requires the same frequency for all data arrays.")

    #check if the data arrays are the same length
    uniform_length = len(acceleration_data_lists[0])
    for data_axis in acceleration_data_lists:
        if len(data_axis) != uniform_length:
            raise ValueError("Data arrays are not the same length. Calculation of MAD requires the same length for all data arrays.")


def calc_mad_in_interval(
        acceleration_data_lists: list, 
        start_position: int, 
        end_position: int, 
    ):
    """
    Calculate MAD in a given time frame.
        current_acceleration = root(x^2 + y^2 + z^2)
        average_acceleration = sum(current_acceleration) / interval_size
        MAD = sum(abs(current_acceleration - average_acceleration)) / interval_size

    ARGUMENTS:
    --------------------------------
    acceleration_data_lists: list
        list of acceleration data arrays (x, y, z - axis)
    start_position: int
        start position of the interval
    end_position: int
        end position of the interval

    RETURNS: 
    --------------------------------
    MAD: float
        mean amplitude deviation of the movement acceleration data in the given interval
    """
    #calculate average acceleration over given interval (start_position to end_position)
    interval_size = end_position - start_position
    average_acceleration = 0
    for i in np.arange(start_position, end_position):
        current_acceleration = 0
        for data_axis in acceleration_data_lists:
            current_acceleration += data_axis[i]**2
        current_acceleration = current_acceleration**0.5
        average_acceleration += current_acceleration
    average_acceleration /= interval_size

    #calculate MAD
    mad = 0
    for i in np.arange(start_position, end_position):
        current_acceleration = 0
        for data_axis in acceleration_data_lists:
            current_acceleration += data_axis[i]**2
        current_acceleration = current_acceleration**0.5
        mad += abs(current_acceleration - average_acceleration)
    mad /= interval_size

    return mad


def calc_mad(
        acceleration_data_lists: list,
        frequencies: list, 
        time_period: int,
    ):
    """
    Calculate mean amplitude deviation (MAD) of movement acceleration data.

    ARGUMENTS:
    --------------------------------
    acceleration_data_lists: list
        list of acceleration data arrays (x, y, z - axis)
    frequencies: list
        list of sampling frequencies of the acceleration data arrays
    time_period: int
        length of the time period in seconds over which the MAD will be calculated
    wrist_acceleration_keys: list
        list of keys of data dictionary that are relevant for MAD calculation

    RETURNS:
    --------------------------------
    MAD: list
        list of MAD values for each interval: MAD[i] = MAD in interval i
    """
    #check if data is valid
    check_mad_conditions(
        acceleration_data_lists = acceleration_data_lists,
        frequencies = frequencies
        )

    #transform time_period to number of samples
    number_of_samples = int(time_period * frequencies[0])

    #calculate MAD in intervals
    MAD = []
    for i in np.arange(0, len(acceleration_data_lists[0]), number_of_samples):
        upper_border = i + number_of_samples
        if upper_border > len(acceleration_data_lists[0]):
            upper_border = len(acceleration_data_lists[0])
        MAD.append(calc_mad_in_interval(
            acceleration_data_lists = acceleration_data_lists,
            start_position = i,
            end_position = upper_border)
            )

    return np.array(MAD)


def calculate_MAD_in_acceleration_data(
        data_directory: str,
        valid_file_types: list,
        wrist_acceleration_keys: list,
        physical_dimension_correction_dictionary: dict,
        mad_time_period_seconds: int,
        results_path: str,
    ):
    """
    Calculate the MAD values for the wrist acceleration data for all valid files in the
    data directory and save them to a pickle file.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    wrist_acceleration_keys: list
        keys for the wrist acceleration data in the data dictionary
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    mad_time_period_seconds: int
        time period in seconds over which the MAD will be calculated
    results_path: str
        path to the pickle file where the MAD values are saved

    RETURNS:
    --------------------------------
    None, but the MAD values are saved to a pickle file as a dictionary in the following
    format:
        {
            "file_name": name of file 1,
            "MAD": MAD values for file 1,
            ...
        }
        ...
    """
    
    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(results_path) + "computation_in_progress.pkl"

    # if the temporary file already exists, something went wrong
    if os.path.isfile(temporary_file_path):
        raise Exception("The file: " + temporary_file_path + " should not exist. Either a previous computation was interrupted or another computation is ongoing.")

    # check if MAD values already exist and if yes ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = results_path,
        dictionary_entry = "MAD",
        additionally_remove_entries = ["MAD_frequency"]
    )
    
    # create list to store unprocessable files
    unprocessable_files = []

    # get all valid files
    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]
   
    # check how many files in the results are left to process
    left_overs = 0
    if not user_answer == "no_file_found":
        # load existing results
        results_generator = load_from_pickle(results_path)

        for generator_entry in results_generator:
                # check if needed dictionary keys exist
                if "file_name" not in generator_entry.keys():
                    continue

                if "MAD" not in generator_entry.keys():
                    left_overs += 1

                # get current file name
                file_name = generator_entry["file_name"]

                if file_name in valid_files:
                    valid_files.remove(file_name)
        
        del results_generator
    
    # create variables to track progress
    total_files = len(valid_files) + left_overs
    progressed_files = 0
    start_time = time.time()

    if total_files > 0:
        print("\nCalculating MAD in the wrist acceleration data in %i files from \"%s\":" % (total_files, data_directory))
    
    # if results file already exists and none of its entries are left to process or reprocess, rename the file 
    # and continue with remaining valid files that were not processed yet
    if not user_answer == "no_file_found" and left_overs == 0:
        os.rename(results_path, temporary_file_path)

    if not user_answer == "no_file_found" and left_overs > 0:
        # load existing results
        results_generator = load_from_pickle(results_path)

        # calculate MAD in the wrist acceleration data
        for generator_entry in results_generator:
            # skip if MAD values already exist and the user does not want to override
            if "MAD" in generator_entry.keys() and user_answer == "n":
                append_to_pickle(generator_entry, temporary_file_path)
                continue

            # show progress
            progress_bar(progressed_files, total_files, start_time)
            progressed_files += 1

            try:
                # get current file name
                file_name = generator_entry["file_name"]

                if file_name in valid_files:
                    valid_files.remove(file_name)

                # create lists to save the acceleration data and frequencies for each axis
                acceleration_data = []
                acceleration_data_frequencies = []

                # try to load the data and correct the physical dimension if needed
                # (get the acceleration data and frequency for each axis)
                for possible_axis_keys in wrist_acceleration_keys:
                    this_axis_signal, this_axis_frequency = read_edf.get_data_from_edf_channel(
                        file_path = data_directory + file_name,
                        possible_channel_labels = possible_axis_keys,
                        physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                    )

                    # append data to corresponding lists
                    acceleration_data.append(this_axis_signal)
                    acceleration_data_frequencies.append(this_axis_frequency)
                
                # calculate MAD values
                this_MAD_values = calc_mad(
                    acceleration_data_lists = acceleration_data,
                    frequencies = acceleration_data_frequencies,
                    time_period = mad_time_period_seconds, 
                    )
                
                # calculate sampling frequency
                mad_sampling_frequency = 1 / mad_time_period_seconds
                if int(mad_sampling_frequency) == mad_sampling_frequency:
                    mad_sampling_frequency = int(mad_sampling_frequency)
                
                # save MAD values
                generator_entry["MAD"] = this_MAD_values

                generator_entry["MAD_frequency"] = mad_sampling_frequency

            except:
                unprocessable_files.append(file_name)
            
            append_to_pickle(generator_entry, temporary_file_path)
    
    # calculate the MAD values for the remaining files
    for file_name in valid_files:
        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        generator_entry = {"file_name": file_name}

        try:
            # create lists to save the acceleration data and frequencies for each axis
            acceleration_data = []
            acceleration_data_frequencies = []

            # try to load the data and correct the physical dimension if needed
            # (get the acceleration data and frequency for each axis)
            for possible_axis_keys in wrist_acceleration_keys:
                this_axis_signal, this_axis_frequency = read_edf.get_data_from_edf_channel(
                    file_path = data_directory + file_name,
                    possible_channel_labels = possible_axis_keys,
                    physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                )

                # append data to corresponding lists
                acceleration_data.append(this_axis_signal)
                acceleration_data_frequencies.append(this_axis_frequency)
            
            # calculate MAD values
            this_MAD_values = calc_mad(
                acceleration_data_lists = acceleration_data,
                frequencies = acceleration_data_frequencies,
                time_period = mad_time_period_seconds, 
                )
            
            # calculate sampling frequency
            mad_sampling_frequency = 1 / mad_time_period_seconds
            if int(mad_sampling_frequency) == mad_sampling_frequency:
                mad_sampling_frequency = int(mad_sampling_frequency)
            
            # save MAD values for this file to the dictionary
            generator_entry["MAD"] = this_MAD_values # type: ignore    
            generator_entry["MAD_frequency"] = mad_sampling_frequency # type: ignore

        except:
            unprocessable_files.append(file_name)
        
        # if more than the file name is in the dictionary, save the dictionary to the pickle file
        if len(generator_entry) > 1:
            append_to_pickle(generator_entry, temporary_file_path)
    
    progress_bar(progressed_files, total_files, start_time)

    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        if os.path.isfile(results_path):
            os.remove(results_path)
        os.rename(temporary_file_path, results_path)

    # print unprocessable files
    if len(unprocessable_files) > 0:
        print("\nFor the following " + str(len(unprocessable_files)) + " files the MAD values could not be calculated:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- Wrist Acceleration channel labels within .edf file are not in the list of possible wrist acceleration keys or labels were not added to physical dimension correction dictionary. (See project_parameters.py – line 39 and 72.)")
        print(" "*5 + "- .edf file contains format errors")
        print(" "*5 + "- No matching label in wrist_acceleration_keys and the files")
        print(" "*5 + "- Physical dimension of label is unknown")
        print(" "*5 + "- Error during calculating of MAD values")


import h5py
import matplotlib.pyplot as plt


def mad_comparison_report(
        mad_comparison_report_dezimal_places: int,
        mad_comparison_report_path: str,
        mad_differences: list, 
        file_names: list
    ):
    """
    Saves results of the MAD comparison to a text file.

    ARGUMENTS:
    --------------------------------
    mad_comparison_report_dezimal_places: int
        number of decimal places to which the MAD differences are rounded
    mad_comparison_report_path: str
        path to the text file where the comparison report is saved
    mad_differences: list
        list of average relative difference between the mad values for each file
    file_names: list
        list of file names for which the MAD values were compared
    """

    # open the file to write the report to
    comparison_file = open(mad_comparison_report_path, "w")

    # write the header
    message = "Comparison of MAD Calculation"
    comparison_file.write(message + "\n")
    comparison_file.write("="*len(message) + "\n\n")

    mad_difference_column_header = "MAD Difference"
    file_name_column_header = "File Name"

    mad_difference_column = [mad_difference_column_header]
    mad_difference_column.append(print_smart_rounding(np.mean(mad_differences), mad_comparison_report_dezimal_places)) # type: ignore
    for mad_difference in mad_differences:
        mad_difference_column.append(print_smart_rounding(mad_difference, mad_comparison_report_dezimal_places)) # type: ignore

    file_name_column = [file_name_column_header]
    file_name_column.append("Mean")
    for file_name in file_names:
        file_name_column.append(file_name)
    
    mad_difference_column_length = max([len(column) for column in mad_difference_column])
    file_name_column_length = max([len(column) for column in file_name_column])
    
    # write the columns
    for i in range(len(mad_difference_column)):
        message = " " + print_in_middle(file_name_column[i], file_name_column_length) + " | " + print_in_middle(mad_difference_column[i], mad_difference_column_length) + " "
        comparison_file.write(message + "\n")
        if i < len(mad_difference_column)-1:
            comparison_file.write("-"*len(message) + "\n")
    
    comparison_file.close()


def mad_comparison(
        path_to_h5file: str,
        results_path: str,
        mad_comparison_report_dezimal_places: int,
        mad_comparison_report_path: str,
    ):
    """
    Compares MAD values calculated and stored in 'results_path' to the available MAD values accessable in 
    'path_to_h5file'.

    For every file, the average relative difference between the MAD values is calculated and saved to a text 
    file.

    ARGUMENTS:
    --------------------------------
    path_to_h5file: str
        path to the h5 file where the available MAD values are stored
    results_path: str
        path to the pickle file where the MAD values are saved
    mad_comparison_report_dezimal_places: int
        number of decimal places to which the MAD differences are rounded
    mad_comparison_report_path: str
        path to the text file where the comparison report is saved
    
    RETURNS:
    --------------------------------
    None, but the comparison report is saved to a text file
    """

    # check if the report already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_file(file_path = mad_comparison_report_path,
            message = "\nMAD comparison report already exists in " + mad_comparison_report_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # access the dataset which provides the valid regions
    h5_dataset = h5py.File(path_to_h5file, 'r')

    # accessing patient ids and rri frequency:
    patients = list(h5_dataset['mad'].keys()) # type: ignore
    available_rri_frequency = h5_dataset["mad"].attrs["freq"]

    # load calculated results
    results_generator = load_from_pickle(results_path)

    mad_difference = list()
    processed_files = list()

    # create variables to track progress
    total_files = get_pickle_length(results_path, " ")
    progressed_files = 0
    start_time = time.time()

    # create lists to store unprocessable files
    unprocessable_files = []
    
    print("\nComparing calculated to available MAD values for %i files:" % (total_files))

    for generator_entry in results_generator:
        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            file_name = generator_entry["file_name"]

            if generator_entry["MAD_frequency"] != available_rri_frequency:
                raise ValueError
            
            # get available MAD values
            patient_id = file_name[:5]
            available_mad = np.array(h5_dataset["mad"][patient_id]) # type: ignore

            # get calculated MAD values
            MAD_values = np.array(generator_entry["MAD"])

            # for some reason the available ones are longer than the original ECG data, so we will shift them
            differences = list()
            length_difference = len(available_mad)-len(MAD_values)

            for i in np.arange(0, length_difference):
                this_difference = np.mean(np.abs(available_mad[i:i+len(MAD_values)] - MAD_values))
                this_max = np.array([max(abs(available_mad[j]), abs(MAD_values[j-i])) for j in np.arange(i, i+len(MAD_values))]) # type: ignore
                differences.append(this_difference / this_max)
            
            mad_difference.append(np.min(differences))
            processed_files.append(file_name)

        except:
            unprocessable_files.append(file_name)
        
    progress_bar(progressed_files, total_files, start_time)

    # print unprocessable files 
    if len(unprocessable_files) > 0:
        print("\nFor the following " + str(len(unprocessable_files)) + " files the MAD values could not be compared:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- Corresponding comparison file not available in the dataset")
        print(" "*5 + "- Error occured during comparing the MAD values")
    
    # write comparison report
    print("\nWriting report for MAD Comparison...")

    mad_comparison_report(
        mad_comparison_report_dezimal_places = mad_comparison_report_dezimal_places,
        mad_comparison_report_path = mad_comparison_report_path,
        mad_differences = mad_difference, 
        file_names = processed_files
    )