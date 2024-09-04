# LOCAL IMPORTS
import read_edf
from side_functions import *


def calculate_rri_from_peaks(
        rpeaks: list,
        ecg_sampling_frequency: int,
        target_sampling_frequency: int,
        signal_length: int
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
    
    RETURNS:
    --------------------------------
    rri: list
        list of RR-intervals
    """

    number_rri_entries = int(signal_length / ecg_sampling_frequency * target_sampling_frequency)

    rpeaks = np.array(rpeaks) # type: ignore
    rpeak_position_seconds = rpeaks / ecg_sampling_frequency # type: ignore

    rri = []

    for i in range(number_rri_entries):
        rri_datapoint_second = i / target_sampling_frequency

        this_rri = 0
        for j in range(1, len(rpeak_position_seconds)):
            if rpeak_position_seconds[j-1] <= rri_datapoint_second and rri_datapoint_second <= rpeak_position_seconds[j]:
                this_rri = rpeak_position_seconds[j] - rpeak_position_seconds[j-1]
                break
            if rri_datapoint_second > rpeak_position_seconds[j]:
                break
        
        rri.append(this_rri)
    
    return rri


def get_rri_from_peaks(
        data_directory: str,
        ecg_keys: list,
        physical_dimension_correction_dictionary: dict,
        rpeak_function_name: str,
        rri_sampling_frequency: int,
        preparation_results_path: str,
        file_name_dictionary_key: str,
        rri_dictionary_key: str,
    ):
    """
    Detected r-peaks can be corrected using the wfdb library. This is useful if the
    detected r-peaks are shifted by a few samples. It also makes the comparison of
    different r-peak detection methods easier.

    (The peak direction depends on how the heart beats in direction to the electrodes.
    Therefore it can be different for different data sets, but is always the same within
    on set of data.) 

    Therefore we let the library decide on the direction of the peaks.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    ecg_keys: list
        list of possible labels for the ECG data
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    rpeak_function_name: str
        name of the r-peak detection function
    rri_sampling_frequency: int
        target sampling frequency of the RR-intervals
    preparation_results_path: str
        path to the pickle file where the valid regions are saved
    file_name_dictionary_key
        dictionary key to access the file name
    rri_dictionary_key: str
        dictionary key to access the RR-intervals

    RETURNS:
    --------------------------------
    None, but the rpeaks are saved as dictionaries to a pickle file in the following format:
    {
        file_name_dictionary_key: file_name_1,
        rri_dictionary_key: rri_1,
        ...
    }
        ...
    """
    
    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(preparation_results_path) + "computation_in_progress.pkl"

    # if the temporary file already exists, it means a previous computation was interrupted
    # ask the user if the results should be overwritten or recovered
    if os.path.isfile(temporary_file_path):
        recover_results_after_error(
            all_results_path = preparation_results_path, 
            some_results_with_updated_keys_path = temporary_file_path, 
            file_name_dictionary_key = file_name_dictionary_key,
        )
    
    # check if correction of r-peaks already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = preparation_results_path,
        dictionary_entry = rri_dictionary_key
    )

    # cancel if needed data is missing
    if user_answer == "no_file_found":
        print("\nFile containing detected r-peaks not found. As they are needed to correct them in the first place, the correction will be skipped.")
        return

    # create lists to store unprocessable files
    unprocessable_files = []

    # load preparation results
    preparation_results_generator = load_from_pickle(preparation_results_path)
    
    # create variables to track progress
    start_time = time.time()
    total_files = get_pickle_length(preparation_results_path, rri_dictionary_key)
    progressed_files = 0

    if total_files > 0:
        print("\nCalculating RR-Intervals from r-peaks detected by %s in %i files:" % (rpeak_function_name, total_files))
    
    # correct rpeaks
    for generator_entry in preparation_results_generator:
        # skip if corrected r-peaks already exist and the user does not want to override
        if user_answer == "n" and rri_dictionary_key in generator_entry.keys():
            append_to_pickle(generator_entry, temporary_file_path)
            continue

        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            # get the valid regions for the ECG data and file name
            file_name = generator_entry[file_name_dictionary_key]

            # try to load the data and correct the physical dimension if needed
            ecg_signal, ecg_sampling_frequency = read_edf.get_data_from_edf_channel(
                file_path = data_directory + file_name,
                possible_channel_labels = ecg_keys,
                physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
            )

            # get the r-peaks
            rpeaks = generator_entry[rpeak_function_name]
            rri = calculate_rri_from_peaks(
                rpeaks = rpeaks,
                ecg_sampling_frequency = ecg_sampling_frequency,
                target_sampling_frequency = rri_sampling_frequency,
                signal_length = len(ecg_signal)
            )
        
            # add the rri to the dictionary
            generator_entry[rri_dictionary_key] = rri

        except:
            unprocessable_files.append(file_name)
        
        append_to_pickle(generator_entry, temporary_file_path)
    
    progress_bar(progressed_files, total_files, start_time)

    # rename the file that stores the calculated data
    if os.path.isfile(temporary_file_path):
        os.remove(preparation_results_path)
        os.rename(temporary_file_path, preparation_results_path)

    # print unprocessable files
    if len(unprocessable_files) > 0:
        print("\nFor the following " + str(len(unprocessable_files)) + " files the rri could not be calculated from the r-peaks:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- Dictionary keys that access the file name and/or r-peaks do not exist in the results. Check keys in file or recalculate them.")
        print(" "*5 + "- .edf file contains format errors")
