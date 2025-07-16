"""
Author: Johannes Peter Knoll

This file contains functions that are used to read EDF files.
"""

# IMPORTS
import pyedflib # https://github.com/holgern/pyedflib
import numpy as np
import os

# LOCAL IMPORTS
from side_functions import *

# https://github.com/holgern/pyedflib


def correct_physical_dimension(signal_key, signal_dimension, dimension_correction_dict):
    """
    Corrects the physical dimensions of the signals. This is important as the physical
    dimensions of the signals might not always be the same in all files.

    ARGUMENTS:
    --------------------------------
    signal_key: str
        label of the signal
    signal_dimension: str
        physical dimension of the signal
    dimension_correction_dict: 
        dictionary containing all possible signal labels as keys and a dictionary as value 
        dictionary value has the following structure:
            {
                "possible_dimensions": list,
                "dimension_correction": list
            }
    
    RETURNS:
    --------------------------------
    correction_value: float
        value which should be multiplied to the signal for correction
    """
    for key in dimension_correction_dict:
        if key == signal_key:
            signal_dim_index = dimension_correction_dict[key]["possible_dimensions"].index(signal_dimension)
            correction_value = dimension_correction_dict[key]["dimension_correction"][signal_dim_index]
            break
    
    return correction_value


def get_data_from_edf_channel(
        file_path: str, 
        possible_channel_labels: list, 
        physical_dimension_correction_dictionary: dict
    ):
    """
    Reads the signal, frequency and physical dimension from an EDF file.
     
    The labels for channels are not consistent in all EDF files. Therefore we first need 
    to find the correct label for the channel in this file.

    After reading the signal, the function checks the physical dimension and corrects the
    signal if the physical dimension is off.
      
    Returns the corrected signal and sampling frequency.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file
    possible_channel_labels: list
        list of possible labels for the signal
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals

    
    RETURNS:
    --------------------------------
    signal: np.array
        signal from the channel
    sample_frequency: int
        frequency of the signal
    """
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()

    for i in np.arange(n):
        if signal_labels[i] in possible_channel_labels:
            channel = signal_labels[i]
            break

    for i in np.arange(n):
        if signal_labels[i] == channel:
            signal = f.readSignal(i)
            sample_frequency = f.getSampleFrequency(i)
            physical_dimension = f.getPhysicalDimension(i)
            break
    f._close()

    signal = np.array(signal)
    dimension_correction_value = correct_physical_dimension(
        signal_key = channel,
        signal_dimension = physical_dimension,
        dimension_correction_dict = physical_dimension_correction_dictionary
    )

    signal = signal * dimension_correction_value

    return signal, sample_frequency


def get_data_length_from_edf_channel(file_path: str, possible_channel_labels: list):
    """
    Reads the length of the signal from an EDF file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file
    possible_channel_labels: list
        list of possible labels for the signal
    
    RETURNS:
    --------------------------------
    data_length: int
        length of the signal
    """
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()

    for i in np.arange(n):
        if signal_labels[i] in possible_channel_labels:
            channel = signal_labels[i]

    for i in np.arange(n):
        if signal_labels[i] == channel:
            data_length = f.getNSamples()[i]
            break
    f._close()

    return data_length


def get_frequency_from_edf_channel(file_path: str, possible_channel_labels: list):
    """
    Reads the frequency of the channel from an EDF file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file
    possible_channel_labels: list
        list of possible labels for the signal
    
    RETURNS:
    --------------------------------
    sample_frequency: int
        frequency of the signal
    """
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()

    for i in np.arange(n):
        if signal_labels[i] in possible_channel_labels:
            channel = signal_labels[i]

    for i in np.arange(n):
        if signal_labels[i] == channel:
            sample_frequency = f.getSampleFrequency(i)
            break
    f._close()

    return sample_frequency


def get_header_from_edf_file(file_path: str):
    """
    Reads the header of the EDF file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file
    
    RETURNS:
    --------------------------------
    header: dict
        dictionary containing the header information
    """
    f = pyedflib.EdfReader(file_path)

    try:
        signals_in_file = f.signals_in_file
    except:
        signals_in_file = None
    
    try:
        file_duration = f.file_duration
    except:
        file_duration = None
    
    try:
        datarecord_duration = f.datarecord_duration
    except:
        datarecord_duration = None
    
    try:
        start_date = str(f.getStartdatetime().year) + "-" + str(f.getStartdatetime().month) + "-" + str(f.getStartdatetime().day)
    except:
        start_date = None
    
    try:
        start_time = str(f.getStartdatetime().hour) + ":" + str(f.getStartdatetime().minute) + ":" + str(f.getStartdatetime().second)
    except:
        start_time = None
    
    try:
        patient_code = f.getPatientCode()
    except:
        patient_code = None

    try:
        gender = f.getSex()
    except:
        gender = None
    
    try:
        birthdate = f.getBirthdate()
    except:
        birthdate = None

    try:
        patient_name = f.getPatientName()
    except:
        patient_name = None

    f._close()
    
    header = {
        "signals_in_file": signals_in_file,
        "file_duration": file_duration,
        "datarecord_duration": datarecord_duration,
        "start_date": start_date,
        "start_time": start_time,
        "patient_code": patient_code,
        "gender": gender,
        "birthdate": birthdate,
        "patient_name": patient_name
    }

    return header


"""
Following functions are not needed for the final implementation, but were required by me
to get an overview of the data.
"""


def get_dimensions_and_signal_labels(directory, valid_file_types = [".edf"]):
    """
    Collects the physical dimensions and signal labels from all valid files in the 
    directory.

    ARGUMENTS:
    --------------------------------
    directory: str
        path to the directory
    
    RETURNS:
    --------------------------------
    all_signal_labels: list
        list of all signal labels
    all_physical_dimensions: list
        list of lists containing the physical dimensions of the signals
    """
   
    # get all valid files in the directory
    all_files = os.listdir(directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    # create lists to store the signal labels and their physical dimensions
    all_signal_labels = []
    all_physical_dimensions = []

    # variables to track the progress
    total_files = len(valid_files)
    progressed_files = 0
    error_in_files = []
    start_time = time.time()

    print("Reading signal labels and their physical dimensions from %i files:" % total_files)
    for file in valid_files:
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            f = pyedflib.EdfReader(directory + file)

            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            sigdims = dict()

            for i in np.arange(n):
                sigdims[signal_labels[i]] = f.getPhysicalDimension(i)
            f._close()
        except:
            error_in_files.append(file)
            continue

        for key in sigdims:
            if key not in all_signal_labels:
                all_signal_labels.append(key)
                all_physical_dimensions.append([])
            key_to_index = all_signal_labels.index(key)
            if sigdims[key] not in all_physical_dimensions[key_to_index]:
                all_physical_dimensions[key_to_index].append(sigdims[key])
    
    progress_bar(progressed_files, total_files, start_time)

    if len(error_in_files) > 0:
        print("Due to an error in the following files, they could not be read and were skipped:")
        print(error_in_files)
    
    return all_signal_labels, all_physical_dimensions


def read_out_channel(
        data_directory: str,
        valid_file_types: list,
        channel_key_to_read_out: list,
        physical_dimension_correction_dictionary: dict,
        results_path: str,
        new_dictionary_key = None
    ):
    """
    Read out and save the channels to the results path.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    keys_to_read_out: list
        list containing possible keys that refer to the same signal
    physical_dimension_correction_dictionary: dict
        dictionary needed to check and correct the physical dimension of all signals
    results_path: str
        path to the pickle file where the valid regions are saved
    new_dictionary_keys: list
        new dictionary key to store the read out signal
        if None, the key is the same as the channel key to read out

    RETURNS:
    --------------------------------
    None, but the channels are saved to the results path:
        {
            "file_name": file_name_1,
            channel_key_1: channel_signal_1_1,
            channel_key_2: channel_signal_1_2,
        }
            ...
    """

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(results_path) + "computation_in_progress.pkl"

    # if the temporary file already exists, something went wrong
    if os.path.isfile(temporary_file_path):
        raise Exception("The file: " + temporary_file_path + " should not exist. Either a previous computation was interrupted or another computation is ongoing.")
    
    if new_dictionary_key is None:
        new_dictionary_key = channel_key_to_read_out[0]
    
    # check if channel key already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override_dictionary_entry(
        file_path = results_path,
        dictionary_entry = new_dictionary_key
    )
    
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
        results_generator = load_from_pickle(results_path)

        for generator_entry in results_generator:
                # check if needed dictionary keys exist
                if "file_name" not in generator_entry.keys():
                    continue

                if new_dictionary_key not in generator_entry.keys():
                    store_previous_dictionary_entries[generator_entry["file_name"]] = generator_entry
                    continue

                # get current file name
                file_name = generator_entry["file_name"]

                if file_name in valid_files:
                    valid_files.remove(file_name)
                
                append_to_pickle(generator_entry, temporary_file_path)
    
    # create variables to track progress
    total_files = len(valid_files)
    progressed_files = 0
    start_time = time.time()

    if total_files > 0:
        print(f"\nReading out and saving entries with matching key from: {channel_key_to_read_out} in {total_files} files from \"{data_directory}\":")

    if user_answer == "y":
        # load existing results
        results_generator = load_from_pickle(results_path)

        for generator_entry in results_generator:
            # show progress
            progress_bar(progressed_files, total_files, start_time)
            progressed_files += 1

            try:
                # get current file name
                file_name = generator_entry["file_name"]

                if file_name in valid_files:
                    valid_files.remove(file_name)

                # try to load the data and correct the physical dimension if needed
                channel_signal, channel_sampling_frequency = get_data_from_edf_channel(
                    file_path = data_directory + file_name,
                    possible_channel_labels = channel_key_to_read_out,
                    physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
                )
                
                # save the channel for this file
                generator_entry[new_dictionary_key] = channel_signal
                generator_entry[new_dictionary_key + "_frequency"] = channel_sampling_frequency

            except:
                unprocessable_files.append(file_name)
            
            append_to_pickle(generator_entry, temporary_file_path)
    
    # read out channel for the remaining files
    for file_name in valid_files:
        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        if file_name in store_previous_dictionary_entries.keys():
            generator_entry = store_previous_dictionary_entries[file_name]
        else:
            generator_entry = {"file_name": file_name}

        try:
            # try to load the data and correct the physical dimension if needed
            channel_signal, channel_sampling_frequency = get_data_from_edf_channel(
                file_path = data_directory + file_name,
                possible_channel_labels = channel_key_to_read_out,
                physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
            )

            # save the channel for this file
            generator_entry[new_dictionary_key] = channel_signal
            generator_entry[new_dictionary_key + "_frequency"] = channel_sampling_frequency    

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
        print("\nFor the following " + str(len(unprocessable_files)) + " files the channel could not be saved:")
        print(unprocessable_files)
        print("\nPossible reasons (decreasing probability):")
        print(" "*5 + "- No channel found that matches one of the provided keys.")
        print(" "*5 + "- ECG file contains format errors.")


def library_overview(file_name):
    """
    This function won't be used in the project. It is just a demonstration of available
    commands in the pyedflib library.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the EDF file
    
    RETURNS:
    --------------------------------
    None
    """
    f = pyedflib.EdfReader(file_name)
    #f = pyedflib.data.test_generator()
    print("\nlibrary version: %s" % pyedflib.version.version) # type: ignore

    print("\ngeneral header:\n")

    # print("filetype: %i\n"%hdr.filetype);
    print("edfsignals: %i" % f.signals_in_file)
    print("file duration: %i seconds" % f.file_duration)
    print("weird: %i" % f.datarecord_duration)
    print("startdate: %i-%i-%i" % (f.getStartdatetime().day,f.getStartdatetime().month,f.getStartdatetime().year))
    print("starttime: %i:%02i:%02i" % (f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second))
    # print("patient: %s" % f.getP);
    # print("recording: %s" % f.getPatientAdditional())
    print("patientcode: %s" % f.getPatientCode())
    print("gender: %s" % f.getSex())
    print("birthdate: %s" % f.getBirthdate())
    print("patient_name: %s" % f.getPatientName())
    print("patient_additional: %s" % f.getPatientAdditional())
    print("admincode: %s" % f.getAdmincode())
    print("technician: %s" % f.getTechnician())
    print("equipment: %s" % f.getEquipment())
    print("recording_additional: %s" % f.getRecordingAdditional())
    print("datarecord duration: %f seconds" % f.getFileDuration())
    print("number of datarecords in the file: %i" % f.datarecords_in_file)
    print("number of annotations in the file: %i" % f.annotations_in_file)

    all_channels = []
    for i in np.arange(f.signals_in_file):
        all_channels.append(f.getLabel(i))
    
    print("\nall channels in the file:")
    print(all_channels)

    channel = 0
    print("\nsignal parameters for the %d.channel:\n\n" % channel)

    print("label: %s" % f.getLabel(channel))
    print("samples in file: %i" % f.getNSamples()[channel])
    # print("samples in datarecord: %i" % f.get
    print("physical maximum: %f" % f.getPhysicalMaximum(channel))
    print("physical minimum: %f" % f.getPhysicalMinimum(channel))
    print("digital maximum: %i" % f.getDigitalMaximum(channel))
    print("digital minimum: %i" % f.getDigitalMinimum(channel))
    print("physical dimension: %s" % f.getPhysicalDimension(channel))
    print("prefilter: %s" % f.getPrefilter(channel))
    print("transducer: %s" % f.getTransducer(channel))
    print("samplefrequency: %f" % f.getSampleFrequency(channel))

    annotations = f.readAnnotations()
    for n in np.arange(f.annotations_in_file):
        print(f"annotation: onset is {annotations[0][n]:f}    duration is {annotations[1][n]}    description is {annotations[2][n]}")

    buf = f.readSignal(channel)
    n = 200
    print("\nread %i samples\n" % n)
    result = ""
    for i in np.arange(n):
        result += ("%.1f, " % buf[i])
    print(result)
    f._close()
    del f


def retrieve_file_header_information(
        data_directory: str,
        results_path: str,
    ):
    """
    Retrieve basic header information and append it to the results file. Theoretically, this function could 
    also be applied to data that could not be processed by other main functions (e.g. calculating MAD values
    or determining valid ecg regions). However, as we do not need this information if we cannot calculate
    the MAD or RRI values, it will only be applied to existing data in the results file.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    results_path: str
        path to the pickle file where the valid regions are saved

    RETURNS:
    --------------------------------
    None, but some file header information is saved as dictionaries to a pickle file in the following format:
    {
        "file_name": file_name_1,
        "start_date": start_date_1,
        "start_time": start_time_1,
        ...
    }
        ...
    """

    # path to pickle file which will store results
    temporary_file_path = get_path_without_filename(results_path) + "computation_in_progress.pkl"

    # if the temporary file already exists, something went wrong
    if os.path.isfile(temporary_file_path):
        raise Exception("The file: " + temporary_file_path + " should not exist. Either a previous computation was interrupted or another computation is ongoing.")

    # create lists to store unprocessable files
    unprocessable_files = []
   
    # create variables to track progress
    total_files = get_pickle_length(results_path, "start_time")
    progressed_files = 0
    start_time = time.time()

    if total_files > 0:
        print("\nRetrieving header information of %i files from \"%s\":" % (total_files, data_directory))
    else:
        return
    
    # load results
    results_generator = load_from_pickle(results_path)
    
    for generator_entry in results_generator:

        # show progress
        progress_bar(progressed_files, total_files, start_time)
        progressed_files += 1

        try:
            # get the file name
            file_name = generator_entry["file_name"]

            # get the header information
            edf_header = get_header_from_edf_file(file_path = data_directory + file_name)
        
            # add the information to the dictionary
            generator_entry["start_date"] = edf_header["start_date"]
            generator_entry["start_time"] = edf_header["start_time"]

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
        print("\nFor the following " + str(len(unprocessable_files)) + " files the header information could not be retrieved:")
        print(unprocessable_files)
        print("Possible reasons (decreasing probability):")
        print(" "*5 + "- Relevant header information not present in .edf file")


def nako_data_distribution(Data_Directories: list, Directory_Group_Name: list):
    """
    """

    information = dict()

    for group_dir, group_name in zip(Data_Directories, Directory_Group_Name):
        start_dates = []
        start_times = []
        datarecord_durations = []
        file_durations = []

        genders = []
        birthdates = []
        
        channels = []

        total_files = 0
        file_success = 0


        for data_dir in group_dir:

            print("\n Reading informations from files in %s:" % data_dir)
            count_progress = 0

            for file in os.listdir(data_dir):

                if get_file_type(file) == ".edf":

                    count_progress += 1
                    print(count_progress, end="\r")

                    path = data_dir + file
                    total_files += 1
                    if not os.path.exists(path):
                        continue
                    
                    try:
                        open_edf_file = pyedflib.EdfReader(path)
                        start_dates.append((open_edf_file.getStartdatetime().day, open_edf_file.getStartdatetime().month, open_edf_file.getStartdatetime().year))
                        start_times.append((open_edf_file.getStartdatetime().hour, open_edf_file.getStartdatetime().minute, open_edf_file.getStartdatetime().second))
                        datarecord_durations.append(open_edf_file.datarecord_duration)
                        file_durations.append(open_edf_file.getFileDuration())

                        genders.append(open_edf_file.getSex())
                        birthdates.append(open_edf_file.getBirthdate())

                        needed_channels = 0
                        this_channels = [open_edf_file.getLabel(i) for i in np.arange(open_edf_file.signals_in_file)]
                        for chnel in this_channels:
                            if "X" in chnel or "Y" in chnel or "Z" in chnel or "ECG" in chnel:
                                needed_channels += 1
                        
                        if needed_channels == 4:
                            channels.append(needed_channels)
                        else:
                            channels.append(this_channels)
                    except:
                        continue

                    file_success += 1
                    print(start_dates, start_times, datarecord_durations, file_durations, channels, genders, birthdates)
                    break

        information[group_name + "_start_dates"] = start_dates
        information[group_name + "_start_times"] = start_times
        information[group_name + "_datarecord_durations"] = datarecord_durations
        information[group_name + "_file_durations"] = file_durations
        information[group_name + "_genders"] = genders
        information[group_name + "_birthdates"] = birthdates
        information[group_name + "_channels"] = channels
        information[group_name + "_total_files"] = total_files
        information[group_name + "_error_files"] = total_files - file_success
    
    with open("nako_information", "wb") as f:
        pickle.dump(information, f)

# EDF_Data_Directories = ["/media/yaopeng/data1/NAKO-33a/", "/media/yaopeng/data1/NAKO-33b/", "/media/yaopeng/data1/NAKO-609/", "/media/yaopeng/data1/NAKO-419/", "/media/yaopeng/data1/NAKO-84/"]

nako_data_distribution(
    Data_Directories = [
        [
            "/media/yaopeng/data1/NAKO-33a/",
            "/media/yaopeng/data1/NAKO-33b/",
            "/media/yaopeng/data1/NAKO-84/"
            "/media/yaopeng/data1/NAKO-419/",
        ],
        [
            "/media/yaopeng/data1/NAKO-609/",
            "/media/yaopeng/data2/NAKO-994/"
        ]
    ],
    Directory_Group_Name = ["baseline", "follow_up"]
)