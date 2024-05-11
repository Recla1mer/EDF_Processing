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

    print("Reading signal labels and their physical dimensions from %i files:" % total_files)
    for file in valid_files:
        progress_bar(progressed_files, total_files)
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
    
    progress_bar(progressed_files, total_files)

    if len(error_in_files) > 0:
        print("Due to an error in the following files, they could not be read and were skipped:")
        print(error_in_files)
    
    return all_signal_labels, all_physical_dimensions


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
    print("\nlibrary version: %s" % pyedflib.version.version)

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


# try_directory = "Data/GIF/SOMNOwatch/"
# print(get_dimensions_and_signal_labels(try_directory))