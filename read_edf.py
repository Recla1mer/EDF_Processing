import pyedflib # https://github.com/holgern/pyedflib
import numpy as np
import os

import pickle

import NN_plot_helper as NNPH

# https://github.com/holgern/pyedflib
test_file_path = "Test_Data/Somnowatch_Messung.edf"


def test_library(file_name):
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

    channel = 3
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
    Read an EDF file and return a dictionary: {"Signal_Label": Signal}.

    ATTENTION: In the actual EDF file, the signals are shown in blocks over time. This was previously not considered in the pyedflib library. Now it seems to be fixed.
    """
    f = pyedflib.EdfReader(file_name)

    duration = f.file_duration

    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    n_min = f.getNSamples()[0]
    sigbufs = dict()
    sigfreqs = dict()

    for i in np.arange(n):
        this_signal = f.readSignal(i)
        sigbufs[signal_labels[i]] = this_signal
        sigfreqs[signal_labels[i]] = f.getSampleFrequency(i)
        if n_min < len(this_signal):
            n_min = len(this_signal)
    f._close()
    return sigbufs, sigfreqs, duration


# file_name = "Neural_Networks/Test_Data/Somnowatch_Messung.edf"
# sigbufs, sigfreqs, duration = get_edf_data(file_name)
# for key, value in sigbufs.items():
#     this_data = value
#     this_time = array_from_duration(duration, sigfreqs[key])
#     NNPH.seperate_plots(this_data, this_time, key, TEMPORARY_FIGURE_DIRECTORY_PATH + "test_data", xlim=[2990, 3000])

# signals = read_edf(file_name)
# save_to_pickle(signals, TEMPORARY_PICKLE_DIRECTORY_NAME + "signals.pkl")

# NNPH.seperate_plots_from_bib(TEMPORARY_PICKLE_DIRECTORY_NAME + "signals.pkl", TEMPORARY_FIGURE_DIRECTORY_PATH)
# clear_directory(TEMPORARY_PICKLE_DIRECTORY_NAME)