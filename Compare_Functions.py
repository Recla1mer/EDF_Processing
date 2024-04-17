"""
There already exists code from previous projects that does what I want to do. 
As it was written a while ago, I rewrite the code to be sure it functions correctly.
In this file, I compare the results of the old code to the new one.
"""

import copy
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import scipy.datasets  # new in scipy 1.10.0, used to be in scipy.misc

# import my secondary python files
import read_edf
import MAD
import rpeak_detection

# imported old secondary python files
from pyedflib import EdfReader
import old_code.mad as old_mad

test_file_path = "Test_Data/Somnowatch_Messung.edf"


def MAD_compare(edf_file_path: str, time_period = 1):
    """
    Compare the results of the new MAD calculation to the old MAD calculation.
    """
    # #generate random data
    # data = {"X": np.random.rand(1000), "Y": np.random.rand(1000), "Z": np.random.rand(1000)}
    # frequency = {"X": 100, "Y": 100, "Z": 100}

    #calculate MAD with old code (refer to old_code/rpeak_detection.py: save_rpeaks_mad for data collection)
    with EdfReader(edf_file_path) as f:
        signal_labels = f.getSignalLabels()
        startdatetime = f.getStartdatetime()
        
    # MAD calculation
    try:
        i_x = signal_labels.index('X')
        i_y = signal_labels.index('Y')
        i_z = signal_labels.index('Z')
        mad_frequency = f.getSampleFrequency(i_x)
        #mad_frequency = int(mad_frequency / f.datarecord_duration) #this line was in the old code, but it is wrong as the pyedflib might have changed since then.
        mad_frequency = int(mad_frequency)
        old_MAD = np.array([
            f.readSignal(i_x),
            f.readSignal(i_y),
            f.readSignal(i_z)
        ])
        start_time = time.time() #start timer
        old_MAD = old_mad.cal_mad_jit(old_MAD, mad_frequency)
    except ValueError:
        old_MAD = None
    end_time = time.time() #end timer
    old_time = end_time - start_time

    #calculate MAD with new code
    sigbufs, sigfreqs, duration = read_edf.get_edf_data(edf_file_path)
    start_time = time.time() #start timer
    # sigbufs = {k: sigbufs[k] for k in ('X', 'Y', 'Z')}
    # sigfreqs = {k: sigfreqs[k] for k in ('X', 'Y', 'Z')}

    # for i in ["X", "Y", "Z"]:
    #     sigfreqs[i] = int(sigfreqs[i])
    new_MAD = MAD.calc_mad(sigbufs, sigfreqs, time_period)
    new_MAD = np.array(new_MAD)
    end_time = time.time() #end timer
    new_time = end_time - start_time

    #compare results
    # print("Old MAD:", old_MAD[:10])
    # print("New MAD:", new_MAD[:10])
    print("Data Difference:", np.sum(np.abs(old_MAD - new_MAD)))
    print("Time Difference (old - new): %f s" % round(old_time - new_time,1))


def rpeak_detection_compare(
        edf_file_path: str,
        relevant_key = "ECG", 
        lower_border = None, 
        interval_size = 2500,
        primary_function = rpeak_detection.get_rpeaks_wfdb,
        secondary_function = rpeak_detection.get_rpeaks_neuro,
        name_primary = "WFDB",
        name_secondary = "Neurokit2"
    ):
    """
    Compare the R-peak detection methods that are implemented in the rpeak_detection.py file.

    ARGUMENTS:
    --------------------------------
    edf_file_path: str
        path to the edf file
    relevant_key: str, default "ECG"
        key of the ECG data in the data dictionary
    lower_border: int, default None
        lower border of the interval in which the R peaks should be detected
        if None, the whole dataset will be used but won't be plotted
    interval_size: int, default 2500
        size of the interval in which the R peaks should be detected
        only relevant if lower_border is not None
    primary_function: function, default rpeak_detection.get_rpeaks_wfdb
        primary R peak detection function
    secondary_function: function, default rpeak_detection.get_rpeaks_neuro
        secondary R peak detection function
    name_primary: str, default "WFDB"
        name of the primary R peak detection function
    name_secondary: str, default "Neurokit2"
        name of the secondary R peak detection function
    
    RETURNS:
    --------------------------------
    None

    ADDITIONAL INFORMATION:
    --------------------------------
    
    Interesting lower border values:    55550 (weird), 
                                        2091000 (normal), 
                                        824500 (normal inversed), 
                                        2207200 (some noise),
                                        2247100 (lots of noise),
                                        4056800 (normal with energy jump)
                                        4046592 (normal with energy jump)
    """
    sigbufs, sigfreqs, duration = read_edf.get_edf_data(edf_file_path)

    if lower_border is None:
        detection_interval = None
    else:
        if lower_border + interval_size > len(sigbufs[relevant_key]):
            raise ValueError("Interval size is too large for the dataset.")
        detection_interval = (lower_border, lower_border + interval_size)


    start_time = time.time()
    rpeaks_primary = primary_function(sigbufs, sigfreqs, relevant_key, detection_interval)
    end_time = time.time()
    primary_time = end_time - start_time
    start_time = time.time()
    rpeaks_secondary = secondary_function(sigbufs, sigfreqs, relevant_key, detection_interval)
    end_time = time.time()
    secondary_time = end_time - start_time
    # print("%s R peaks:" % name_primary, rpeaks_primary)
    # print("%s R peaks:" % name_secondary, rpeaks_secondary)
    print("%s Total Time: %f s" % (name_primary, primary_time))
    print("%s Total Time: %f s" % (name_secondary, secondary_time))
    print("Time Difference (%s - %s): %f s" % (name_primary, name_secondary, round(primary_time - secondary_time,3)))
    if lower_border is not None:
        print("Datapoints per second (%s): %f" % (name_primary, interval_size / primary_time))
        print("Datapoints per second (%s): %f" % (name_secondary, interval_size / secondary_time))
    else:
        print("Datapoints per second (%s): %f" % (name_primary, len(sigbufs[relevant_key]) / primary_time))
        print("Datapoints per second (%s): %f" % (name_secondary, len(sigbufs[relevant_key]) / secondary_time))


    print("Number of R peaks in %s: %d" % (name_primary, len(rpeaks_primary)))
    print("Number of R peaks in %s: %d" % (name_secondary, len(rpeaks_secondary)))

    intersecting_rpeaks = np.intersect1d(rpeaks_primary, rpeaks_secondary)
    removed_rpeaks_primary = np.setdiff1d(rpeaks_primary, intersecting_rpeaks)
    removed_rpeaks_secondary = np.setdiff1d(rpeaks_secondary, intersecting_rpeaks)

    print("Percentage of intersecting R peaks (%s): %f %%" % (name_primary, (len(intersecting_rpeaks) / len(rpeaks_primary) * 100)))
    print("Percentage of intersecting R peaks (%s): %f %%" % (name_secondary, (len(intersecting_rpeaks) / len(rpeaks_secondary) * 100)))

    if detection_interval is not None:
        fig, ax = plt.subplots()
        ax.plot(np.arange(detection_interval[0], detection_interval[1]), sigbufs[relevant_key][detection_interval[0]:detection_interval[1]], label="ECG")
        # ax.plot(rpeaks_primary, sigbufs[relevant_key][rpeaks_primary], "ro", label=name_primary)
        ax.plot(intersecting_rpeaks, sigbufs[relevant_key][intersecting_rpeaks], "bo", label="Both")
        ax.plot(removed_rpeaks_primary, sigbufs[relevant_key][removed_rpeaks_primary], "ro", label=name_primary)
        ax.plot(removed_rpeaks_secondary, sigbufs[relevant_key][removed_rpeaks_secondary], "go", label=name_secondary)
        ax.legend()
        plt.show()


#MAD_compare(test_file_path)
#rpeak_detection_compare(test_file_path)
rpeak_detection_compare(test_file_path, lower_border = 0, interval_size = 10000, secondary_function=rpeak_detection.get_rpeaks_old, name_secondary="Old")