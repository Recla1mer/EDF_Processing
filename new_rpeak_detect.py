import pandas as pd
import matplotlib.pyplot as plt
from biosppy.signals.ecg import christov_segmenter, hamilton_segmenter
import rpeak_detection as rpd
import read_edf as redf
import numpy as np
import time
# from ecgdetectors import Detectors


voltage_dimensions = ["uV", "mV"]
voltage_correction = [1, 1e3]

force_dimensions = ["mg"]
force_correction = [1]

physical_dimension_correction_dictionary = {
    "ECG": {"possible_dimensions": voltage_dimensions, "dimension_correction": voltage_correction},
    "X": {"possible_dimensions": force_dimensions, "dimension_correction": force_correction},
    "Y": {"possible_dimensions": force_dimensions, "dimension_correction": force_correction},
    "Z": {"possible_dimensions": force_dimensions, "dimension_correction": force_correction}
}

file_path = "Data/GIF/SOMNOwatch/SL001_SL001_(1).edf"
# file_path = "Data/GIF/SOMNOwatch/SL317_SL317_(1).edf"
interval_size = 50000
lower_bound = 750288
interval = [lower_bound, lower_bound + interval_size]
interval = [716288, int(10694656)]
#interval = [0, 10100000] #10132480

if __name__ == "__main__":
    data_file = "no_pain_alone_dyad18.txt"
    df = pd.read_csv(data_file, sep='\t')
    t = df['Time (s)']
    ECG = df['PDA2_Bio Ch 1'].values

    rpeaks_hamilton_old = hamilton_segmenter(ECG, 500)["rpeaks"]
    rpeaks_christov_old = christov_segmenter(ECG, 500)["rpeaks"]

    rpeaks_hamilton = rpd.get_rpeaks_hamilton(ECG, 500, None)
    rpeaks_christov = rpd.get_rpeaks_christov(ECG, 500, None)
    plt.figure(figsize=(15, 4))
    plt.plot(t, ECG)
    # plt.scatter(t[rpeaks_christov], ECG[rpeaks_christov], color='red', marker='o',
    #             label='R-peaks from Christov', alpha=0.5)
    # plt.scatter(t[rpeaks_hamilton], ECG[rpeaks_hamilton], color='green', 
    #             marker='o', label='R-peaks from Hamilton', alpha=0.5)
    plt.scatter(t[rpeaks_christov_old], ECG[rpeaks_christov_old], color='orange', marker='s',
                label='R-peaks from Christov', alpha=0.5)
    plt.scatter(t[rpeaks_hamilton_old], ECG[rpeaks_hamilton_old], color='blue', 
                marker='*', label='R-peaks from Hamilton', alpha=0.5)
    # plt.scatter(t[rpeaks_wfdb], ECG[rpeaks_wfdb], color='blue', 
    #             marker='o', label='R-peaks from wfdb', alpha=0.5)
    # plt.scatter(t[rpeaks_old], ECG[rpeaks_old], color='black', 
    #             marker='o', label='R-peaks from old', alpha=0.5)
    plt.legend()
    plt.xlim(95, 100)
    plt.show()

    """
    bla
    """
    # try to load the data and correct the physical dimension if needed
    # ECG, ecg_sampling_frequency = redf.get_data_from_edf_channel(
    #     file_path = file_path,
    #     possible_channel_labels = ["ECG"],
    #     physical_dimension_correction_dictionary = physical_dimension_correction_dictionary
    # )

    # start_time = time.time()
    # rpeaks_hamilton = rpd.get_rpeaks_hamilton(ECG, ecg_sampling_frequency, interval)
    # print("Hamilton: ", time.time() - start_time)
    # start_time = time.time()
    # rpeaks_wfdb = rpd.get_rpeaks_wfdb(ECG, ecg_sampling_frequency, interval)
    # print("WFDB: ", time.time() - start_time)
    # start_time = time.time()
    # rpeaks_old = rpd.get_rpeaks_ecgdetectors(ECG, ecg_sampling_frequency, interval)
    # print("Old: ", time.time() - start_time)
    # start_time = time.time()
    # rpeaks_christov = rpd.get_rpeaks_christov(ECG, ecg_sampling_frequency, interval)
    # print("Christov: ", time.time() - start_time)

    # print("Hamilton: ", len(rpeaks_hamilton), rpeaks_hamilton)
    # print("WFDB: ", len(rpeaks_wfdb), rpeaks_wfdb)
    # print("Old: ", len(rpeaks_old), rpeaks_old)
    # print("Christov: ", len(rpeaks_christov), rpeaks_christov)

    # plt.figure(figsize=(15, 4))
    # plt.plot(np.arange(interval[0], interval[1]), ECG[interval[0]:interval[1]])
    # plt.scatter(rpeaks_christov, ECG[rpeaks_christov], color='red', marker='o',
    #             label='R-peaks from Christov', alpha=0.5)
    # plt.scatter(rpeaks_hamilton, ECG[rpeaks_hamilton], color='green', 
    #             marker='^', label='R-peaks from Hamilton', alpha=0.5)
    # plt.scatter(rpeaks_wfdb, ECG[rpeaks_wfdb], color='blue', 
    #             marker='o', label='R-peaks from wfdb', alpha=0.5)
    # plt.scatter(rpeaks_old, ECG[rpeaks_old], color='black', 
    #             marker='o', label='R-peaks from old', alpha=0.5)
    # plt.legend()
    # plt.xlim(95, 100)
    # plt.show()