import pandas as pd
import matplotlib.pyplot as plt
from biosppy.signals.ecg import christov_segmenter, hamilton_segmenter
import rpeak_detection as rpd
# from ecgdetectors import Detectors


if __name__ == "__main__":
    data_file = "no_pain_alone_dyad18.txt"
    df = pd.read_csv(data_file, sep='\t')
    t = df['Time (s)']
    ECG = df['PDA2_Bio Ch 1'].values
    rpeaks = christov_segmenter(ECG, 500)['rpeaks']
    rpeaks_hamilton = hamilton_segmenter(ECG, 500)['rpeaks']
    rpeaks_wfdb = rpd.get_rpeaks_wfdb(ECG, 500, None)
    rpeaks_old = rpd.get_rpeaks_old(ECG, 500, None)
    rpeaks_christov = rpd.get_rpeaks_christov(ECG, 500, None)
    plt.figure(figsize=(15, 4))
    plt.plot(t, ECG)
    plt.scatter(t[rpeaks], ECG[rpeaks], color='red', marker='v',
                label='R-peaks from Christov', alpha=0.5)
    plt.scatter(t[rpeaks_christov], ECG[rpeaks_christov], color='red', marker='o',
                label='R-peaks from Christov', alpha=0.5)
    plt.scatter(t[rpeaks_hamilton], ECG[rpeaks_hamilton], color='green', 
                marker='^', label='R-peaks from Hamilton', alpha=0.5)
    # plt.scatter(t[rpeaks_wfdb], ECG[rpeaks_wfdb], color='blue', 
    #             marker='o', label='R-peaks from wfdb', alpha=0.5)
    # plt.scatter(t[rpeaks_old], ECG[rpeaks_old], color='black', 
    #             marker='o', label='R-peaks from old', alpha=0.5)
    plt.legend()
    plt.xlim(95, 100)
    plt.show()
