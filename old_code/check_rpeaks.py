import numpy as np
import pyedflib

from load_data import read_rri, read_mad
import matplotlib.pyplot as plt

# def plot_ECG(t, ecg, rpeaks):
    # raise NotImplemented


def visual_ecg_rpeaks_mad(edffile, rrifile, madfile, newrrifile):
    rpeaks, freq, _ = read_rri(rrifile)
    rpeaks2, freq, _ = read_rri(newrrifile)
    mad, _ = read_mad(madfile)
    with pyedflib.EdfReader(edffile) as f:
        labels = f.getSignalLabels()
        i = labels.index("ECG")
        ecg = f.readSignal(i)
    t = np.arange(ecg.size) / freq

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(mad)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("MAD")
    ax[1].plot(rpeaks[1:] / freq, np.diff(rpeaks) / freq)
    ax[1].plot(rpeaks2[1:] / freq, np.diff(rpeaks2) / freq)
    ax[1].set_ylabel("RRI/s")
    ax[1].set_ylim(0.2, 2)
    ax[2].plot(t, ecg, 'k-')
    ax[2].plot(t[rpeaks], ecg[rpeaks], 'r*')
    ax[3].plot(t, ecg, 'k-')
    ax[3].plot(t[rpeaks2], ecg[rpeaks2], 'r*')
    plt.show()


if __name__ == "__main__":
    folder = "NAKO-84"
    import os
    # folder = "NAKO-609"
    # folder = "NAKO-33a"
    # subject = "105555"
    subject = "100006"
    # 84 100007
    # subject = "10017"
    # subject = "10076"
    visual_ecg_rpeaks_mad(f"/media/yaopeng/data1/{folder}/{subject}.edf",
                        #   f"RRI/{folder}/{subject}.rri",
                          f"cache/{subject}.rri",
                          f"/home/yaopeng/data/MAD/{folder}/{subject}_mad.npz",
                          f"/home/yaopeng/data/RRI/{folder}/{subject}.rri",)
                        #   f"cache/{subject}.rri",)