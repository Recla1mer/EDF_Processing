"""
Load rri files and mad files

author: Yaopeng Ma sdumyp@126.com or mayaope@biu.ac.il

date 04/2023
"""

import datetime
import numpy as np
import pandas as pd


def read_rri(rri_file: str):
    """
    load rri file, return start time (datetime), rri, and frequency
    Parameters
    --------------
    rri_file: str
        the path of .rri file
    
    Returns
    --------------
    rpeaks' index: numpy array [int]
    
    sample rate: int
        sample rate of the ECG recording.

    recording starttime: datetime
    """
    f = open(rri_file, 'r')
    f.readline()
    f.readline()
    """
    Get start time of this rri file
    """
    rr_startdate = f.readline()
    index_start = rr_startdate.find("=") + 1
    rr_startdate = rr_startdate[index_start:-1]
    rr_starttime = f.readline()
    index_start = rr_starttime.find("=") + 1
    rr_starttime = rr_starttime[index_start:-1]
    file_starttime = rr_startdate + " " + rr_starttime

    try:
        file_starttime = datetime.datetime.strptime(file_starttime,
                                                    "%d.%m.%Y %H:%M:%S")
    except ValueError:
        file_starttime = datetime.datetime.strptime(file_starttime,
                                                    "%Y-%m-%d %H:%M:%S")
    """
    get frequency of rri
    """
    line = f.readline()
    index_start = line.find("=") + 1
    freq = int(line[index_start:])

    f.close()
    df = pd.read_csv(rri_file,
                     skiprows=7,
                     sep="\t",)
    # df = df[df.type == "N"]
    return df.iloc[:, 0].values, freq, file_starttime


def read_mad(mad_file:str):
    """
    load the mad file (npz format)
    Parameters
    ------------
    mad_file: str
        the path of the mad file
    
    Return
    ------------
    mad signal: numpy array
        (by default it is 1Hz)
    
    starttime: datetime
    """
    mad = np.load(mad_file, allow_pickle=True)
    return mad['mad'], mad['starttime']


if __name__ == '__main__':
    rpeaks, freq, rstarttime= read_rri("./cache/10005.rri")
    mad, madstarttime = read_mad("./cache/10005.mad.npz")
    print(madstarttime)
    print(rstarttime)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(np.log10(mad))
    axs[0].set_ylabel("log10 mad")
    axs[1].plot(rpeaks[1:] / freq, np.diff(rpeaks) / freq)
    axs[1].set_ylabel('rri')
    plt.xlabel("Seconds(s)")
    plt.show()
