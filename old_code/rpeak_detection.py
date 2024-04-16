"""
Detect R peaks and calculate mad using stationary wavelet transform (SWT) method from py-ecg-detector
And generate the Rpeaks file

the function save_rpeaks_mad() will do the process and generate the file

For load the generated file, use the file load_data.py

author: Yaopeng Ma sdumyp@126.com or mayaope@biu.ac.il
date: 04/2023
"""
from pyedflib import EdfReader
from biosppy.signals.ecg import correct_rpeaks
from biosppy.signals.tools import filter_signal
from old_code.fix_edf import edf_deidentify
from old_code.mad import cal_mad_jit 
from ecgdetectors import Detectors
import numpy as np


def get_rpeaks(ecg_signal, samplingrate):
    """
    detect Rpeaks by SWT method then correct by the function from biosppy/
    
    Parameters
    --------------
    ecg_signal: 1D numpy array
        contains ECG signal
    samplingrate: int
        sampling rate of ECG
    """
    detector = Detectors(samplingrate)
    rpeaks = detector.swt_detector(ecg_signal)
    (rpeaks, ) = correct_rpeaks(signal=ecg_signal, rpeaks=rpeaks, sampling_rate=samplingrate, tol=0.15)
    (rpeaks, ) = correct_rpeaks(signal=ecg_signal, rpeaks=rpeaks, sampling_rate=samplingrate, tol=0.05)
    # detect large gap (10 s)
    threshold = 4
    rpeaks = np.append(rpeaks, ecg_signal.size - 1)
    rri = np.diff(rpeaks) / samplingrate
    index = np.where(rri > threshold)[0]
    # if no gap, return
    if index.size == 0:
        return rpeaks
    real_rpeaks = [rpeaks]
    num_inserts = np.zeros(index.size)
    # loop all the gaps
    for i, ind in enumerate(index):
        # skip the peaks
        start_i = rpeaks[ind] + int(samplingrate * 0.02)
        end_i = rpeaks[ind + 1] - int(samplingrate * 0.02)

        duration = (end_i - start_i) / samplingrate
        sub_ecg = ecg_signal[start_i: end_i]
        sub_rpeaks = get_rpeaks(sub_ecg, samplingrate)
        if sub_rpeaks.size == 0 or (sub_ecg[sub_rpeaks]).mean() < 100:
            continue
        
        sub_rpeaks = sub_rpeaks + start_i
        num_inserts[i] = sub_rpeaks.size / duration
        real_rpeaks.append(sub_rpeaks)
    if np.max(num_inserts) < 0.2:
        return rpeaks[:-1]

    rpeaks = np.concatenate(real_rpeaks)
    rpeaks = np.sort(rpeaks)[:-1]
    return rpeaks
       

def save_rpeaks_mad(edffilepath, rrfilename, madfilename=None, deidentify=False):
    """
    Process an EDF file to detect R peaks and calculate the 1Hz mad (optional)
    
    Parameters:
    --------------
    effilepath: str
    
    rrfilename: str
        the filepath of the rri file to save
        
    madfilename: str, default None
        the filepath of the mad file to save, None if the MAD is not needed
    
    deidentify: bool, default is False
        whether to deidentify the edffile or not. More about the deidentify, check the file: fix_edf.py
    
    Returns:
    --------------
        None
    """
    if deidentify:
        edf_deidentify(edffilepath, overwrite=deidentify)
    with EdfReader(edffilepath) as f:
        signal_labels = f.getSignalLabels()
        startdatetime = f.getStartdatetime()
        for i, l in enumerate(signal_labels):
            if l.find("ECG") >= 0:
                break
        else:
            i = -1
        # MAD calculation
        if madfilename is not None:
            try:
                i_x = signal_labels.index('X')
                i_y = signal_labels.index('Y')
                i_z = signal_labels.index('Z')
                mad_frequency = f.getSampleFrequency(i_x)
                mad_frequency = int(mad_frequency / f.datarecord_duration)
                MAD = np.array([
                    f.readSignal(i_x),
                    f.readSignal(i_y),
                    f.readSignal(i_z)
                ])
                MAD = cal_mad_jit(MAD, mad_frequency)
            except ValueError:
                MAD = None
        else:
            MAD = None

        # ECG preprocess
        if i >= 0:
            samplingrate = f.getSampleFrequency(i)
            samplingrate = int(samplingrate / f.datarecord_duration)
            ecg_signal = f.readSignal(i)
            # R peak detection
            rpeaks = get_rpeaks(ecg_signal, samplingrate) 
        
    # Save mad file
    if MAD is not None:
        np.savez(madfilename, mad=MAD, starttime=startdatetime)
    # Save rr file
    if i >= 0:
        with open(rrfilename, 'w') as f:
            f.write("author: Yaopeng Ma, used python with pyedflib and py-ecg-detectors, stationary wavelet transform method\n")
            f.write(f"number_qrs={rpeaks.size}\n")
            f.write(f"rec-date=" + startdatetime.strftime("%d.%m.%Y") + '\n')
            f.write(f"rec-time=" + startdatetime.strftime("%H:%M:%S") + '\n')
            f.write(f"samplerate={samplingrate}\n")
            f.write(f"pos-type=qrs_pos\n")
            f.write("-" * 50 + '\n')
            for rpeak in rpeaks:
                f.write(f"{rpeak}\n")
    if MAD is None and i == -1:
        raise ValueError(edffilepath, " XYZ and ECG are not Found !")
    elif MAD is None and i >= 0:
        raise ValueError(edffilepath, " XYZ are not Found !")
    elif i == -1 and MAD is not None:
        raise ValueError(edffilepath, " ECG is not Found !")

    return


if __name__ == "__main__":
    import time
    import os
    folder = '/media/yaopeng/data1/NAKO-84'
    subject = "100006"
    t1 = time.time()
    save_rpeaks_mad(os.path.join(folder, subject + ".edf"),
                    f'cache/{subject}.rri',
                    f'cache/{subject}.mad',
                    True)
    t2 = time.time()
    print(t2 - t1)
    # import os
    # folderspath = "/media/yaopeng/data1/"
    # folder_touse = ['NAKO-33a-copy']
    # for folder in folder_touse:
    #     folder_path = os.path.join(folderspath, folder)
    #     filenames = os.listdir(folder_path)
    #     filepathes = [os.path.join(folder_path, f) for f in filenames]
    #     filepathes = list(filter(lambda x: x[:-4] != '.edf', filepathes))
    # print(filepathes)
