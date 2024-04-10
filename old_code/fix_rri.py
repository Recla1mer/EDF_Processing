from pyedflib import EdfReader
import numpy as np
from ntpath import basename
from multiprocessing import Process
import os

from load_data import read_rri
from rpeak_detection import get_rpeaks


def fix_rrifile(rrifile, edffile, low_thershold=4, up_thershold=60):
    rpeaks, samplerate, starttime = read_rri(rrifile)
    rri = np.diff(rpeaks) / samplerate
    if up_thershold is not None:
        index = np.where((rri > low_thershold) & (rri < up_thershold))[0]
    else:
        index = np.where((rri > low_thershold))[0]
    if index.size == 0:
        raise ValueError("No Need To Be Fixed")
    with EdfReader(edffile) as f:
        i = f.getSignalLabels().index("ECG")
        ecg = f.readSignal(i)
        last_point = f.getFileDuration()
    rpeaks = np.append(rpeaks, last_point)

    num_inserts = np.zeros(index.size)
    real_rpeaks = [rpeaks]
    for i, ind in enumerate(index):
        duration = (rpeaks[ind+1] - rpeaks[ind]) / samplerate
        sub_ecg = ecg[rpeaks[ind]:rpeaks[ind+1]]
        sub_rpeaks = get_rpeaks(sub_ecg, samplerate)
        if sub_rpeaks.size == 0 or (sub_ecg[sub_rpeaks]).mean() < 100:
            continue
        sub_rpeaks = sub_rpeaks + rpeaks[ind]
        num_inserts[i] = sub_rpeaks.size / duration
        real_rpeaks.append(sub_rpeaks)
    if np.max(num_inserts) < 0.2:
        raise ValueError("No Need To Be Fixed")

    real_rpeaks = np.concatenate(real_rpeaks)
    return np.sort(real_rpeaks)[:-1], samplerate, starttime


def save_rrifile(saved_file, rpeaks, startdatetime, samplingrate):
    with open(saved_file, 'w') as f:
            f.write("author: Yaopeng Ma, used python with pyedflib and py-ecg-detectors, stationary wavelet transform method (fixed)\n")
            f.write(f"number_qrs={rpeaks.size}\n")
            f.write(f"rec-date=" + startdatetime.strftime("%d.%m.%Y") + '\n')
            f.write(f"rec-time=" + startdatetime.strftime("%H:%M:%S") + '\n')
            f.write(f"samplerate={samplingrate}\n")
            f.write(f"pos-type=qrs_pos\n")
            f.write("-" * 50 + '\n')
            for rpeak in rpeaks:
                f.write(f"{rpeak}\n")


def fix_saverrifile(rrifile, edffile, saved_file):
     rpeaks, fs, starttime = fix_rrifile(rrifile, edffile)
     save_rrifile(saved_file, rpeaks, starttime, fs)


def process_fix(rrifiles, edffiles, fixedrri_folder):
    if not os.path.exists(fixedrri_folder):
        os.mkdir(fixedrri_folder)
    
    for rrifile, edffile in zip(rrifiles, edffiles):
        rribasename = basename(rrifile)
        newrripath = os.path.join(fixedrri_folder, rribasename)
        try:
            fix_saverrifile(rrifile, edffile, newrripath)
            print(f"Fix DONE: {rrifile}")
        except Exception:
            print(f"{rrifile}: No need to modify")
            continue



def main():
    num_workers = 8
    RRI_folderpath = "./RRI_fixed/"
    # RRI_folderpath = "./RRI/"
    edf_folderpath = "/media/yaopeng/data1"
    nakofolders = ["NAKO-33a", "NAKO-33b", 
                   "NAKO-84", "NAKO-84n1", 
                   "NAKO-84n2", "NAKO-419", 
                   "NAKO-419k", "NAKO-609"]
    
    for folder in nakofolders:
        edffolder_path = os.path.join(edf_folderpath, folder)
        rrifolder_path = os.path.join(RRI_folderpath, folder)
        fixedrri_folder = os.path.join("RRI_fixed", folder)
        subjects = [f.split('.')[0] for f in os.listdir(rrifolder_path)]
        rrifiles = [os.path.join(rrifolder_path, s + '.rri') for s in subjects]
        edffiles = [os.path.join(edffolder_path, s + '.edf') for s in subjects]
        process_list = []
        for i in range(num_workers):
            p = Process(target=process_fix,
                        args=(rrifiles[i::num_workers],
                              edffiles[i::num_workers],
                              fixedrri_folder))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()


if __name__ == "__main__":
    # subject = "10014"
    # test_rrifile = f"./RRI_fixed/NAKO-33a/{subject}.rri"
    # test_edffile = f"/media/yaopeng/data1/NAKO-33a/{subject}.edf"
    # fix_saverrifile(test_rrifile, test_edffile, test_rrifile)
    main()
