import numpy as np
import os

import pickle

# import secondary python files
import read_edf
import MAD
import NN_plot_helper as NNPH
import rpeak_detection


TEMPORARY_PICKLE_DIRECTORY_NAME = "Temporary_Pickles/"
TEMPORARY_FIGURE_DIRECTORY_PATH = "Temporary_Figures/"
if not os.path.isdir(TEMPORARY_PICKLE_DIRECTORY_NAME):
    os.mkdir(TEMPORARY_PICKLE_DIRECTORY_NAME)
if not os.path.isdir(TEMPORARY_FIGURE_DIRECTORY_PATH):
    os.mkdir(TEMPORARY_FIGURE_DIRECTORY_PATH)

def clear_directory(directory):
    """
    Clear the directory of everything
    """
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            if os.path.isdir(file_path):
                clear_directory(file_path)
        except Exception as e:
            print(e)


clear_directory(TEMPORARY_PICKLE_DIRECTORY_NAME)
clear_directory(TEMPORARY_FIGURE_DIRECTORY_PATH)


def save_to_pickle(data, file_name):
    """
    Save data to a pickle file.
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def array_from_duration(duration, freq):
    """
    Return an array of time points from 0 to duration with frequency freq.
    """
    return np.arange(0, duration, 1 / freq)


file_name = "Test_Data/Somnowatch_Messung.edf"

# sigbufs, sigfreqs, duration = read_edf.get_edf_data(file_name)
# for key, value in sigbufs.items():
#     this_data = value
#     this_time = array_from_duration(duration, sigfreqs[key])
#     NNPH.seperate_plots(this_data, this_time, key, TEMPORARY_FIGURE_DIRECTORY_PATH + "test_data", xlim=[2990, 3000])


sigbufs, sigfreqs, duration = read_edf.get_edf_data(file_name)
#print(MAD.calc_mad(sigbufs, sigfreqs, 60))
save_to_pickle(sigbufs["ECG"][0:1000000], "Calibration_Data/ecg_1.pkl")
save_to_pickle(sigbufs["ECG"][7000000:8000000], "Calibration_Data/ecg_2.pkl")
save_to_pickle(sigbufs["ECG"][2000000:3000000], "Calibration_Data/ecg_3.pkl")
print(sigbufs["ECG"][0:1000000])


