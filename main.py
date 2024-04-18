"""
Author: Johannes Peter Knoll

Main python file for the neural network project.
"""

# import libraries
import numpy as np
import os
import pickle

# import secondary python files
import read_edf
import MAD
import rpeak_detection
import check_data
import plot_helper as NNPH


"""
In this section we create the needed directories for the project.
We will also implement functions to manipulate the directories and save data to pickle files.
"""


# define directory and file names (will always be written in capital letters)
TEMPORARY_PICKLE_DIRECTORY_NAME = "Temporary_Pickles/"
TEMPORARY_FIGURE_DIRECTORY_PATH = "Temporary_Figures/"

THRESHOLD_DIRECTORY = "Thresholds/"

CHECK_ECG_DATA_THRESHOLDS = THRESHOLD_DIRECTORY + "Check_ECG_Data_Thresholds.pkl"

CALIBRATION_DATA_PATH = "Calibration_Data/Somnowatch_Messung.edf"

# create directories if they do not exist
if not os.path.isdir(TEMPORARY_PICKLE_DIRECTORY_NAME):
    os.mkdir(TEMPORARY_PICKLE_DIRECTORY_NAME)
if not os.path.isdir(TEMPORARY_FIGURE_DIRECTORY_PATH):
    os.mkdir(TEMPORARY_FIGURE_DIRECTORY_PATH)
if not os.path.isdir(THRESHOLD_DIRECTORY):
    os.mkdir(THRESHOLD_DIRECTORY)


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


def save_to_pickle(data, file_name):
    """
    Save data to a pickle file.
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(file_name):
    """
    Load data from a pickle file.
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


"""
In this section we can set the parameters for the project (kwargs). It also contains a 
function to easily create a sub dictionary of the main one (kwargs).
"""
kwargs = {
    "file_path": CALIBRATION_DATA_PATH, # path to the EDF file for threshold calibration
    "ecg_threshold_multiplier": 0.75, # multiplier for the thresholds in check_data.check_ecg() (between 0 and 1)
    "show_calibration_data": False, # if True, the calibration data in the manually chosen intervals will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
    "calculate_thresholds": False, # if True, you will have the option to recalculate the thresholds for various functions
    "threshold_dezimal_places": 2, # number of dezimal places for the thresholds in the pickle files
    "time_interval_seconds": 10, # time interval considered when calculating thresholds
    "min_valid_length_minutes": 5, # minimum length of valid data in minutes
    "allowed_invalid_region_length_seconds": 30 # data region (see above) still considered valid if the invalid part is shorter than this
}

if not isinstance(kwargs["file_path"], str):
    raise ValueError("'file_path' parameter must be a string.")
if not isinstance(kwargs["ecg_threshold_multiplier"], (int, float)):
    raise ValueError("'ecg_threshold_multiplier' parameter must be an integer or a float.")
if kwargs["ecg_threshold_multiplier"] <= 0 or kwargs["ecg_threshold_multiplier"] > 1:
    raise ValueError("'ecg_threshold_multiplier' parameter must be between 0 and 1.")
if not isinstance(kwargs["show_calibration_data"], bool):
    raise ValueError("'show_calibration_data' parameter must be a boolean.")
if not isinstance(kwargs["calculate_thresholds"], bool):
    raise ValueError("'calculate_thresholds' parameter must be a boolean.")
if kwargs["show_calibration_data"] and kwargs["calculate_thresholds"]:
    raise ValueError("'show_calibration_data' and 'calculate_thresholds' parameter cannot both be True at the same time.")
if not isinstance(kwargs["threshold_dezimal_places"], int):
    raise ValueError("'threshold_dezimal_places' parameter must be an integer.")
if not isinstance(kwargs["time_interval_seconds"], int):
    raise ValueError("'time_interval_seconds' parameter must be an integer.")
if not isinstance(kwargs["min_valid_length_minutes"], int):
    raise ValueError("'min_valid_length_minutes' parameter must be an integer.")
if not isinstance(kwargs["allowed_invalid_region_length_seconds"], int):
    raise ValueError("'allowed_invalid_region_length_seconds' parameter must be an integer.")


def create_sub_dict(dictionary, keys):
    """
    Create a sub dictionary of the main one.
    """
    return {key: dictionary[key] for key in keys}


"""
In this section is the function provided to calculate the thresholds needed for various 
other functions.

Check that the test data and the intervals in which it is used align with the purpose.
Also examine whether the test data used is suitable for the actual data, e.g. the physical
units match, etc.
"""


def calculate_thresholds(
        file_path: str, 
        ecg_threshold_multiplier: float,
        threshold_dezimal_places: int,
        show_calibration_data = False
    ):
    """
    This function calculates the thresholds needed in various functions.
    Please note that the intervals are chosen manually and might need to be adjusted, if 
    you can't use the test data. In this case, you can use this function to plot the data 
    in the given intervals to see what the test data should look like (see ARGUMENTS).

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file for threshold calibration
    ecg_threshold_multiplier: float
        multiplier for the thresholds in check_data.check_ecg()
    threshold_dezimal_places: int
        number of dezimal places for the thresholds in the pickle files
    show_graphs: bool, default False
        if True, the data will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
        if False, the thresholds will be calculated and saved to THRESHOLD_DIRECTORY

    RETURNS:
    --------------------------------
    None, but the thresholds are saved to a pickle file
    """

    # Load the data
    sigbufs, sigfreqs, duration = read_edf.get_edf_data(file_path)

    # Calculate thresholds for check_data.check_ecg()
    if os.path.isfile(CHECK_ECG_DATA_THRESHOLDS) and not show_calibration_data:
        first_try = True
        while True:
            if first_try:
                user_answer = input("Thresholds for check_data.check_ecg() already exist. Are you sure you want to overwrite them? (y/n)")
            else:
                user_answer = input("Please answer with 'y' or 'n'.")
            if user_answer == "y":
                os.remove(CHECK_ECG_DATA_THRESHOLDS)
                break
            elif user_answer == "n":
                print("Thresholds for check_data.check_ecg() were not overwritten. Continuing with the existing data.")
                break
            else:
                first_try = False
                print("Answer not recognized.")
    if not os.path.isfile(CHECK_ECG_DATA_THRESHOLDS) and not show_calibration_data:
        user_answer = "y"
   
    # Calibration intervals for check_data.check_ecg()
    interval_size = 2560 # 10 seconds for 256 Hz
    lower_borders = [
        2091000, # 2h 17min 10sec for 256 Hz
        6292992, # 6h 49min 41sec for 256 Hz
        2156544, # 2h 20min 24sec for 256 Hz
        1781760 # 1h 56min 0sec for 256 Hz
        ]
    detection_intervals = [(border, border + interval_size) for border in lower_borders]

    # Plot the data if show_graphs is True
    if show_calibration_data:
        names = ["perfect_ecg", "fluctuating_ecg", "noisy_ecg", "negative_peaks_ecg"]
        for interval in detection_intervals:
            NNPH.simple_plot(sigbufs["ECG"][interval[0]:interval[1]], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + names[detection_intervals.index(interval)] + "_ten_sec.png")
        return
    
    # Calculate and save the thresholds
    if user_answer == "y":

        threshold_values = check_data.eval_thresholds_for_check_ecg(
            sigbufs, 
            detection_intervals,
            threshold_multiplier = ecg_threshold_multiplier,
            threshold_dezimal_places = threshold_dezimal_places,
            relevant_key = "ECG",
            )
        
        check_ecg_thresholds = dict()
        check_ecg_thresholds["check_ecg_std_min_threshold"] = threshold_values[0]
        check_ecg_thresholds["check_ecg_std_max_threshold"] = threshold_values[1]
        check_ecg_thresholds["check_ecg_distance_std_ratio_threshold"] = threshold_values[2]
        
        save_to_pickle(check_ecg_thresholds, CHECK_ECG_DATA_THRESHOLDS)


"""
Main Part of the code.
"""

# make sure temporary directories are empty
clear_directory(TEMPORARY_PICKLE_DIRECTORY_NAME)
clear_directory(TEMPORARY_FIGURE_DIRECTORY_PATH)


# calculate the thresholds or show how calibration data needed for this should look like
calculate_thresholds_args = create_sub_dict(kwargs, ["file_path", "ecg_threshold_multiplier", "show_calibration_data"])

if kwargs["show_calibration_data"]:
    calculate_thresholds(**calculate_thresholds_args)
    raise SystemExit(0)

if kwargs["calculate_thresholds"]:
    calculate_thresholds(**calculate_thresholds_args)

del calculate_thresholds_args

check_ecg_thresholds_dict = load_from_pickle(CHECK_ECG_DATA_THRESHOLDS)
kwargs.update(check_ecg_thresholds_dict)
del check_ecg_thresholds_dict

# load the data
sigbufs, sigfreqs, duration = read_edf.get_edf_data(kwargs["file_path"])

# evaluate valid regions for the ECG data
check_ecg_args = create_sub_dict(
    kwargs, ["check_ecg_std_min_threshold", "check_ecg_std_max_threshold", 
             "check_ecg_distance_std_ratio_threshold", "time_interval_seconds", 
             "min_valid_length_minutes", "allowed_invalid_region_length_seconds"]
    )

lower_border = 2091000
#lower_border = 19059968
interval_size = 153600
interval_size += 3000
sigbufs["ECG"] = sigbufs["ECG"][lower_border:lower_border+interval_size] # 5 minutes
print(check_data.check_ecg(sigbufs, sigfreqs, **check_ecg_args))


#print(MAD.calc_mad(sigbufs, sigfreqs, 60))

lower_border = 2091000 # normal -> (2.0075213675213663, 197.83946745575457, 199.84698882327595, 8.274984047278469, 8.19185941583606)
lower_border = 6292992 # normal with fluktuations -> (4.659731379731384, 266.368146782294, 271.0278781620254, 7.913620804139733, 7.777563408721908)
lower_border = 2156544 # normal but noisier -> (-27.746617826617822, 255.46153773709977, 227.71491991048197, 8.45031380399717, 7.622415231769258)
lower_border = 1781760 # normal but negative peaks -> (-6.9907692307692315, 205.32352678537083, 198.33275755460159, 8.068507699004185, 7.802839882852345)
lower_border = 661504 # hard noise -> (106.51741147741147, 683.9100807116326, 790.4274921890441, 6.121519633418348, 5.2965882739914845)
lower_border = 19059968 # extreme overkill -> (1215.042735042735, 3564.7805272632563, 4779.823262305991, 2.2436294007100166, 1.673293333065164)
lower_border = 18344704 # not as extreme overkill -> (131.8173382173382, 773.3280754044366, 905.1454136217749, 6.737446338711472, 5.7562644983290765)
lower_border = 17752064 # hard noise ->
#lower_border = 10756096 # small but weird spikes -> (-29.99404151404151, 13.22985961137362, -16.76418190266789, 7.97397013977477, 2.440652110238981)
#lower_border = 10788096 # continous flat, one large spike -> (-20.894163614163613, 21.48801387793023, 0.5938502637666154, 36.18452510085921, 18.345767573613273)
#lower_border = 10792704 # continous flat -> (-18.863980463980457, 1.626617368352779, -17.23736309562768, 15.61327567929753, 1.2394379902742687)
#lower_border = 15378176 # lots of noise -> (-7.123614163614163, 8.544405291879885, 1.420791128265722, 7.087873116133069, 3.86530414604664)
#lower_border = 15381248 # lots of noise, one spike -> (-6.0905494505494495, 15.850078468610418, 9.759529018060968, 21.323120806150065, 15.40398657770408)

#print(check_data.useful_thresholds(sigbufs, sigfreqs, relevant_key = "ECG", detection_interval = (lower_border, lower_border + 2500)))


