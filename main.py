"""
Author: Johannes Peter Knoll

Main python file for the neural network project.
"""

# import libraries
import numpy as np
import os
import pickle
import time

# import secondary python files
import read_edf
import MAD
import rpeak_detection
import check_data
import plot_helper


"""
--------------------------------
PARAMETERS AND FILE SECTION
--------------------------------

In this section we set the parameters for the project and define the file/directory names.
"""

# define directory and file names (will always be written in capital letters)
DATA_DIRECTORY = "Data/"

TEMPORARY_PICKLE_DIRECTORY_NAME = "Temporary_Pickles/"
TEMPORARY_FIGURE_DIRECTORY_PATH = "Temporary_Figures/"

PREPARATION_DIRECTORY = "Preparation/"

CHECK_ECG_DATA_THRESHOLDS_PATH = PREPARATION_DIRECTORY + "Check_ECG_Data_Thresholds.pkl"
VALID_ECG_REGIONS_PATH = PREPARATION_DIRECTORY + "Valid_ECG_Regions.pkl"

CERTAIN_RPEAKS_PATH = PREPARATION_DIRECTORY + "Certain_Rpeaks.pkl"
UNCERTAIN_PRIMARY_RPEAKS_PATH = PREPARATION_DIRECTORY + "Uncertain_Primary_Rpeaks.pkl"
UNCERTAIN_SECONDARY_RPEAKS_PATH = PREPARATION_DIRECTORY + "Uncertain_Secondary_Rpeaks.pkl"

CALIBRATION_DATA_PATH = "Calibration_Data/Somnowatch_Messung.edf"

# create directories if they do not exist
if not os.path.isdir(TEMPORARY_PICKLE_DIRECTORY_NAME):
    os.mkdir(TEMPORARY_PICKLE_DIRECTORY_NAME)
if not os.path.isdir(TEMPORARY_FIGURE_DIRECTORY_PATH):
    os.mkdir(TEMPORARY_FIGURE_DIRECTORY_PATH)
if not os.path.isdir(PREPARATION_DIRECTORY):
    os.mkdir(PREPARATION_DIRECTORY)

# set the parameters for the project
parameters = dict()

file_params = {
    "file_path": CALIBRATION_DATA_PATH, # path to the EDF file for threshold calibration
    "data_directory": DATA_DIRECTORY, # directory where the data is stored
    "valid_file_types": [".edf"], # valid file types in the data directory
    "ecg_key": "ECG", # key for the ECG data in the data dictionary
    "wrist_acceleration_keys": ["X", "Y", "Z"], # keys for the wrist acceleration data in the data dictionary
}
check_ecg_params = {
    "determine_valid_ecg_regions": True, # if True, the valid regions for the ECG data will be determined
    "calculate_thresholds": True, # if True, you will have the option to recalculate the thresholds for various functions
    "show_calibration_data": False, # if True, the calibration data in the manually chosen intervals will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
    "ecg_threshold_multiplier": 0.5, # multiplier for the thresholds in check_data.check_ecg() (between 0 and 1)
    "check_ecg_threshold_dezimal_places": 2, # number of dezimal places for the check ecg thresholds in the pickle files
    "check_ecg_time_interval_seconds": 10, # time interval considered when determining the valid regions for the ECG data
    "min_valid_length_minutes": 5, # minimum length of valid data in minutes
    "allowed_invalid_region_length_seconds": 30, # data region (see above) still considered valid if the invalid part is shorter than this
}
detect_rpeaks_params = {
    "detect_rpeaks": True, # if True, the R peaks will be detected in the ECG data
    "rpeak_primary_function": rpeak_detection.get_rpeaks_wfdb, # primary R peak detection function
    "rpeak_secondary_function": rpeak_detection.get_rpeaks_old, # secondary R peak detection function
    "rpeak_name_primary": "wfdb", # name of the primary R peak detection function
    "rpeak_name_secondary": "ecgdetectors", # name of the secondary R peak detection function
    "rpeak_distance_threshold_seconds": 0.1, # max 50ms
}

parameters.update(file_params)
parameters.update(check_ecg_params)
parameters.update(detect_rpeaks_params)

del file_params
del check_ecg_params
del detect_rpeaks_params

# following parameters are calculated in the PREPARATION section. They are written here for better overview and explanation
params_to_be_calculated = {
    "check_ecg_std_min_threshold": 97.84, # if the standard deviation of the ECG data is below this threshold, the data is considered invalid
    "check_ecg_std_max_threshold": 530.62, # if the standard deviation of the ECG data is above this threshold, the data is considered invalid
    "check_ecg_distance_std_ratio_threshold": 1.99, # if the ratio of the distance between two peaks and twice the standard deviation of the ECG data is above this threshold, the data is considered invalid
    "valid_ecg_regions": dict() # dictionary containing the valid regions for the ECG data
}

# check the parameters
if not isinstance(parameters["file_path"], str):
    raise ValueError("'file_path' parameter must be a string.")
if not isinstance(parameters["data_directory"], str):
    raise ValueError("'data_directory' parameter must be a string.")
if not isinstance(parameters["valid_file_types"], list):
    raise ValueError("'valid_file_types' parameter must be a list.")
if not isinstance(parameters["ecg_key"], str):
    raise ValueError("'ecg_key' parameter must be a string.")
if not isinstance(parameters["wrist_acceleration_keys"], list):
    raise ValueError("'wrist_acceleration_keys' parameter must be a list.")

if not isinstance(parameters["determine_valid_ecg_regions"], bool):
    raise ValueError("'determine_valid_ecg_regions' parameter must be a boolean.")
if not isinstance(parameters["calculate_thresholds"], bool):
    raise ValueError("'calculate_thresholds' parameter must be a boolean.")
if not isinstance(parameters["show_calibration_data"], bool):
    raise ValueError("'show_calibration_data' parameter must be a boolean.")
if parameters["show_calibration_data"] and parameters["calculate_thresholds"]:
    raise ValueError("'show_calibration_data' and 'calculate_thresholds' parameter cannot both be True at the same time.")
if not isinstance(parameters["ecg_threshold_multiplier"], (int, float)):
    raise ValueError("'ecg_threshold_multiplier' parameter must be an integer or a float.")
if parameters["ecg_threshold_multiplier"] <= 0 or parameters["ecg_threshold_multiplier"] > 1:
    raise ValueError("'ecg_threshold_multiplier' parameter must be between 0 and 1.")
if not isinstance(parameters["check_ecg_threshold_dezimal_places"], int):
    raise ValueError("'check_ecg_threshold_dezimal_places' parameter must be an integer.")
if not isinstance(parameters["check_ecg_time_interval_seconds"], int):
    raise ValueError("'check_ecg_time_interval_seconds' parameter must be an integer.")
if not isinstance(parameters["min_valid_length_minutes"], int):
    raise ValueError("'min_valid_length_minutes' parameter must be an integer.")
if not isinstance(parameters["allowed_invalid_region_length_seconds"], int):
    raise ValueError("'allowed_invalid_region_length_seconds' parameter must be an integer.")

if not isinstance(parameters["detect_rpeaks"], bool):
    raise ValueError("'detect_rpeaks' parameter must be a boolean.")
if not callable(parameters["rpeak_primary_function"]):
    raise ValueError("'rpeak_primary_function' parameter must be a function.")
if not callable(parameters["rpeak_secondary_function"]):
    raise ValueError("'rpeak_secondary_function' parameter must be a function.")
if not isinstance(parameters["rpeak_name_primary"], str):
    raise ValueError("'rpeak_name_primary' parameter must be a string.")
if not isinstance(parameters["rpeak_name_secondary"], str):
    raise ValueError("'rpeak_name_secondary' parameter must be a string.")
if not isinstance(parameters["rpeak_distance_threshold_seconds"], float):
    raise ValueError("'rpeak_distance_threshold_seconds' parameter must be a float.")


"""
--------------------------------
HELPER FUNCTIONS SECTION
--------------------------------

In this section we provide small functions to keep the code a little cleaner.
"""
def print_percent_done(index, total, bar_len=50, title='Please wait'):
    '''
    Source: https://stackoverflow.com/questions/6169217/replace-console-output-in-python

    index is expected to be 0 based index. 
    0 <= index < total
    '''
    percent_done = index/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')

    if round(percent_done) == 100:
        print('\t✅')

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


def get_file_type(file_name):
    """
    Get the file type of a file.
    """
    return os.path.splitext(file_name)[1]


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


def ask_for_permission_to_override(file_path: str, message: str):
    """
    If a file already exists, ask the user if they want to overwrite it.

    RETURNS:
    --------------------------------
    user_answer: str
        "y" if the user wants to overwrite the file, "n" if not
    """
    if os.path.isfile(file_path):
        first_try = True
        while True:
            if first_try:
                user_answer = input(message + " already exist. Are you sure you want to overwrite them? (y/n)")
            else:
                user_answer = input("Please answer with 'y' or 'n'.")
            if user_answer == "y":
                os.remove(file_path)
                break
            elif user_answer == "n":
                print(message + " were not overwritten. Continuing with the existing data.")
                break
            else:
                first_try = False
                print("Answer not recognized.")
    else:
        user_answer = "y"
    
    return user_answer


def create_sub_dict(dictionary, keys):
    """
    Create a sub dictionary of the main one.
    """
    return {key: dictionary[key] for key in keys}


"""
Following functions are needed in the PREPARATION section of the project.

These are used to calculate thresholds and evaluate valid regions for the ECG data.

ATTENTION:
Check that the test data and the intervals in which it is used align with the purpose.
Also examine whether the test data used is suitable for the actual data, e.g. the physical
units match, etc.
"""


def calculate_thresholds(
        file_path: str, 
        ecg_threshold_multiplier: float,
        check_ecg_threshold_dezimal_places: int,
        show_calibration_data: bool,
        ecg_key: str,
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
    check_ecg_threshold_dezimal_places: int
        number of dezimal places for the check ecg thresholds in the pickle files
    show_graphs: bool, default False
        if True, the data will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
        if False, the thresholds will be calculated and saved to THRESHOLD_DIRECTORY

    RETURNS:
    --------------------------------
    None, but the thresholds are saved to a pickle file
    """
    # Load the data
    sigbufs, sigfreqs, duration = read_edf.get_edf_data(file_path)

    # check if ecg thresholds already exist and if yes: ask for permission to override
    if show_calibration_data:
        user_answer = "n"
    else:
        user_answer = ask_for_permission_to_override(file_path = CHECK_ECG_DATA_THRESHOLDS_PATH, 
                                        message = "Thresholds for check_data.check_ecg()")
   
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
        names = ["perfect_ecg", "fluctuating_ecg", "noisy_ecg", "negative_peaks"]
        for interval in detection_intervals:
            plot_helper.simple_plot(sigbufs[ecg_key][interval[0]:interval[1]], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + names[detection_intervals.index(interval)] + "_ten_sec.png")
    
    # Calculate and save the thresholds for check_data.check_ecg()
    if user_answer == "y":

        threshold_values = check_data.eval_thresholds_for_check_ecg(
            sigbufs, 
            detection_intervals,
            threshold_multiplier = ecg_threshold_multiplier,
            threshold_dezimal_places = check_ecg_threshold_dezimal_places,
            ecg_key = ecg_key,
            )
        
        check_ecg_thresholds = dict()
        check_ecg_thresholds["check_ecg_std_min_threshold"] = threshold_values[0]
        check_ecg_thresholds["check_ecg_std_max_threshold"] = threshold_values[1]
        check_ecg_thresholds["check_ecg_distance_std_ratio_threshold"] = threshold_values[2]
        
        save_to_pickle(check_ecg_thresholds, CHECK_ECG_DATA_THRESHOLDS_PATH)

        del threshold_values
    del detection_intervals
    
    #end for check_ecg thresholds


def determine_valid_ecg_regions(
        data_directory: str,
        valid_file_types: list,
        check_ecg_std_min_threshold: float, 
        check_ecg_std_max_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        check_ecg_time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        ecg_key: str
    ):
    """
    Determine the valid ECG regions for all valid file types in the given data directory.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    others: see check_data.check_ecg()

    RETURNS:
    --------------------------------
    None, but the valid regions are saved to a pickle file
    
    """
    user_answer = ask_for_permission_to_override(file_path = VALID_ECG_REGIONS_PATH,
                                                message = "Valid regions for the ECG data")
    
    if user_answer == "n":
        return
    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    total_files = len(valid_files)
    progressed_files = 0

    valid_regions = dict()

    print("Calculating valid regions for the ECG data in %i files:" % total_files)
    for file in valid_files:
        print_percent_done(progressed_files, total_files)
        sigbufs, sigfreqs, duration = read_edf.get_edf_data(data_directory + file)
        valid_regions[file] = check_data.check_ecg(
            sigbufs, 
            sigfreqs, 
            check_ecg_std_min_threshold = check_ecg_std_min_threshold, 
            check_ecg_std_max_threshold = check_ecg_std_max_threshold, 
            check_ecg_distance_std_ratio_threshold = check_ecg_distance_std_ratio_threshold,
            time_interval_seconds = check_ecg_time_interval_seconds, 
            min_valid_length_minutes = min_valid_length_minutes,
            allowed_invalid_region_length_seconds = allowed_invalid_region_length_seconds,
            ecg_key = ecg_key
            )
        progressed_files += 1
    print_percent_done(progressed_files, total_files)
    
    save_to_pickle(valid_regions, VALID_ECG_REGIONS_PATH)


def detect_rpeaks_in_ecg_data(
        data_directory: str,
        valid_file_types: list,
        ecg_key: str,
        rpeak_primary_function,
        rpeak_secondary_function,
        rpeak_distance_threshold_seconds: float,
    ):
    """
    Detect R peaks in the ECG data and compare the results of two different functions.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    others: see rpeak_detection.compare_rpeak_detection_methods()

    RETURNS:
    --------------------------------
    None, but the valid regions are saved to a pickle file
    """
    user_answer = ask_for_permission_to_override(file_path = CERTAIN_RPEAKS_PATH,
                                                message = "Detected R peaks")
    
    if user_answer == "n":
        return
    
    os.remove(UNCERTAIN_PRIMARY_RPEAKS_PATH)
    os.remove(UNCERTAIN_SECONDARY_RPEAKS_PATH)

    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    total_files = len(valid_files)
    progressed_files = 0

    certain_rpeaks = dict()
    uncertain_primary_rpeaks = dict()
    uncertain_secondary_rpeaks = dict()

    # load valid ecg regions
    valid_ecg_regions = load_from_pickle(VALID_ECG_REGIONS_PATH)

    # detect rpeaks in the valid regions of the ECG data
    print("Detecting R peaks in the ECG data in %i files:" % total_files)
    for file in valid_files:
        print_percent_done(progressed_files, total_files)
        try:
            detection_intervals = valid_ecg_regions[file]
            progressed_files += 1
        except KeyError:
            print("Valid regions for the ECG data in " + file + " are missing. Skipping this file.")
            continue
        sigbufs, sigfreqs, duration = read_edf.get_edf_data(data_directory + file)
        this_certain_rpeaks = np.array([], dtype = int)
        this_uncertain_primary_rpeaks = np.array([], dtype = int)
        this_uncertain_secondary_rpeaks = np.array([], dtype = int)
        for interval in detection_intervals:
            this_result = rpeak_detection.combined_rpeak_detection_methods(
                sigbufs, 
                sigfreqs, 
                ecg_key,
                interval, 
                rpeak_primary_function,
                rpeak_secondary_function,
                rpeak_distance_threshold_seconds,
                )
            this_certain_rpeaks = np.append(this_certain_rpeaks, this_result[0])
            this_uncertain_primary_rpeaks = np.append(this_uncertain_primary_rpeaks, this_result[1])
            this_uncertain_secondary_rpeaks = np.append(this_uncertain_secondary_rpeaks, this_result[2])
        
        certain_rpeaks[file] = this_certain_rpeaks
        uncertain_primary_rpeaks[file] = this_uncertain_primary_rpeaks
        uncertain_secondary_rpeaks[file] = this_uncertain_secondary_rpeaks
    
    print_percent_done(progressed_files, total_files)
    
    save_to_pickle(certain_rpeaks, CERTAIN_RPEAKS_PATH)
    save_to_pickle(uncertain_primary_rpeaks, UNCERTAIN_PRIMARY_RPEAKS_PATH)
    save_to_pickle(uncertain_secondary_rpeaks, UNCERTAIN_SECONDARY_RPEAKS_PATH)
        


"""
--------------------------------
PREPARATION SECTION
--------------------------------

In this section we will make preparations for the main part of the project. Depending on
the parameters set in the kwargs dictionary, we will calculate the thresholds needed for
various functions, evaluate the valid regions for the ECG data or just load these
informations, if this was already done before.
"""

def preparation_section():
            
    # make sure temporary directories are empty
    clear_directory(TEMPORARY_PICKLE_DIRECTORY_NAME)
    clear_directory(TEMPORARY_FIGURE_DIRECTORY_PATH)

    # calculate the thresholds or show how calibration data needed for this should look like
    calculate_thresholds_args = create_sub_dict(
        parameters, ["file_path", "ecg_threshold_multiplier", "check_ecg_threshold_dezimal_places",
                    "show_calibration_data", "ecg_key"]
        )

    if parameters["show_calibration_data"]:
        calculate_thresholds(**calculate_thresholds_args)
        raise SystemExit(0)

    if parameters["calculate_thresholds"]:
        calculate_thresholds(**calculate_thresholds_args)

    del calculate_thresholds_args

    # load the thresholds to the parameters dictionary
    check_ecg_thresholds_dict = load_from_pickle(CHECK_ECG_DATA_THRESHOLDS_PATH)
    parameters.update(check_ecg_thresholds_dict)
    del check_ecg_thresholds_dict

    # evaluate valid regions for the ECG data
    if parameters["determine_valid_ecg_regions"]:
        determine_ecg_region_args = create_sub_dict(
            parameters, ["data_directory", "valid_file_types", "check_ecg_std_min_threshold", 
                        "check_ecg_std_max_threshold", "check_ecg_distance_std_ratio_threshold", 
                        "check_ecg_time_interval_seconds", "min_valid_length_minutes", 
                        "allowed_invalid_region_length_seconds", "ecg_key"]
            )
        determine_valid_ecg_regions(**determine_ecg_region_args)
        del determine_ecg_region_args

    # detect R peaks in the valid regions of the ECG data
    if parameters["detect_rpeaks"]:
        detect_rpeaks_args = create_sub_dict(
            parameters, ["data_directory", "valid_file_types", "ecg_key", "rpeak_primary_function",
                        "rpeak_secondary_function", "rpeak_distance_threshold_seconds"]
            )
        detect_rpeaks_in_ecg_data(**detect_rpeaks_args)
        del detect_rpeaks_args


"""
--------------------------------
MAIN SECTION
--------------------------------

In this section we will run the functions we have created until now.
"""

preparation_section()

# Testing
"""
sigbufs, sigfreqs, duration = read_edf.get_edf_data(parameters["file_path"])
lower_border = 2091000
#lower_border = 19059968
interval_size = 153600
#interval_size += 3000
sigbufs[parameters["ecg_key"]] = sigbufs[parameters["ecg_key"]][lower_border:lower_border+interval_size] # 5 minutes
print(check_data.check_ecg(sigbufs, sigfreqs, parameters["check_ecg_std_min_threshold"], parameters["check_ecg_std_max_threshold"], parameters["check_ecg_distance_std_ratio_threshold"], parameters["check_ecg_time_interval_seconds"], parameters["min_valid_length_minutes"], parameters["allowed_invalid_region_length_seconds"], parameters["ecg_key"]))
"""

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


