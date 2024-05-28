import main
import random

PREPARATION_DIRECTORY = main.PREPARATION_DIRECTORY
PREPARATION_RESULTS_NAME = main.PREPARATION_RESULTS_NAME

parameters = main.parameters

# choose a random file
data_directory = "Data/GIF/SOMNOwatch/"
file_data_name = "SL001_SL001_(1).edf"

file_data_path = data_directory + file_data_name
additions_results_path = parameters["additions_results_path"]

# choose size of interval
interval_size = 2560

# get r-peak function names ("wfdb", "ecgdetectors", "hamilton", "christov", "gif_classification")
first_rpeak_function_name = "wfdb"
second_rpeak_function_name = "gif_classification"

# load the valid regions
additions_results_generator = main.load_from_pickle(additions_results_path)
for generator_entry in additions_results_generator:
    if generator_entry[parameters["file_name_dictionary_key"]] == file_data_name:
        first_rpeaks = generator_entry[first_rpeak_function_name]
        second_rpeaks = generator_entry[second_rpeak_function_name]
        break

# load the ECG data
ECG, frequency = main.read_edf.get_data_from_edf_channel(
    file_path = file_data_path,
    possible_channel_labels = parameters["ecg_keys"],
    physical_dimension_correction_dictionary = parameters["physical_dimension_correction_dictionary"]
    )

# combine the r-peaks, retrieve the intersected r-peaks and the r-peaks that are only in the first or second list
rpeaks_intersected, rpeaks_only_primary, rpeaks_only_secondary = main.rpeak_detection.combine_rpeaks(
    rpeaks_primary = first_rpeaks,
    rpeaks_secondary = second_rpeaks,
    frequency = frequency,
    rpeak_distance_threshold_seconds = parameters["rpeak_distance_threshold_seconds"]
)

# choose random r-peak for plotting
random_first_rpeak = random.choice(rpeaks_only_primary)
random_second_rpeak = random.choice(rpeaks_only_secondary)
random_rpeak = random.choice([random_first_rpeak, random_second_rpeak])
print("Random r-peak location: %d" % random_rpeak)

# nice values for plotting
# random_rpeak = 10625000 # for Data/GIF/SOMNOwatch/SL001_SL001_(1).edf, wfdb, gif_classification

x_lim = [int(random_rpeak-interval_size/2), int(random_rpeak+interval_size/2)]

main.plot_helper.plot_rpeak_detection(
    ECG = ECG,
    rpeaks = [rpeaks_only_primary, rpeaks_only_secondary, rpeaks_intersected],
    rpeaks_name = ["only " + first_rpeak_function_name, "only " + second_rpeak_function_name, "intersected"],
    xlim = x_lim)