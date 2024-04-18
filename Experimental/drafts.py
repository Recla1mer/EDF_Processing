"""
This file contains functions that were discarded during the project.
"""


def create_calibration_data(
        file_path, 
        show_graphs = False
    ):
    """
    Create calibration data. Please note that the intervals are chosen manually and might 
    need to be adjusted, if you can't use the test data. In this case, you should at least
    plot the data in the given intervals to see what the data should look like 
    (see ARGUMENTS).

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the EDF file
    show_graphs: bool, default False
        if True, the data will be plotted and saved to TEMPORARY_FIGURE_DIRECTORY_PATH
        if False, the data will be saved to pickle files

    RETURNS:
    --------------------------------
    None
    """
    if len(os.listdir(CALIBRATION_DATA_DIRECTORY)) > 0 and not show_graphs:
        first_try = True
        while True:
            if first_try:
                user_answer = input("Calibration data already exists. Are you sure you want to overwrite it? (y/n)")
            else:
                user_answer = input("Please answer with 'y' or 'n'.")
            if user_answer != "y":
                clear_directory(CALIBRATION_DATA_DIRECTORY)
            elif user_answer == "n":
                print("Calibration data was not overwritten. Continuing with the existing data.")
                return
            else:
                first_try = False
                print("Answer not recognized.")

    sigbufs, sigfreqs, duration = read_edf.get_edf_data(file_path)
   
    # Calibration data for check_data
    interval_size = 2560 # 10 seconds for 256 Hz
    
    lower_border = 2091000 # 2h 17min 10sec for 256 Hz  
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "perfect_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], PERFECT_ECG_TEN_SEC)

    lower_border = 6292992 # 6h 49min 41sec for 256 Hz
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "fluctuating_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], FLUCTUATING_ECG_TEN_SEC)

    lower_border = 2156544 # 2h 20min 24sec for 256 Hz
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "noisy_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], NOISY_ECG_TEN_SEC)

    lower_border = 1781760 # 1h 56min 0sec for 256 Hz
    if show_graphs:
        NNPH.simple_plot(sigbufs["ECG"][lower_border:lower_border + interval_size], np.arange(interval_size), TEMPORARY_FIGURE_DIRECTORY_PATH + "negative_peaks_ecg_ten_sec.png")
    else:
        save_to_pickle(sigbufs[lower_border:lower_border + interval_size], NEGATIVE_PEAKS_ECG_TEN_SEC)