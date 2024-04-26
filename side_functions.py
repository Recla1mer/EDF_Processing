"""
Author: Johannes Peter Knoll

--------------------------------
SIDE FUNCTIONS 
--------------------------------

In this file we provide functions to keep the other files a little cleaner and more intuitive.
"""

# IMPORTS
import os
import pickle

def progress_bar(index, total, bar_len=50, title='Please wait'):
    '''
    Source: https://stackoverflow.com/questions/6169217/replace-console-output-in-python
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


def get_file_name_from_path(file_path):
    """
    Separate the file name and the extension of a file.
    """
    for i in range(len(file_path)-1, -1, -1):
        if file_path[i] == "/":
            return file_path[i+1:]


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
                user_answer = input(message + " Are you sure you want to overwrite them? (y/n)")
            else:
                user_answer = input("Please answer with 'y' or 'n'.")
            if user_answer == "y":
                os.remove(file_path)
                break
            elif user_answer == "n":
                print("Existing Data was not overwritten. Continuing...")
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


def create_rpeaks_pickle_path(Directory, rpeak_function_name):
    """
    Create the path for the pickle file where the rpeaks are saved for each method.
    """
    return Directory + "RPeaks_" + rpeak_function_name + ".pkl"