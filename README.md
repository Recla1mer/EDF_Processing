# EDF File Processing

Python project for processing .edf files to retrieve ECG and wrist accelerometry signals, enabling 
calculation of RR intervals and Mean Amplitude Deviation (MAD – values characterizing motion activity). 
This tool is designed to prepare data for use cases such as deep convolutional neural network-based sleep 
stage classification. (Teaser for my project: `Sleep_Stage_Classification` :D)

## Ensuring Smooth Data Access

Before running the code, open the file `project_parameters.py` to configure the file parameters in the 
'SETTING UNIFORM PARAMETERS' section. Here, you must specify which keys access the ECG and wrist accelerometry 
data in your .edf files. If your data uses different keys across files, add them all to ensure they can be 
accessed.

**Note:** If your data is not stored in .edf format, you can still use this project. Check section:
`No EDF? No Problem!` below!

## Quick Start

To begin processing, open `main.py` and use the `Data_Processing` function. 
Provide a list of paths to directories containing your .edf files and define the output path for saving results.

During processing, the program will execute the following:

- ECG Validation: Evaluate where ECG data was recorded correctly to determine evaluatable segments
- R-peak Detection: Detect R-peak locations in the valid segments of the ECG data using specified detectors
- RRI Calculation: Calculate RR-Intervals from detected R-peak locations
- MAD Calculation: Calculate Mean Amplitude Deviation from wrist accelerometry data

During the `Data_Processing` function, RRI values are calculated only for specific time periods where the 
ECG data is deemed evaluable. In contrast, MAD values are computed over the entire signal length, as their 
calculation is less affected by noise.

To align the MAD values with the same time periods as the RRI values, you can use the `Extract_RRI_MAD` 
function. This process generates a separate file that requires significantly less storage space.

## Accessing Results

The structure and format of the `Data_Processing` results are detailed in the documentation. However, if your 
primary interest lies in obtaining the RRI and MAD values, you can skip this and refer directly to the 
documentation for the `Extract_RRI_MAD` function instead.

The Jupyter Notebook `Access_Results.ipynb` demonstrates how to access and visualize the results
of your computation.

## Comparing Results

If you have existing results for ECG validation, R-peak detection, RRI and MAD calculation that you'd like to
compare with those generated by this project, you can use the `Data_Processing_and_Comparing` function instead. 
For this comparison, you’ll need to specify additional information, such as the path to the existing results.

The program will then generate four .txt files, providing a summary of correspondence between the compared 
results.

## No EDF? No Problem!

This project is designed to process .edf files conveniently. It nevertheless shouldn't take you too much
effort to use it for any kind of data source. As I see it, you have two options:

### Accessing the functions used for the individual processing operations

The Jupyter Notebook `Processing_Demo.ipynb` was designed to test if my processing functions work correctly. 
It is therefore a quick summary of the important processing functions. Just replace the data source in the 
cells and you are good to go!

### Replace the Data Access function

This one takes a little more work, but should still be accomplishable rather quickly.
Simply replace the functionality of the `get_data_from_edf_channel` function in `read_edf.py` to handle your 
data format. Then, modify the file parameters (as outlined in section: `Ensuring Smooth Data Access`) to 
specify the file type you are working with.

Afterwards you should be able to use the main.py file as documented.