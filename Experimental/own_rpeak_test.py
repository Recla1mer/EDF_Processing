
import numpy as np
import matplotlib.pyplot as plt
import read_edf
edf_file_path = "Test_Data/Somnowatch_Messung.edf"
sigbufs, sigfreqs, duration = read_edf.get_edf_data(edf_file_path)

gap = 1250
gap = 2500
lower_border = 2090000
lower_border =  2247100
lower_border = 4046592
upper_border = lower_border + gap

def detect_in_interval(signal, interval, sampling_rate, steps=100, accumulate_steps=3, threshold=0.2):
    """
    Detect the peaks in the interval of the signal.
    """
    unusual_low_peak_distance_in_seconds = 0.2
    max_peak_width = unusual_low_peak_distance_in_seconds * sampling_rate
    max_peak_width = int(max_peak_width)

    max_sig = max(signal[interval[0]:interval[1]])
    min_sig = min(signal[interval[0]:interval[1]])
    step_size = int((max_sig - min_sig) / steps)
    collection_values = np.arange(min_sig, max_sig, step_size)
    num_of_points_below_value = np.zeros(len(collection_values))
    for sig_i in np.arange(interval[0], interval[1]):
        for val_i in range(len(collection_values)):
            if signal[sig_i] > collection_values[val_i]:
                num_of_points_below_value[val_i] += 1
            else:
                break
    
    difference_between_points = np.diff(num_of_points_below_value)
    nested_differences = []
    for i in range(0,len(difference_between_points-accumulate_steps), accumulate_steps):
        nested_differences.append(abs(np.sum(difference_between_points[i:i+accumulate_steps])))
    for i in range(0,len(nested_differences)-1):
        if abs(nested_differences[i] - nested_differences[i+1]) < threshold*nested_differences[i]:
            line_at = i
            break
    line_at = line_at*accumulate_steps+1
    #print("Line at:", line_at, "Value:", collection_values[line_at])
    rpeaks = []
    collection_of_positions = []
    last_sig_i = interval[0]
    for sig_i in np.arange(interval[0], interval[1]):
        if signal[sig_i] > collection_values[line_at]:
            if sig_i - last_sig_i > max_peak_width:
                rpeaks.append(collection_of_positions)
                last_sig_i = sig_i
                collection_of_positions = []
            collection_of_positions.append(sig_i)

    for i in range(len(rpeaks)-1,-1,-1):
        if len(rpeaks[i]) == 0:
            rpeaks.pop(i)

    real_peak = []
    for i in range(len(rpeaks)):
        index_of_max = np.argmax(signal[rpeaks[i]])
        real_peak.append(rpeaks[i][index_of_max])
    print("R peaks:", real_peak)
    
    return real_peak
            

rpeaks = detect_in_interval(sigbufs["ECG"], [lower_border, upper_border], sigfreqs["ECG"])

fig, ax = plt.subplots()

ax.plot(np.arange(lower_border, upper_border), sigbufs["ECG"][lower_border:upper_border], label="ECG")
ax.plot(rpeaks, sigbufs["ECG"][rpeaks], "ro", label="R peaks")
ax.legend(loc="best")

plt.show()
        
        