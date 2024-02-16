import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from matplotlib.colors import Normalize
from matplotlib import colors
from scipy.integrate import cumtrapz
import numpy as np
from scipy.io import wavfile
import wave
import struct
from PIL import Image

def compute_absolute_values(vector):
    absolute_values = [abs(sample) for sample in vector]
    return absolute_values

def normalize_audio(file_path):
    # Open the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Convert byte buffer to numpy array
    audio_data = data

    # Normalize audio levels between -1 and 1
    normalized_data = audio_data / np.max(np.abs(audio_data))

    return normalized_data

def find_peaks(normalized_data, threshold):
    peaks = []
    peak_positions = []
    i = 0

    while i < len(normalized_data):
        if normalized_data[i] > threshold:
            start = i
            end = i

            # Variables to track the largest peak within the range
            max_peak = normalized_data[start]
            max_peak_position = start

            # Find consecutive peaks within a range of 100 samples
            while end - start < 100 and end < len(normalized_data) - 1 :
                end += 1

                # Update the largest peak within the range
                if normalized_data[end] > max_peak:
                    max_peak = normalized_data[end]
                    max_peak_position = end

            # Keep the largest peak and its position
            peaks.append(max_peak)
            peak_positions.append(max_peak_position)

            # Skip to the next sample after the found peaks
            i = end + 1
        else:
            i += 1

    return peaks, peak_positions

def subtract_consecutive_elements(input_vector):
    result_vector = []
    for i in range(len(input_vector) - 1):
        diff = input_vector[i+1] - input_vector[i]
        result_vector.append(diff)
    return result_vector

def find_pulse_widths(signal_vector, pulse_indices):
    widths = []

    for pulse_index in pulse_indices:
        pulse_start = max(0, pulse_index - 5)
        pulse_end = min(len(signal_vector), pulse_index + 6)
        pulse = signal_vector[pulse_start:pulse_end]
        peak = max(pulse)
        threshold = 0.1 * peak
        
        before_pulse_width = 0
        after_pulse_width = 0
        
        # Search backwards from the peak
        while ((pulse_index - before_pulse_width >= 0)) and (signal_vector[pulse_index - before_pulse_width] > threshold or signal_vector[pulse_index - before_pulse_width-1] > threshold or signal_vector[pulse_index - before_pulse_width-2] > threshold ):
            before_pulse_width += 1

        
        # Search forwards from the peak
        while (pulse_index + after_pulse_width + 1 < len(signal_vector)) and (signal_vector[pulse_index + after_pulse_width] > threshold or signal_vector[pulse_index + after_pulse_width+1] > threshold or signal_vector[pulse_index + after_pulse_width+2] > threshold ):
            after_pulse_width += 1
        
        pulse_width = before_pulse_width + after_pulse_width
        widths.append(pulse_width)

    return widths

def find_peak_frequency(vector, indices, sample_rate):
    peak_frequencies = []

    for index in indices:
        start_index = max(0, index - 70)
        end_index = min(len(vector), index + 71)
        range_samples = vector[start_index:end_index]

        fft_result = np.fft.fft(range_samples)
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.fftfreq(len(range_samples), d=1/sample_rate)
        
        positive_frequencies = frequencies[:len(frequencies)//2]
        positive_magnitudes = magnitudes[:len(frequencies)//2]
        
        peak_magnitude_index = np.argmax(positive_magnitudes)
        peak_frequency = positive_frequencies[peak_magnitude_index]

        peak_frequencies.append(peak_frequency)

    return peak_frequencies

def calculate_average(vector):
    if len(vector) == 0:
        return None

    total = sum(vector)
    average = total / len(vector)
    return average

def remove_greater_than(vector):
    modified_vector = [x for x in vector if x <= 1]
    return modified_vector

wav_file_path = 'C:\\Users\\Luis\\Documents\\MATLAB\\Blainville_2.WAV'
threshold_level = 0.3

normalized_audio = normalize_audio(wav_file_path)
peaks_above_threshold, positions = find_peaks(normalized_audio, threshold_level)
time_intervals= np.array(subtract_consecutive_elements(positions)) /96000
power_signal = compute_absolute_values(normalized_audio)
withs = np.array(find_pulse_widths(power_signal,positions))*1e6 /96000
frequencies = find_peak_frequency(normalized_audio,positions,96000)
print(time_intervals)
time_intervals = remove_greater_than(time_intervals)

print("ICI: ",calculate_average(time_intervals))
print("Click: ", calculate_average(withs))
print("Frequency: ",calculate_average(frequencies))
