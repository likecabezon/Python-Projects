import numpy as np
from scipy.interpolate import CubicSpline

def add_sine_wave_with_complex_and_frequency_vectors(sample_rate, sample_values, complex_amplitudes, frequency_values):
    """
    Add a sine wave to the sample values with complex vector of amplitudes and vector of frequency values.

    Args:
        sample_rate (float): The sample rate of the input signal.
        sample_values (numpy.ndarray): The input signal as a 1D numpy array.
        complex_amplitudes (numpy.ndarray): Complex amplitudes (amplitude and phase) for each time frame.
        frequency_values (numpy.ndarray): Frequency values for each time frame (in Hz).

    Returns:
        numpy.ndarray: The modified input signal with the added sine wave.
    """
    num_samples = len(sample_values)
    
    # Ensure complex_amplitudes and frequency_values have the same length as sample_values
    if len(complex_amplitudes) != num_samples:
        # Perform spline interpolation for complex amplitudes
        interp_amp = CubicSpline(np.arange(len(complex_amplitudes)), complex_amplitudes, axis=0)
        complex_amplitudes = interp_amp(np.linspace(0, len(complex_amplitudes) - 1, num_samples))
    
    if len(frequency_values) != num_samples:
        # Perform spline interpolation for frequency values
        interp_freq = CubicSpline(np.arange(len(frequency_values)), frequency_values, axis=0)
        frequency_values = interp_freq(np.linspace(0, len(frequency_values) - 1, num_samples))
    
    time = np.arange(len(sample_values)) / sample_rate
    sine_wave = np.abs(complex_amplitudes) * np.sin(2 * np.pi * frequency_values * time + np.angle(complex_amplitudes))
    modified_signal = sample_values + sine_wave
    return modified_signal

# Example usage:
sample_rate = 44100  # Sample rate in Hz
duration = 1.0  # Duration of the signal in seconds

# Generate a sample signal
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
sample_values = np.random.random(len(t))  # Replace with your own input signal

# Generate complex amplitude vector and frequency vector over time
num_samples = len(sample_values)
complex_amplitudes = np.array([0.5 * np.exp(1j * np.pi / 4 * (i / num_samples)) for i in range(num_samples)])
frequency_values = np.linspace(1000, 2000, num_samples)  # Varying frequency values

# Add the sine wave with varying amplitudes and frequencies to the input signal
modified_signal = add_sine_wave_with_complex_and_frequency_vectors(sample_rate, sample_values, complex_amplitudes, frequency_values)