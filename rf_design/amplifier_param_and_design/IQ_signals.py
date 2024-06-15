import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_spectrum(iq_signal, carrier_freq, sample_rate):
    """
    Plot the frequency spectrum of the modulated carrier with the IQ signal.

    Parameters:
    - iq_signal: numpy array of complex values representing the IQ signal
    - carrier_freq: carrier frequency in Hz
    - sample_rate: sampling rate in Hz
    """
    # Number of samples
    N = len(iq_signal)

    # Time array
    t = np.arange(N) / sample_rate

    # Modulated carrier signal
    carrier = np.exp(2j * np.pi * carrier_freq * t)
    modulated_signal = iq_signal * carrier

    # Compute the frequency spectrum
    spectrum = np.fft.fft(modulated_signal)/sample_rate
    freq = np.fft.fftfreq(N, 1/sample_rate)

    # Shift the zero frequency component to the center
    spectrum = np.fft.fftshift(spectrum)
    freq = np.fft.fftshift(freq)

    # Plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freq, spectrum)
    plt.title('Frequency Spectrum of the Modulated Carrier')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

# Example usage:

# Carrier frequency in Hz
carrier_freq = 1e5  

# Sampling rate in Hz
sample_rate = 1e6  

# IQ signal: array of complex values
iq_signal = (2*np.cos(100*2*np.pi*np.arange(1e6)/sample_rate) + 1j * np.sin(100*2*np.pi*np.arange(1e6)/sample_rate))/(2)

plot_frequency_spectrum(iq_signal, carrier_freq, sample_rate)