import wave
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt


def remove_dc_component(signal):
    dc_component = np.mean(signal)
    signal_without_dc = signal - dc_component
    return signal_without_dc

def compute_fft(signal, sample_rate):
    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1/sample_rate)
    fft_values = np.fft.fft(signal)
    return frequencies, fft_values


def linear_sweep(start_freq, end_freq, duration, sampling_rate):
    # Calculate the number of samples
    num_samples = int(duration * sampling_rate)
    
    # Generate time vector
    time_vector = np.linspace(0, duration, num_samples, endpoint=False)
    freqs = np.linspace(start_freq,end_freq,num_samples)
    # Generate the sweep signal
    sweep_signal = np.sin( 2 * np.pi * freqs *  time_vector)
    
    return time_vector, sweep_signal

time, sweep = linear_sweep(1, 7.5e3,10,44100)




samp_rate, time_domain_response = scipy.io.wavfile.read("MISC\\venv\\Respuesta_generador_sweep.wav")
print(time_domain_response)

freqs, fft_response = compute_fft(time_domain_response,samp_rate)
plt.plot(fft_response.real )
plt.show()
