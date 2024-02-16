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

audio1 = scipy.io.wavfile.read("measured_signal.wav")
audio2 = scipy.io.wavfile.read("target_signal.wav")
samp_rate = audio2[0]

measured_signal = []
targeted_signal = []
for i in range(len(audio1[1])):
    measured_signal.append(audio1[1][i][0])
    targeted_signal.append(audio2[1][i][0])

freqs, measured_signal_fft = compute_fft(measured_signal, samp_rate)
freqs, targeted_signal_fft = compute_fft(targeted_signal, samp_rate)

correction_filter = np.array(targeted_signal_fft)/np.array(measured_signal_fft) 
output_signal_fft = np.array(targeted_signal) * correction_filter
output_signal = np.fft.ifft(output_signal_fft)

#plt.plot( measured_signal)

#plt.show()
def linear_sweep(start_freq, end_freq, duration, sampling_rate):
    # Calculate the number of samples
    num_samples = int(duration * sampling_rate)
    
    # Generate time vector
    time_vector = np.linspace(0, duration, num_samples, endpoint=False)
    freqs = np.linspace(start_freq,end_freq,num_samples)
    print(time_vector)
    # Generate the sweep signal
    sweep_signal = np.sin( 2*np.pi * freqs *  time_vector)
    
    return time_vector, sweep_signal

time, sweep = linear_sweep(1, 15e3,10,37.5e3)



print(sweep)

scipy.io.wavfile.write("barrido.wav", 37500, sweep.astype(np.float32))

