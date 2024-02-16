import numpy as np
import matplotlib.pyplot as plt

def generate_constant_amplitude_phase_signal(n, amplitude, phase_degrees):
    # Convert phase from degrees to radians
    phase_radians = np.deg2rad(phase_degrees)

    # Create a complex IQ signal with constant amplitude and phase
    iq_signal = amplitude * np.exp(1j * phase_radians) * np.ones(n)

    return iq_signal

def plot_iq_fft(iq_signal, sample_rate, title):
    # Calculate the FFT of the IQ signal
    fft_result = np.fft.fftshift(np.fft.fft(iq_signal))
    
    # Calculate the frequency axis values
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(iq_signal), 1 / sample_rate))

    # Create a plot for the linear FFT spectrum
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(fft_freq, np.abs(fft_result), label='Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(-sample_rate / 2, sample_rate / 2)
    plt.grid(True)
    plt.legend()
    plt.title(title + 'Linear FFT Spectrum')

    # Create a plot for the log FFT spectrum
    plt.subplot(1, 2, 2)
    plt.plot(fft_freq, 10 * np.log10(np.abs(fft_result) ** 2), label='Power Spectrum (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.xlim(-sample_rate / 2, sample_rate / 2)
    plt.grid(True)
    plt.legend()
    plt.title(title + 'Log FFT Spectrum')

    # Adjust the layout
    plt.tight_layout()

    # Show the plots
    plt.show()


def plot_iq_constellation(iq_vector, title):
    # Extract the in-phase (I) and quadrature (Q) components
    i_component = iq_vector.real
    q_component = iq_vector.imag

    # Create a scatter plot for the IQ constellation
    plt.figure(figsize=(8, 8))
    plt.scatter(i_component, q_component, s=5)
    
    # Set axis labels
    plt.xlabel('In-phase (I)')
    plt.ylabel('Quadrature (Q)')
    
    # Set plot limits
    plt.xlim(min(i_component), max(i_component))
    plt.ylim(min(q_component), max(q_component))
    
    # Show the plot with the specified title
    plt.grid(True)
    plt.title(title)
    plt.show()

def amplify_iq_signal(iq_signal, in_phase_gain_db, quadrature_gain_db):
    # Convert dB gains to linear scale
    in_phase_gain = 10 ** (in_phase_gain_db / 20.0)
    quadrature_gain = 10 ** (quadrature_gain_db / 20.0)

    # Separate the I and Q components
    i_component = iq_signal.real
    q_component = iq_signal.imag

    # Amplify the I and Q components independently
    amplified_i_component = i_component * in_phase_gain
    amplified_q_component = q_component * quadrature_gain

    # Create the amplified IQ signal
    amplified_iq_signal = amplified_i_component + 1j * amplified_q_component

    return amplified_iq_signal

def generate_iq_noise(n, sample_rate, spectral_density):
    # Calculate the variance of the noise based on spectral density
    variance = spectral_density * sample_rate / 2.0

    # Generate random complex noise samples with the specified variance
    noise_samples = np.random.normal(0, np.sqrt(variance), size=(n, 2))

    # Convert the noise samples to a complex IQ vector
    iq_noise = noise_samples[:, 0] + 1j * noise_samples[:, 1]

    return iq_noise

def calculate_iq_fft(iq_signal, sample_rate):
    # Calculate the FFT of the IQ signal
    fft_result = np.fft.fftshift(np.fft.fft(iq_signal))
    
    # Calculate the frequency axis values
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(iq_signal), 1 / sample_rate))
    
    return fft_result, fft_freq

def plot_fft_log(fft_result, fft_freq, title):
    # Calculate the power spectrum in dB
    power_spectrum_db = 10 * np.log10(np.abs(fft_result) ** 2)

    # Create a plot for the log FFT spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(fft_freq, power_spectrum_db, label='Power Spectrum (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    plt.legend()
    plt.title(title + ' Log FFT Spectrum')

    # Show the plot
    plt.show()


samples = 500
samp_rate= 1000
average_fft_noisy_signal = np.zeros(samples)
for i in range(1,100000):
    thermal_noise = generate_iq_noise(samples,samp_rate,0.1)
    signal = generate_constant_amplitude_phase_signal(samples,15,0)
    noisy_signal = signal + thermal_noise
    single_fft_noisy_signal,fft_freqs = calculate_iq_fft(noisy_signal,samp_rate)
    average_fft_noisy_signal = average_fft_noisy_signal + single_fft_noisy_signal/100000

plot_iq_constellation(noisy_signal,'Noisy signal IQ Constellation')
plot_fft_log(average_fft_noisy_signal,fft_freqs,'Averaged noisy signal ')
amplified_signal = amplify_iq_signal(noisy_signal,60,-60)
plot_iq_constellation(amplified_signal,'Amplified signal IQ constellation')
plot_iq_fft(amplified_signal,samp_rate,'Amplified signal')
