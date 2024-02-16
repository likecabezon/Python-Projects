#Codigo que crea un sonido de un tiempo determinado y a una frecuencia de muestreo determinada. 
#Para crear la señal hay una matriz de harmónicos, cada fila de la matriz corresponde a un determinado harmónico.
#En dichas filas hay valores para la amplitud y la fase de dicho harmónico para cada momento del tiempo representados en notación compleja (a+bj) siendo j la unidad imaginaria por lo tanto la amplitud para cada momento sería raizcuadrada(a^2+b^2) y la fase sería θ = atan(b/a).
#Si no queremos definir cada valor de amplitud y fase para cada muestra podemos definir unos valores teniendo en cuenta que serán equtemporales y el código interporalá el resto de valores
#Ademas tenemos un vector de frecuencias que contiene la evolución temporal de la frecuencia principal y cada harmónico por lo tanto podemos hacer barridos, cambiar notas simulando un teclado o generar patrones de frecuencia más complejos 
#Por lo tanto manipulando el contenido de cada harmónico podemos construir ondas triangulares, de diente de sierra, ondas cuadradas, pulsos de ancho variable, y con envoltorios con formas arbitrarias
#Proximamente se mejorará el sistema de discriminación de altas frecuencias usando un algoritmo real de filtro paso bajo, evitando que se genere "aliasing" con la frecuencia de muestreo y sin generar distorsión en el espectro
#También se añadirá un sistema para definir cualquier forma de filtro/equalizador de forma arbitraria y dinámica con el tiempo para generar todo tipo de efectos durante la pista
#Para el futuro queda desarrollar un sistema de visualización del espectro en tiempo real con la pista y/o algún modo de grafico frecuencia/tiempo mixto como un wavelet 

import numpy as np
from scipy.interpolate import CubicSpline
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import spectrogram
from scipy.signal import butter, sosfreqz, sosfilt

def plot_spectrogram(signal, sample_rate, plot_type):
    """
    Plot a spectrogram of the signal over time, with color representing power at different frequencies and times.

    Args:
        signal (numpy.ndarray): The input signal as a 1D numpy array.
        sample_rate (int): The sample rate of the signal.
        plot_type (str, optional): Type of plot ('color' or '3d'). Default is 'color'.
    """
    # Calculate the spectrogram
    f, t, Sxx = spectrogram(signal, fs=sample_rate)
    
    if plot_type == 'color':
        # Create a color plot of the spectrogram
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', cmap='inferno')
        plt.title('Spectrogram of the Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power (dB)')
        plt.show()
    elif plot_type == '3d':
        # Create a 3D surface plot of the spectrogram
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        t, f = np.meshgrid(t, f)
        ax.plot_surface(t, f, 10 * np.log10(Sxx), cmap='inferno')
        ax.set_title('3D Spectrogram of the Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel('Power (dB)')
        plt.show()
    else:
        print("Invalid plot_type. Please use 'color' or '3d'.")

def plot_complex_matrix_3d(complex_matrix, ending_time):
    """
    Create a 3D plot to visualize complex values over time, with different colors for different rows.
    
    It is assumed that each point in a row is equitemporally spaced and that each vector starts at t=0.

    Args:
        complex_matrix (numpy.ndarray): A 2D numpy array with complex values, where each row represents a vector of complex values at equitemporally spaced time points.
        ending_time (float): The ending time value for the time dimension.
    """
    num_rows, num_time_points = complex_matrix.shape

    # Create a colormap for different rows
    colors = plt.cm.viridis(np.linspace(0, 1, num_rows))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_rows):
        # Generate a time vector for the current row
        time_vector = np.linspace(0, ending_time, num_time_points)
        
        # Extract complex values for the current row
        complex_values = complex_matrix[i, :]

        # Plot complex values against time
        ax.plot(time_vector, np.real(complex_values), np.imag(complex_values), label=f'Harmonic {i+1}', color=colors[i])

    # Set labels and a legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Real')
    ax.set_zlabel('Imaginary')
    ax.legend()

    plt.title('3D Plot of Complex Values Over Time')
    plt.show()

def plot_complex_matrix_with_time(complex_matrix, ending_time):
    """
    Plot complex vectors in a matrix over time, assuming that the start time is t=0 and the ending time is provided.

    Args:
        complex_matrix (numpy.ndarray): A 2D numpy array with complex values, where each row represents a vector of complex values.
        ending_time (float): The ending time value for the time dimension.
    """
    num_rows, num_time_points = complex_matrix.shape
    
    # Create subplots for amplitude and phase
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i in range(num_rows):
        # Generate a time vector for the current row
        time_vector = np.linspace(0, ending_time, num_time_points)
        
        # Extract complex values for the current row
        complex_values = complex_matrix[i, :]
        
        # Calculate the amplitude and phase
        amplitude = np.abs(complex_values)
        phase_degrees = np.degrees(np.angle(complex_values))
        
        # Plot the amplitude vs. time
        axes[0].plot(time_vector, amplitude, label=f'Harmonic {i+1}')
        
        # Plot the phase vs. time (in degrees)
        axes[1].plot(time_vector, phase_degrees, label=f'Harmonic {i+1}')
    
    # Set labels and legends for amplitude and phase plots
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].set_title('Amplitude vs. Time')
    
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].legend()
    axes[1].set_title('Phase vs. Time (Degrees)')
    
    plt.tight_layout()
    plt.show()

def plot_fft(signal, sample_rate):
    """
    Compute and plot the FFT of a signal in a log-frequency and log-power axis.

    Args:
        signal (numpy.ndarray): The input signal.
        sample_rate (int): The sample rate of the signal (samples per second).
    """
    # Compute the FFT of the signal
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1 / sample_rate)
    fft_values = np.fft.fft(signal)
    
    # Calculate the magnitude and convert to dB scale
    magnitude = np.abs(fft_values)
    db_magnitude = 20 * np.log10(magnitude + 1e-6)  # Add a small constant to avoid log(0)

    # Plot the FFT on a log-frequency and log-power axis
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(freqs, db_magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT in Log-Frequency and Log-Power Axis")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.log10(freqs), db_magnitude)
    plt.xlabel("Log(Frequency) (log10(Hz))")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_vector_with_sample_rate(data, sample_rate):
    """
    Plot a vector with the specified sample rate.

    Args:
        data (numpy.ndarray): The data to be plotted.
        sample_rate (int): The sample rate of the data (samples per second).
    """
    num_samples = len(data)
    time = np.arange(num_samples) / sample_rate
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Vector Plot")
    plt.grid(True)
    plt.show()

def create_wav_file(audio_vector, sample_rate, output_filename):
    """
    Create a WAV file from an audio vector and sample rate.

    Args:
        audio_vector (numpy.ndarray): A 1D numpy array containing audio data.
        sample_rate (int): The sample rate of the audio data (samples per second).
        output_filename (str): The name of the WAV file to be generated.
    """
    # Normalize the audio data to the appropriate sample width (16-bit)
    audio_vector = (audio_vector* 0.7 * 32767).astype(np.int16)
    
    # Save the audio data to a WAV file
    wavfile.write(output_filename, sample_rate, audio_vector)
    print(f"WAV file '{output_filename}' created.")

def resize_complex_matrix(complex_matrix, input_vector):
    """
    Resize each row of a complex matrix to match the size of an input vector using spline interpolation.

    Args:
        complex_matrix (numpy.ndarray): A 2D numpy array with complex values, where each row represents a vector of complex values.
        input_vector (numpy.ndarray): The input vector with the desired size.

    Returns:
        numpy.ndarray: The resized complex matrix with rows matching the size of the input vector.
    """
    num_rows, num_time_points = complex_matrix.shape
    num_input_points = len(input_vector)
    
    # Create an empty array to store the resized complex matrix
    resized_complex_matrix = np.empty((num_rows, num_input_points), dtype=complex_matrix.dtype)
    
    for i in range(num_rows):
        # Generate a time vector for the current row
        time_vector = np.linspace(0, 1, num_time_points)
        
        # Perform spline interpolation to resize the current row
        interp = CubicSpline(time_vector, complex_matrix[i, :])
        resized_complex_matrix[i, :] = interp(np.linspace(0, 1, num_input_points))
    
    return resized_complex_matrix

def apply_low_pass_filter(signal, sample_rate, cutoff_freq=17000):
    """
    Apply a low-pass filter to a signal with a given sample rate and cutoff frequency.

    Args:
        signal (numpy.ndarray): The input signal as a 1D numpy array.
        sample_rate (int): The sample rate of the signal.
        cutoff_freq (int, optional): Cutoff frequency for the low-pass filter in Hz. Default is 17000 Hz.

    Returns:
        numpy.ndarray: The filtered signal.
    """
    # Design a low-pass Butterworth filter
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    sos = butter(10, normal_cutoff, btype='low', analog=False, output='sos')
    
    # Frequency response of the filter
    frequencies, response = sosfreqz(sos, worN=2000)
    
    # Apply the filter to the signal
    filtered_signal = sosfilt(sos, signal)
    
    return filtered_signal

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
    sine_wave = apply_low_pass_filter(np.abs(complex_amplitudes) * np.sin(2 * np.pi * frequency_values * time + np.angle(complex_amplitudes)),sample_rate)
    modified_signal = sample_values + sine_wave
    return modified_signal


def harmonic_content_manager(harmonic_matrix, sample_values, sample_rate, frequencies_vector):

    harmonics_number = harmonic_matrix.shape[0]
    
    for i in range(harmonics_number):
        
        sample_values = add_sine_wave_with_complex_and_frequency_vectors(sample_rate, sample_values, harmonic_matrix[i,:], frequencies_vector * i)
    
    return sample_values

# Example usage:
sample_rate = 44100  # Sample rate in Hz
duration = 0.9  # Duration of the signal in seconds

# Generate a sample signal
init_signal = np.zeros(int(duration*sample_rate))
t = time = np.arange(len(init_signal)) / sample_rate

complex_amplitudes =  np.exp(-1*(32*t-4)**2) * (1+1j)
frequency_values = np.ones_like(t)*220

harmonic_matrix = np.empty((10, 4))
for i in range(10):
    harmonic_matrix[i,0]=0
    harmonic_matrix[i,1]=((-1)**i+1)/(i+1)
    harmonic_matrix[i,2]= ((-1)**i+1)*((-1)**i+1)/((i+1)**2)
    harmonic_matrix[i,3]= 0

modified_signal = harmonic_content_manager(harmonic_matrix,init_signal,sample_rate,frequency_values)
print(modified_signal)
plot_vector_with_sample_rate(modified_signal,sample_rate)
plot_fft(modified_signal,sample_rate)
harmonic_matrix = resize_complex_matrix(harmonic_matrix,t)
plot_complex_matrix_3d(harmonic_matrix,duration)
plot_complex_matrix_with_time(harmonic_matrix,duration)
plot_spectrogram(modified_signal,sample_rate,'color')
create_wav_file(modified_signal,sample_rate,'Spline_harmonics_test4.wav')
