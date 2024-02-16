import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from rtlsdr import RtlSdr


def exchange_top_bottom(vector):
    """
    Exchanges the top half of a vector with the bottom half.

    Parameters:
    - vector: numpy array or list, the vector to be modified
    """
    length = len(vector)
    if length % 2 != 0:
        # If the length is odd, round up to the nearest even number
        length = length + 1

    mid_point = length // 2
    top_half = vector[mid_point:]
    bottom_half = vector[:mid_point]

    # Exchange the top and bottom halves
    exchanged_vector = np.concatenate((top_half, bottom_half))

    return exchanged_vector

def plot_fft_in_db(freq_axis, fft_vector, title='', x_label='Frequency (Hz)', y_label='Magnitude (dB)'):
    """
    Plots the FFT values in a dB scale.

    Parameters:
    - freq_axis: numpy array or list, the frequency axis
    - fft_vector: numpy array or list, the FFT values
    - title: str, the title of the plot (default is an empty string)
    - x_label: str, the label for the x-axis (default is 'Frequency (Hz)')
    - y_label: str, the label for the y-axis (default is 'Magnitude (dB)')
    """
    # Calculate the magnitude spectrum in dB
    magnitude_spectrum_db = 20 * np.log10(np.abs(fft_vector))

    # Plot the frequency axis vs. magnitude spectrum in dB
    plt.figure(figsize=(10, 6))
    plt.plot(freq_axis, magnitude_spectrum_db)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def plot_vector(vector, title='', x_label='Index', y_label='Value'):
    """
    Plots a vector.

    Parameters:
    - vector: numpy array or list, the vector to be plotted
    - title: str, the title of the plot (default is an empty string)
    - x_label: str, the label for the x-axis (default is 'Index')
    - y_label: str, the label for the y-axis (default is 'Value')
    """
    plt.figure(figsize=(10, 6))
    plt.plot(vector)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def animate_plot(frame, line, x_data, y_data,sdr,samp_rate,samples_per_bin,init_freq,end_freq):
    # Generate new data for the plot
    y,x = s_a_loop(sdr,samp_rate,samples_per_bin,init_freq,end_freq)
    
    # Update the data for the plot
    line.set_data(x, 20 * np.log10(np.abs(y) + 1e-10))  # Convert to dB, avoiding log(0)
    
    return line,

def plot_animation(sdr,samp_rate,samples_per_bin,init_freq,end_freq):
    # Initialize the plot
    sdr.sample_rate = samp_rate
    sdr.gain = 20
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(-30, 70) 
    ax.set_xlim(init_freq,end_freq) # Adjust the y-axis limits as needed
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Real-time Plot in dB')

    y_data,x_data = s_a_loop(sdr,samp_rate,samples_per_bin,init_freq,end_freq)

    # Create the animation
    animation = FuncAnimation(fig, animate_plot, fargs=(line, x_data, y_data,sdr,samp_rate,samples_per_bin,init_freq,end_freq), blit=True, interval =0)

    # Show the plot
    plt.show()

def catch_espectrum(sdr,center_freq,num_samples):
    sdr.center_freq = center_freq
    return np.array(sdr.read_samples(num_samples))
    
def s_a_loop(sdr,samp_rate,samples_per_bin,init_freq,end_freq):
    longfft =np.array([])
    longfreqaxis=np.array([])
    center_freq = init_freq
    while (center_freq<end_freq):
        # Inside the loop, 'center_freq' will take values from start_freq to end_freq with the specified bandwidth
        local_sample = np.array(catch_espectrum(sdr,center_freq, samples_per_bin))
        local_freq_spectrum = exchange_top_bottom(np.fft.fft(local_sample))
        local_freq_axis =exchange_top_bottom(np.fft.fftfreq(len(local_sample), 1/samp_rate)+ center_freq)
        longfft = np.append(longfft, local_freq_spectrum)
        longfreqaxis = np.append(longfreqaxis, local_freq_axis)
        center_freq = center_freq +samp_rate


    return longfft ,longfreqaxis

# Example usage:
if __name__ == "__main__":
    # Create an RTL-SDR object
    sdr = RtlSdr()

    center_freq = 100e6  # Replace with your desired center frequency in Hz
    num_samples = 1000  # Replace with the desired number of samples
    sample_rate = 2e6  # 2 MHz sample rate
    #fft,freq = s_a_loop(sdr,sample_rate,num_samples,108e6,138e6)
    #plot_fft_in_db(freq,fft)
    plot_animation(sdr,sample_rate,num_samples,432e6,436e6)
    
    sdr.close()
    



