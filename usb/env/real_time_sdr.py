import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from rtlsdr import RtlSdr

FULL_FFT_VEC=np.array([])

def get_spectrum_chunk(sdr, center_freq, samples):
    # Capture IQ data from the SDR
    sdr.center_freq = center_freq
    samples_data = sdr.read_samples(samples)

    # Perform FFT on the samples
    fft_result = np.fft.fftshift(np.fft.fft(samples_data))

    return fft_result

def init_plot(freqs):
    fig, ax = plt.subplots()
    FULL_FFT_VEC=np.zeros_like(freqs)
    line, = ax.plot(freqs,FULL_FFT_VEC )
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(0, 50)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Spectrum Analyzer')
    

    return fig, ax, line

def update_plot(i, sdr, center_freqs, samples_per_chunk, line, freqs):
    global FULL_FFT_VEC
    center_freq = center_freqs[i]
    fft_result = get_spectrum_chunk(sdr, center_freq, samples_per_chunk)
    start_index = i * samples_per_chunk
    end_index = start_index + 1023
    FULL_FFT_VEC[start_index:end_index] = np.abs(fft_result)
    line.set_ydata(FULL_FFT_VEC)
    return line,
def animate(sdr, start_freq, end_freq, sample_rate, samples_per_chunk):
    i=start_freq +sample_rate/2
    center_freqs=np.array([])
    while(i<end_freq):
        center_freqs = np.append(center_freqs,i)
        i+=sample_rate
    complete_freqs=np.array([])
    for cf in center_freqs:
        complete_freqs = np.append(complete_freqs,np.fft.fftshift(np.fft.fftfreq(samples_per_chunk, 1/sample_rate))+cf)

    # Initialize the plot
    fig, ax, line = init_plot(complete_freqs)
    
    # Define the animation function
    animation = FuncAnimation(
        fig,
        update_plot,
        fargs=(sdr, center_freqs, samples_per_chunk, line,complete_freqs),
        frames=len(center_freqs),
        interval=100,  # Delay between frames in milliseconds
        blit=True
    )

    # Display the animated plot
    plt.show()

if __name__ == "__main__":
    # Initialize the RTL-SDR device
    sdr = RtlSdr()

    # Set SDR parameters
    start_freq = 100e6  # Start frequency in Hz
    end_freq = 120e6  # End frequency in Hz
    sample_rate = 2.048e6  # Sample rate in Hz
    samples_per_chunk = 1024  # Number of samples per spectrum chunk

    # Configure SDR device
    sdr.sample_rate = sample_rate
    sdr.center_freq = start_freq
    sdr.gain = 20

    # Run the animation
    animate(sdr, start_freq, end_freq, sample_rate, samples_per_chunk)

