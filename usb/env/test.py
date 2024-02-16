import pygame
import sys
import numpy as np
from rtlsdr import RtlSdr

def get_spectrum_chunk(sdr,cf,  samples):
    # Capture IQ data from the SDR
    sdr.center_freq = cf
    samples_data = sdr.read_samples(samples)

    # Perform FFT on the samples
    fft_result = np.abs(np.fft.fftshift(np.fft.fft(samples_data)))

    return fft_result

def generate_pixel_matrix(sdr,cf,x_size,vert_size):
    # Generate a 512x512 matrix of boolean values
    fft =np.round(((10*np.log10(get_spectrum_chunk(sdr, cf,  x_size)))-20)/80*vert_size,0)
    return fft

def update_color(fft,pos,chunk_size,vert_size):
    global square_colors
    square_colors[pos:pos+chunk_size,:]=(255,255,255)

    vec = np.array(range(chunk_size))+pos
    
    for i in vec:
    
        square_colors[i,int(fft[i-pos]):vert_size-1]=(0,0,0)

def draw_window(sdr,start,stop,bw,hor_size,vert_size):
    # Initialize Pygame
    pygame.init()
    num_steps = int((stop-start)/bw)
    cfs= np.array([])
    for i in range(num_steps):
        cfs = np.append(cfs,bw*i)
    cfs = cfs + start + bw/2
    samples_per_step = int(hor_size/num_steps)
    # Set the dimensions of the window
    window_size = (1024+360, 1024)
    

    # Create the window
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Pixel Matrix Square")
    # Fill the window with a white background
    window.fill((255, 255, 255))
    # Set the dimensions and position of the square
    square_size = (hor_size, vert_size)
    square_position = (window_size[0] - square_size[0], 0)

    # Generate the initial pixel matrix
    pixel_matrix = generate_pixel_matrix(sdr,cfs[0],samples_per_step,vert_size)

    # Set the initial colors based on the pixel matrix
    global square_colors
    square_colors = np.full((hor_size, vert_size, 3), 255, dtype=np.uint8)  # Initialize with white
    # Set the clock to control the frame rate

    clock = pygame.time.Clock()
    
    # Set a timer event to update the pixel matrix every...
    pygame.time.set_timer(pygame.USEREVENT, 50)
    sweep_loop_counter=0
    # Run the spectrum analyzer loop
    running = True
    while running:
        if sweep_loop_counter == num_steps-1:
            sweep_loop_counter = 0
        else:
            sweep_loop_counter+=1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.USEREVENT:
                # Update the pixel matrix and colors every second
                pixel_matrix = generate_pixel_matrix(sdr,cfs[sweep_loop_counter],samples_per_step,vert_size)
                update_color(pixel_matrix,sweep_loop_counter*samples_per_step,samples_per_step,vert_size)

        # Create a surface with the same dimensions as the square
        square_surface = pygame.Surface(square_size)

        # Fill the surface with the colors from square_colors
        pygame.surfarray.blit_array(square_surface, square_colors)

        # Draw the colored square at the top right
        window.blit(square_surface, square_position)

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        clock.tick(100)  # Cap the frame rate at 60 frames per second

    # Quit Pygame
    pygame.quit()
    sys.exit()

# Call the function to draw the window
sdr = RtlSdr()
sdr.gain = 25
start,stop= 118e6, 137e6
bw = 2.8e6

sdr.sample_rate = bw
draw_window(sdr,start,stop,bw,1024,820)