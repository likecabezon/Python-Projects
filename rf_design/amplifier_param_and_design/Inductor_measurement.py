import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pyvisa
import time
import csv
from decimal import Decimal, getcontext
import math
import cmath
import pandas as pd
import os


def evaluate_with_random_variation(w, L):
    # Evaluate the main expression: (w**2 * L**2) / (2500 + w**2 * L**2)
    expression_result = (w**2 * L**2) / (2500 + w**2 * L**2)

    # Calculate a random value within the range of 0% to 10% of the expression result
    variation = expression_result * np.random.uniform(-0.1, 0.1)

    # Apply the random variation

    result_with_variation = expression_result + variation
    

    return result_with_variation

def plot_log_vectors(x_values, y_values, x_label='X', y_label='Y', title='Logarithmic Plot'):
    # Create the plot
    plt.figure()

    # Set the x and y values for the plot
    plt.plot(x_values, y_values, marker='o', linestyle='-')

    # Set labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Set the x and y axes to be in log scale
    plt.xscale('linear')
    plt.yscale('log')

    # Show the plot
    plt.show()

def evaluate_vector_with_random_variation(w_values,L):
    results = []
    for w in w_values:
        result = evaluate_with_random_variation(w, L)
        results.append(result)
    return results

def evaluate_expression(w, L):
    return (w**2 * L**2) / (2500 + w**2 * L**2)

def log_evaluate_expresion(w,L):
    return 10*np.log10((w**2 * L**2) / (2500 + w**2 * L**2))

def lin_squared_difference_error(w_values, measured_values,L):


    # Evaluate the expression for all 'w' values with the given 'L'
    evaluation_values = [evaluate_expression(w, L) for w in w_values]

    # Calculate the squared difference between measured values and evaluation values
    error = np.sum((np.array(measured_values) - np.array(evaluation_values))**2)

    return error

def log_squared_difference_error(w_values,log_values,L):
    # Evaluate the expression for all 'w' values with the given 'L'
    evaluation_values = [log_evaluate_expresion(w, L) for w in w_values]

    # Calculate the squared difference between measured values and evaluation values
    error = np.sum((np.array(log_values) - np.array(evaluation_values))**2)

    return error

def oscilloscope_frequency_trigger(desired_freq, chan, scope, tolerance=0.001):
    scope.write(':MEASure:COUNter:SOURce ' + str(chan))

    while True:
        readed_freq = float(scope.query(':MEASure:COUNter:VALue?').strip())
        if abs(desired_freq - readed_freq) < (desired_freq * tolerance):
            trigger = True
            break
        else:
            trigger = False
              # Wait for 0.1 second before checking again

    return readed_freq, trigger

def cal_measurement_loop(desired_freqs, chan, scope):
    V_cal = []
    vertscale = 0.5
    scope.write(':CHANnel'+ str(chan)+':VERNier ON')
    print('Connect signal to calibration jig port 1 and calibration jig port 2 to channel '+ str(chan))
    for desired_freq in desired_freqs:
        readed_freq, trigger = oscilloscope_frequency_trigger(desired_freq, chan, scope)
        if trigger:
            print(f"Trigger activated for desired frequency: {desired_freq} Hz")
            # Perform additional actions or measurements
            time_div = horizontal_autoscale(scope,desired_freq)
            vertscale,norm_wave,v1 = custom_vertical_autoscale(scope,vertscale,chan,desired_freq,time_div)
            print(str(v1)+'V')
        
        V_cal.append(v1)

    return V_cal

def coil_measurement_loop(desired_freqs,chan,scope):
    V_coil =[]
    vertscale = 0.5
    scope.write(':CHANnel'+ str(chan)+':VERNier ON')
    print('Connect signal to pot 1 of Inductor measurement jig and port 2 of Inductor measurement jig to channel '+str(chan))
    for desired_freq in desired_freqs:
        print('Change the frequency of the signal to '+str(desired_freq)+ 'Hz')
        input('Press enter to measure this frequency')
        time_div = horizontal_autoscale(scope,desired_freq)
        vertscale,norm_wave,v2 = custom_vertical_autoscale(scope,vertscale,chan,desired_freq,time_div)
        print(str(v2) + 'V')
        V_coil.append(v2)
    return V_coil

def save_matrix_as_csv(matrix, filename):

    # Ensure that the matrix is a NumPy array
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Transpose the matrix to have vectors as columns
    transposed_matrix = matrix.T

    # Save the transposed matrix as a CSV file
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in transposed_matrix:
            csv_writer.writerow(row)

def error_vs_L(w_values, measured_values, L_values):
    errors = []

    for L in L_values:
        error = lin_squared_difference_error(w_values, measured_values, L)
        errors.append(error)

    # Plot the graph
    plt.figure()
    plt.plot(L_values, errors, marker='o', linestyle='-')
    plt.xlabel('L')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Error')
    plt.title('Error vs. L')
    plt.grid(True)
    plt.show()

def golden_section_search(w_values, measured_values, a, b, tol=1e-11):
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Calculate the initial interior points
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi

    # Evaluate the function at the interior points
    f_x1 = lin_squared_difference_error(w_values, measured_values, x1)
    f_x2 = lin_squared_difference_error(w_values, measured_values, x2)

    while (b - a) > tol:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            x1 = b - (b - a) / phi
            f_x2 = f_x1
            f_x1 = lin_squared_difference_error(w_values, measured_values, x1)
        else:
            a = x1
            x1 = x2
            x2 = a + (b - a) / phi
            f_x1 = f_x2
            f_x2 = lin_squared_difference_error(w_values, measured_values, x2)

    # Return the optimal value of 'L' as the average of the final interval
    optimal_L = (a + b) / 2

    return optimal_L

def calculate_gain_loss(calibration_table, measured_vector):
    if len(calibration_table) != len(measured_vector):
        raise ValueError("Input vectors must have the same length.")
    
    gain_loss_vector = np.subtract(measured_vector, calibration_table)
    return gain_loss_vector

def db_to_linear(db_vector):
    linear_vector = 10 ** (db_vector / 10)
    return linear_vector

def custom_vertical_autoscale(scope, v_div_initial ,channel,frequency,time_per_div):
    decreasing = False
    increasing = False
    v_div = v_div_initial
    while (not decreasing) or (not increasing):
        
        adjust_vertical_scale(scope,channel,v_div)
        norm_data = get_waveform_vector(scope,channel)
        maxpeak = np.max(norm_data)
        minpeak = np.min(norm_data)
        first_v_div = v_div
        if (maxpeak/255)< 0.8 or (minpeak/255)> 0.2 :
            dist = np.max((0.8-(maxpeak/255),(minpeak/255)- 0.2))
            decreasing = True
            v_div = v_div * ((1 - dist)**4)
            
        else:
            dist = np.max(((maxpeak/255) - 0.8, 0.2 - (minpeak/255)))
            increasing = True
            v_div = v_div * ((1 + dist)**4)         
    N = len(norm_data)
    time = np.arange(0, N) * time_per_div/100
    sine_wave = np.sin(2 * np.pi * frequency * time)
    cosine_wave = np.cos(2 * np.pi * frequency * time)
    real_part = np.sum(norm_data * cosine_wave)/N
    imaginary_part = np.sum(norm_data * sine_wave)/N
    print('real,im: '+str((real_part,imaginary_part)))
    amplitude = np.sqrt(real_part**2 + imaginary_part**2)*first_v_div/25
    return first_v_div,norm_data/255,amplitude

def get_waveform_vector(scope, channel):
    scope.write(":WAVeform:SOURce CHANnel"+str(channel))
    scope.write(":WAVeform:FORMat BYTE")
    scope.write(":WAVeform:MODE NORMal")
    waveform_data = scope.query_binary_values(":WAVeform:DATA?", datatype='B', is_big_endian=True)
    return np.array(waveform_data)

def adjust_vertical_scale(scope, channel, v_div):
    scope.write(f":CHANnel{channel}:SCALe {v_div}")

def horizontal_autoscale(scope,frequency):
    period = 1/frequency
    ideal_time_div = period/12
    scope.write(':TIMebase:MAIN:SCALe '+str(ideal_time_div))
    real_time_div = float(scope.query(':TIMebase:MAIN:SCALe?').strip())
    return real_time_div

def wave_phase_estimation(waveform_vector,frequency,duration):
    crossings = []
    for i in range(1, len(waveform_vector)):
        if waveform_vector[i - 1] <= 0 < waveform_vector[i]:
            crossing_position = i - (0 - waveform_vector[i - 1]) / (waveform_vector[i] - waveform_vector[i - 1])
            crossings.append(crossing_position)
    center_sample = len(waveform_vector) / 2
    nearest_crossing = min(crossings, key=lambda x: abs(x - center_sample))
    position_diference = center_sample-nearest_crossing
    time_difference = (duration/len(waveform_vector)) * position_diference
    period = 1/frequency
    phase_shift= 360*(time_difference/period)
    return phase_shift

def scope_method_evaluation(w,L):
    return (w * L)/np.sqrt(2500 + ((w * L)**2))

def polar_to_cartesians(gains,Phase_shifts):
    cartesian_vector = []
    for magnitude, phase_deg in zip(gains, Phase_shifts):
        phase_rad = math.radians(phase_deg)  # Convert phase to radians
        real_part = magnitude * math.cos(phase_rad)
        imag_part = magnitude * math.sin(phase_rad)
        cartesian_vector.append(complex(real_part, imag_part))
    return cartesian_vector

def scope_method_error(L,w_vec,gains):
    acumulated_error = 0
    for i in range(len(w_vec)):
        computed_gain = scope_method_evaluation(w_vec[i],L)
        error = (gains[i]-computed_gain) ** 2
        acumulated_error += error
    return acumulated_error

def scope_golden_ratio_search(error_function, w_vec, gains, a, b, epsilon):
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    while abs(c - d) > epsilon:
        error_c = error_function(c, w_vec, gains)
        error_d = error_function(d, w_vec, gains)

        if error_c < error_d:
            b = d
            d = c
            c = b - gr * (b - a)
        else:
            a = c
            c = d
            d = a + gr * (b - a)

    return (b + a) / 2

def scope_plot_error_vs_L(w_vec, gains, a, b, num_points):
    L_values = np.linspace(a, b, num_points)
    errors = [scope_method_error(L, w_vec, gains) for L in L_values]

    plt.plot(L_values, errors)
    plt.xlabel('L Value')
    plt.ylabel('Error')
    plt.title('Error vs. L Value')
    plt.grid()
    plt.show()

def plot_vectors_as_x(vector_x, vector_y1, vector_y2, x_label="X-axis", y1_label="Y1-axis", y2_label="Y2-axis", title="Vector Plot"):

    plt.figure(figsize=(8, 4))  # Set the figure size

    # Plot the vectors
    plt.plot(vector_x, vector_y1, label=y1_label)
    plt.plot(vector_x, vector_y2, label=y2_label)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y1_label + " / " + y2_label)
    plt.title(title)

    # Add a legend to distinguish between the two vectors
    plt.legend()

    # Display the plot
    plt.grid(True)  # Add grid lines
    plt.show()

def csv_to_matrix(csv_file):
    try:
        # Read the CSV file using numpy
        data = np.genfromtxt(csv_file, delimiter=',')
        
        # Transpose the data to stack columns as vectors
        matrix = data.T
        
        return matrix
    except Exception as e:
        print(f"Error: {e}")
        return None

def read_csv_folder(folder_path):
    try:
        # Get a list of all files in the specified folder
        all_files = os.listdir(folder_path)
        
        # Filter the list to keep only CSV files
        csv_files = [file for file in all_files if file.endswith('.csv')]
        
        # Sort the CSV file names alphabetically
        csv_files.sort()
        
        return csv_files
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_csv_files(folder_path):
    csv_files = read_csv_folder(folder_path)
    vectors = []

    if csv_files:
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            matrix = np.array(csv_to_matrix(file_path))
            if matrix is not None:
                freqs = matrix[0]
                gains = matrix[1]
                w_values = 2 * np.pi * freqs
                vectors.append((w_values, gains))  # Store X and Y values as tuples
    
    else:
        print("No CSV files found in the folder.")
    
    return vectors

rm = pyvisa.ResourceManager()
rm.list_resources()
DS1202Z_E = rm.open_resource('USB0::0x1AB1::0x0517::DS1ZE231404736::INSTR')
DS1202Z_E.baud_rate = 96000000



#freqs = np.arange(2e6, 30e6, 1e6)
#w_values = freqs * 2 *np.pi
#v1 = cal_measurement_loop(freqs,1,DS1202Z_E)
#v2 = coil_measurement_loop(freqs,1,DS1202Z_E)
w_test = np.arange(5e5,30e6,1e5) *2 * np.pi
test = np.array(scope_method_evaluation(w_test,100e-9))
#gains= np.array(v2)/(np.array(v1)*2)
#plot_vectors_as_x(w_values,test,gains)
#scope_plot_error_vs_L(w_values,gains,1e-9,1e-6,2000)
#print('Coil value: '+str(scope_golden_ratio_search(scope_method_error,w_values,gains,1e-9,1e-6,1e-11)))
#measurement_matrix = np.stack((freqs,gains))
#save_matrix_as_csv(measurement_matrix,'handcoil5test8csv')

Cal_data =[-1.5,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.5,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.5,-1.5,-1.5,-1.4,-1.6,-1.6,-1.6,-1.5,-1.5,-1.2,-1.1,-1]
Mes_data = [-35.6,-30,-26.8,-24.4,-22.4,-20.4,-19.2,-18.4,-17.6,-16.8,-16,-15.2,-14.8,-14,-12.8,-12,-10.8,-9.6,-8.4,-7.6,-6.8,-6.4,-5.6,-4.8,-4,-3.2]
l_values = []

folder_path = "Coil5"
vectors = process_csv_files(folder_path)
print(vectors)
print(w_test,test)
#vectors = vectors.append((w_test,test))


# Create a list of unique colors for each vector
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Create a plot for each vector with a different color
for i, (x, y) in enumerate(vectors):
    color = colors[i % len(colors)]  # Use modulo to cycle through colors
    plt.plot(x, y, label=f'Vector {i+1}', color=color)

plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.title('Multiple Vectors')
plt.show()



ftrue = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20,22,24,26,28,30,33,36,39,42])
wtrue = ftrue * np.pi * 2 * 1000000
Vmeas = db_to_linear(calculate_gain_loss(Cal_data,Mes_data))
plot_log_vectors(wtrue,Vmeas)
computedL = np.logspace(-9, -6, num=100, base=10)
print(golden_section_search(wtrue,Vmeas,1e-9,1e-6,))
error_vs_L(wtrue,Vmeas,computedL)
