import math
import numpy as np
import matplotlib.pyplot as plt



# Function to calculate impedance of a series combination
def series_impedance(components):
    total_impedance = sum(components)
    return total_impedance

# Function to calculate impedance of a parallel combination
def parallel_impedance(components):
    inverse_impedances = [1 / impedance for impedance in components]
    total_inverse_impedance = sum(inverse_impedances)
    total_impedance = 1 / total_inverse_impedance
    return total_impedance

# Calculate the imaginary impedance of the given expression as a function of frequency
def calculate_imaginary_impedance(frequencies):
    component_1 = 3e-9  # 3nH
    component_2 = 0.08e-12  # 0.08pF
    component_3 = 50  # 50 ohm
    component_4 = 5e3  # 5kohm
    component_5 = 0.1e-12  # 0.1pF

    imaginary_impedances = []

    for frequency in frequencies:
        angular_frequency = 2 * math.pi * frequency

        # Calculate the impedance of the inner parallel combination: (component_3 || (component_4 || component_5))
        inner_parallel_impedance = parallel_impedance([component_4, component_5])

        # Calculate the reactances of the components
        reactance_1 = angular_frequency * component_1
        reactance_2 = 1 / (angular_frequency * component_2)
        reactance_3 = 1 / (angular_frequency * inner_parallel_impedance)

        # Calculate the impedance of the entire expression: component_3 + (reactance_1 + (reactance_2 || reactance_3))
        expression_impedance = component_3 + series_impedance([reactance_1, reactance_2, reactance_3])

        imaginary_impedances.append(expression_impedance.imag)

    return imaginary_impedances

# Define the frequency range
start_frequency = 1e6  # 1 MHz
end_frequency = 1e9  # 1 GHz
num_points = 100  # Number of frequency points

frequencies = np.linspace(start_frequency, end_frequency, num_points)

# Calculate the imaginary impedance as a function of frequency
imaginary_impedances = calculate_imaginary_impedance(frequencies)

# Plot the imaginary impedance as a function of frequency
plt.plot(frequencies, imaginary_impedances)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Imaginary Impedance (ohms)')
plt.title('Imaginary Impedance vs. Frequency')
plt.grid(True)
plt.show()
