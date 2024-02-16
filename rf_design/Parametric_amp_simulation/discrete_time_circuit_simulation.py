import numpy as np
import matplotlib.pyplot as plt

# Time-varying capacitance function (example: sinusoidal variation)
def capacitance(t):
    return 1.0 + 0.5 * np.sin(4 * np.pi * t )

# Circuit parameters
R = 1.0  # Ohms

# Time parameters
dt = 0.01  # Time step
end_time = 5.0

# Generator voltage function (example: sinusoidal waveform)
def generator_voltage(t):
    return np.sin(2 * np.pi * t)

# Initialize variables
time = np.arange(0, end_time, dt)
voltage_across_capacitor = np.zeros_like(time)
generator_voltage_values = np.zeros_like(time)

# Initial condition
voltage_across_capacitor[0] = 0.0

# Numerical solution using Euler method with time-varying capacitance
for i in range(1, len(time)):
    generator_voltage_values[i] = generator_voltage(time[i])
    capacitance_value = capacitance(time[i])
    voltage_across_capacitor[i] = voltage_across_capacitor[i-1] + (1 / (R * capacitance_value)) * (generator_voltage_values[i-1] - voltage_across_capacitor[i-1]) * dt

# Plot the results
plt.plot(time, generator_voltage_values, label='Generator Voltage')
plt.plot(time, voltage_across_capacitor, label='Voltage across Capacitor')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()