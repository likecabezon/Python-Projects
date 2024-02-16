import cmath
import math

def transmission_line_distributed_parameters(Rd, Gd, Ld, Cd, frequency):
    # Calculate angular frequency
    omega = 2 * math.pi * frequency
    
    # Calculate characteristic impedance (Z0)
    Z0 = cmath.sqrt((Rd + 1j * omega * Ld) / (Gd + 1j * omega * Cd))
    
    # Calculate propagation constant (gamma)
    gamma = 1j * cmath.sqrt((Rd + 1j * omega * Ld) * (Gd + 1j * omega * Cd))
    
    return Z0, gamma

def transmission_line_lumped_parameters(Z0,gamma,frequency):
    omega = 2 * math.pi * frequency

    series_impedance = gamma*Z0
    paralel_impedance = gamma/Z0

    Rd = series_impedance.real
    Gd = paralel_impedance.real
    Ld = series_impedance.imag/omega
    Cd = paralel_impedance.imag/omega

    return Rd, Gd, Ld, Cd

def distributed_parameters_from_measurement(Zin_short, Zin_open, length):
    
    # Calculate gamma
    gamma =  cmath.atanh(cmath.sqrt(Zin_short / Zin_open))/length
    
    # Calculate characteristic impedance (Z0)
    Z0 = cmath.sqrt(Zin_short * Zin_open)
    
    return Z0,gamma 

# Example usage:
Zin_short = 1376.5287 + 122.8519j
Zin_open = 377.42 - 186.53j
Rd = 2.227e-2  # per-unit length resistance
Gd = 3.428e-8  # per-unit length conductance
Ld = 382.18e-9  # per-unit length inductance
Cd = 2.114e-12  # per-unit length capacitance
frequency = 1500  # 1 GHz

Z0, gamma = distributed_parameters_from_measurement(Zin_short,Zin_open,20000)
Rd, Gd, Ld, Cd = transmission_line_lumped_parameters(Z0,gamma,frequency)
print("Characteristic Impedance (Z0):", Z0)
print("Propagation Constant (gamma):", gamma)
print(Rd,Gd,Ld,Cd)