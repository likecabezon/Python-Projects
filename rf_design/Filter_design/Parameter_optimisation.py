import random
import numpy as np

def find_best_components_monte_carlo(resistor_values, cap_values, target_Q, target_Fc, max_iterations):
    best_combination = None
    best_Q_diff = float('inf')
    best_Fc_diff = float('inf')
    
    for _ in range(max_iterations):
        r1, r2 = random.sample(resistor_values, 2)
        c1, c2 = random.sample(cap_values, 2)
        
        Q_value = (np.sqrt(r1 * r2 * c1 * c2))/(r1*c1 + r2*c1)
        Fc_value = 1/ (2* np.pi * np.sqrt(r1 * r2 * c1 * c2))
        
        Q_diff = abs(Q_value - target_Q)/abs(target_Q)
        Fc_diff = abs(Fc_value - target_Fc)/abs(target_Fc)
        
        if Q_diff + Fc_diff < best_Q_diff + best_Fc_diff:
            best_combination = (r1, r2, c1, c2)
            best_Q_diff = Q_diff
            best_Fc_diff = Fc_diff
            
    return best_combination, best_Q_diff, best_Fc_diff

# Example usage:
resistor_values = [1.0e2, 2.0e2, 2.2e2, 3.3e2, 7.0e2, 8.0e2, 8.2e2, 9.0e2, 1.0e3, 2.2e3, 3.0e3, 5.0e3, 6.0e3, 7.0e3, 9.0e3, 1.0e4, 1.1e4, 1.2e4, 1.3e4, 2.0e4, 3.0e4, 3.3e4, 4.0e4, 4.7e4, 5.0e4, 7.0e4, 8.2e4]  # list of resistor values in ohms
cap_values = [1.0e-10, 2e-10, 3e-10, 4.3e-10, 5e-10, 6.2e-10, 7.5e-10, 8.2e-10, 9.1e-10, 1.0e-9, 2.0e-9, 3.3e-9, 4.0e-9, 5.0e-9, 6.2e-9, 7.5e-9, 8.2e-9, 9.1e-9, 1.0e-8, 2.0e-8, 3.0e-8, 4.3e-8, 5.0e-8, 6.2e-8, 7.5e-8, 1.0e-7, 2.0e-7, 3.0e-7, 4.3e-7, 5.6e-7, 6.2e-7, 7.5e-7, 8.2e-7] # list of capacitor values in farads
target_Q = 3.56  # target Q value 
target_Fc = 0.9932 * 3500  # target Cutoff frequency value in Hertz

best_combination, Q_diff, Fc_diff = find_best_components_monte_carlo(resistor_values, cap_values, target_Q, target_Fc, max_iterations=100000)

print(f"Best combination: {best_combination}")
print(f"Relative Q difference: {Q_diff}")
print(f"Relative Fc difference: {Fc_diff}")