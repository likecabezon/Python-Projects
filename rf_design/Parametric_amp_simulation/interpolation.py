import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the function to fit
def func(X,d, a, b, c):
    return d *np.exp(X * a)* np.cos(X * b + c)**8


# Define the function to minimize (error)
def error_func(X, Y, a, b, c):
    return (func(X, a, b, c) - Y)**2

# Sample data for X and Y
X_data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36])  # Replace with your actual X data
Y_data = np.array([0,0,0,0,1,4.5,10.5,14,13.5,10,7,4.8,2.4,1.2,0.4,0.2,0,0,0,0,0.4,1.2,2.4,6.2,9.6,12,10.6,5,4.2,2.8,1.2,0.4,0,0,0])  # Replace with your actual Y data

# Initial guess for parameters a, b, and c
initial_guess = [15,-0.01, 0.2, 0]

# Perform the curve fitting using least squares method
optimized_params, _ = curve_fit(func, X_data, Y_data, initial_guess)

# Display the optimized parameters
d_opt, a_opt, b_opt, c_opt = optimized_params
print(f"Optimized parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}, d = {d_opt}")

X_fit = np.linspace(0, max(X_data), 100)  # Adjust the range accordingly
Y_fit = func(X_fit,d_opt, a_opt, b_opt, c_opt)

# Plot the data and the fitted expression
plt.scatter(X_data, Y_data, label='Data')
plt.plot(X_fit, Y_fit, label='Fitted Expression', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()