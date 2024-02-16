import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import wave

def compute_potential_values(vector_matrix, potential_mat, Dx, Dy):
    x_size, y_size = vector_matrix.shape[:2]
    # Compute values for the bottom x border
    for x in range(x_size):
        
        if x > 0 and x < x_size-1:
            Px_1 = potential_mat[0,x+1]
            Px_2 = potential_mat[0,x-1]
            Py_1 = potential_mat[1,x]
            Vx_1 = vector_matrix[0,x+1,0]
            Vx_2 = vector_matrix[0,x-1,0]
            Vy_1 = vector_matrix[1,x,1]
            Pxy = potential_mat[0,x]
            dot_prods = (Dx*(Dx*(Vx_1**2+Vx_2**2) + Vx_1*(Pxy-Px_1) + Vx_2*(Px_2-Pxy)) + Dy*(Vy_1**2 * Dy - Vy_1 * Py_1 + Vy_1 * Pxy))/3
            Aver_pot = (Px_1 + Px_2 + Py_1)/3
            potential_mat[0,x] = dot_prods 
        elif x == x_size - 1:
            dot_prods = (Vx_2*Dx*(Vx_2*Dx + Px_2 - Pxy) + Vy_1*Dy*(Vy_1*Dy - Py_1 + Pxy))/2 #Corner of the row
            Aver_pot = (Px_2 + Py_1)/2
            potential_mat[0,x] = dot_prods 
        else:
            potential_mat[0, 0] = 0  # First point is fixed at 0

    # Compute values for the right y border
    for y in range(1, y_size):
        if y < y_size - 1:
            Px_2 = potential_mat[y,-2]
            Py_1 = potential_mat[y+1,-1]
            Py_2 = potential_mat[y-1,-1]
            Pxy = potential_mat[y,-1]
            Vx_2 = vector_matrix[y,-2,0]
            Vy_1 = vector_matrix[y+1,-1,1]
            Vy_2 = vector_matrix[y-1,-1,1]
            dot_prods = (Dy*(Dy*(Vy_1**2+Vy_2**2) + Vy_1*(Pxy-Py_1) + Vy_2*(Py_2-Pxy)) + Vx_2*Dx*(Vx_2*Dx + Px_2 - Pxy))/3
            Aver_pot = (Py_2 + Py_1 + Px_2)/3
            potential_mat[y,-1] = dot_prods 
        else:
            dot_prods = (Vx_2*Dx*(Vx_2*Dx + Px_2 - Pxy) + Vy_2*Dy*(Vy_2*Dy + Py_2 - Pxy))/2
            Aver_pot = (Px_2 + Py_2)/2
            potential_mat[y,-1] = dot_prods 
            
    # Compute values for the top x border
    for x in range(1,x_size):
        if x < x_size-1:
            Px_1 = potential_mat[-1,-x]
            Px_2 = potential_mat[-1,-x-2]
            Py_2 = potential_mat[-2,-x-1]
            Pxy = potential_mat[-1,-x-1]
            Vx_1 = vector_matrix[-1,-x,0]
            Vx_2 = vector_matrix[-1,-x-2,0]
            Vy_2 = vector_matrix[-2,-x-1,1]
            dot_prods = (Dx*(Dx*(Vx_1**2+Vx_2**2) + Vx_1*(Pxy-Px_1) + Vx_2*(Px_2-Pxy)) + Vy_2*Dy*(Vy_2*Dy + Py_2 - Pxy))/3
            Aver_pot = (Px_1 + Px_2 + Py_2)/3
            potential_mat[-1,-x-1] = dot_prods 
        else:
            dot_prods = (Vx_1*Dx*(Vx_1*Dx - Px_1 + Pxy) + Vy_2*Dy*(Vy_2*Dy + Py_2 - Pxy))/2
            Aver_pot = (Py_2 + Px_1)/2
            potential_mat[-1,-x-1] = dot_prods 
            
    # Compute values for the left y border
    for y in range(1,y_size-1):
        Px_1 = potential_mat[-y-1,1]
        Py_1 = potential_mat[-y,0]
        Py_2 = potential_mat[-y-2,0]
        Pxy = potential_mat[-y-1,0]
        Vx_1 = vector_matrix[-y-1,1,0]
        Vy_1 = vector_matrix[-y,0,1]
        Vy_2 = vector_matrix[-y-2,0,1]
        dot_prods = (Dy*(Dy*(Vy_1**2+Vy_2**2) + Vy_1*(Pxy-Py_1) + Vy_2*(Py_2-Pxy)) + Vx_1*Dx*(Vx_1*Dx - Px_1 + Pxy))/3
        Aver_pot = (Px_1 + Py_1 + Py_2)/3
        potential_mat[-y-1,0] = dot_prods 

    for x in range(1,x_size-1):
        for y in range(1,y_size-1):
            Px_1 = potential_mat[y,x+1]
            Px_2 = potential_mat[y,x-1]
            Py_1 = potential_mat[y+1,x]
            Py_2 = potential_mat[y-1,x]
            Pxy = potential_mat[y,x]
            Vx_1 = vector_matrix[y,x+1,0]
            Vx_2 = vector_matrix[y,x-1,0]
            Vy_1 = vector_matrix[y+1,x,1]
            Vy_2 = vector_matrix[y-1,x,1]
            dot_prods = (Dy*(Dy*(Vy_1**2+Vy_2**2) + Vy_1*(Pxy-Py_1) + Vy_2*(Py_2-Pxy)) + Dx*(Dx*(Vx_1**2+Vx_2**2) + Vx_1*(Pxy-Px_1) + Vx_2*(Px_2-Pxy)))/4
            Aver_pot = (Px_1 + Px_2 + Py_1 + Py_2)/4
            potential_mat[y,x] = dot_prods 

    return potential_mat

def compute_potential_values1(vector_matrix, potential_mat, Dx, Dy):
    x_size, y_size = vector_matrix.shape[:2]

    # Compute values for the bottom x border
    for x in range(x_size):
        if x > 0 and x < x_size-1:
            dot_prods = ((vector_matrix[0, x-1, 0] * Dx) + (vector_matrix[0, x+1, 0] * -Dx) + (vector_matrix[1, x, 1] * -Dy))/3
            Aver_pot = (potential_mat[0, x-1] + potential_mat[0, x+1] + potential_mat[1, x])/3
            potential_mat[0, x] = dot_prods + Aver_pot
        elif x == x_size - 1:
            dot_prods = ((vector_matrix[0, x-1, 0] * Dx) + (vector_matrix[1, x, 1] * -Dy))/2 #Corner of the row
            Aver_pot = (potential_mat[0, x-1] + potential_mat[1, x])/2
            potential_mat[0, x] = dot_prods + Aver_pot
        else:
            potential_mat[0, 0] = 0  # First point is fixed at 0

    # Compute values for the right y border
    for y in range(1, y_size - 1):  # Exclude the last element
        dot_prods = (vector_matrix[y-1, -1, 1] * Dy) + (vector_matrix[y+1, -1, 1] * -Dy) + (vector_matrix[y, -2, 0] * Dx) / 3
        Aver_pot = (potential_mat[y-1, -1] + potential_mat[y+1, -1] + potential_mat[y, -2]) / 3
        potential_mat[y, x_size - 1] = dot_prods + Aver_pot

    # Compute values for the top x border
    for x in range(1, x_size - 1):  # Exclude the last element
        dot_prods = (vector_matrix[-1, -2-x, 0] * Dx) + (vector_matrix[-1, -x, 0] * -Dx) + (vector_matrix[-2, -x-1, 1] * Dy) / 3
        Aver_pot = (potential_mat[-1, -2-x] + potential_mat[-1, -x] + potential_mat[-2, -x-1]) / 3
        potential_mat[-1, -x] = dot_prods + Aver_pot

    # Compute values for the left y border
    for y in range(1, y_size - 1):  # Exclude the last element
        dot_prods = (vector_matrix[-2-y, 1, 1] * Dy) + (vector_matrix[-y, 1, 1] * -Dy) + (vector_matrix[-y-1, 2, 0] * -Dx) / 3
        Aver_pot = (potential_mat[-2-y, 1] + potential_mat[-y, 1] + potential_mat[-y-1, 2]) / 3
        potential_mat[y, 0] = dot_prods + Aver_pot

    return potential_mat

def compute_scalar_field(vector_field, dx, dy):
    M, N, _ = vector_field.shape
    scalar_field_x = np.zeros((M, N))
    scalar_field_y = np.zeros((M, N))

    # Compute the integral of Fx with respect to x
    scalar_field_x[:, 0] = 0.5 * (vector_field[:, 0, 0] + vector_field[:, -1, 0]) * dx
    for j in range(1, N):
        scalar_field_x[:, j] = scalar_field_x[:, j-1] + 0.5 * (vector_field[:, j, 0] + vector_field[:, j-1, 0]) * dx

    # Compute the integral of Fy with respect to y
    scalar_field_y[0, :] = 0.5 * (vector_field[0, :, 1] + vector_field[-1, :, 1]) * dy
    for i in range(1, M):
        scalar_field_y[i, :] = scalar_field_y[i-1, :] + 0.5 * (vector_field[i, :, 1] + vector_field[i-1, :, 1]) * dy

    # Compute the scalar field ϕ(x, y) by adding the integrals of Fx and Fy
    scalar_field = scalar_field_x + scalar_field_y

    return scalar_field


def gradient_vectors(function, x, y):
    # Calculate the partial derivatives
    fx = np.gradient(function, x, axis=0)
    fy = np.gradient(function, y, axis=1)
    
    # Combine the partial derivatives into gradient vectors
    gradient_x = np.stack([fx, fy], axis=2)
    
    return gradient_x

# Example function: f(x, y) = x^2 + 2y^2
def example_function(x, y):
    return x*y

# Define the x-y mesh
x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)

# Create the x-y mesh grid
X, Y = np.meshgrid(x, y)

# Evaluate the function on the mesh grid
Z = example_function(X, Y)
initialval= np.zeros_like(Z)
# Calculate the gradient vectors
gradient = gradient_vectors(Z, x, y)

print(gradient),print(Z)
Dx = 2/5
Dy = 2/5
result = compute_scalar_field(gradient, Dx, Dy)
print(result)