import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


def compute_electric_field_array(charges, x, y):
    """Computes the electric field at each point in a mesh due to an array of charges."""
    k = 1/(4*np.pi*8.85e-12)  # Coulomb's constant
    X, Y = np.meshgrid(x, y)

    dx = X[..., np.newaxis] - charges[:, 0]
    dy = Y[..., np.newaxis] - charges[:, 1]
    r = np.sqrt(dx**2 + dy**2)
    magnitude = k * charges[:, 2] / (r**3)
    fieldx = (magnitude*dx).sum(axis=-1)
    fieldy = (magnitude*dy).sum(axis=-1)
    electric_field_vec = [(x, y) for x, y in zip(fieldx.flat, fieldy.flat)]
    electric_field_vec = np.array(electric_field_vec).reshape(fieldx.shape + (2,))
    electric_field = np.sqrt(np.sum(electric_field_vec**2, axis=-1))
    return  electric_field, electric_field_vec

def compute_field(charges, x_grid, y_grid):
    field = np.zeros((len(y_grid), len(x_grid), 2))
    e0 = 8.854e-12  # Permittivity of free space

    for charge in charges:
        charge_x, charge_y, charge_value = charge

        dx = x_grid - charge_x
        dy = y_grid - charge_y

        distance_cubed = np.sqrt(dx**2 + dy**2)**3
        field_strength = (1 / (4 * np.pi * e0)) * (charge_value / distance_cubed)

        field[:, :, 0] += field_strength * dx
        field[:, :, 1] += field_strength * dy

    return field

def recta(linden, i, f, n):
    # create an array of positions from i to f
    positions = np.linspace(i, f, n, endpoint=False)

    # create a vector of constant values
    constants = np.full((n, 1), (linden / n) * np.linalg.norm(f-i))

    # concatenate the positions with the constant values and return the result
    return np.concatenate((positions, constants), axis=1)

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
    scalar_field = -scalar_field_x - scalar_field_y

    return scalar_field

def campo_vec_2d(x,y,data):
    norms = np.sqrt(np.sum(data**2, axis=2))  # Compute the norms of each vector
    u = data[:,:,0] / norms  # Normalize the x components of the vectors
    v = data[:,:,1] / norms  # Normalize the y components of the vectors
    # Create a quiver plot with colored arrows
    fig, ax = plt.subplots()
    Q = ax.quiver(x, y, u, v, norms, cmap='coolwarm',  pivot='mid')
    Q.norm.vmin = np.amin(norms)
    Q.norm.vmax = np.amax(norms)
    # Add a color bar to the plot
    cbar = plt.colorbar(Q, ax=ax)
    cbar.set_label('Vector Length')
    # Set the axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Field of Vectors with Color Gradient')
    # Display the plot
    plt.show()
def log_campo_vec_2d(x,y,data):
    norms = np.sqrt(np.sum(data**2, axis=2))  # Compute the norms of each vector
    log_norms = np.log10(norms)  # Take the base 10 logarithm of the norms
    u = data[:,:,0] / norms  # Normalize the x components of the vectors
    v = data[:,:,1] / norms  # Normalize the y components of the vectors
    # Create a quiver plot with colored arrows
    fig, ax = plt.subplots()
    Q = ax.quiver(x, y, u, v, log_norms, cmap='coolwarm',  pivot='mid')
    Q.norm.vmin = np.amin(log_norms)
    Q.norm.vmax = np.amax(log_norms)
    # Add a color bar to the plot
    cbar = plt.colorbar(Q, ax=ax)
    cbar.set_label('Base-10 Logarithm of Vector Length')
    # Set the axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Field of Vectors with Color Gradient')
    # Display the plot
    plt.show()
def plot_field(x, y, z, min,max):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(min, max)
    plt.show()
def log_plot_field(x, y, z):
    log_z = np.log10(z)  # Take the base 10 logarithm of z

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, log_z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Base-10 Logarithm of Z')
    ax.set_zlim(np.amin(log_z), np.amax(log_z))
    plt.show()

# Define charges
charges = recta(-1e-7,np.array([-1,-1]),np.array([1,1]),100)

# Define grid of points to compute electric field
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
dx = 1
dy = 1
X,Y = np.meshgrid(x,y)
# Compute electric field
electricfvec = compute_field(charges, X, Y)



V = compute_scalar_field(electricfvec,dx,dy)

plot_field(X, Y, V,-20000000,0)
log_campo_vec_2d(x, y, electricfvec)