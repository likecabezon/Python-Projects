import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate some sample data
np.random.seed(0)
x_data = np.sort(5 * np.random.rand(80))
y_data = 2 * x_data + 1 + 0.2 * np.random.randn(80)

# Define the model function (a straight line)
def model_function(x, m, b):
    return m * x + b

# Define the least squares error function
def least_squares_error(m, b):
    y_pred = model_function(x_data, m, b)
    return np.sum((y_data - y_pred) ** 2)

# Initialize animation components
fig, ax = plt.subplots()
m_values = np.linspace(0, 3, 100)
b_values = np.linspace(0, 2, 100)
M, B = np.meshgrid(m_values, b_values)
Z = np.vectorize(least_squares_error)(M, B)
contour = ax.contourf(M, B, Z, levels=20, cmap="viridis")
sc = ax.scatter([], [], c='red', marker='o')

# Animation update function
def update(frame):
    m = frame / 50  # Adjust the frame value for animation
    b = 1.2
    y_pred = model_function(x_data, m, b)
    sc.set_offsets(np.column_stack((m, b)))
    return sc,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 101), blit=True)

# Add labels and a colorbar
plt.xlabel('Slope (m)')
plt.ylabel('Y-Intercept (b)')
plt.colorbar(contour, label='Least Squares Error')

# Show the animation
plt.show()
