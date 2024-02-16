import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import matplotlib.animation as animation
import ipywidgets as widgets
from ipywidgets import interact, IntSlider

def compute_electric_field_array_3d(charges, x, y, z):
    """Computes the electric field at each point in a mesh due to an array of charges."""
    k = 1/(4*np.pi*8.85e-12)  # Coulomb's constant
    X, Y, Z = np.meshgrid(x, y, z)

    dx = X[..., np.newaxis, np.newaxis] - charges[:, 0]
    dy = Y[..., np.newaxis, np.newaxis] - charges[:, 1]
    dz = Z[..., np.newaxis, np.newaxis] - charges[:, 2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    magnitude = k * charges[:, 3] / (r**3)
    fieldx = (magnitude*dx).sum(axis=-1)
    fieldy = (magnitude*dy).sum(axis=-1)
    fieldz = (magnitude*dz).sum(axis=-1)
    electric_field_vec = [(x, y, z) for x, y, z in zip(fieldx.flat, fieldy.flat, fieldz.flat)]
    electric_field_vec = np.array(electric_field_vec).reshape(fieldx.shape + (3,))
    electric_field = np.sqrt(np.sum(electric_field_vec**2, axis=-1))
    return  electric_field, electric_field_vec

def create_animation(x, y, z, EM):
    fig, ax = plt.subplots()
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    EZ0 = EM[:,:,0]
    
    # Set minimum and maximum values of color scale
    min_val = np.min(EM)
    max_val = 20000
    
    im = ax.imshow(EZ0, cmap='coolwarm', extent=[x[0], x[-1], y[0], y[-1]], vmin=min_val, vmax=max_val)
    fig.colorbar(im)

    def update(frame):
        EZ0 = EM[:,:,frame]
        im.set_data(EZ0)
        ax.set_title('Z = {:.2f}'.format(z[frame]))

    # Create animation object
    anim = animation.FuncAnimation(fig, update, frames=len(z), interval=100)

    plt.show()
def recta3d(linden, start, end, n):

    # create an array of positions from start to end for x, y, and z coordinates
    x = np.linspace(start[0], end[0], n, endpoint=False)
    y = np.linspace(start[1], end[1], n, endpoint=False)
    z = np.linspace(start[2], end[2], n, endpoint=False)
    constants = np.full((n, 1), (linden / n)*np.linalg.norm(end - start)) 
    positions = np.column_stack((x, y, z,constants))
     
    return positions 
charges = np.concatenate((recta3d(-0.5e-6, np.array([-5,-5,1]), np.array([5,5,1]), 70), recta3d(0.5e-6, np.array([-5,5,-1]), np.array([5,-5,-1]), 70)), axis=0)
print(charges)
x = np.linspace(-5,5,70)
y = np.linspace(-5,5,70)
z = np.linspace(-5,5,70)
EM, EV = compute_electric_field_array_3d(charges, x, y, z)

create_animation(x,y,z,EM)