
"""
This program was created by ChatGPT.  I asked it to create a particle swarm simulation in python
where there is a central larger mass and other particles are initialized to be moving around
the central mass.  I also asked it to use FuncAnimation.

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
box_size = 10.0 # size of simulation box
dt = 0.01 # time step
num_particles = 100 # number of particles
G = 1.0 # gravitational constant
m = 1.0 # mass of each particle
M_center = 100.0 # mass of center object
num_steps = 1000 # number of simulation steps

# Initialize particle positions and velocities
pos = np.zeros((num_particles, 3))
vel = np.zeros((num_particles, 3))
theta = np.random.uniform(low=0.0, high=2*np.pi, size=num_particles)
r = np.random.uniform(low=0.0, high=box_size/2, size=num_particles)
pos[:,0] = r * np.cos(theta) + box_size/2
pos[:,1] = r * np.sin(theta) + box_size/2
vel[:,0] = -r * np.sin(theta)
vel[:,1] = r * np.cos(theta)
forces = np.zeros((num_particles, 3))

# Add center object
pos_center = np.array([box_size/2, box_size/2, box_size/2])
vel_center = np.zeros(3)
forces_center = np.zeros(3)

# Calculate gravitational forces
def calculate_forces(pos):
    forces = np.zeros((num_particles, 3))
    for j in range(num_particles):
        for k in range(num_particles):
            if j != k:
                r = pos[k] - pos[j]
                f = G * m**2 / np.linalg.norm(r)**3 * r
                forces[j] += f
    for j in range(num_particles):
        forces[j] += G * m * M_center / np.linalg.norm(pos[j] - pos_center)**3 * (pos_center - pos[j])
    return forces

# Update particle positions and velocities
def update_positions(i):
    global pos, vel, forces, pos_center, vel_center, forces_center
    pos += vel * dt + 0.5 * forces / m * dt**2
    new_forces = calculate_forces(pos)
    vel += 0.5 * (forces + new_forces) / m * dt
    forces = new_forces
    pos_center += vel_center * dt + 0.5 * forces_center / M_center * dt**2
    new_forces_center = G * m * np.sum(pos - pos_center, axis=0) / np.linalg.norm(pos_center)**3
    vel_center += 0.5 * (forces_center + new_forces_center) / M_center * dt
    forces_center = new_forces_center
    scat.set_offsets(pos)
    scat_center.set_offsets(pos_center[:2])
    return scat, scat_center

# Create initial scatter plot
fig, ax = plt.subplots()
ax.set_xlim([0, box_size])
ax.set_ylim([0, box_size])
scat = ax.scatter(pos[:,0], pos[:,1], s=2)
scat_center = ax.scatter(pos_center[0], pos_center[1], s=200, color='red')

# Run animation
ani = FuncAnimation(fig, update_positions, frames=num_steps, interval=1, blit=True)
plt.show()
