import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
box_size = 10.0 # size of simulation box
dt = 0.01 # time step
num_particles = 10 # number of particles
G = 1.0 # gravitational constant
m = 1.0 # mass of each particle
num_steps = 10000 # number of simulation steps

# Initialize particle positions and velocities
pos = np.random.uniform(low=0.0, high=box_size, size=(num_particles, 3))
vel = np.random.normal(loc=0.0, scale=1.0, size=(num_particles, 3))
forces = np.zeros((num_particles, 3))

# Calculate gravitational forces
def calculate_forces(pos):
    forces = np.zeros((num_particles, 3))
    for j in range(num_particles):
        for k in range(num_particles):
            if j != k:
                r = pos[k] - pos[j]
                f = G * m**2 / np.linalg.norm(r)**3 * r
                forces[j] += f
    return forces

# Update particle positions and velocities
def update_positions(i):
    global pos, vel, forces
    pos += vel * dt + 0.5 * forces / m * dt**2
    new_forces = calculate_forces(pos)
    vel += 0.5 * (forces + new_forces) / m * dt
    forces = new_forces
    scat.set_offsets(pos)
    return scat,

# Create initial scatter plot
fig, ax = plt.subplots()
ax.set_xlim([0, box_size])
ax.set_ylim([0, box_size])
scat = ax.scatter(pos[:,0], pos[:,1], s=2)

# Run animation
ani = FuncAnimation(fig, update_positions, frames=num_steps, interval=1, blit=True)
plt.show()
