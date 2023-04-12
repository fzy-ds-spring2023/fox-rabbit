"""

John Rachlin
DS 2000: Intro to Programming with Data

Filename: 
    
Description: 

    
"""




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TWOPI = 2*np.pi

fig, ax = plt.subplots()

# Plot sine wave
t = np.arange(0.0, TWOPI, 0.001)
s = np.sin(t)
l = plt.plot(t, s)

ax = plt.axis([0,TWOPI,-1,1])

# Plot red dot at starting position
redDot, = plt.plot(0, np.sin(0), 'ro')

def animate(i):
    redDot.set_data(i, np.sin(i))
    return redDot,

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(0.0, TWOPI, 0.1),
                                      interval=10, blit=False, repeat=True)
    

plt.show()