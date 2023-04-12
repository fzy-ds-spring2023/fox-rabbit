"""

John Rachlin
DS 3500: Advanced Programming with Data

Filename: modmath.py

Description: Experiment with modular math and animation


"""


# Usually we use `%matplotlib inline`. However we need `notebook` for the anim to render in the notebook.

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def f(x, y):
    return x*y


def animate_func(i, arr):
    arr = arr % (i+1)
    im.set_array(arr)
    plt.title("xy mod m, m=" + str(i))
    return im,


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(6, 6))

SZ = 500
init = np.zeros(shape=(SZ, SZ), dtype=int)
im = plt.imshow(init, cmap='inferno', interpolation='gaussian', aspect='auto', vmin=0, vmax=255)


base = np.fromfunction(f, (SZ, SZ), dtype=int)
anim = animation.FuncAnimation(fig, animate_func, fargs=(base,),
                               frames=5000, interval=1, repeat=False)

plt.show()
print('Done!')
