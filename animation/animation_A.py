"""

John Rachlin
DS 3500: Advanced to Programming with Data

Filename: animation_A.py
    
Description: Demo A of animation

    
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import matplotlib; matplotlib.use("TkAgg")



FIGSIZE = 8
SIZE = 100

def animate_func(i, *fargs):

    elapsed = (time.time_ns() - start) / 10 ** 9
    fps = i / elapsed
    plt.title(f"Frame: {i} Elapsed: {elapsed:.2f} FPS: {fps:.1f}")
    im = fargs[0]
    im.set_array(np.random.rand(SIZE,SIZE))
    return [im]


def main():


    # Initial frame
    global start
    start = time.time_ns()

    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
    arr = np.random.rand(SIZE, SIZE)

    # global im - avoid fargs
    im = plt.imshow(arr) #, interpolation='none')

    # configure the animiation
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        fargs=(im,),
        frames = 10000,
        interval = 500, # in ms
        repeat=False
    )

    # go
    plt.show()

main()
