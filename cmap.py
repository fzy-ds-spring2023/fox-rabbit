'''
    define a custom colormap for the simulation
'''

import matplotlib as plt
import numpy as np

unoccupied_color = '#D2B48C' # tan
grass_color = '#228b22' # green
rabbit_color = '#0096ff' # blue
fox_color = '#EE4B2B' # red

colors = [unoccupied_color, grass_color, rabbit_color, fox_color]
values = [0, 1, 2, 3]
pp_map = plt.colors.ListedColormap(colors)
bounds = np.arange(-0.5, 4, 1)
norm = plt.colors.BoundaryNorm(bounds, pp_map.N)

