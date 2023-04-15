'''

Felix Yang, Emma Shek, and Sabrina Zhou
DS 3500: Advanced to Programming with Data
Filename: fox-rabbit.py
Description: animation of fox and rabbit prey and predator model

'''

import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns
from cmap import pp_map, norm
from collections import defaultdict
import argparse

SIZE = 200  # The dimensions of the field
GRASS_RATE = 0.05 # Probability that grass grows back at any location in the next season.
WRAP = False # Does the field wrap around on itself when rabbits move?
RABBITS = 100 # Initial number of rabbits
FOXES = 50 # Initial number of foxes
FOX_HUNGER = 10 # Number of days a fox can go without eating before dying
FOX_SPEED = 2 # How fast a fox can move
SIM_LENGTH = 1000 # Number of days to simulate

def new_params():
    """ Set arguments from command line """
    
    # User defined command-line arguments
    parser = argparse.ArgumentParser(description='Command line arguments to test fox-rabbit.py')

    parser.add_argument('-g', '--grass_rate', default=0.05, help='Input grass growth rate')
    parser.add_argument('-k', '--fox_k_value', default=10, help='Input k cycle that foxes can go without food')
    parser.add_argument('-fs', '--field_size', default=200, help='Input field dimension')
    parser.add_argument('-f', '--num_foxes', default=50, help='Input number of foxes')
    parser.add_argument('-r', '--num_rabbits', default=100, help='Input number of rabbits')
    parser.add_argument('-fm', '--fox_move', default=2, help='Input movement foxes can take each cycle')
    parser.add_argument('-sim', '--sim_length', default=1000, help='Input number of generations to simulate')
    parser.add_argument('-w', '--wrapped', default=False, help='Input whether field is wrapped or not')
    
    # set global variables
    global GRASS_RATE, FOXES, FOX_HUNGER, FOX_SPEED, RABBITS, SIM_LENGTH, SIZE, WRAP
    args = parser.parse_args()
    GRASS_RATE = float(args.grass_rate)
    FOXES = int(args.num_foxes)
    FOX_HUNGER = int(args.fox_k_value)
    FOX_SPEED = int(args.fox_move)
    RABBITS = int(args.num_rabbits)
    SIM_LENGTH = int(args.sim_length)
    SIZE = int(args.field_size)
    WRAP = bool(args.wrapped)

def flatten(l):
    """ flatten a list of lists """
    return [item for sublist in l for item in sublist]

class Animal:
    """ A furry creature roaming a field to survive. """

    def __init__(self, aid, max_offspring, speed, starve, eats=[1]):
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.id = aid
        self.max_offspring = max_offspring
        self.speed = speed
        self.eaten = 0
        self.days_hungry = 0
        self.starve = starve # store the number of days going hungry
        self.eats = eats

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """

        born = []
        if self.eaten == 1:
            self.eaten = 0
            for _ in range(rnd.randint(1,self.max_offspring)):
                born.append(copy.deepcopy(self))
        return born

    def eat(self, food):
        """ if food is available, eat it and reset days_hungry to 0 """
        if food:
            self.eaten = 1
            self.days_hungry = 0
        else:
            self.days_hungry += 1

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.randint(-self.speed, self.speed)) % SIZE
            self.y = (self.y + rnd.randint(-self.speed, self.speed)) % SIZE
        else:
            self.x = min(SIZE-1, max(0, (self.x + rnd.randint(-self.speed, self.speed))))
            self.y = min(SIZE-1, max(0, (self.y + rnd.randint(-self.speed, self.speed))))


class Field:
    """ A field is a patch of grass with 0 or more animals moving around
    in search of food. Grass is 1 """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no animals """
        self.animals = defaultdict(list)
        self.field = np.ones(shape=(SIZE,SIZE), dtype=int)
        self.nanimals = defaultdict(list)
        self.ngrass = []

    def add_animal(self, animal):
        """ A new animal is added to the field """
        self.animals[animal.id].append(animal)

    def move(self):
        """ Animals move around the field """
        for a in flatten(self.animals.values()):
            a.move()

    def eat(self):
        """ Animals eat (if they find their food where they are) """
        for aid in self.animals.keys():
            for a in self.animals[aid]:
                # iterate over the list of foods the animal eats
                for food in a.eats:
                    # grass
                    if food == 1:
                        a.eat(self.field[a.x,a.y])
                        self.field[a.x,a.y] = 0
                    # other animals
                    elif food in self.animals.keys():
                        available = self.animals[food]
                        self.animals[food] = [f for f in available if f.x != a.x or f.y != a.y]
                        # if number of animals are different, food was eaten
                        a.eat(len(available) != len(self.animals[food]))

    def survive(self):
        """ Rabbits who eat some grass live to eat another day """
        for aid in self.animals.keys():
            self.animals[aid] = [a for a in self.animals[aid] if a.days_hungry <= a.starve]

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        for aid in self.animals.keys():
            born = []
            for animal in self.animals[aid]:
                born += animal.reproduce()
            self.animals[aid] += born
            self.nanimals[aid].append(self.num_animals(aid))

        self.ngrass.append(self.amount_of_grass())

    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_animal(self, aid):
        """ Return a matrix of animal locations """
        animal = np.zeros(shape=(SIZE,SIZE), dtype=int)
        for a in self.animals[aid]:
            animal[a.x, a.y] = aid
        return animal

    def num_animals(self, aid):
        """ How many rabbits are there in the field ? """
        return len(self.animals[aid])

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self):
        """ Run one cycle of the simulation """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()

    def history(self, labels, marker='.'):
        """ Plot the history of the field
            A time-series of the population """
        plt.figure(figsize=(12,6))
        plt.xlabel("generation #")
        plt.ylabel("% population/grass coverage")

        # get generations and grass coverage
        xg = list(range(len(self.ngrass)))
        grass_cov = [grass/(SIZE**2) for grass in self.ngrass[:]]

        # build the percentage of each animal
        for aid in self.nanimals.keys():
            max_a = max(self.nanimals[aid])
            ys = np.array(self.nanimals[aid])/max_a
            plt.plot(xg, ys, marker=marker, label=labels[aid])

        plt.plot(xg, grass_cov, marker=marker, label=labels[1])

        plt.grid()
        plt.suptitle("Populations over Time", fontweight='bold', fontsize=16, x=0.125, y=.97, ha="left")
        plt.title(f"foxes: {FOXES}, fox k-value: {FOX_HUNGER}, fox movement: {FOX_SPEED}, rabbits: {RABBITS}, size: {SIZE}, grass: {GRASS_RATE}", fontsize=14, pad=10, loc="left")
        plt.savefig(f"plots/{FOXES}{FOX_HUNGER}{FOX_SPEED}{RABBITS}{SIZE}{GRASS_RATE}.png", bbox_inches='tight')
        plt.legend()
        sns.set()
        plt.show()


def animate(i, field, im):
    # break after SIM_LENGTH generations
    if i > SIM_LENGTH:
        return
    field.generation()

    # get the highest value of the field and animals (for plotting)
    arr = [field.field]
    arr += [field.get_animal(aid) for aid in field.animals.keys()]
    im.set_array(np.maximum.reduce(arr))
    plt.title("generation = " + str(i))
    return im,

def main():
    
    # Set the parameters
    new_params()
    
    # Create the ecosystem
    field = Field()

    # rabbit grass eater
    for _ in range(RABBITS):
        field.add_animal(Animal(2, max_offspring=2, speed=1, starve=0, eats=[1]))

    # fox rabbit eater
    for _ in range(FOXES):
        field.add_animal(Animal(3, max_offspring=1, speed=FOX_SPEED, starve=FOX_HUNGER, eats=[2]))

    # Plot the ecosystem
    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(array, cmap=pp_map, norm=norm, interpolation='None', aspect='auto')
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    map_idx = {1: 'grass', 2: 'rabbit', 3: 'fox'}
    field.history(labels=map_idx)


if __name__ == '__main__':
    main()




