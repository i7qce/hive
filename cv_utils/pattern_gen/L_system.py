import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def derivation(axiom, steps, system_rules):
    """Generates an L-System sequence for a given axiom and number of steps."""
    derived = axiom
    for _ in range(steps):
        derived = ''.join(system_rules.get(char, char) for char in derived)
    return derived


def generate_coordinates(sequence, seg_length, initial_heading, angle_increment):
    """
    Generates a list of coordinates based on the L-System sequence.

    Parameters:
        sequence (str): The L-System sequence to interpret.
        seg_length (float): The length of each forward step.
        initial_heading (float): The initial direction of drawing in degrees.
        angle_increment (float): The angle increment for each rotation command.

    Returns:
        list of tuples: Each tuple contains (x, y) coordinates for plotting.
    """
    x, y = 0, 0  # Starting position
    heading = initial_heading  # Start with the initial heading
    coordinates = [(x, y)]
    stack = []

    for command in sequence:
        if command in "FGRL":
            # Move forward in the current direction
            x += seg_length * math.cos(math.radians(heading))
            y += seg_length * math.sin(math.radians(heading))
            coordinates.append((x, y))
        elif command == "+":
            heading -= angle_increment  # Rotate clockwise
        elif command == "-":
            heading += angle_increment  # Rotate counterclockwise
        elif command == "[":
            stack.append((x, y, heading))
        elif command == "]":
            x, y, heading = stack.pop()
            coordinates.append((x, y))

    return coordinates


def plot_l_system(coordinates):
    """
    Plots L-System with matplotlib lines
    """
    plt.figure(figsize=(8, 8))
    plt.plot(*zip(*coordinates), lw=0.5)
    plt.axis("equal")
    plt.axis("off")
    plt.show()

def plot_l_system_array(coordinates):
    """
    Plots L-System with matplotlib array
    """
    c = np.array(coordinates)
    # Subtract offset so all coordinates are positive
    c = c - np.array([c[:,0].min(), c[:,1].min()])
    # Get required image size
    y_size, x_size = int(np.ceil(c[:,0].max())), int(np.ceil(c[:,1].max()))
    img = np.zeros((x_size, y_size, 3))
    # Draw lines
    for idx in range(1,c.shape[0]):
        img = cv2.line(img, c[idx-1].astype(np.uint16), c[idx].astype(np.uint16), (255, 255, 255), 1)
    plt.imshow(img)
    plt.axis("equal")
    plt.axis("off")
    return img

def run(L_params):
    system_rules = L_params['system_rules']
    axiom = L_params['axiom']
    iterations = L_params['iterations']
    segment_length = L_params['segment_length']
    initial_heading = L_params['initial_heading']
    angle_increment = L_params['angle_increment']
    plot_type = L_params['plot_type']

    # Generate L-System sequence
    final_sequence = derivation(axiom, iterations, system_rules)

    # Generate coordinates for plotting with both heading and angle
    coordinates = generate_coordinates(final_sequence, segment_length, initial_heading, angle_increment)

    # Plot the L-System
    if plot_type == 'LINE':
        plot_l_system(coordinates)
    elif plot_type == 'ARRAY':
        img = plot_l_system_array(coordinates)
        return img

S_CARPET = {
    'system_rules': {
        'G': 'GGG', 
        'F': 'F+F-F-F-G+F+F+F-F',
    },
    'axiom': 'F',
    'iterations': 5,
    'segment_length': 5,
    'initial_heading': 45,
    'angle_increment': 90,
}

S_TRIANGLE = {
    'system_rules': {
        'L': 'R-L-R', 
        'R': 'L+R+L',
    },
    'axiom': 'L',
    'iterations': 7,
    'segment_length': 5,
    'initial_heading': 0,
    'angle_increment': 60,
}

HILBERT = {
    'system_rules': {
        'L': '+RF-LFL-FR+', 
        'R': '-LF+RFR+FL-',
    },
    'axiom': 'L',
    'iterations': 5,
    'segment_length': 5,
    'initial_heading': 0,
    'angle_increment': 90,
}

SNOWFLAKE = {
    'system_rules': {
        'X': 'X+YF++YF-FX--FXFX-YF+', 
        'Y': '-FX+YFYF++YF+FX--FX-Y',
    },
    'axiom': 'FX',
    'iterations': 5,
    'segment_length': 5,
    'initial_heading': 0,
    'angle_increment': 60,
}

PEANO = {
    'system_rules': {
        'X': 'XFYFX+F+YFXFY-F-XFYFX', 
        'Y': 'YFXFY-F-XFYFX+F+YFXFY',
    },
    'axiom': 'X',
    'iterations': 3,
    'segment_length': 5,
    'initial_heading': 0,
    'angle_increment': 90,
}

FERN = {
    'system_rules': {
        'X': 'F+[[X]-X]-F[-FX]+X', 
        'F': 'FF',
    },
    'axiom': '-X',
    'iterations': 5,
    'segment_length': 5,
    'initial_heading': 0,
    'angle_increment': 25,
}

KOCH_1 = {
    'system_rules': {
        'F': 'F-F+F+FF-F-F+F', 
    },
    'axiom': 'F-F-F-F',
    'iterations': 3,
    'segment_length': 5,
    'initial_heading': 90,
    'angle_increment': 90,
}

KOCH_2 = {
    'system_rules': {
        'F': 'FF-F-F-F-F-F+F', 
    },
    'axiom': 'F-F-F-F',
    'iterations': 3,
    'segment_length': 5,
    'initial_heading': 90,
    'angle_increment': 90,
}

KOCH_3 = {
    'system_rules': {
        'F': 'F+f-FF+F+FF+Ff+FF-f+FF-F-FF-Ff-FFF', 
        'f': 'ffffff',
    },
    'axiom': 'F+F+F+F',
    'iterations': 2,
    'segment_length': 5,
    'initial_heading': 90,
    'angle_increment': 90,
}

QUAD_SNOWFLAKE = {
    'system_rules': {
        'F': 'F+F-F-F+F', 
    },
    'axiom': '-F',
    'iterations': 3,
    'segment_length': 5,
    'initial_heading': 90,
    'angle_increment': 60,
}

# Same as snowflake?
HEX_GOSPER = {
    'system_rules': {
        'L': 'L+R++R-L--LL-R+', 
        'R': '-L+RR++R+L--L-R', 
    },
    'axiom': 'L',
    'iterations': 4,
    'segment_length': 5,
    'initial_heading': 60,
    'angle_increment': 60,
}

# Same as Fern?
AXIAL_TREE = {
    'system_rules': {
        'X': 'F-[[X]+X]+F[+FX]-X', 
        'F': 'FF', 
    },
    'axiom': 'X',
    'iterations': 5,
    'segment_length': 5,
    'initial_heading': 90,
    'angle_increment': 22.5,
}

DRAGON = {
    'system_rules': {
        'X': 'X+YF+', 
        'Y': '-FX-Y', 
    },
    'axiom': 'FX',
    'iterations': 12,
    'segment_length': 5,
    'initial_heading': 0,
    'angle_increment': 90,
}

"""
Usage:

import importlib
import L_system
importlib.reload(L_system)

G = L_system.HILBERT
G.update({'plot_type': 'ARRAY'})
L_system.run(G)

"""