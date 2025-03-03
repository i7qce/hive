import random
import os
import numpy as np
from PIL import Image
from IPython import display
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import ccu

from enum import Enum, auto

import copy

import numpy as np
import cv2

import dataclasses

# Reference implementation:
# https://terbium.io/2018/11/wave-function-collapse/

def blend_many(imgs):
    """
    Blends a sequence of images.

    ims: a list of images

    """

    imgs = [x.astype(np.float64) for x in imgs]
    blended_img = (1/len(imgs)*(sum(imgs))).astype(np.uint8)

    return blended_img

def blend_tiles(choices, tiles):
    """
    Given a list of states (True if ruled out, False if not) for each tile,
    and a list of tiles, return a blend of all the tiles that haven't been
    ruled out.
    """
    to_blend = [tiles[i].image for i in range(len(choices)) if choices[i]]
    return blend_many(to_blend)

def show_state(potential, tiles):
    """
    Given a list of states for each tile for each position of the image, return
    an image representing the state of the global image.
    """
    rows = []
    for row in potential:
        rows.append([np.asarray(blend_tiles(t, tiles)) for t in row])

    rows = np.array(rows)
    # print(rows.shape)
    n_rows, n_cols, tile_height, tile_width, _ = rows.shape
    images = np.swapaxes(rows, 1, 2)
    return Image.fromarray(images.reshape(n_rows*tile_height, n_cols*tile_width, 3))

def find_true(array):
    """
    Like np.nonzero, except it makes sense.
    """
    transform = int if len(np.asarray(array).shape) == 1 else tuple
    return list(map(transform, np.transpose(np.nonzero(array))))

def rotate_dict(input_dict, n):
    """
    Rotate dict values clockwise
    """
    orders = ['up', 'right', 'down', 'left']
    ordered_dict = {order: input_dict[order] for order in orders}
    rotated_vals = list(ordered_dict.values())[n:] + list(ordered_dict.values())[:n]
    return dict(zip(ordered_dict.keys(), rotated_vals))


@dataclasses.dataclass
class tile_data:
    image: np.ndarray
    connections: dict
    weight: int

class Tiles:
    """
    Tile class for wavefunction collapse

    Pipeline is as follows:

    Supply either 
    a) Example image to extract features from
    b) Tiles, with a config that specifies which tiles connect to which

    If a), then convert to b)

    Then, generate all possible rotations and reflections. Then, supply a list of all 
    possible tiles that will be used in the rendered image


    input_tiles = {
        image_1: {
            'connection_plugs': {'up': 1, 'down': 2, 'right': 1, 'left': 3}
        }
    }


    """
    # rot90 = 'rot90'
    # rot180 = 'rot180'
    # rot270 = 'rot270'
    # flipud = 'flipud'
    # fliplr = 'fliplr'

    def __init__(self, input, **kwargs):
        if isinstance(input, list):
            print(f"Initializing Tiles with a list of tiles")
            self.ref_image = None
            self.tiles = input
        elif isinstance(input, np.ndarray):
            print(f"Initializing Tiles with an image")
            self.ref_image = input
            self.tiles = None
            self.patch_size = kwargs.get('patch_size', 3)
            
        
    def extract_patches(self):
        pass

    @staticmethod
    def analyze_symmetry(img):
        """
        Determine the symmetries of input image [WxHxC]

        Also, return all versions, with rules similarly rotated
        """
        if np.all(np.rot90(img,1) == img):
            rot90 = True
        else:
            rot90 = False
        
        if np.all(np.rot90(img,2) == img):
            rot180 = True
        else:
            rot180 = False

        if np.all(np.rot90(img,3) == img):
            rot270 = True
        else:
            rot270 = False
        
        if np.all(np.flip(img,axis=0) == img):
            flipud = True
        else:
            flipud = False

        if np.all(np.flip(img,axis=1) == img):
            fliplr = True
        else:
            fliplr = False

        flips = [fliplr, flipud]
        rots = [rot90, rot180, rot270]
        
        if all(flips + rots):
            # Don't return anything, original image is enough
            return 'X'
        elif not any(flips + rots):
            # return all flips/rotations
            return 'L'
        elif all(flips) and rot180:
            return 'I'
        elif (not any(rots)) and fliplr:
            return 'T'
        elif rot180 and (not any([rot90, rot270] + flips)):
            return '\\'
    
    def apply_transforms(self, img, connections, transformations):
        return_imgs = [img]
        return_connections = [connections]

        if 'rot90' in transformations:
            print(f"Applying rot90")
            return_imgs.extend([np.rot90(img)])
            return_connections.extend([rotate_dict(connections,1)])
            
        if 'rot180' in transformations:
            print(f"Applying rot180")
            return_imgs.extend([np.rot90(img,2)])
            return_connections.extend([rotate_dict(connections, 2)])

        if 'rot270' in transformations:
            print(f"Applying rot270")
            return_imgs.extend([np.rot90(img,3)])
            return_connections.extend([rotate_dict(connections, 3)])

        if 'flipud' in transformations:
            print(f"Applying flipud")
            return_imgs.extend([np.flip(img,axis=0)])
            new_connections = copy.deepcopy(connections)
            new_connections_u, new_connections_d = new_connections['up'], new_connections['down']
            new_connections['up'] = new_connections_d
            new_connections['down'] = new_connections_u
            return_connections.extend([new_connections])

        if 'fliplr' in transformations:
            print(f"Applying fliplr")
            return_imgs.extend([np.flip(img,axis=1)])
            new_connections = copy.deepcopy(connections)
            new_connections_l, new_connections_r = new_connections['left'], new_connections['right']
            new_connections['left'] = new_connections_r
            new_connections['right'] = new_connections_l
            return_connections.extend([new_connections])

        return return_imgs, return_connections, [1/len(return_imgs)]*len(return_imgs)

    def apply_symmetry_to_images_and_connections(self, img, connections):
        """
        Given some symmetry class and list of connections, 
        """

        # if symmetry
        symmetry_class = self.analyze_symmetry(img)
        print(f"Symmetry Class is {symmetry_class}")
        if symmetry_class == 'X':#all(flips + rots):
            # Don't return anything, original image is enough
            transformations = []
        elif symmetry_class == 'L':#not any(flips + rots):
            # return all flips/rotations
            transformations = ['rot90', 'rot180', 'rot270', 'flipud', 'fliplr']
        elif symmetry_class == 'I':#all(flips) and rot180:
            # return rot90
            transformations = ['rot90']
        elif symmetry_class == 'T':#(not any(rots)) and fliplr:
            # return flipud
            transformations = ['flipud', 'rot90', 'rot270']
        elif symmetry_class == '\\':#rot180 and (not any([rot90, rot270] + flips)):
            # return 
            transformations = ['rot90', 'rot270', 'flipud', 'fliplr']
        print(transformations)

        return self.apply_transforms(img, connections, transformations)

    
    def generate_tiles(self):
        return_imgs = []
        return_connections = []
        return_weights = []
        for tile in self.tiles:
            imgs, connections, weights = self.apply_symmetry_to_images_and_connections(tile['image'], tile['connections'])
            print(f"For tile, found {len(imgs)} and {len(connections)} entries")
            return_imgs.extend(imgs)
            return_connections.extend(connections)
            return_weights.extend(weights)
        return return_imgs, return_connections, return_weights

    @property
    def all_tiles(self):
        tiles, connections, weights = self.generate_tiles()
        return [tile_data(*x) for x in zip(tiles, connections, weights)]


def run_iteration(old_potential, weights):
    potential = old_potential.copy()
    to_collapse = location_with_fewest_choices(potential) #3
    if to_collapse is None:                               #1
        raise StopIteration()
    elif not np.any(potential[to_collapse]):              #2
        raise Exception(f"No choices left at {to_collapse}")
    else:                                                 #4 ↓
        nonzero = find_true(potential[to_collapse])
        tile_probs = weights[nonzero]/sum(weights[nonzero])
        selected_tile = np.random.choice(nonzero, p=tile_probs)
        potential[to_collapse] = False
        potential[to_collapse][selected_tile] = True
        propagate(potential, to_collapse)                 #5
    return potential

def location_with_fewest_choices(potential):
    num_choices = np.sum(potential, axis=2, dtype='float32')
    num_choices[num_choices == 1] = np.inf
    candidate_locations = find_true(num_choices == num_choices.min())
    location = random.choice(candidate_locations)
    if num_choices[location] == np.inf:
        return None
    return location



class Direction(Enum):
    RIGHT = 'right'; UP = 'up'; LEFT = 'left'; DOWN = 'down'
    
    def reverse(self):
        return {Direction.RIGHT: Direction.LEFT,
                Direction.LEFT: Direction.RIGHT,
                Direction.UP: Direction.DOWN,
                Direction.DOWN: Direction.UP}[self]

def neighbors(location, height, width):
    res = []
    x, y = location
    if x != 0:
        res.append((Direction.UP, x-1, y))
    if y != 0:
        res.append((Direction.LEFT, x, y-1))
    if x < height - 1:
        res.append((Direction.DOWN, x+1, y))
    if y < width - 1:
        res.append((Direction.RIGHT, x, y+1))
    return res

def propagate(potential, start_location):
    height, width = potential.shape[:2]
    needs_update = np.full((height, width), False)
    needs_update[start_location] = True
    while np.any(needs_update):
        needs_update_next = np.full((height, width), False)
        locations = find_true(needs_update)
        for location in locations:
            possible_tiles = [tiles[n] for n in find_true(potential[location])]
            for neighbor in neighbors(location, height, width):
                neighbor_direction, neighbor_x, neighbor_y = neighbor
                neighbor_location = (neighbor_x, neighbor_y)
                was_updated = add_constraint(potential, neighbor_location,
                                             neighbor_direction, possible_tiles)
                needs_update_next[location] |= was_updated
        needs_update = needs_update_next

def add_constraint(potential, location, incoming_direction, possible_tiles):
    neighbor_constraint = {t.connections[incoming_direction.value] for t in possible_tiles}
    outgoing_direction = incoming_direction.reverse()
    changed = False
    for i_p, p in enumerate(potential[location]):
        if not p:
            continue
        if tiles[i_p].connections[outgoing_direction.value] not in neighbor_constraint:
            potential[location][i_p] = False
            changed = True
    if not np.any(potential[location]):
        raise Exception(f"No patterns left at {location}")
    return changed

def run_wfc(tiles1, plot=False):
    global tiles
    tiles= tiles1
    weights = np.asarray([t.weight for t in tiles])
    potential = np.full((30, 30, len(tiles)), True)
    display.display(show_state(potential, tiles))

    p = potential
    images = [show_state(p, tiles)]
    print(f"Running propagation...")
    p_stack = []
    while True:
        try:
            pass
            p_stack.append(p)
            p = run_iteration(p, weights)
            if plot:
                images.append(show_state(p, tiles))  # Move me for speed
                ccu.plot_images_as_video_frame(images[-1])
        except StopIteration as e:
            break
        except Exception as e:
            print(e)
            print(f"Resetting to earlier state...")
            for _ in range(5):
                p_stack.pop()
            p = p_stack[-1]
            continue


    print("Done!")
    if plot:
        ccu.plot_images_as_video_frame(show_state(p, tiles))
    return show_state(p, tiles)