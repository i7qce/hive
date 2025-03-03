# %% This is a test
import os

import cv2
import matplotlib.pyplot as plt

import importlib

import wfc
importlib.reload(wfc)

# %% 
hive_path = '../..'

straight_image = cv2.imread(os.path.join(hive_path, 'data/wfc/straight.png'))[:,:,::-1]
bend_image = cv2.imread(os.path.join(hive_path, 'data/wfc/bend.png'))[:,:,::-1]
blank_image = cv2.imread(os.path.join(hive_path, 'data/wfc/blank.png'))[:,:,::-1]
cross_image = cv2.imread(os.path.join(hive_path, 'data/wfc/cross.png'))[:,:,::-1]
t_image = cv2.imread(os.path.join(hive_path, 'data/wfc/t.png'))[:,:,::-1]

A = wfc.Tiles(
    input = [
        {
        'image': straight_image,
        'connections': {'up': 1, 'down': 1, 'right': 0, 'left': 0}
        },
        {
        'image': bend_image,
        'connections': {'up': 0, 'down': 1, 'right': 1, 'left': 0}
        },
        {
        'image': blank_image,
        'connections': {'up': 0, 'down': 0, 'right': 0, 'left': 0}
        },
        # {
        # 'image': cross_image,
        # 'connections': {'up': 1, 'down': 1, 'right': 1, 'left': 1}
        # },
        {
        'image': t_image,
        'connections': {'up': 1, 'down': 0, 'right': 1, 'left': 1}
        },
    ]
)

wfc.run_wfc(A.all_tiles, plot=True)
# %%
