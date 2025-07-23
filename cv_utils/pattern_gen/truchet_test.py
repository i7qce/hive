# %% 
import cv2
import numpy as np
import matplotlib.pyplot as plt

import importlib

import sys
sys.path.append('..')

import ccu

# %%
import truchet
importlib.reload(truchet)
importlib.reload(ccu)

np.random.seed(0)
result, result1 = truchet.generate(tile_type=0, tile_size=23, num_tiles=23)
plt.imshow(result)

# %%
retval, labels, stats, centroids = cv2.connectedComponentsWithStats(1-result1.astype(np.uint8))
plt.imshow(labels%15, cmap='magma')

# %%
for idx in range(100):
    ccu.plot_images_as_video_frame(labels%idx, cmap='magma')
