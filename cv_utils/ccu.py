"""
Common Canvas Utilities
"""

import time

import cv2
import matplotlib.pyplot as plt

from IPython.display import clear_output


def wfc():
    pass

def _wfc_from_patterns():
    pass

def _extract_patterns():
    pass

def plot_images_as_video_frame(img, title=None):
    plt.figure(dpi=150)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()
    time.sleep(0.001)
    plt.close()
    clear_output(wait=True)

def clear_outputs():
    plt.show()
    time.sleep(0.001)
    plt.close()
    clear_output(wait=True)