import numpy as np

def rgb_to_grayscale(image):
    return 0.2990 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]