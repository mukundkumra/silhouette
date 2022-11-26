import numpy as np

def dilate(mask):
    ways = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    mask = mask > 0
    result = mask
    for w in ways:
        result = np.logical_or(result, np.roll(mask, w, axis=(1,0)))
    return result.astype(dtype=np.uint8) * 255

def erode(mask):
    ways = [(1,0),(0,1),(0,-1),(-1,0)]
    mask = mask > 0
    result = mask
    for w in ways:
        result = np.logical_and(result, np.roll(mask, w, axis=(1,0)))
    return result.astype(dtype=np.uint8) * 255