import numpy as np

def dilate(mask):
    shape = mask.shape
    result = np.zeros(shape, dtype=np.uint8)
    for y in range(1,shape[0]-1):
        for x in range(1,shape[1]-1):
            result[y,x] = np.max(mask[y-1:y+2,x-1:x+2])
    return result

def erode(mask):
    shape = mask.shape
    result = np.zeros(shape, dtype=np.uint8)
    for y in range(1,shape[0]-1):
        for x in range(1,shape[1]-1):
            result[y,x] = np.min(mask[y-1:y+2,x-1:x+2])
    return result