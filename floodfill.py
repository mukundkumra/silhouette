import numpy as np
import morphology as morph

def fill_background(mask, window_size):
    shape = mask.shape
    mask = mask > 0
    mask = morph.dilate(mask)

    # find empty space moving window
    stack = []
    for x in np.arange(0, shape[0], window_size):
        x2 = max(x + window_size, shape[1])
        if not np.any(mask[x:x2,0]):
            stack.append((x,0))
        if not np.any(mask[x:x2,-1]):
            stack.append((x,-1))

    for y in np.arange(0, shape[1], window_size):
        y2 = max(y + window_size, shape[0])
        if not np.any(mask[0,y:y2]):
            stack.append((0,y))
        if not np.any(mask[-1,y:y2]):
            stack.append((-1,y))

    checked = np.zeros(shape, dtype=bool)
    for t in stack:
        checked[t] = True
    ways = [(0,1), (0,-1), (1,0), (-1,0)]
    while len(stack) > 0:
        c = stack.pop()
        for w in ways:
            d = (c[0] + w[0], c[1] + w[1])
            if d[0] < 0 or d[0] >= shape[0] or d[1] < 0 or d[1] >= shape[1]:
                continue
            if checked[d] or mask[d]:
                continue
            checked[d] = True
            stack.append(d)

    return np.logical_not(checked).astype(dtype=np.uint8) * 255

        
