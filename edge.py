import numpy as np

# sobel edge detection
def Sobel(gray_img):
    shape = gray_img.shape
    
    dx = np.zeros(shape)
    dx += np.roll(gray_img,1,axis=1)
    dx -= np.roll(gray_img,-1,axis=1)
    gx = 2 * dx + np.roll(dx,1,axis=0) + np.roll(dx,-1,axis=0)
    
    dy = np.zeros(shape)
    dy += np.roll(gray_img,1,axis=0)
    dy -= np.roll(gray_img,-1,axis=0)
    gy = 2 * dy + np.roll(dy,1,axis=1) + np.roll(dy,-1,axis=1)
    
    result = np.sqrt(gx*gx+gy*gy).astype(np.uint8)
    theta = np.arctan2(gy,gx) / np.pi * 180
    return result, theta

# canny edge detection
def Canny(gray_img, threshold1, threshold2):
    shape = gray_img.shape
    sobel, theta = Sobel(gray_img)
    
    # Non-Maximum Suppression
    result = np.zeros(shape)
    for y in range(1,shape[0]-1):
        for x in range(1,shape[1]-1):
            t = theta[y,x]
            if t < 0:
                t += 180
            if t < 22.5 or t > 157.5:
                # horizontal maximum check
                result[y,x] = sobel[y,x] if (sobel[y,x-1] < sobel[y,x] > sobel[y,x+1]) else 0
            elif t < 67.5:
                # increasing diagonal maximum check
                result[y,x] = sobel[y,x] if (sobel[y-1,x-1] < sobel[y,x] > sobel[y+1,x+1]) else 0
            elif t > 112.5:
                # decreasing diagonal maximum check
                result[y,x] = sobel[y,x] if (sobel[y-1,x+1] < sobel[y,x] > sobel[y+1,x-1]) else 0
            else:
                # vertical maximum check
                result[y,x] = sobel[y,x] if (sobel[y-1,x] < sobel[y,x] > sobel[y+1,x]) else 0

    # Hysteresis edge tracking
    ways = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    edges = np.zeros(shape, dtype=np.uint8)
    checked = np.zeros(shape, dtype=bool)
    stack = []
    for y in range(1,shape[0]-1):
        for x in range(1,shape[1]-1):
            if checked[y,x]:
                continue
            checked[y,x] = True
            if result[y,x] > threshold2:
                edges[y,x] = 255
                stack.append((y,x))
                while len(stack) > 0:
                    c = stack.pop()
                    for w in ways:
                        d = (c[0] + w[0], c[1] + w[1])                        
                        if checked[d]:
                            continue
                        checked[d] = True
                        if result[d] > threshold1:
                            edges[d] = 255
                            stack.append(d)

    return edges