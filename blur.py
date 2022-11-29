import numpy as np

def make_gaussian_mask(height, width):
    half_height_index = int(height / 2)
    half_width_index = int(width / 2)
    mask = np.full((height, width), 1)
    mask[half_height_index, :] += np.full((width,), 1).T
    mask[:, half_width_index] += np.full((height,), 1)
    return mask

# blur_constant means mask's width(or height). it can be only odd number
def blurring(image, blur_constant):
    half_mask_index = int(blur_constant / 2)
    [img_height, img_width] = np.shape(image)
    gaussian_mask = make_gaussian_mask(blur_constant, blur_constant)    
    blurred_image = np.zeros_like(image)
    for y in range(0, img_height):
        for x in range(0, img_width):
            l = t = r = b = half_mask_index                        
            if x < half_mask_index:
                l = x
            if y < half_mask_index:
                t = y
            if x >= img_width - half_mask_index:
                r = img_width - x
            if y >= img_height - half_mask_index:
                b = img_height - y
            masked_img = image[y-t:y+b,x-l:x+r] * gaussian_mask[half_mask_index-t:half_mask_index+b,half_mask_index-l:half_mask_index+r]
            total_bgr = np.sum(masked_img) / ((l + r + 1) * (t + b + 1))
            blurred_image[y, x] = total_bgr

    return blurred_image.astype(dtype=np.uint8)
