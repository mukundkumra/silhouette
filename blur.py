import numpy as np
import cv2 as cv

def make_gaussian_mask(height, width):
    total_pixels = height * width + height + width
    value = 1 / total_pixels

    half_height_index = int(height / 2)
    half_width_index = int(width / 2)
    mask = np.full((height, width, 3), value)
    mask[half_height_index, :, :] += np.full((width, 3), value)
    mask[:, half_width_index, :] += np.full((height, 3), value)
    return mask

# blur_constant means mask's width(or height). it can be only odd number
def blurring(image, blur_constant):
    half_mask_index = int(blur_constant / 2)
    [img_height, img_width, rgb] = np.shape(image)

    blurred_image = np.zeros_like(image)
    for y in range(0, img_height):
        for x in range(0, img_width):
            total_bgr = np.full(3, 0)

            start_y = y - half_mask_index
            if start_y < 0:
                start_y = 0

            start_x = x - half_mask_index
            if start_x < 0:
                start_x = 0

            end_y = y + half_mask_index
            if end_y > img_height:
                end_y = img_height

            end_x = x + half_mask_index
            if end_x > img_width:
                end_x = img_width

            mask_height = end_y - start_y
            mask_width = end_x - start_x
            sliced_img_size = mask_height * mask_width
            sliced_img = image[start_y:end_y, start_x:end_x, :]
            gaussian_mask = make_gaussian_mask(mask_height, mask_width)
            masked_img = sliced_img * gaussian_mask
            reshaped_sliced_img = masked_img.reshape(sliced_img_size, 3)

            total_bgr = np.sum(reshaped_sliced_img, axis=0, keepdims=True)
            blurred_image[y, x] = total_bgr

    cv.imshow('original image', image)
    cv.imshow('blurred image', blurred_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return blurred_image