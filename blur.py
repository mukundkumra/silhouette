import numpy as np
import cv2 as cv

blur_constant = 21

def blurring(image_path):
    image = cv.imread(image_path)

    mask_size = blur_constant * blur_constant
    half_mask_index = int(blur_constant / 2)

    [img_height, img_width, rgb] = np.shape(image)
    end_height = img_height - half_mask_index
    end_width = img_width - half_mask_index

    new_image = np.zeros_like(image)
    for y in range(half_mask_index, end_height):
        for x in range(half_mask_index, end_width):
            [total_blue, total_green, total_red] = np.full(3, 0)

            for mask_y in range(-half_mask_index, half_mask_index):
                for mask_x in range(-half_mask_index, half_mask_index):
                    img_bgr = image[y + mask_y][x + mask_x]
                    total_blue += img_bgr[0]
                    total_green += img_bgr[1]
                    total_red += img_bgr[2]

            total_blue /= mask_size
            total_red /= mask_size
            total_green /= mask_size
            new_image[y][x] = [total_blue, total_green, total_red]

    cv.imshow('original image', image)
    cv.imshow('blurred image', new_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit(0)