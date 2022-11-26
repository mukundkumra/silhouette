import cv2
import numpy as np
import yaml
from pathlib import Path
import edge
from gaussian_blur import gaussian_blur
from contour import find_contours
import morphology as morph
from utils import rgb_to_grayscale

from blur import blurring

def read_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def remove_background(config: dict, image):
        # Taken from https://towardsdatascience.com/background-removal-with-python-b61671d1508a

        # Convert image to grayscale    
        image_gray = rgb_to_grayscale(image)
        
        # Apply Canny Edge Dection
        #edges = cv2.Canny(image_gray, config['canny_low'], config['canny_high'])
        edges = edge.Canny(image_gray, config['canny_low'], config['canny_high'])

        #edges = cv2.dilate(edges, None)
        edges = morph.dilate(edges)
        #edges = cv2.erode(edges, None)
        edges = morph.erode(edges)

        # get the contours and their areas
        contours = find_contours(edges)
        # contours = find_contours(edges)
        # print(contours)

        # cont = cv2.contourArea(contours[0])
        contour_info = [(c, cv2.contourArea(c),) for c in contours]

        # Get the area of the image as a comparison
        image_area = image.shape[0] * image.shape[1]

        # calculate max and min areas in terms of pixels
        max_area = config['max_area'] * image_area
        min_area = config['min_area'] * image_area

        # Set up mask with a matrix of 0's
        mask = np.zeros(edges.shape, dtype = np.uint8)

        # Go through and find relevant contours and apply to mask
        for contour in contour_info:
            # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
            if contour[1] > min_area and contour[1] < max_area:
                # Add contour to mask
                mask = cv2.fillConvexPoly(mask, contour[0], (255))
        
        # use dilate, erode, and blur to smooth out the mask
        mask = cv2.dilate(mask, None, iterations=config['dilate_iter'])
        mask = cv2.erode(mask, None, iterations=config['erode_iter'])
        # mask = cv2.GaussianBlur(mask, (config['blur'], config['blur']), 0)
        mask_stack = np.stack([mask, mask, mask])
        mask_stack = np.transpose(mask_stack, axes=(1, 2, 0))

        # Ensures data types match up
        mask_stack = mask_stack.astype('float32') / 255.0           
        image = image.astype('float32') / 255.0

        # Blend the image and the mask
        masked = (mask_stack * image) + ((1-mask_stack) * config['mask_color'])
        masked = (masked * 255).astype('uint8')

        return masked


def process_images(input_path: str, output_path: str, process, config: dict) -> None:
    image = cv2.imread(input_path)
    processed_image = process(config, image)
    cv2.imwrite(output_path, processed_image)


if __name__ == "__main__":
    config = read_yaml('config.yaml')
    images_dir = Path('images')
    input_path = str(images_dir / 'portrait.jpg')
    output_path = str(images_dir / 'blurred_3.jpg')
    image = cv2.imread(input_path)
    cv2.imwrite(output_path, blurring(image, 3))
    #process_images(input_path, output_path, remove_background, config)
    exit(0)
def test():
    config = read_yaml('config.yaml')
    images_dir = Path('images')
    input_path = str(images_dir / 'portrait.jpg')
    output_path = str(images_dir / 'silhouette.jpg')
    process_images(input_path, output_path, remove_background, config)
