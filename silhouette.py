import cv2
import numpy as np
import yaml
from pathlib import Path
import edge
import morphology as morph
import floodfill

def read_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def remove_background(config: dict, image):
        # Taken from https://towardsdatascience.com/background-removal-with-python-b61671d1508a

        # Convert image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        # Apply Canny Edge Dection
        edges = edge.Canny(image_gray, config['canny_low'], config['canny_high'])
        cv2.imshow('', edges)
        cv2.waitKey()

        edges = morph.dilate(edges)
        edges = morph.erode(edges)
        mask = floodfill.fill_background(edges)
        
        cv2.imshow('', mask)
        cv2.waitKey()

        # use dilate, erode, and blur to smooth out the mask
        edges = morph.dilate(edges)
        edges = morph.erode(edges)
        mask = cv2.GaussianBlur(mask, (config['blur'], config['blur']), 0)

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
    output_path = str(images_dir / 'silhouette.jpg')
    process_images(input_path, output_path, remove_background, config)