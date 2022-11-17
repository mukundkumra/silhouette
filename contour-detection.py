import cv2
import numpy as np
import yaml


image_path = 'images/portrait1.jpg'
image = cv2.imread(image_path)

def read_yaml(path: str) -> dict:
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

config = read_yaml('config.yaml')

def contour_detection(config: dict, image):
        # Convert image to grayscale        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Canny Edge Dection
        edges = cv2.Canny(image_gray, config['canny_low'], config['canny_high'])

        cv2.imwrite('images/grayscale.jpg', image_gray)

contour_detection(config, image)