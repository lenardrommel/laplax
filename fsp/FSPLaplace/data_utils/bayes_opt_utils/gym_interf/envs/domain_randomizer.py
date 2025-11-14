from cv2 import cv2
import os
import numpy as np


class DomainRandomizer(object):
    def __init__(self, data_folder):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.masks = []
        for image_name in os.listdir(os.path.join(package_directory, data_folder)):
            images = np.load(os.path.join(package_directory, data_folder, image_name))
            for image in images:
                image = cv2.resize(image, (64, 64))
                mask = np.sqrt(image.astype(np.float) / np.max(image))
                self.masks.append(mask)

    def get_mask(self):
        rnd = np.random.randint(0, len(self.masks))
        img = self.masks[rnd]
        (h, w) = img.shape
        center = (w / 2, h / 2)
        angle = np.random.randint(0, 360)
        matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated = cv2.warpAffine(img, matrix, (h, w))
        return rotated
