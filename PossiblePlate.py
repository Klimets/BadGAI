# PossiblePlate.py

import cv2
import numpy as np


class PossiblePlate:

    # constructor
    def __init__(self):
        self.img_plate = None
        self.img_grayscale = None
        self.img_thresh = None

        self.rrLocationOfPlateInScene = None

        self.str_chars = ""  # end constructor
