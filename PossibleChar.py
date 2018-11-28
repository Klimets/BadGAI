# PossibleChar.py

import cv2
import math


class PossibleChar:

    # constructor
    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)

        [int_x, int_y, int_width, int_height] = self.boundingRect

        self.intBoundingRectX = int_x
        self.intBoundingRectY = int_y
        self.intBoundingRectWidth = int_width
        self.intBoundingRectHeight = int_height

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (
            self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (
            self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt(
            (self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(
            self.intBoundingRectWidth) / float(self.intBoundingRectHeight)
    # end constructor
