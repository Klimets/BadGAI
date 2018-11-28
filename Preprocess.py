# Preprocess.py

import cv2
import numpy as np
import math

# module level variables
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


def preprocess(img_original):
    img_grayscale = extract_value(img_original)

    img_max_contrast_grayscale = maximize_contrast(img_grayscale)

    height, width = img_grayscale.shape

    img_blurred = cv2.GaussianBlur(
        img_max_contrast_grayscale,
        GAUSSIAN_SMOOTH_FILTER_SIZE,
        0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        255.0,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_THRESH_BLOCK_SIZE,
        ADAPTIVE_THRESH_WEIGHT)

    return img_grayscale, img_thresh
# end function


def extract_value(img_original):

    img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

    img_hue, img_saturation, img_value = cv2.split(img_hsv)

    return img_value
# end function


def maximize_contrast(img_grayscale):

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_top_hat = cv2.morphologyEx(
        img_grayscale,
        cv2.MORPH_TOPHAT,
        structuring_element)
    img_black_hat = cv2.morphologyEx(
        img_grayscale,
        cv2.MORPH_BLACKHAT,
        structuring_element)

    img_grayscale_plus_top_hat = cv2.add(img_grayscale, img_top_hat)
    img_grayscale_plus_top_hat_minus_black_hat = cv2.subtract(
        img_grayscale_plus_top_hat, img_black_hat)

    return img_grayscale_plus_top_hat_minus_black_hat
# end function
