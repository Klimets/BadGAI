# DetectChars.py
import os

import cv2
import numpy as np
import math
import random

import PlateRecognition
import Preprocess
import PossibleChar

kNearest = cv2.ml.KNearest_create()

# module level variables
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def load_and_train():

    try:
        classifications = np.loadtxt("classifications.txt", np.float32)
    except BaseException:
        print("error, unable to open classifications.txt, exiting program\n")
        return False
    try:
        flattened_images = np.loadtxt("flattened_images.txt", np.float32)
    except BaseException:
        print("error, unable to open flattened_images.txt, exiting program\n")
        return False

    classifications = classifications.reshape(
        (classifications.size, 1))
    kNearest.setDefaultK(1)
    kNearest.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)

    return True
# end function

def detect_chars_in_plates(list_of_possible_plates):

    if len(list_of_possible_plates) == 0:
        return list_of_possible_plates

    for possiblePlate in list_of_possible_plates:

        possiblePlate.img_grayscale, possiblePlate.img_thresh = Preprocess.preprocess(
            possiblePlate.imgPlate)

        possiblePlate.img_thresh = cv2.resize(
            possiblePlate.img_thresh, (0, 0), fx=1.6, fy=1.6)

        thresh_old_value, possiblePlate.img_thresh = cv2.threshold(
            possiblePlate.img_thresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        list_of_possible_chars_in_plate = find_possible_chars_in_plate(
            possiblePlate.img_grayscale, possiblePlate.img_thresh)

        if PlateRecognition.showSteps:
            height, width, num_channels = possiblePlate.imgPlate.shape
            contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for possible_char in list_of_possible_chars_in_plate:
                contours.append(possible_char.contour)

            cv2.drawContours(contours, contours, -1,
                             PlateRecognition.SCALAR_WHITE)

            cv2.imshow("6", contours)

        list_of_lists_of_matching_chars = find_list_of_lists_of_matching_chars(
            list_of_possible_chars_in_plate)

        if PlateRecognition.showSteps:
            contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for list_of_matching_chars in list_of_lists_of_matching_chars:
                random_blue = random.randint(0, 255)
                random_green = random.randint(0, 255)
                random_red = random.randint(0, 255)

                for matching_char in list_of_matching_chars:
                    contours.append(matching_char.contour)
                cv2.drawContours(contours, contours, -
                                 1, (random_blue, random_green, random_red))
            cv2.imshow("7", contours)

        if len(list_of_lists_of_matching_chars) == 0:
            possiblePlate.strChars = ""
            continue

        for i in range(0, len(list_of_lists_of_matching_chars)):
            list_of_lists_of_matching_chars[i].sort(
                key=lambda matching_char: matching_char.intCenterX)
            list_of_lists_of_matching_chars[i] = remove_inner_overlapping_chars(
                list_of_lists_of_matching_chars[i])

        if PlateRecognition.showSteps:
            contours = np.zeros((height, width, 3), np.uint8)

            for list_of_matching_chars in list_of_lists_of_matching_chars:
                random_blue = random.randint(0, 255)
                random_green = random.randint(0, 255)
                random_red = random.randint(0, 255)

                del contours[:]

                for matching_char in list_of_matching_chars:
                    contours.append(matching_char.contour)

                cv2.drawContours(contours, contours, -
                                 1, (random_blue, random_green, random_red))
            cv2.imshow("8", contours)

        len_of_longest_list_of_chars = 0
        index_of_longest_list_of_chars = 0

        for i in range(0, len(list_of_lists_of_matching_chars)):
            if len(
                    list_of_lists_of_matching_chars[i]) > len_of_longest_list_of_chars:
                len_of_longest_list_of_chars = len(
                    list_of_lists_of_matching_chars[i])
                index_of_longest_list_of_chars = i

        longest_list_of_matching_chars = list_of_lists_of_matching_chars[
            index_of_longest_list_of_chars]

        if PlateRecognition.showSteps:
            contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matching_char in longest_list_of_matching_chars:
                contours.append(matching_char.contour)

            cv2.drawContours(contours, contours, -1,
                             PlateRecognition.SCALAR_WHITE)

        possiblePlate.strChars = recognize_chars_in_plate(
            possiblePlate.img_thresh, longest_list_of_matching_chars)

    return list_of_possible_plates
# end function

def find_possible_chars_in_plate(img_grayscale, img_thresh):
    list_of_possible_chars = []
    img_thresh_copy = img_thresh.copy()

    # find all contours in plate
    contours, contours, hierarchy = cv2.findContours(
        img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possible_char = PossibleChar.PossibleChar(contour)

        if check_if_possible_char(possible_char):
            list_of_possible_chars.append(possible_char)
    return list_of_possible_chars


def check_if_possible_char(possible_char):
    if (possible_char.intBoundingRectArea > MIN_PIXEL_AREA and
            possible_char.intBoundingRectWidth > MIN_PIXEL_WIDTH and
            possible_char.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possible_char.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def find_list_of_lists_of_matching_chars(list_of_possible_chars):
    list_of_lists_of_matching_chars = []

    for possible_char in list_of_possible_chars:
        list_of_matching_chars = find_list_of_matching_chars(
            possible_char, list_of_possible_chars)

        list_of_matching_chars.append(possible_char)

        if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        list_of_lists_of_matching_chars.append(list_of_matching_chars)
        list_of_chars_with_current_matches_removed = list(
            set(list_of_possible_chars) - set(list_of_matching_chars))

        recursivelist_of_lists_of_matching_chars = find_list_of_lists_of_matching_chars(
            list_of_chars_with_current_matches_removed)

        for recursivelist_of_matching_chars in recursivelist_of_lists_of_matching_chars:
            list_of_lists_of_matching_chars.append(recursivelist_of_matching_chars)

        break

    return list_of_lists_of_matching_chars
# end function


def find_list_of_matching_chars(possible_char, list_of_chars):
    list_of_matching_chars = []

    for possiblematching_char in list_of_chars:
        if possiblematching_char == possible_char:
            continue

        int_distance_between_chars = distance_between_chars(
            possible_char, possiblematching_char)

        int_angle_between_chars = angle_between_chars(
            possible_char, possiblematching_char)

        change_in_area = float(
            abs(possiblematching_char.intBoundingRectArea - possible_char.intBoundingRectArea)) / float(
            possible_char.intBoundingRectArea)

        change_in_width = float(
            abs(possiblematching_char.intBoundingRectWidth - possible_char.intBoundingRectWidth)) / float(
            possible_char.intBoundingRectWidth)
        change_in_height = float(
            abs(possiblematching_char.intBoundingRectHeight - possible_char.intBoundingRectHeight)) / float(
            possible_char.intBoundingRectHeight)

        if (int_distance_between_chars < (possible_char.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                int_angle_between_chars < MAX_ANGLE_BETWEEN_CHARS and
                change_in_area < MAX_CHANGE_IN_AREA and
                change_in_width < MAX_CHANGE_IN_WIDTH and
                change_in_height < MAX_CHANGE_IN_HEIGHT):
            list_of_matching_chars.append(possiblematching_char)

    return list_of_matching_chars
# end function

def distance_between_chars(first_char, second_char):
    int_x = abs(first_char.intCenterX - second_char.intCenterX)
    int_y = abs(first_char.intCenterY - second_char.intCenterY)

    return math.sqrt((int_x ** 2) + (int_y ** 2))
# end function

def angle_between_chars(first_char, second_char):
    adj = float(abs(first_char.intCenterX - second_char.intCenterX))
    opp = float(abs(first_char.intCenterY - second_char.intCenterY))

    if adj != 0.0:
        angle_in_rad = math.atan(opp / adj)
    else:
        angle_in_rad = 1.5708

    angle_in_deg = angle_in_rad * (180.0 / math.pi)
    return angle_in_deg
# end function

def remove_inner_overlapping_chars(list_of_matching_chars):
    list_of_chars_with_inner_char_removed = list(list_of_matching_chars)

    for currentChar in list_of_matching_chars:
        for otherChar in list_of_matching_chars:
            if currentChar != otherChar:
                if distance_between_chars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in list_of_chars_with_inner_char_removed:
                            list_of_chars_with_inner_char_removed.remove(
                                currentChar)
                    else:
                        if otherChar in list_of_chars_with_inner_char_removed:
                            list_of_chars_with_inner_char_removed.remove(
                                otherChar)

    return list_of_chars_with_inner_char_removed
# end function

def recognize_chars_in_plate(img_thresh, list_of_matching_chars):
    str_chars = ""

    height, width = img_thresh.shape

    img_thresh_color = np.zeros((height, width, 3), np.uint8)

    list_of_matching_chars.sort(key=lambda matching_char: matching_char.intCenterX)
    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR, img_thresh_color)
    for currentChar in list_of_matching_chars: 
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(
            img_thresh_color,
            pt1,
            pt2,
            PlateRecognition.SCALAR_GREEN,
            2)

        img_roi = img_thresh[
            currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
            currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        img_roi_resized = cv2.resize(
            img_roi, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))

        roi_resized = img_roi_resized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
        roi_resized = np.float32(roi_resized)
        retval, results, neigh_resp, dists = kNearest.findNearest(
            roi_resized, k=1)
        str_current_char = str(chr(int(results[0][0])))

        str_chars = str_chars + str_current_char

    return str_chars
# end function
