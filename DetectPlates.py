# DetectPlates.py

import cv2
import numpy as np
import math
import PlateRecognition
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detect_plates_in_scene(img_original_scene):
    list_of_possible_plates = []  # this will be the return value

    height, width, num_of_channels = img_original_scene.shape

    img_contours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    img_grayscale_scene, img_thresh_scene = Preprocess.preprocess(img_original_scene)

    list_of_possible_chars_in_scene = find_possible_chars_in_scene(img_thresh_scene)

    list_of_lists_of_matching_chars = DetectChars.find_list_of_lists_of_matching_chars(
        list_of_possible_chars_in_scene)

    for list_of_matching_chars in list_of_lists_of_matching_chars:
            random_blue = random.randint(0, 255)
            random_green = random.randint(0, 255)
            random_red = random.randint(0, 255)

            contours = []

            for matching_char in list_of_matching_chars:
                contours.append(matching_char.contour)

            cv2.drawContours(img_contours, contours, -1, 
                             (random_blue, random_green, random_red))

    for list_of_matching_chars in list_of_lists_of_matching_chars:
        possible_plate = extract_plate(img_original_scene, list_of_matching_chars)

        if possible_plate.imgPlate is not None:   # if plate was found
            list_of_possible_plates.append(possible_plate)
        # end if
    # end for

    print("\n" + str(len(list_of_possible_plates)) + " possible plates found")

    for i in range(0, len(list_of_possible_plates)):
            rect_points = cv2.boxPoints(
                list_of_possible_plates[i].rrLocationOfPlateInScene)

            cv2.line(
                img_contours, tuple(
                    rect_points[0]), tuple(
                    rect_points[1]), PlateRecognition.SCALAR_RED, 2)
            cv2.line(
                img_contours, tuple(
                    rect_points[1]), tuple(
                    rect_points[2]), PlateRecognition.SCALAR_RED, 2)
            cv2.line(
                img_contours, tuple(
                    rect_points[2]), tuple(
                    rect_points[3]), PlateRecognition.SCALAR_RED, 2)
            cv2.line(
                img_contours, tuple(
                    rect_points[3]), tuple(
                    rect_points[0]), PlateRecognition.SCALAR_RED, 2)

            cv2.imshow("4b", list_of_possible_plates[i].imgPlate)
            cv2.waitKey(0)


    return list_of_possible_plates
# end function


def find_possible_chars_in_scene(img_thresh):
    list_of_possible_chars = []

    count_of_possible_chars = 0

    img_thresh_copy = img_thresh.copy()

    img_contours, contours, hierarchy = cv2.findContours(
        img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img_thresh.shape
    img_contours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):

        if PlateRecognition.showSteps:
            cv2.drawContours(
                img_contours,
                contours,
                i,
                PlateRecognition.SCALAR_WHITE)

        possible_char = PossibleChar.PossibleChar(contours[i])

        if DetectChars.check_if_possible_char(possible_char):
            count_of_possible_chars = count_of_possible_chars + 1
            list_of_possible_chars.append(possible_char)

    return list_of_possible_chars
# end function


def extract_plate(img_original, list_of_matching_chars):
    possible_plate = PossiblePlate.PossiblePlate()

    list_of_matching_chars.sort(key=lambda matching_char: matching_char.intCenterX)

    plate_center_x = (list_of_matching_chars[0].intCenterX + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterX) / 2.0
    plate_center_y = (list_of_matching_chars[0].intCenterY + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterY) / 2.0

    plate_center = plate_center_x, plate_center_y

    plate_width = int((list_of_matching_chars[len(
        list_of_matching_chars) - 1].intBoundingRectX + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intBoundingRectWidth - list_of_matching_chars[
                             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    total_of_char_heights = 0

    for matching_char in list_of_matching_chars:
        total_of_char_heights = total_of_char_heights + matching_char.intBoundingRectHeight

    average_char_height = total_of_char_heights / len(list_of_matching_chars)

    plate_height = int(average_char_height * PLATE_HEIGHT_PADDING_FACTOR)

    opposite = list_of_matching_chars[len(
        list_of_matching_chars) - 1].intCenterY - list_of_matching_chars[0].intCenterY
    hypotenuse = DetectChars.distance_between_chars(list_of_matching_chars[0], list_of_matching_chars[
        len(list_of_matching_chars) - 1])
    correction_angle_in_rad = math.asin(opposite / hypotenuse)
    correction_angle_in_deg = correction_angle_in_rad * (180.0 / math.pi)

    possible_plate.rrLocationOfPlateInScene = (
        tuple(plate_center), (plate_width, plate_height), correction_angle_in_deg)

    rotation_matrix = cv2.getRotationMatrix2D(
        tuple(plate_center), correction_angle_in_deg, 1.0)

    height, width, num_of_channels = img_original.shape

    img_rotated = cv2.warpAffine(img_original, rotation_matrix, (width, height))
    img_cropped = cv2.getRectSubPix(
        img_rotated, (plate_width, plate_height), tuple(plate_center))

    possible_plate.imgPlate = img_cropped

    return possible_plate
# end function
