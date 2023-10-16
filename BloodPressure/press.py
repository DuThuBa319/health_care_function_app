import os
import cv2
from BloodPressure.roi_bp import ( crop_image,
                                 roi_blood_pressure, find_closest_tuple)
import numpy as np
import time
import matplotlib.pyplot as plt



FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
DIGITSDICT_tuple = [
    (1, 1, 1, 1, 1, 1, 0),
    (0, 1, 1, 0, 0, 0, 0),
    (1, 1, 0, 1, 1, 0, 1),
    (1, 1, 1, 1, 0, 0, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (1, 1, 1, 0, 0, 1, 1),
    (1, 0, 1, 1, 0, 1, 1),
    (1, 0, 1, 1, 1, 1, 1),
    (1, 1, 1, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 0, 1, 1),
]
DIGITSDICT = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 0, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


def edged_img(cv_img, num_position=False, first_num=False, first_num_3=False):
    num_channels = cv_img.shape[2]
    if num_channels > 1:
        try:
            roi = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        except:
            pass
    roi = cv2.GaussianBlur(roi, (5, 5), 1)
    roi = cv2.bilateralFilter(roi, 0, sigmaColor=5, sigmaSpace=50)
    edged = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, C=2)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        (_, _, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) >= 50:
            filtered_contours += [contour]
    # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 50]
    edged = np.zeros_like(roi)
    cv2.drawContours(edged, contours=filtered_contours, contourIdx=-1,
                     color=(255, 255, 255), thickness=cv2.FILLED)
    if first_num:
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_max = 0
        contour_cur = 0
        for contour in contours:
            try:
                if cv2.contourArea(contour) >= cv2.contourArea(contour_cur):
                    contour_max = contour
                    contour_cur = contour_max
            except:
                contour_max = contour
                contour_cur = contour_max
        try:
            (x, y, w, h) = cv2.boundingRect(contour_max)
        except:
            print('digit in img are: [0]')
            return 0
        if w < 50:
            if w * h > 700:
                print("Digits in img are: [1]")
                return 1
        print("Digits in img are: [0]")
        return 0
    if num_position:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 9))
        dilated = cv2.dilate(edged, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
        dilated = cv2.dilate(dilated, kernel, iterations=1)

        erosion_kernel = np.ones((1, 2), np.uint8)
        eroded = cv2.erode(dilated, erosion_kernel, iterations=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
        dilated = cv2.dilate(dilated, kernel, iterations=1)

        erosion_kernel = np.ones((4, 3), np.uint8)
        eroded = cv2.erode(dilated, erosion_kernel, iterations=1)
    #####################
    if first_num_3:
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_max = 0
        contour_cur = 0
        for contour in contours:
            try:
                if cv2.contourArea(contour) >= cv2.contourArea(contour_cur):
                    contour_max = contour
                    contour_cur = contour_max
            except:
                contour_max = contour
                contour_cur = contour_max
        try:
            (x, y, w, h) = cv2.boundingRect(contour_max)
            print(w, h)
        except:
            print('digit in img are: [0]')
            return 0
        if h > 50:
            if cv2.contourArea(contour_max) > 1000:
                return 1
        else:
            return 0  
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if h > 30:
            digits_contours += [contour]
    sorted_digits = sorted(digits_contours, key=lambda contour: cv2.boundingRect(contour)[0])
    digits = []
    canvas = roi.copy()
    area_max = 0
    area_pre = 0
    for i, cnt in enumerate(sorted_digits):
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = eroded[y: y + h, x: x + w]
        area_c = cv2.contourArea(cnt)
        if area_c > area_pre:
            area_max = i
            area_pre = area_c
        print(f"W:{w}, H:{h}")
        qW, qH = int(w * 0.25), int(h * 0.15)
        fractionH, halfH, fractionW = int(h * 0.07), int(h * 0.5), int(w * 0.3)
        # ưiki
        sevensegs = [
            ((0, 0), (w, qH)),  # a (top bar)
            ((w - qW, 0), (w, halfH)),  # b (upper right)
            ((w - qW - 3, halfH), (w - 3, h)),  # c (lower right)
            ((0, h - qH), (w, h)),  # d (lower bar)
            ((0, halfH), (qW, h)),  # e (lower left)
            ((0, 0), (qW, halfH)),  # f (upper left)
            # ((0, halfH - fractionH), (w, halfH + fractionH)) # center
            (
                (0 + fractionW, halfH - fractionH),
                (w - fractionW, halfH + fractionH),
            ),  # center
        ]

        on = [0] * 7
        for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
            region = roi[p1y:p2y, p1x:p2x]
           
            if np.sum(region == 255) > region.size * 0.5:
                on[i] = 1
        if on not in DIGITSDICT_tuple:
            closest_tuple = find_closest_tuple(on, DIGITSDICT_tuple)
            index = DIGITSDICT_tuple.index(closest_tuple)
            on = DIGITSDICT_tuple[index]
        digit = DIGITSDICT[tuple(on)]
        if digit != 1:
            if w < 70:
                digit = 1
        # print(f"Digit is: {digit}")
        digits += [digit]
    
    if len(digits) == 0:
        digits = [0]
    return digits[area_max]



def roi_press(image_path):
    num = []
    number = []
    digit = roi_blood_pressure(image_path, canny=20, num_canny=100)
    digit1_png = crop_image(digit, 85, 25, 150, 134)
    digit1_png = cv2.resize(digit1_png, None, None, fx=2, fy=1)
    num += [edged_img(digit1_png, first_num=True)]
    digit2_png = crop_image(digit, 152, 25, 217, 134)
    digit2_png = cv2.resize(digit2_png, None, None, fx=2, fy=1)
    num += [edged_img(digit2_png)]
    digit3_png = crop_image(digit, 219, 30, 284, 139)
    digit3_png = cv2.resize(digit3_png, None, None, fx=2, fy=1)
    num += [edged_img(digit3_png)]
    number += [num[0] * 100 + num[1] * 10 + num[2]]
    digit7_png = crop_image(digit, 170, 295, 198, 379)
    digit7_png = cv2.resize(digit7_png, (120, 130))
    num += [edged_img(digit7_png, num_position=True, first_num_3=True)]
    digit8_png = crop_image(digit, 200, 295, 242, 379)
    digit8_png = cv2.resize(digit8_png, (120, 130))
    num += [edged_img(digit8_png, num_position=True)]
    digit9_png = crop_image(digit, 245, 295, 287, 379)
    digit9_png = cv2.resize(digit9_png, (120, 130))
    num += [edged_img(digit9_png, num_position=True)]
    number += [num[3] * 100 + num[4] * 10 + num[5]]
    return number
