import os
import cv2
from BloodPressure.roi_bp import (cv2_to_pillow, pillow_to_cv2, crop_image,
                                 roi_blood_pressure, find_closest_tuple)
import numpy as np
import time
import matplotlib.pyplot as plt

def img_roi(digit):
    digit1_png = crop_image(digit, 85, 25, 150, 134)
    digit1_png = cv2.resize(digit1_png, None, None, fx=2, fy=1)

    digit2_png = crop_image(digit, 152, 25, 217, 134)
    digit2_png = cv2.resize(digit2_png, None, None, fx=2, fy=1)

    digit3_png = crop_image(digit, 219, 30, 284, 139)
    digit3_png = cv2.resize(digit3_png, None, None, fx=2, fy=1)

    digit4_png = crop_image(digit, 85, 174, 150, 283)
    digit4_png = cv2.resize(digit4_png, None, None, fx=2, fy=1)

    digit5_png = crop_image(digit, 150, 174, 215, 283)
    digit5_png = cv2.resize(digit5_png, None, None, fx=2, fy=1)

    digit6_png = crop_image(digit, 219, 174, 284, 283)
    digit6_png = cv2.resize(digit6_png, None, None, fx=2, fy=1)

    digit7_png = crop_image(digit, 201, 295, 241, 379)
    digit7_png = cv2.resize(digit7_png, (109, 130))

    digit8_png = crop_image(digit, 245, 295, 285, 379)
    digit8_png = cv2.resize(digit8_png, (109, 130))

    roi, (ax1, ax2, ax3, ax4, ax5,
          ax6, ax7, ax8) = plt.subplots(1, 8)
    ax1.imshow(digit1_png)
    ax2.imshow(digit2_png)
    ax3.imshow(digit3_png)
    ax4.imshow(digit4_png)
    ax5.imshow(digit5_png)
    ax6.imshow(digit6_png)
    ax7.imshow(digit7_png)
    ax8.imshow(digit8_png)
    plt.show()


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


def edged_img(cv_img, num_position=False, first_num=False):
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
    ##################################################
    # cv2.imshow('edge', edged)
    # # cv2.waitKey(10)
    # # cv2.destroyAllWindows()
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         cv2.destroyAllWindows()
    #         break
    ##################################################
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        (_, _, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) >= 50:
            filtered_contours += [contour]
    # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 50]
    edged = np.zeros_like(roi)
    cv2.drawContours(edged, contours=filtered_contours, contourIdx=-1,
                     color=255, thickness=cv2.FILLED)

    # cv2.imshow('new_edge', edged)
    # cv2.waitKey(10)
    # cv2.destroyAllWindows()
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         cv2.destroyAllWindows()
    #         break
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
    ##################################################
    # cv2.imshow("Eroded", eroded)
    # # cv2.waitKey(10)
    # # cv2.destroyAllWindows()
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         cv2.destroyAllWindows()
    #         break
    ##################################################
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
            # print(f'{i}:{(np.sum(region == 255))/(region.size * 0.5)}')
            # cv2.imshow('region', region)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('a'):
            #         cv2.destroyAllWindows()
            #         break
            # print(
            #     f"{i}: Sum of 1: {np.sum(region == 255)}, Size: {region.size * 0.5}"
            # )
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
    print(f"Digits in img are: {digits[area_max]}")
    if len(digits) == 0:
        digits = [0]
    return digits[area_max]


def roi_3(image_path):
    num_3 = []
    digit = roi_blood_pressure(image_path, canny=20, num_canny=100)
    digit1 = crop_image(digit, 85, 23, 284, 134)
    digit1 = cv2.resize(digit1, None, None, fx=2, fy=1)
    num_3 += [edged_img(digit1)]

    digit2 = crop_image(digit, 85, 174, 284, 283)
    digit2 = cv2.resize(digit2, None, None, fx=2, fy=1)
    num_3 += [edged_img(digit2)]

    digit3 = crop_image(digit, 203, 295, 283, 379)
    digit3 = cv2.resize(digit3, None, None, fx= 2.2, fy = 1.2)
    num_3 += [edged_img(digit3)]
    print(num_3)
    return num_3

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
    digit4_png = crop_image(digit, 85, 174, 150, 283)
    digit4_png = cv2.resize(digit4_png, None, None, fx=2, fy=1)
    num += [edged_img(digit4_png, first_num=True)]
    digit5_png = crop_image(digit, 150, 174, 215, 283)
    digit5_png = cv2.resize(digit5_png, None, None, fx=2, fy=1)
    num += [edged_img(digit5_png)]
    digit6_png = crop_image(digit, 219, 174, 284, 283)
    digit6_png = cv2.resize(digit6_png, None, None, fx=2, fy=1)
    num += [edged_img(digit6_png)]
    number += [num[0+3] * 100 + num[1+3] * 10 + num[2+3]]
    digit7_png = crop_image(digit, 200, 295, 242, 379)
    digit7_png = cv2.resize(digit7_png, (120, 130))
    num += [edged_img(digit7_png, num_position=True)]
    digit8_png = crop_image(digit, 245, 295, 287, 379)
    digit8_png = cv2.resize(digit8_png, (120, 130))
    num += [edged_img(digit8_png, num_position=True)]
    number += [num[6] * 10 + num[7]]
    print(number)
    return number

if __name__ == '__main__':
    folder_path = 'test_2_img'
    images = os.listdir(folder_path)
    print(images)
    for image in images:
        image_name = os.path.splitext(image)[0]
        print(image_name)
        image_path = f'{folder_path}/{image}'
        sTime = time.time()
        num = roi_press(image_path)
        # digit = roi_blood_pressure(image_path, canny=20, num_canny=100)
        # img_roi(digit)
        eTime = time.time()
        print('time run:', eTime - sTime)
        cv2.imshow(f'{num}', cv2.resize(cv2.imread(image_path), None, None, fx=0.5, fy=0.5))
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('a'):
        #         cv2.destroyAllWindows()
        #         break
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        print('**************%%%%%%%%%%%%%%%%%$$$$$$$$$$$$$$$$')