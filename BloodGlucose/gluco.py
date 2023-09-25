import cv2
import numpy as np
import os
from BloodGlucose.roi_bp import crop_image, roi_blood_pressure, find_closest_tuple
import time

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


def edged_img(cv_img, thresh):
    num_channels = cv_img.shape[2]
    if num_channels > 1:
        try:
            roi = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        except:
            pass
    roi = cv2.GaussianBlur(roi, (5, 5), 1)
    roi = cv2.bilateralFilter(roi, 0, sigmaColor=5, sigmaSpace=50)
    edged = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, C=thresh)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 40]
    edged = np.zeros_like(roi)
    cv2.drawContours(edged, contours=filtered_contours, contourIdx=-1,
                     color=255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    dilated = cv2.dilate(dilated, kernel, iterations=1)
    erosion_kernel = np.ones((3, 2), np.uint8)
    eroded = cv2.erode(dilated, erosion_kernel, iterations=1)
    cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_cnts = []
    canvas = roi.copy()
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if h > 40:
            digits_cnts += [cnt]

    sorted_digits = sorted(digits_cnts, key=lambda cnt: cv2.boundingRect(cnt)[0])
    digits = []
    area_max = 0
    area_pre = 0
    for i, cnt in enumerate(sorted_digits):
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = eroded[y: y + h, x: x + w]
        area_c = cv2.contourArea(cnt)
        if area_c > area_pre:
            area_max = i
            area_pre = area_c
        # print(f"W:{w}, H:{h}")
        qW, qH = int(w * 0.25), int(h * 0.15)
        fractionH, halfH, fractionW = int(h * 0.05), int(h * 0.5), int(w * 0.25)
        sevensegs = [
            ((0, 0), (w, qH)),  # a (top bar)
            ((w - qW, 0), (w, halfH)),  # b (upper right)
            ((w - qW, halfH), (w, h)),  # c (lower right)
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
        digits += [digit]
    if len(digits) == 0:
        digits = [0]
    return digits[area_max]


def roi_glu(image_path, flash):
    num = []
    digit = roi_blood_pressure(image_path, canny=100, num_canny=100)
    if flash == 'true':
        print('flash on')
        thresh = 4
    else:
        print('flash off')
        thresh = 2
    image3 = crop_image(digit, 40, 137, 97, 273)
    image3 = cv2.resize(image3, None, None, fx=2.3, fy=1)
    num += [edged_img(image3, thresh=thresh)]

    image2 = crop_image(digit, 120, 137, 177, 273)
    image2 = cv2.resize(image2, None, None, fx=2.3, fy=1)
    num += [edged_img(image2, thresh=thresh)]

    image1 = crop_image(digit, 190, 137, 257, 273)
    image1 = cv2.resize(image1, None, None, fx=2.3, fy=1)
    num += [edged_img(image1, thresh=thresh)]
    num_str = f'{num[0] * 100 + num[1] * 10 + num[2]}'
    if num_str == '888':
        num_str = '8.8.8'
    print(num_str)
    return num_str, digit

if __name__ == '__main__':
    folder_path = 'ori_img_1'
    imgs = os.listdir(folder_path)
    print(imgs)
    for img in imgs:
        name = os.path.splitext(img)[0]
        print(name)
        path = f'{folder_path}/{img}'
        stime = time.time()
        number, digit = roi_glu(image_path=path)
        etime = time.time()
        print('running time:', etime - stime)
        cv2.imshow(f'{number}', cv2.resize(digit, (240, 240)))
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('a'):
        #         cv2.destroyAllWindows()
        #         break
