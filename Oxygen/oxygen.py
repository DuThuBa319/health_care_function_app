import cv2
import os
import time
import numpy as np
from Oxygen.roi_bp import find_closest_tuple
import matplotlib.pyplot as plt


DIGITSDICT_tuple = [
    (1, 1, 1, 1, 1, 1, 0),
    (0, 1, 1, 0, 0, 0, 0),
    (1, 1, 0, 1, 1, 0, 1),
    (1, 1, 1, 1, 0, 0, 1),
    (0, 1, 1, 1, 0, 0, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (1, 1, 1, 0, 0, 1, 1),
    (1, 0, 1, 1, 0, 1, 1),
    (1, 0, 1, 1, 1, 1, 1),
    (1, 1, 1, 0, 0, 0, 0),
    (1, 1, 1, 0, 0, 0, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 0, 1, 1),
]
DIGITSDICT = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 0, 0): 7,
    (1, 1, 1, 0, 0, 0, 1): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}
def histogram(channel, i):
    histogram = cv2.calcHist([channel], [i], None, [256], [0, 255])
    start_bin = 200
    end_bin = 255
    selected_histogram = histogram[start_bin:end_bin + 1]
    max_index = np.argmax(selected_histogram)
    peak_value = max_index + start_bin
    # print("max hist", peak_value)
    return peak_value


def oxygen_meter(image_path):
    num = []
    #image = cv2.imread(image_path)
    image = cv2.resize(image_path, (600, 300))
    b, g, r = cv2.split(image)
    threshold_high_old = int(histogram(r, 0))
    threshold_high = int(threshold_high_old + 40)
    threshold_low = 210
    if threshold_high > 255:
        threshold_high = 255
    if int(threshold_low) > threshold_high_old:
        threshold_low = threshold_high_old - 80
    print(threshold_low, '-->', threshold_high)
    red_channel_mask = cv2.inRange(r, threshold_low, threshold_high)
    edged = np.zeros_like(r)
    edged[red_channel_mask > 0] = 255
    #old threshold
    # red_channel = image[:, :, 2]
    # threshold_value = 170
    # _, edged = cv2.threshold(red_channel, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filters_contours_1 = []
    for contour in contours:
        if (cv2.contourArea(contour) > 60):
            filters_contours_1 += [contour]
    edged = np.zeros_like(edged)
    height, width, _ = image.shape
    padding = 20
    for contour in filters_contours_1:
        x, y, w, h = cv2.boundingRect(contour)
        if x < padding or y < padding or x + w > width - padding or y + h > height - padding:
            continue
        cv2.drawContours(edged, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)







    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edged)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > w:
            contour_mask = np.zeros_like(edged)
            cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            kernel = np.array([[0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1],
                               [0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0],], dtype=np.uint8)

            dilated_mask = cv2.dilate(contour_mask, kernel, iterations=3)
            mask = cv2.bitwise_or(mask, dilated_mask)

    edged = cv2.bitwise_or(edged, mask)
    # cv2.imshow('{edged}', edged)
    dilated = edged.copy()
    while True:
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_main = sorted(contours, key=cv2.contourArea, reverse=True)
        total_area = sum(cv2.contourArea(contour) for contour in contours)
        total_area_main = sum(cv2.contourArea(contour) for contour in contours_main[:2])
        min_percentage = 0.8
        if total_area_main / total_area > min_percentage:
            break
        else:
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(dilated, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (1, 1))
    eroded = cv2.erode(dilated, kernel, iterations=1)
    # cv2.imshow('eroded', eroded)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         cv2.destroyAllWindows()
    #         break


    ############################################################
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_pos_num = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(min(len(contours), 2)):
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        # print( x, y, w, h)
        list_pos_num += [(x, y, w, h)]
    list_pos_num = sorted(list_pos_num, key=lambda rect: rect[0] + rect[2])
    canvas = image.copy()
    for x, y, w, h in list_pos_num:
        num += [edged_img(eroded, x, y, w, h, image)]
    





    number = num[0] * 10 + num[1]
    return number


def edged_img(eroded, X, Y, W, H, image):
    roi = eroded[Y: Y + H, X: X + W]
    img = image[Y: Y + H, X: X + W]
    roi = cv2.resize(roi, (122 * 2, 170 * 2))
    img = cv2.resize(img, (122 * 2, 170 * 2))
    h, w = roi.shape[0], roi.shape[1]


    
    
    
    
    
    
    qW, qH = int(w * 0.25), int(h * 0.15)
    fractionH, halfH, fractionW = int(h * 0.08), int(h * 0.5), int(w * 0.3)
    sevensegs = [
        ((0 + int(qW * 0.6), 0), (w - int(qW * 0.6), qH)),  # a (top bar)
        ((w - qW, qH), (w, halfH)),  # b (upper right)
        ((w - qW - 10, halfH), (w - 10, h-qH)),  # c (lower right)
        ((0, h - qH), (w, h)),  # d (lower bar)
        ((0, halfH), (qW, h-qH)),  # e (lower left)
        ((0, qH), (qW, halfH)),  # f (upper left)
        # ((0, halfH - fractionH), (w, halfH + fractionH)) # center
        (
            (0 + fractionW, halfH - fractionH),
            (w - fractionW, halfH + fractionH),
        ),  # center
    ]
    on = [0] * 7
    canvas = roi.copy()
    for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
        cv2.rectangle(canvas, (p1x+5, p1y+5), (p2x-5, p2y-5), (0, 0, 255), thickness=3)
        region = roi[p1y+5:p2y-5, p1x+5:p2x-5]
        # print(f'{i}:{(np.sum(region == 255))/(region.size * 0.5)}')
        if np.sum(region == 255) > region.size * 0.5:
            on[i] = 1
    if on not in DIGITSDICT_tuple:
        closest_tuple = find_closest_tuple(on, DIGITSDICT_tuple)
        index = DIGITSDICT_tuple.index(closest_tuple)
        on = DIGITSDICT_tuple[index]
    digit = DIGITSDICT[tuple(on)]
    if digit != 1:
        if w < 100:
            digit = 1
    return digit
