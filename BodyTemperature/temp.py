import cv2
import numpy as np
import time
import os
from BodyTemperature.roi_bp import find_closest_tuple

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)
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
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}




def temp(image, flash=False):
    roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(cv2.resize(roi, (600, 170)), (11, 11), 5)
    roi = cv2.bilateralFilter(roi, 1, sigmaColor=10, sigmaSpace=75)
    #roi = cv2.resize(roi, None, None, fx=2.2, fy=1.5)
    if flash == 'true':
        # print('flash on')
        edged = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, C=4)
    else:
        # print('flash off')
        edged = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, C=2)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    filtered_contours = []
    border_thickness = 5
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (cv2.contourArea(contour) > 380) & (w < 200) & (h < 250) & (x > border_thickness or x + w > edged.shape[1] - border_thickness):
            filtered_contours += [contour]
    edged = np.zeros_like(roi)
    cv2.drawContours(edged, contours=filtered_contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
    
    
    
    
    
    filtered_contours = []
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(roi)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > w:
            # Tạo mask cho contour hiện tại
            contour_mask = np.zeros_like(roi)
            cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            kernel = np.ones((8, 1), np.uint8)
            dilated_mask = cv2.dilate(contour_mask, kernel, iterations=3)
            mask = cv2.bitwise_or(mask, dilated_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    left, right, top, bottom = x, x + w, y, y + h
    canvas = roi.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        left = min(left, x)
        right = max(right, x + w)
        top = min(top, y)
        bottom = max(bottom, y + h)
    # print(left, right, bottom, top)
    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 0), 2)
    # cv2.imshow(' can', canvas)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         cv2.destroyAllWindows()
    #         break

    edged = cv2.bitwise_or(edged, mask)
    erosion_kernel = np.ones((4, 4), np.uint8)
    eroded = cv2.erode(edged, erosion_kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (2, 5))
    eroded = cv2.dilate(eroded, kernel, iterations=2)
    ##################################################
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    list_pos_num = []
    closest_distance = 0
    closest_pair = (0, 0)
    for i in range(min(len(contours), 3)):
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        # print( x, y, w, h)
        list_pos_num += [(x, y, w, h)]
    list_pos_num = sorted(list_pos_num, key=lambda rect: rect[0] + rect[2])
    closest_pair = None
    closest_distance = float('inf')
    for _, _, w_chuan, h_chuan in list_pos_num:
        ratio = h_chuan / w_chuan
        distance = abs(ratio - 1.18)
        if distance < closest_distance:
            closest_distance = distance
            closest_pair = (w_chuan, h_chuan)
    w_o, h_o = closest_pair
    # print(w_o, h_o)
    num = []
    for x, y, w, h in list_pos_num:
        # print('y+h, right:', w+x, right)
        if x+w > right:
            # print('oke')
            roi_num = roi[y + h - h_o:y + h, right - w_o:right]
            num += [edged_img(roi_num)]
            
            # cv2.imshow(' Image', roi_num)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('a'):
            #         cv2.destroyAllWindows()
            #         break
        else:
            # print(w / h)
            if (w / h < 0.5) or (w / h > 1.12):
                roi_num = roi[y + h - h_o:y + h, x + w - w_o:x + w]
            else:
                roi_num = roi[y:y + h, x:x + w]
            num += [edged_img(roi_num)]
            # cv2.imshow(' Image', roi_num)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('a'):
            #         cv2.destroyAllWindows()
            #         break
    print(num)
    if len(num) == 3:
        num = str(float(num[0] * 10 + num[1] + num[2] / 10))
    else:
        num = 'Error'
    return num



def edged_img(roi):
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
        # print('area:', cv2.contourArea(contour), w, h)
        if (cv2.contourArea(contour) >= 75) & (h > 10) & (w < roi.shape[1]):
            filtered_contours += [contour]
    # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= 50]
    edged = np.zeros_like(roi)
    cv2.drawContours(edged, contours=filtered_contours, contourIdx=-1,
                     color=(255, 255, 255), thickness=cv2.FILLED)

    # cv2.imshow('new_edge', edged)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         cv2.destroyAllWindows()
    #         break

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    dilated = cv2.dilate(dilated, kernel, iterations=1)

    erosion_kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(dilated, erosion_kernel, iterations=1)
    #################################################
    # cv2.imshow("Eroded", eroded)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         cv2.destroyAllWindows()
    #         break
    #################################################
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
        qW, qH = int(w * 0.25), int(h * 0.2)
        fractionH, halfH, fractionW = int(h * 0.07), int(h * 0.5), int(w * 0.3)
        # ưiki
        sevensegs = [
            ((0, 0), (w, qH)),  # a (top bar)
            ((w - qW, 0), (w, halfH)),  # b (upper right)
            ((w - qW - 3, halfH), (w - 3, h)),  # c (lower right)
            ((0, h - qH), (w, h)),  # d (lower bar)
            ((0 + 3, halfH), (qW, h + 3)),  # e (lower left)
            ((0 + 3, 0), (qW, halfH + 3)),  # f (upper left)
            # ((0, halfH - fractionH), (w, halfH + fractionH)) # center
            (
                (0 + fractionW, halfH - fractionH),
                (w - fractionW, halfH + fractionH),
            ),  # center
        ]
        on = [0] * 7
        for (i, ((p1x, p1y), (p2x, p2y))) in enumerate(sevensegs):
            region = roi[p1y:p2y, p1x:p2x]
            
            
            
            
            
            
            
            
             
            if np.sum(region == 255) > region.size * 0.6:
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
    #print(f"Digits in img are: {digits[area_max]}")
    if len(digits) == 0:
        digits = [0]
    return digits[area_max]
  #  print('--------------------------------------------++')
   

# if __name__ == '__main__':
#     folder_path = 'ori_img_5'
#     images = os.listdir(folder_path)
#     print(images)
#     for image in images:
#         file_path = os.path.join(folder_path, image)
#         if os.path.isfile(file_path) and image.lower().endswith(('.jpg', '.jpeg', '.png')):
#             name = os.path.splitext(image)[0]
#             print(name)
#             image_path = f'{folder_path}/{image}'
#             temp(image_path)
#             # cv2.imshow(f'{name}', roi)
#             # while True:
#             #     if cv2.waitKey(1) & 0xFF == ord('a'):
#             #         cv2.destroyAllWindows()
#             #         break
