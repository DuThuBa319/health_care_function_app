import cv2
import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import math
PURPLE = (75, 0, 130)
YELLOW = (0, 255, 255)
THICKNESS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
def get_values_around_number(number, count):
    values = [number]
    offset = 1
    for _ in range(count):
        values.append(number + offset)
        values.append(number - offset)
        if number - offset < 4 or number + offset > 11:
            break
        offset += 1
    return values




def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy


def calculate_discrete_density(image):
    unique_labels, label_counts = np.unique(image, return_counts=True)
    non_zero_labels = label_counts[1:]
    discrete_density = np.mean(non_zero_labels)
    return discrete_density


def brightness_thresh(image):
    cv2.imwrite('threshold.png', image)
    image_th = cv2.imread('threshold.png', cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image_th, 0, 255, cv2.THRESH_BINARY)
    total_pixels = binary_image.size
    white_pixels = cv2.countNonZero(binary_image)
    percentage_white = (white_pixels / total_pixels)
    return percentage_white


def brightness_mean(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright = np.mean(image)
    return bright

def euclidean_distance(tuple1, tuple2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(tuple1, tuple2)))


def compare_contour(image_check, image_sample):
    # image and image_sample must be in cv_grayscale_image
    contours1, _ = cv2.findContours(image_check, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_check, contours1, -1, (50, 130, 125), 5)
    contours2, _ = cv2.findContours(image_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_sample, contours2, -1, (50, 130, 125), 5)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image_check)
    ax1.set_title('Image check')
    ax2.imshow(image_sample)
    ax2.set_title('Image sample')
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    hu_moments1 = cv2.HuMoments(cv2.moments(contours1[0])).flatten()
    hu_moments2 = cv2.HuMoments(cv2.moments(contours2[0])).flatten()
    distance = cv2.norm(hu_moments1, hu_moments2, cv2.NORM_L2)
    return distance


def contour_shape(image1, image2):
    match_result = 0
    contours1, _ = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour1 in contours1:
        for contour2 in contours2:
            match_result = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
            if match_result < 0.1:
                cv2.drawContours(image1, [contour1], -1, (0, 255, 0), 2)
                cv2.drawContours(image2, [contour2], -1, (0, 255, 0), 2)
    fig, (axis1, axis2) = plt.subplots(1, 2)
    axis1.imshow(image1)
    axis2.imshow(image2)
    axis1.axis('Off')
    axis2.axis('Off')
    plt.tight_layout()
    plt.show()
    return match_result


def find_closest_tuple(target_list, tuple_list):
    closest_tuple = None
    closest_distance = float('inf')
    for tuple_item in tuple_list:
        distance = euclidean_distance(target_list, tuple_item)
        if distance < closest_distance:
            closest_tuple = tuple_item
            closest_distance = distance
    return closest_tuple


def replace_tuple_if_not_present(target_tuple, tuple_list):
    try:
        index = tuple_list.index(target_tuple)
    except ValueError:
        closest_tuple = find_closest_tuple(target_tuple, tuple_list)
        index = tuple_list.index(closest_tuple)
    tuple_list[index] = target_tuple

def increase_contrast(image, factor):
    img_array = np.array(image)
    mean_value = np.mean(img_array)
    contrasted_array = (img_array - mean_value) * factor + mean_value
    contrasted_array = np.clip(contrasted_array, 0, 255)
    contrasted_image = Image.fromarray(contrasted_array.astype('uint8'))
    return contrasted_image

def crop_image(image, left, top, right, bottom):
    image = cv2_to_pillow(image)
    cropped_image = image.crop((left, top, right, bottom))
    # cv2.imwrite(f'roi_trash/{str(uuid.uuid4())}.png', pillow_to_cv2(cropped_image))
    return pillow_to_cv2(cropped_image)

def cv2_to_pillow(cv2_image):
    pillow_image = Image.fromarray(cv2_image)
    return pillow_image

def pillow_to_cv2(pillow_image):
    cv2_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
    return cv2_image



def reduce_blur(pil_image, sigma):
    img_array = np.array(pil_image)
    blurred_array = gaussian_filter(img_array, sigma=sigma)
    blurred_pil_image = Image.fromarray(blurred_array.astype('uint8'))
    return blurred_pil_image


def roi_blood_pressure(img_path, canny=100, num_canny=100, dsize=(480, 640)):
    peri = 0
    peri_pre = 0
    x_cur, y_cur, w_cur, h_cur, area_cur, peri_cur = 0, 0, 0, 0, 0, 0
    #img_color = cv2.imread(img_path)
    img_color = cv2.resize(img_path, dsize)
    img_color_c = img_color.copy()
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img, (5, 5), 10)
    blurred = cv2.bilateralFilter(blurred, 5, sigmaColor=50, sigmaSpace=50)
    edged = cv2.Canny(blurred, canny, 150, 255)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:num_canny]
    cv2.drawContours(img_color, cnts, 0, PURPLE, THICKNESS)

    for i in range(len(cnts)):
        cv2.drawContours(img_color, cnts, i, PURPLE, THICKNESS)
        # print(f"ContourArea:{cv2.contourArea(cnts[i])}")
        x, y, w, h = cv2.boundingRect(cnts[i])
        cv2.rectangle(img_color, (x, y), (x + w, y + h), YELLOW, THICKNESS)
        area = round(cv2.contourArea(cnts[i]), 1)
        peri = round(cv2.arcLength(cnts[i], closed=True), 1)
        if peri >= peri_pre:
            peri_pre = peri
            x_cur, y_cur, w_cur, h_cur, area_cur, peri_cur = x, y, w, h, area, peri

        # print(f"ContourArea:{area}, Peri: {peri}")
        # cv2.putText(img_color, "Area:" + str(area), (x, y - 15), FONT, 0.4, PURPLE, 1)
        # cv2.putText(img_color, "Perimeter:" + str(peri), (x, y - 5), FONT, 0.4, PURPLE, 1)
    roi = img_color_c[y_cur: y_cur + h_cur, x_cur: x_cur + w_cur]
    # cv2.imshow('roi', roi)
    # cv2.waitKey(0)
    roi = cv2.resize(roi, (296, 385))
    # roi = pillow_to_cv2(increase_contrast(cv2_to_pillow(roi), 1.25)) # có thể gây mất nét khi đọc contour
    # image_name_with_extension = os.path.basename(img_path)
    # image_name, _ = os.path.splitext(image_name_with_extension)
    # roi_path = image_name + '-roi.png'
    # cv2.imwrite(f'inter/{roi_path}', roi)
    return roi




