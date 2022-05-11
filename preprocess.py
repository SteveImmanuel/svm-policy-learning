from re import L
import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from constants import *
from utils import *


def rotate(img, angle, rot_center=None):
    """
    Rotate the image by angle degrees.
    """
    if rot_center is None:
        rot_center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_mat = cv.getRotationMatrix2D(rot_center, angle, 1.0)
    return cv.warpAffine(img, rot_mat, img.shape[:2])


def get_state_vector(pts, img):
    data_pts = np.array(pts, dtype=np.float64).reshape(len(pts), 2)
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    center = (int(mean[0, 0]), int(mean[0, 1]))

    axis = []
    for i in range(4):
        axis.append({})
    axis[0]['dir'] = normalize_vector(eigenvectors[0])
    axis[1]['dir'] = normalize_vector(eigenvectors[1])

    axis[0]['intersection'] = find_intersection2(center, axis[0]['dir'], img, (255, 255, 255))
    axis[1]['intersection'] = find_intersection2(center, axis[1]['dir'], img, (255, 255, 255))

    axis[0]['len'] = np.linalg.norm(axis[0]['intersection'][0] - axis[0]['intersection'][1])
    axis[1]['len'] = np.linalg.norm(axis[1]['intersection'][0] - axis[1]['intersection'][1])

    major_idx = np.argmax([axis[0]['len'], axis[1]['len']])
    minor_idx = 1 - major_idx

    new_center = (axis[major_idx]['intersection'][1] + axis[major_idx]['intersection'][0]) // 2
    axis[0]['center'] = new_center
    axis[1]['center'] = new_center

    axis[minor_idx]['intersection'] = find_intersection2(
        new_center,
        axis[minor_idx]['dir'],
        img,
        (255, 255, 255),
    )
    axis[minor_idx]['len'] = np.linalg.norm(axis[minor_idx]['intersection'][0] -
                                            axis[minor_idx]['intersection'][1])

    axis[2]['center'] = (new_center + axis[major_idx]['intersection'][0]) // 2
    axis[3]['center'] = (new_center + axis[major_idx]['intersection'][1]) // 2
    axis[2]['intersection'] = find_intersection2(
        axis[2]['center'],
        axis[minor_idx]['dir'],
        img,
        (255, 255, 255),
    )
    axis[3]['intersection'] = find_intersection2(
        axis[3]['center'],
        axis[minor_idx]['dir'],
        img,
        (255, 255, 255),
    )

    if axis[major_idx]['dir'][0] < 0:
        axis[major_idx]['dir'] *= -1
    angle = atan2(axis[major_idx]['dir'][1], axis[major_idx]['dir'][0])

    cv.circle(img, axis[0]['center'], 2, (255, 0, 255), 2)
    cv.circle(img, axis[2]['center'], 2, (0, 255, 0), 2)
    cv.circle(img, axis[3]['center'], 2, (0, 255, 0), 2)
    cv.circle(img, axis[minor_idx]['intersection'][0], 2, (0, 0, 255), 2)
    cv.circle(img, axis[minor_idx]['intersection'][1], 2, (0, 0, 255), 2)
    cv.circle(img, axis[major_idx]['intersection'][0], 2, (255, 0, 0), 2)
    cv.circle(img, axis[major_idx]['intersection'][1], 2, (255, 0, 0), 2)
    cv.circle(img, axis[2]['intersection'][0], 2, (0, 255, 0), 2)
    cv.circle(img, axis[2]['intersection'][1], 2, (0, 255, 0), 2)
    cv.circle(img, axis[3]['intersection'][0], 2, (0, 255, 0), 2)
    cv.circle(img, axis[3]['intersection'][1], 2, (0, 255, 0), 2)

    cv.line(img, axis[major_idx]['intersection'][0], axis[major_idx]['intersection'][1],
            (255, 0, 0), 1)
    cv.line(img, axis[minor_idx]['intersection'][0], axis[minor_idx]['intersection'][1],
            (0, 0, 255), 1)
    cv.line(img, axis[2]['intersection'][0], axis[2]['intersection'][1], (0, 255, 0), 1)
    cv.line(img, axis[3]['intersection'][0], axis[3]['intersection'][1], (0, 255, 0), 1)

    return axis, major_idx, angle


img = cv.imread('test_images/3.png')
cv.imshow('img', img)
# img = rotate(img, -40)

blank = np.zeros(img.shape, np.uint8)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)

ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
# cv.imshow('thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for i, points in enumerate(contours):
    area = cv.contourArea(points)
    if area < AREA_MIN_THRESHOLD or area > AREA_MAX_THRESHOLD:
        continue

    cv.drawContours(blank, contours, i, (255, 255, 255), 2)
    info = get_state_vector(points, blank)
    print(info[0][2])
    print(info[0][3])

cv.imshow('contours', blank)
cv.waitKey(0)
cv.destroyAllWindows()