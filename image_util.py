#!/usr/bin/env python
# encoding: utf-8

import cv2 as cv

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated

def noise(img):
    for i in range(20): # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def erode(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img = cv.erode(img, kernel)
    return img

def dilate(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img = cv.dilate(img, kernel)
    return img

def random_aug(self, img_list=[], noise=True, dilate=True, erode=True):
    aug_list = copy.deepcopy(img_list)
    for i in range(len(img_list)):
        im = img_list[i]
        if noise and random.random() < 0.5:
            im = noise(im)
        if dilate and random.random() < 0.5:
            im = dilate(im)
        elif erode and random.random() < 0.5:
            im = erode(im)
        aug_list.append(im)
    return aug_list
