#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import os
import sys
import cv2 as cv
import numpy as np
import logging
import logging.handlers

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(lineno)s : %(message)s')
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s : %(message)s')

handler = logging.handlers.RotatingFileHandler("segment.log", maxBytes = 10*1020*1024, backupCount = 3)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
log.addHandler(handler)

class ImageUtil(object):
    def __init__(self, path):
        self.path = path
        self.img = None

    def read_gray(self):
        if self.img is None:
            self.img = cv.imread(self.path, cv.IMREAD_GRAYSCALE)

    def read(self):
        if self.img is None:
            self.img = cv.imread(self.path, cv.IMREAD_COLOR)

    @staticmethod
    def save(img, dstpath):
        if img is not None:
            cv.imwrite(dstpath, img)
        else:
            raise ValueError("write before read, please read first")

    def enlarge(self, heigh_scale, width_scale, interpolation=cv.INTER_LINEAR):
        assert(int(heigh_scale) >= 1)
        assert(int(width_scale) >= 1)
        self.read()
        height, width = self.img.shape[:2]
        # cv.INTER_CUBIC
        enlarge = cv.resize(self.img, (0, 0),
                            fx = width_scale,
                            fy = heigh_scale,
                            interpolation=interpolation)
        self.img = enlarge
        return enlarge

    def shrink(self, heigh_scale, width_scale):
        assert(int(heigh_scale) == 0)
        assert(int(width_scale) == 0)
        self.read()
        height, width = self.img.shape[:2]
        size = (int(width * width_scale), int(height * heigh_scale))
        shrinked = cv.resize(self.img, size, interpolation=cv.INTER_AREA)
        self.img = shrinked
        return shrinked

    # 切换到图片显示页面，按下 q 退出
    @staticmethod
    def show(img, name="test", timeout_sec=0):
        assert(img.shape[0] > 0)
        assert(img.shape[1] > 0)
        print(img.shape)
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(timeout_sec * 1000)
        cv.destroyWindow(name)

    def binarization(self, threshold, maxval):
        self.read()
        if len(self.img.shape) == 3 and self.img.shape[2] == 1:
            img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        else:
            img = self.img
        # 大律法，全局自适应，0 可以修改为任意数字，但不起作用
        ret, dst = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # TRIANGLE，全局自适应，0 可以修改为任意数字，但不起作用，适用于单个波峰
        #ret, dst = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.TRIANGLE)
        #ret, dst = cv.threshold(img, threshold, maxval, cv.THRESH_BINARY)
        #ret, dst = cv.threshold(img, threshold, maxval, cv.THRESH_BINARY_INV)
        #ret, dst = cv.threshold(img, threshold, maxval, cv.THRESH_TRUNC)
        #ret, dst = cv.threshold(img, threshold, maxval, cv.THRESH_TOZERO)
        self.img = dst
        return dst

    @staticmethod
    def resize(img, newsize, keep_aspect = True):
        if keep_aspect:
            height, width = newsize.shape[:2]
            cur_height, cur_width = img.shape[:2]

            ratio_w = float(width)/float(cur_width)
            ratio_h = float(height)/float(cur_height)
            ratio = min(ratio_w, ratio_h)

            new_size = (min(int(cur_width*ratio), width),
                        min(int(cur_height*ratio), height))

            new_size = (max(new_size[0], 1),
                        max(new_size[1], 1),)
            resized_img = cv2.resize(img, new_size)
            return resized_img
        else:
            new_size = (max(newsize[0], 1),
                        max(newsize[1], 1),)
            resized_img = cv2.resize(img, newsize)
            return resized_img



class Segmentation(object):
    def __init__(self, img):
        self.img = img

    def cut_one_line(self, img, begin, end):
        return img[begin:end+1, :]

    def cut_one_colum(self, img, begin, end):
        return img[:, begin:end+1]


    # img is binarization, only 0 and 255
    def get_text_projection(self, img, mode):
        img = np.array(img)
        print(img.shape)
        assert(len(img.shape) == 2)
        rows, cols = img.shape
        # 统计每行中包含字符的数量(0 为黑色)
        if mode == "row":
            row_num = [0 for i in range(rows)]
            index = np.where(img == 0)[0]
            for i in index:
                row_num[i] += 1
            return row_num
        elif mode == "colum":
            col_num = [0 for i in range(cols)]
            index = np.where(img == 0)[1]
            for i in index:
                col_num[i] += 1
            return col_num
        else:
            raise ValueError("mode only support colum | row")

    def test_get_text_projection(self):
        a = [[0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
        print("origin: ", a)
        print("row projection: ", self.get_text_projection(a, "row"))
        print("colum projection: ", self.get_text_projection(a, "colum"))


    def get_peek_range(self, hist, min_valid_char = 2, min_pix_per_char = 5):
        self.debug_peek_range(hist, min_valid_char, min_pix_per_char)
        hist = np.array(hist)
        assert(len(hist.shape) == 1)
        indexes = np.arange(hist.shape[0])
        valid_index = indexes[hist >= min_valid_char]
        log.debug(valid_index)
        begin = valid_index[0]
        ret  = []
        log.debug(valid_index.shape[0])
        max_pix = 0
        min_pix = valid_index.shape[0]
        for i in range(1, valid_index.shape[0]):
            if valid_index[i] == valid_index[i-1] + 1 and i < valid_index.shape[0] - 1:
                continue
            if i == valid_index.shape[0] - 1:
                end = valid_index[i]
            else:
                end = valid_index[i-1]
            char_pix = end - begin + 1
            if (char_pix >= min_pix_per_char):
                ret.append((begin, end))
            else:
                print("the char's height is not enough")
            begin = valid_index[i]
        return ret


    def test_get_peek_range(self):
        a = [0, 3, 3, 3, 1, 2, 4, 4, 4, 4, 4, 1, 1, 4, 4, 1,1, 5,5,5,5]
        print(self.get_peek_range(np.array(a), 2, 3))

    def debug_peek_range(self, hist, min_valid_char = 2, min_pix_per_char = 5):
        hist = np.array(hist)
        assert(len(hist.shape) == 1)
        indexes = np.arange(hist.shape[0])
        valid_index = indexes[hist >= min_valid_char]
        log.info("max_num_chars {0}, min_num_chars {1}, chars_num_thresh {2}".format(
            np.max(hist), np.min(hist[np.where(hist > 0)]), min_valid_char))
        log.debug(valid_index)
        begin = valid_index[0]
        log.debug(valid_index.shape[0])
        max_pix = 0
        min_pix = valid_index.shape[0]
        for i in range(1, valid_index.shape[0]):
            if valid_index[i] == valid_index[i-1] + 1 and i < valid_index.shape[0] - 1:
                continue
            if i == valid_index.shape[0] - 1:
                end = valid_index[i]
            else:
                end = valid_index[i-1]
            char_pix = end - begin + 1
            if (char_pix >= min_pix_per_char):
                #for debug
                if (char_pix > max_pix):
                    max_pix = char_pix
                if (char_pix < min_pix):
                    min_pix = char_pix
            else:
                #for debug
                if (char_pix > max_pix):
                    max_pix = char_pix
                if (char_pix < min_pix):
                    min_pix = char_pix
                print("the char's height is not enough")
            begin = valid_index[i]
        log.info("max_pix {0}, min_pix {1}, min_pix_thresh {2}".format(
            max_pix, min_pix, min_pix_per_char))

    # 列切割
    def colum_split(self, img, min_valid_pix = 2, min_char_width = 5):
        char_list = []
        # 统计图片每列包含字符的像素个数
        col_hist = self.get_text_projection(img, "colum")
        # 每列像素超过 2 个，并且超过连续 10 列，就认为该部分为字符。
        valid_colums = self.get_peek_range(col_hist, min_valid_pix, min_char_width)
        log.debug(valid_colums)
        log.info("colum total %d pair", len(valid_colums))

        chars = []
        for i in range(len(valid_colums)):
            char = self.cut_one_colum(img, valid_colums[i][0], valid_colums[i][1])
            chars.append(char)
        return chars

    # 行切割
    def row_split(self, img, min_valid_pix = 2, min_char_heigh = 2):
        # 统计图片每行包含字符的像素个数
        row_hist = self.get_text_projection(img, "row")
        # 每行像素超过 10 个，并且超过连续 10 行，就认为该部分为字符。
        valid_rows = self.get_peek_range(row_hist, min_valid_pix, min_char_heigh)
        log.debug(valid_rows)
        log.info("row total %d pair", len(valid_rows))

        lines = []
        # 获取每行有效的字符
        for i in range(len(valid_rows)):
            log.debug(valid_rows[i][0], valid_rows[i][1])
            line = self.cut_one_line(self.img, valid_rows[i][0], valid_rows[i][1])
            log.debug(line.shape)
            lines.append(line)
        return lines

    @staticmethod
    def print_img_pix(img):
        limit = 0
        one_line = 50
        for row in img:
            limit += 1
            if limit == 50:
                break
            count = int(len(row) / one_line)
            remain = len(row) % one_line
            for i in range(1, count + 1):
                print(row[(i-1) * one_line:i * one_line])
            print(row[count * one_line:])
            print("\n")

    def run(self, debug = True):
        valid_rows = self.row_split(self.img, 10, 3)
        chars_images = []
        for i, row in enumerate(valid_rows):
            if debug:
                ImageUtil.show(row, "row" + str(i))
            chars = self.colum_split(row, 2, 5)
            chars_images.extend(chars)
        if debug:
            count = 1
            for i, char in enumerate(chars_images):
                count += 1
                if count == 20:
                    break
                ImageUtil.show(char, "char" + str(i))
        return chars_images


def debug(path):
    if not os.path.exists(path):
        raise ValueError("path {0} doesn't exist".format(path))
    img = ImageUtil(path)
    img.read_gray()
    img.enlarge(2, 2)
    #img.shrink(0.1, 0.1)
    print(img.img.shape)
    #img.shrink(0.1, 0.1)
    #img.shrink(0.5, 0.5)
    img.binarization(0, 255)
    seg = Segmentation(img.img)
    chars_images = seg.run()

def main(path):
    if not os.path.exists(path):
        raise ValueError("path {0} doesn't exist".format(path))
    img = ImageUtil(path)
    img.read_gray()
    img.enlarge(2, 2)
    #img.shrink(0.1, 0.1)
    print(img.img.shape)
    #img.shrink(0.1, 0.1)
    #img.shrink(0.5, 0.5)
    img.binarization(0, 255)
    seg = Segmentation(img.img)
    chars_images = seg.run(False)
    target_dir = "test"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    for i, img in enumerate(chars_images):
        ImageUtil.save(img, os.path.join(target_dir, str(i) + ".png"))



if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("%s %s"%(sys.argv[0], "img_file"))
        os._exit(1)
    main(sys.argv[1])
