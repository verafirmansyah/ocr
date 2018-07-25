#!/usr/bin/env python
# encoding: utf-8

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np

# 根据字体生成图像
# 1. 支持生成黑色字符白色背景 或 支持生成白色字符，黑色背景
# 2. 字符填充整个图片 或 字符保留原来大小
# 3. TODO 控制字符与图片边界的距离
# 4. 目前当旋转某些角度时，字符底部存在遮挡问题
class Font2Image(object):

    def __init__(self, size, resize = False):
        """ 初始化

        Args:
            size: (width, height)
            resize: 如果字符大小不够是否 resize 到 width, height
        """
        self.width, self.height = size
        self.resize = resize

    @staticmethod
    def black_boundingbox(img, bg_thresold = 0):
        """获取图片非背景的边框，默认为二值图片，黑色为背景，
        这里显然对彩色图片表现不好，因为很难确定边界值

        Args:
          bg_thresold: 默认 0

        Return:
          包含图像的四个方向的偏移，格式 (left, top, right, low)
        TODO: 用 np 的方式会不会更高效
        """
        assert(len(img.shape) == 2)
        height = img.shape[0]
        width = img.shape[1]
        v_max = np.max(img, axis=0)
        h_max = np.max(img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1
        # 从左往右扫描，遇到非零像素点就以此为字体的左边界
        for i in range(width):
            if v_max[i] > bg_thresold:
                left = i
                break
        # 从右往左扫描，遇到非零像素点就以此为字体的右边界
        for i in range(width - 1, -1, -1):
            if v_max[i] > bg_thresold:
                right = i
                break
        # 从上往下扫描，遇到非零像素点就以此为字体的上边界
        for i in range(height):
            if h_max[i] > bg_thresold:
                top = i
                break
        # 从下往上扫描，遇到非零像素点就以此为字体的下边界
        for i in range(height - 1, -1, -1):
            if h_max[i] > bg_thresold:
                low = i
                break
        return (left, top, right, low)


    @staticmethod
    def white_boundingbox(img, bg_thresold = 255):
        """获取图片非背景的边框，默认为二值图片，白色为背景，
        这里显然对彩色图片表现不好，因为很难确定边界值

        Args:
          bg_thresold: 默认 255

        Return:
          包含图像的四个方向的偏移，格式 (left, top, right, low)
        TODO: 用 np 的方式会不会更高效
        """
        assert(len(img.shape) == 2)
        height = img.shape[0]
        width = img.shape[1]
        v_min = np.min(img, axis=0)
        h_min = np.min(img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1
        # 从左往右扫描，遇到非零像素点就以此为字体的左边界
        for i in range(width):
            if v_min[i] < bg_thresold:
                left = i
                break
        # 从右往左扫描，遇到非零像素点就以此为字体的右边界
        for i in range(width - 1, -1, -1):
            if v_min[i] < bg_thresold:
                right = i
                break
        # 从上往下扫描，遇到非零像素点就以此为字体的上边界
        for i in range(height):
            if h_min[i] < bg_thresold:
                top = i
                break
        # 从下往上扫描，遇到非零像素点就以此为字体的下边界
        for i in range(height - 1, -1, -1):
            if h_min[i] < bg_thresold:
                low = i
                break
        return (left, top, right, low)

    @classmethod
    def boundingbox(cls, img, bg_black):
        if bg_black:
            # img.getbbox()
            return cls.black_boundingbox(img)
        else:
            return cls.white_boundingbox(img)

    @staticmethod
    def rotate(img, angle, bg_black = False):
        """
        目前只支持白色背景，黑色背景
        参考 https://www.licoy.cn/3103.html
        """
        if bg_black:
            #out = img.rotate(angle)
            img2 = img.convert('RGBA')
            rot = img2.rotate(angle, expand=1)
            fff = Image.new('RGBA', img2.size, (0,)*4)
            out = Image.composite(rot, fff, rot)
            out.convert(img.mode)
        else:
            img2 = img.convert('RGBA')
            rot = img2.rotate(angle, expand=1)
            fff = Image.new('RGBA', img2.size, (255,)*4)
            out = Image.composite(rot, fff, rot)
            out.convert(img.mode)
        return out

    def build(self, font_path, char, angle=0, bg_black = False):
        """ 生成 RGB 图片
        Args:
          font_path: 字符字体
          char: 字符
          angle: 字符旋转角度
          background: 图片背景颜色，默认为白色
          charcolor:  字符颜色，默认为黑色

        Return:
          Image object
        """
        if bg_black:
            background = (0, 0, 0)
            charcolor = (255, 255, 255)
        else:
            charcolor = (0, 0, 0)
            background = (255, 255, 255)

        img = Image.new("RGB", (self.width, self.height), background)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width * 0.7))
        # 由于这里无法将字符保存在图片中间，因此，需要后续找到图片边框
        draw.text((0, 0), str(char), charcolor, font=font, align="center")
        if angle != 0:
            img = self.rotate(img, angle, bg_black)
        # 这里相当于只取 R 这个 channel
        np_img = np.asarray(img.getdata(), dtype='uint8')
        np_img = np_img[:, 0]
        np_img = np_img.reshape((self.height, self.width))
        # 获取字符边框
        left, upper, right, lower = self.boundingbox(np_img, bg_black)
        # img = img.crop((left, upper, right, lower))
        np_img = np_img[upper: lower + 1, left: right + 1]
        croped_height, croped_width = np_img.shape[:2]
        left_offset = (self.height - croped_height) // 2
        upper_offset = (self.width - croped_width) // 2
        bg = Image.fromarray(np_img)
        if self.resize:
            target_image = bg.resize((self.width, self.height))
            return target_image
        else:
            target_image = Image.new("RGB", (self.width, self.height), background)
            target_image.paste(Image.fromarray(np_img), (upper_offset, left_offset))
            return target_image

def test():
    font2img = Font2Image((32, 64), True)
    font2img.build("chinese_fonts/DroidSansFallbackFull.ttf", 0, True).show("resize_0")
    font2img.build("chinese_fonts/DroidSansFallbackFull.ttf", 0, False).show("resize_0")
    font2img.build("chinese_fonts/DroidSansFallbackFull.ttf", 0, 45, True).show("resize_45")
    font2img.build("chinese_fonts/DroidSansFallbackFull.ttf", 0, 45, False).show("resize_45")
    font2img2 = Font2Image((32, 64), False)
    font2img2.build("chinese_fonts/DroidSansFallbackFull.ttf", 1, True).show("keep_0")
    font2img2.build("chinese_fonts/DroidSansFallbackFull.ttf", 1, False).show("keep_0")
    font2img2.build("chinese_fonts/DroidSansFallbackFull.ttf", 1, 45, True).show("keep_45")
    font2img2.build("chinese_fonts/DroidSansFallbackFull.ttf", 1, 45, False).show("keep_45")

if __name__ == "__main__":
    test()
