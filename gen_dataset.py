#!/usr/bin/env python
# encoding: utf-8

import argparse
from argparse import RawTextHelpFormatter
import os
import random

import image_util
from font2image import Font2Image

# 每个字符生成的图片数量
# 字体类型数量 *  rotate / rotate_step * 2

def get_label_dict(label_file):
    """ 文件每行一个标签，格式 ID:字符

    Args:
      label_file: 标签文件名
    """
    id_label = {}
    with open(label_file, 'r') as f:
        for line in f:
            item = line.split(":")
            id_label[item[0]] = item[1].strip("\n")
    return id_label

def args_parse():

    description = '''
python gen_dataset.py --label_file [label_file] --out_dir [dataset_dir] \
			--font_dir [fonts] --test_ratio [ratio default 0.2]\
			--width [img_width] --height [img_height] \
                        --rotate [rotate_angle] --rotate_step [rotate_step]\
                        --bg_black --resize
    '''
    #解析输入参数
    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--label_file', dest='label_file',
                        type=str, default=None, required=True,
                        help='label_file for record the id:character map')
    parser.add_argument('--out_dir', dest='out_dir',
                        type=str, default=None, required=True,
                        help='write a caffe dir')
    parser.add_argument('--font_dir', dest='font_dir',
                        type=str, default=None, required=True,
                        help='font dir to to produce images')
    parser.add_argument('--test_ratio', dest='test_ratio',
                        type=float, default=0.2, required=False,
                        help='test dataset size')
    parser.add_argument('--width', dest='width',
                        type=int, default=None, required=True,
                        help='image width')
    parser.add_argument('--height', dest='height',
                        type=int, default=None, required=True,
                        help='image height')
    parser.add_argument('--resize', dest='resize', required=False,
                        help='whether char is full image, default False', action='store_true')
    parser.add_argument('--bg_black', dest='bg_black', required=False,
                        help='whether black as background, default False', action='store_true')
    parser.add_argument('--rotate', dest='rotate',
                        type=int, default=0, required=False,
                        help='abs(rotate) degree 0-45')
    parser.add_argument('--rotate_step', dest='rotate_step',
                        type=int, default=0, required=False,
                        help='rotate step for the rotate angle')
    parser.add_argument('--need_aug', dest='need_aug', required=False,
                        help='need data augmentation', action='store_true')
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":

    options = args_parse()
    out_dir = os.path.expanduser(options['out_dir'])
    font_dir = os.path.expanduser(options['font_dir'])
    test_ratio = float(options['test_ratio'])
    width = int(options['width'])
    height = int(options['height'])
    bg_black = options['bg_black']
    resize = options['resize']
    rotate = int(options['rotate'])
    need_aug = options['need_aug']
    rotate_step = int(options['rotate_step'])
    train_image_dir_name = "train"
    test_image_dir_name = "test"

    # 将 dataset 分为 train 和 test 两个文件夹分别存储
    train_images_dir = os.path.join(out_dir, train_image_dir_name)
    test_images_dir = os.path.join(out_dir, test_image_dir_name)

    if os.path.isdir(train_images_dir):
        print("%s is not empty, backup and run again" % train_images_dir)
        os._exit(1)
    os.makedirs(train_images_dir)

    if os.path.isdir(test_images_dir):
        print("%s is not empty, backup and run again" % test_images_dir)
        os._exit(1)
    os.makedirs(test_images_dir)

    # 以下部分对于大量字符生成有问题，用迭代器的方式应该更合适。
    label_dict = get_label_dict(options['label_file'])

    char_list=[]  # 字符列表
    value_list=[] # label 列表
    for (value,chars) in label_dict.items():
        print (value,chars)
        char_list.append(chars)
        value_list.append(value)

    # 合并成新的映射关系表：（character：id）
    lang_chars = dict(zip(char_list,value_list))

    roate = abs(rotate)
    # 所有角度
    all_rotate_angles = []
    if rotate > 0 and rotate <= 45:
        for i in range(0, rotate+1, rotate_step):
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)
        #print(all_rotate_angles)
        if len(all_rotate_angles) == 0:
            print("set rotate but not one is match")
            os._exit(1)
    else:
        if rotate != 0:
            print("rotate bigger 45 is not support, your rotate is %d " %(rotate))
            os._exit(1)

    # 找到所有字体路径
    # TODO 检查字体有效性
    verified_font_paths = []
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        verified_font_paths.append(path_font_file)

    font2image = Font2Image((width, height), resize)

    # 外层循环是字
    for (char, label) in lang_chars.items():
        # 每种字符的图片列表
        image_list = []
        print (char,label)
        # 内层循环是字体
        for j, verified_font_path in enumerate(verified_font_paths):
            if rotate == 0:
                image = font2image.build(verified_font_path, char, 0, bg_black)
                image_list.append(image)
            else:
                for angle in all_rotate_angles:
                    image = font2image.build(verified_font_path, char, angle, bg_black)
                    image_list.append(image)

        if need_aug:
            image_list = image_util.random_aug(image_list)

        test_num = len(image_list) * test_ratio
        index = [i for i in range(len(image_list))]
        #print(index)
        assert(len(index) > 0)
        random.shuffle(index)  # 图像顺序打乱
        count = 0
        for i in index:
            #print(i)
            img = image_list[i]
            if count < test_num :
                char_dir = os.path.join(test_images_dir, "%d" % int(label))
            else:
                char_dir = os.path.join(train_images_dir, "%d" % int(label))
            count += 1

            if not os.path.isdir(char_dir):
                os.makedirs(char_dir)

            image_path = os.path.join(char_dir,"%d.png" % count)
            img.save(image_path)
