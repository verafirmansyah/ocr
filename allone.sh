#!/bin/bash

#python gen_dataset.py --label_file digital_label --out_dir dataset --font_dir chinese_fonts --width 64 --height 64 --rotate 40 --rotate_step 2 --resize
python ocr.py --mode test
python ocr.py --mode train
python ocr.py --mode test


