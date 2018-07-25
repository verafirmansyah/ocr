#!/usr/bin/env python
# encoding: utf-8

import sys

label_file = sys.argv[1]
inference_file = "inference_result"


def get_result(filepath):
    content = open(filepath, "r").read()
    return content.split(",")

label = get_result(label_file)
inference = get_result(inference_file)

index = 0
match = 0
diff_set = set()
for k, v in zip(inference, label):
    if k != v:
      diff_set.add((k, v))
      print("index {0} inference is {1} but label is {2}".format(index, k, v))
    else:
      match += 1
    index += 1

print("accuracy is ", float(match)/index)
print("difference set is ", diff_set)
