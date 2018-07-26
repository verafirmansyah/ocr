

## 测试环境

python3.6
tensorflow 1.7


## 流程

### 处理流程

1. 生成数据集 : 根据字体，角度生成数据集
2. 训练
3. 验证
4. 测试

### 生成数据集

    python gen_dataset.py --label_file digital_label --out_dir dataset --font_dir chinese_fonts --width 64 --height 64 --rotate 40 --rotate_step 2 --resize

### 训练

    python ocr.py --mode test
    python ocr.py --mode train


### 验证

1. 将测试图片分割为字符，保存在  test 文件夹

    python segment.py pic/numbers.png

2. 进行验证，从 test 文件夹读图片并预测

    python ocr.py --mode inference

3. 比较验证结果

    python summary.py pic/numbers_label

比较预测结果与分割图片的标签，计算准确率，错误识别的样本

更详细信息可以参考 ocr.log 文件

注：测试 pic 内其他图片之前，要将之前分割生成的图片文件夹 test 删除 (rm -rf test)


验证 0-299

    python segment.py pic/0-209.png
    python ocr.py --mode inference
    python summary.py pic/0-299_label

验证 pi.png

    python segment.py pic/pi.png
    python ocr.py --mode inference
    python summary.py pic/pi_label

验证 pi_01.png

    python segment.py pic/pi_01.png
    python ocr.py --mode inference
    python summary.py pic/pi_01_label

验证 glod_ratio.png

    python segment.py pic/glod_ratio.png
    python ocr.py --mode inference
    python summary.py pic/glod_ratio_label

## 结果

经过 500 次 mini-batch 训练之后的结果：

#### pic/0-299.png 识别结果

accuracy is  1.0
difference set is  set()

#### pic/number.png 识别结果：

    index 3 inference is 7 but label is 1
    index 38 inference is 7 but label is 1
    index 48 inference is 7 but label is 1
    index 75 inference is 7 but label is 1
    index 98 inference is 7 but label is 1
    index 106 inference is 7 but label is 1
    index 120 inference is 7 but label is 1
    accuracy is  0.9435483870967742
    difference set is  {('7', '1')}

#### glod_ratio.png 识别结果

    index 1 inference is 1 but label is .
    accuracy is  0.9995251661918328
    difference set is  {('1', '.')}


#### pic/pi.png 识别结果

    index 1 inference is 1 but label is .
    accuracy is  0.9995249406175772
    difference set is  {('1', '.')}

#### pic/pi_01.png 识别结果

    index 1 inference is 1 but label is .
    index 2 inference is 7 but label is 1
    index 4 inference is 7 but label is 1
    index 38 inference is 7 but label is 1
    index 41 inference is 7 but label is 1
    index 50 inference is 7 but label is 1
    index 69 inference is 7 but label is 1
    index 95 inference is 7 but label is 1
    index 96 inference is 7 but label is 1
    index 104 inference is 7 but label is 1
    index 139 inference is 7 but label is 1
    index 149 inference is 7 but label is 1
    index 176 inference is 7 but label is 1
    index 199 inference is 7 but label is 1
    index 207 inference is 7 but label is 1
    index 221 inference is 7 but label is 1
    accuracy is  0.9292035398230089
    difference set is  {('1', '.'), ('7', '1')}

经过查看日志文件，发现如果是 top2 的话，应该是 99% 以上。


## TODO

1. 将生成数据集生成 tfrecord
2. 将 net 部分提出来
3. 用 pipeline 的方式处理数据
4. tensorboard 增加更多监控指标
5. 修改为类似 models/slim 的架构
6. 增加透视矫正和水平矫正

注意:

1. 只要对 label_file 修改并重复上述流程，就可以识别字母和汉字。
2. 目前基于分割的方法有其局限性，最新的 OCR 方法已经完全端到端实现，请参考相关论文



