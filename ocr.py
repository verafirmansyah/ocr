#! /usr/bin/env python
# -*- coding: utf-8 -*-

# top 1 accuracy 0.99826 top 5 accuracy 0.99989
from __future__ import print_function

import os
# 这里可以根据系统读取或手动配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import cv2
from tensorflow.python.ops import control_flow_ops
import sys
from segment import ImageUtil
from export_model import inference as infer

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(lineno)s : %(message)s')
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(lineno)s : %(message)s')

handler = logging.handlers.RotatingFileHandler("ocr.log", maxBytes = 10*1020*1024, backupCount = 3)
log.setLevel(logging.INFO)
handler.setFormatter(formatter)
log.addHandler(handler)

# 输入参数解析
tf.app.flags.DEFINE_integer('charset_size', 10, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_integer('max_steps', 500, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 20, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 100, "the steps to save")
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('debug', False, 'debug the input pipeline')
tf.app.flags.DEFINE_integer('train_epoch', 10, 'Number of epoches to train')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'test', 'Running mode. One of {"train", "test", "inference"}')

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './dataset/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './dataset/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('inference_data_dir', './test/', 'the inference dataset dir')
tf.app.flags.DEFINE_string('inference_result', './inference_result', 'the inference dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('label_file', './digital_label', 'the characters you want to train')


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        self.image_names = []
        assert(len(os.listdir(data_dir)) == FLAGS.charset_size)
        for d in os.listdir(data_dir):
            sub_fold = os.path.join(data_dir, d)
            _ = int(d)
            self.image_names += [os.path.join(sub_fold, file_path) for file_path in os.listdir(sub_fold)]
        random.shuffle(self.image_names)
        self.labels = [int(file_name.split(os.sep)[-2]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        # 镜像变换
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        # 图像亮度变化
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        # 对比度变化
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    #  创建两个队列，一个队列从数据中读取数据 num_epochs，一个队列从前一个队列每次读取 batch_size
    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        """ 每次读取 batch_size 张图片，总共读取 num_epochs 个 epoch
        Args:
          batch_size : 每次读取的图片数量
          num_epochs : 总共读取 num_epochs 个 epoch
        Return:
          image_batch : batch_size 张图片，每张图片 [x, y, z]
          label_batch : batch_size 个标签，每个标签 [label]
        """
        # list 转 tensor
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        # 读 images_tensor, labels_tensor 共 num_epochs 个 epoch，之后抛出 OutOfRangeError，返回一个队列
        # 此处会打乱顺序
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        # 转为 float32 类型，这步必不可少
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
            #images = tf.subtract(images, 128.0)
            #images = tf.div(images, 128.0)
        images = tf.image.per_image_standardization(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        # 确保输入大小一致，缩放和放大应该分别处理，TODO
        images = tf.image.resize_images(images, new_size)
        # 没有数据可读时，会抛出 OutOfRangeError
        image_batch, label_batch, image_filename_batch = tf.train.shuffle_batch([images, labels, input_queue[0]],
                                                num_threads=2,
                                                batch_size=batch_size,
                                                capacity=10*batch_size,
                                                min_after_dequeue=2*batch_size)
        # print 'image_batch', image_batch.get_shape()
        return image_batch, label_batch, image_filename_batch

def get_optimizer(optimizer_method, learning_rate, **kwargs):
  """Configures the optimizer used for training.

  Args: learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if optimizer is not recognized.
  """
  if optimizer_method == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho = kwargs["adadelta_rho"] if "adadelta_rho" in kwargs else 0.95,
        epsilon = kwargs["opt_epsilon"] if "opt_epsilon" in kwargs else 1e-08)
  elif optimizer_method == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value =
          kwargs["adagrad_initial_accumulator_value"] if "adagrad_initial_accumulator_value" in kwargs else 0.1)
  elif optimizer_method == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1 = kwargs["adam_beta1"] if "adam_beta1" in kwargs else 0.9,
        beta2 = kwargs["adam_beta2"] if "adam_beta2" in kwargs else 0.999,
        epsilon = kwargs["opt_epsilon"] if "opt_epsilon" in kwargs else 1e-08)
  elif optimizer_method == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power= kwargs["ftrl_learning_rate_power"] if "ftrl_learning_rate_power" in kwargs else -0.5,
        initial_accumulator_value = kwargs["ftrl_initial_accumulator_value"] if "ftrl_initial_accumulator_value" in kwargs else 0.1,
        l1_regularization_strength= kwargs["ftrl_l1"] if "ftrl_l1" in kwargs else 0.0,
        l2_regularization_strength= kwargs["ftrl_l2"] if "ftrl_l2" in kwargs else 0.0)
  elif optimizer_method == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum = kwargs["momentum"] if "momentum" in kwargs else 0.9,
        use_nesterov = kwargs["use_nesterov"] if "use_nesterov" in kwargs else False,
        name='Momentum')
  elif optimizer_method == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay = kwargs["rmsprop_decay"] if "rmsprop_decay" in kwargs else 0.9,
        momentum = kwargs["rmsprop_momentum"] if "rmsprop_momentum" in kwargs else 0.9,
        epsilon= kwargs["opt_epsilon"] if "opt_epsilon" in kwargs else 1e-10)
  elif optimizer_method == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer_method)
  return optimizer


def build_graph():
    """
    from lenet
    """
    top_k = tf.placeholder(dtype=tf.int32, shape=[], name="top_k")
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob') # dropout打开概率
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    with tf.device('/gpu:0'):
        #给slim.conv2d和slim.fully_connected准备了默认参数：batch_norm
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            # 为什么增加了 weights_initializer 导致梯度爆炸
                            #weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            activation_fn=tf.nn.relu,
                            #normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            conv3_1 = slim.conv2d(images, 32, [5, 5], 1, padding='SAME', scope='conv3_1')
            # 32 x 32 x 32
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='VALID', scope='pool1')
            conv3_2 = slim.conv2d(max_pool_1, 64, [5, 5], padding='SAME', scope='conv3_2')
            # 16 x 16 x 64
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='VALID', scope='pool2')

            # 1 x 1 x 16384
            flatten = slim.flatten(max_pool_2)
            fc1 = slim.fully_connected(flatten, 1024, scope='fc1')
            dropout = slim.dropout(fc1, keep_prob, is_training = is_training)
            logits = slim.fully_connected(dropout, FLAGS.charset_size, activation_fn=None, scope='fc2')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

        # TODO
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        optimizer = get_optimizer("momentum", 0.01)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
        probabilities = tf.nn.softmax(logits, name="probabilities")

        # 绘制 loss accuracy 曲线
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        # top K precdiction, accuracy
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k, name="top_k_predict")
        predicted_val_top_k = tf.identity(predicted_val_top_k, name="predict_val")
        predicted_index_top_k = tf.identity(predicted_index_top_k, name="predict_index")
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, tf.cast(top_k, tf.int64)), tf.float32))

    return {'images': images,
            'labels': labels,
            'top_k' : top_k,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def train():
    log.info('Begin training')
    # 初始化训练的全部文件名和标签
    train_feeder = DataIterator(data_dir=FLAGS.train_data_dir)
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)
    log.info("total train example {0}".format(len(train_feeder.labels)))
    log.info("total test example {0}".format(len(test_feeder.labels)))
    model_name = 'digital_ocr'
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        #  从训练和测试集中的队列中批量读取
        train_images, train_labels, train_files = train_feeder.input_pipeline(FLAGS.batch_size, FLAGS.train_epoch, True)
        test_images, test_labels, test_files = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph()
        # 这里会保存所有可以保存的变量
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state
        # 设置多线程协调器
        coord = tf.train.Coordinator()
        # 开启线程从队列开始读数据，如果没有该函数，直接读取队列数据将阻塞。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        # 可以从上一次的最后一个 step 恢复模型继续训练
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        log.info(':::Training Start:::')
        try:
            i = 0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                # 从队列中读取一个批次的数据
                train_images_batch, train_labels_batch, train_files_batch = sess.run([train_images, train_labels, train_files])
                if FLAGS.debug:
                    for m, img in enumerate(train_images_batch):
                        ImageUtil.show(img, "test" + str(m))
                        print("index " + str(m), str(train_labels_batch[m]),
                                str(train_files_batch[m]).split(os.sep)[-2],
                                str(train_labels_batch[m]) == str(train_files_batch[m]).split(os.sep)[-2])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['top_k']: 1,
                             graph['keep_prob']: 0.8,
                             graph['is_training']: True}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                log.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
                    train_writer.close()
                    test_writer.close()
                    coord.request_stop()
                    break
                if step % FLAGS.eval_steps == 0:
                    test_images_batch, test_labels_batch, test_files_batch = sess.run([test_images, test_labels, test_files])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['top_k']: 1,
                                 graph['keep_prob']: 1.0,
                                 graph['is_training']: False}
                    accuracy_test, test_summary = sess.run([graph['accuracy'], graph['merged_summary_op']], feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    log.info('===============Eval a batch=======================')
                    log.info('the step {0} test accuracy: {1}'.format(step, accuracy_test))
                    log.info('total run {0} epoch'.format(step * FLAGS.batch_size / train_feeder.size))
                if step % FLAGS.save_steps == 0:
                    log.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),
                               global_step=graph['global_step'])
        except tf.errors.OutOfRangeError as e:
            log.error(e.message)
            log.info('================== Train Finished ================')
        except KeyboardInterrupt as e:
            log.info('================== Train Interrup ================')
        finally:
            # 达到最大训练迭代数的时候清理关闭线程
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
            train_writer.close()
            test_writer.close()
            coord.request_stop()
        coord.join(threads)


def validation():
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)
    log.info("total test example {0}".format(len(test_feeder.labels)))
    for l, m in zip(test_feeder.labels, test_feeder.image_names):
        assert(str(l) == str(m).split(os.sep)[-2])

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        test_images, test_labels, test_files = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        graph = build_graph()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        log.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                # Caution: you must run test_images, test_labels, test_files togegher
                test_images_batch, test_labels_batch, test_files_batch = sess.run([test_images, test_labels, test_files])
                if FLAGS.debug:
                    for m, img in enumerate(test_images_batch):
                        ImageUtil.show(img, "test" + str(m))
                        print("index " + str(m),
                                str(test_labels_batch[m]),
                                str(test_files_batch[m]).split(os.sep)[-2],
                                str(test_labels_batch[m]) == str(test_files_batch[m]).split(os.sep)[-2])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['top_k']: 5,
                             graph['keep_prob']: 1.0,
                             graph['is_training']: False}
                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict)
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                log.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                            .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError as e:
            log.error(e.message)
            log.info('================== Validation Finished ================')
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
            log.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        except KeyboardInterrupt as e:
            log.info('================== Validation Interrup ================')
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


def get_file_list(img_dir):
    """
    Args:
        path: only files in it and named by number, such as 111.png
    Return:
        the file path list sored by file number
    """
    files = os.listdir(img_dir)
    file_index_list = []
    ext = None
    for f in files:
        slic = f.split(".")
        if ext is not None:
            if ext != slic[1]:
                raise ValueError("file extention doesn't uniformity, origin {0}, now {1}".format(ext, slic[1]))
        ext = slic[1]
        file_name = slic[0]
        file_index_list.append(int(file_name))
    file_index_list.sort()
    list_name=[]
    for file_index in file_index_list:
        file_path = os.path.join(img_dir, str(file_index) + ".png")
        list_name.append(file_path)
    return list_name


def get_label_dict(label_file):
    """
    Args:
        label_file : file with character set，is formated as id:char
    Return:
        parsed dict with id:char
    """
    id_label = {}
    with open(label_file, 'r') as f:
        for line in f:
            item = line.split(":")
            id_label[item[0]] = item[1].strip("\n")
    return id_label


def inference(name_list):
    # image preprocess
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        log.info('========start inference============')
        graph = build_graph()
        saver = tf.train.Saver()
        # restore model from checkpoint_dir
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            print("restore from the checkpoint {0}".format(ckpt))
            saver.restore(sess, ckpt)
        val_list=[]
        idx_list=[]
        image_batch = []
        ele_num = len(name_list)
        index = 0
        for image in name_list:
            #print(image)
            left = ele_num - index
            percentage = int(float(index) / ele_num * 100)
            if percentage % 10 == 0:
                log.info("complete {0} percentage".format(percentage))
            index += 1
            images_content = tf.read_file(image)
            images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
            images = tf.image.per_image_standardization(images)
            new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
            images = tf.image.resize_images(images, new_size)
            image_batch.append(images)
            if len(image_batch) == FLAGS.batch_size or \
                   (left < FLAGS.batch_size and index == ele_num):
                #img = sess.run(tf.expand_dims(images, 0))
                images = sess.run(tf.convert_to_tensor(image_batch))
                predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                                  feed_dict={graph['images']: images,
                                                             graph['top_k']: 3,
                                                             graph['keep_prob']: 1.0,
                                                             graph['is_training']: False})
                for p in predict_val:
                    val_list.append(p)
                for p in predict_index:
                    idx_list.append(p)
                image_batch = []
    return val_list,idx_list


def predict(name_list):
    val_list=[]
    idx_list=[]
    image_batch = []
    ele_num = len(name_list)
    index = 0
    for image in name_list:
        #print(image)
        left = ele_num - index
        percentage = int(float(index) / ele_num * 100)
        if percentage % 10 == 0:
            log.info("complete {0} percentage".format(percentage))
        index += 1
        images_content = tf.read_file(image)
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        images = tf.image.per_image_standardization(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch.append(images)
        if len(image_batch) == FLAGS.batch_size or \
               (left < FLAGS.batch_size and index == ele_num):
            #img = sess.run(tf.expand_dims(images, 0))
            output = _predict(image_batch)
            predict_val = output[0]
            predict_index = output[1]
            for p in predict_val:
                val_list.append(p)
            for p in predict_index:
                idx_list.append(p)
            image_batch = []
    return val_list,idx_list


def _predict(image_batch):
    with tf.Session() as sess:
        images = tf.convert_to_tensor(image_batch)
        input_map = {
            "keep_prob" : tf.constant(1.0, tf.float32),
            "top_k" : tf.constant(3, tf.int32),
            "image_batch" : images,
            "is_training" : tf.constant(False, tf.bool),
        }
        pb_path = "ocr.pb"
        #output_elements = [ "top_k_predict:0", "top_k_predict:1", "predict_val:0", "predict_index:0", "probabilities:0"]
        output_elements = [ "top_k_predict:0", "top_k_predict:1"]
        return infer(pb_path, input_map, output_elements)


def main(_):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'test':
        validation()
    elif FLAGS.mode == 'inference':
        label_dict = get_label_dict(FLAGS.label_file)
        name_list = get_file_list(FLAGS.inference_data_dir)
        final_predict_val, final_predict_index = inference(name_list)
        # save top-1 prediction
        top_1 =[]
        # top 3 precdict
        for i in range(len(final_predict_val)):
            candidate1 = final_predict_index[i][0]
            candidate2 = final_predict_index[i][1]
            candidate3 = final_predict_index[i][2]
            top_1.append(label_dict[str(candidate1)])
            log.info('[index {0} ] image: {1} predict: {2} {3} {4}; predict index {5} predict_val {6}'.format(
                i, name_list[i], label_dict[str(candidate1)],label_dict[str(candidate2)],label_dict[str(candidate3)],
                final_predict_index[i],final_predict_val[i]))
        print(top_1)
        with open(FLAGS.inference_result, "w") as ff:
            ff.write(",".join(top_1))
    elif FLAGS.mode == 'predict':
        label_dict = get_label_dict(FLAGS.label_file)
        name_list = get_file_list(FLAGS.inference_data_dir)
        final_predict_val, final_predict_index = predict(name_list)
        # save top-1 prediction
        top_1 =[]
        # top 3 precdict
        for i in range(len(final_predict_val)):
            candidate1 = final_predict_index[i][0]
            candidate2 = final_predict_index[i][1]
            candidate3 = final_predict_index[i][2]
            top_1.append(label_dict[str(candidate1)])
            log.info('[index {0} ] image: {1} predict: {2} {3} {4}; predict index {5} predict_val {6}'.format(
                i, name_list[i], label_dict[str(candidate1)],label_dict[str(candidate2)],label_dict[str(candidate3)],
                final_predict_index[i],final_predict_val[i]))
        print(top_1)
        with open(FLAGS.inference_result, "w") as ff:
            ff.write(",".join(top_1))
    else:
        print("mode with one of train|test|inference|predict support")

if __name__ == "__main__":
    tf.app.run()
