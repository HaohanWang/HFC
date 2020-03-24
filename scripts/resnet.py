from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json
import argparse
import numpy as np
import pickle
import tensorflow as tf

version = sys.version_info

sys.path.append('../')

from utility.pgd_attack import LinfPGDAttack

from utility.dataLoader import loadDataCifar10, loadDataCifar10AdvFast
from utility.attackHelper import attackAModelCIFAR
from utility.frequencyHelper import generateSmoothKernel


def getSaveName(args):

    # todo: generate a name to save the model according to the args

    saveName = ''
    return saveName


class CIFAR10Data(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).
    Inputs to constructor
    =====================
        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.
    """

    def __init__(self, path):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(
                os.path.join(path, fname))
            train_images[ii * 10000: (ii + 1) * 10000, ...] = cur_images
            train_labels[ii * 10000: (ii + 1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        with open(os.path.join(path, metadata_filename), 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape(
                (10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])


class AugmentedCIFAR10Data(object):
    """
    Data augmentation wrapper over a loaded dataset.
    Inputs to constructor
    =====================
        - raw_cifar10data: the loaded CIFAR10 dataset, via the CIFAR10Data class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """

    def __init__(self, raw_cifar10data, sess, model):
        assert isinstance(raw_cifar10data, CIFAR10Data)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(
            tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
                           self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(
            lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_cifar10data.train_data, sess,
                                              self.x_input_placeholder,
                                              self.augmented)
        self.eval_data = AugmentedDataSubset(raw_cifar10data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.label_names = raw_cifar10data.label_names


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                       reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder: raw_batch[0]}), raw_batch[1]


def oneHotRepresentation(y, num=10):
    r = []
    for i in range(y.shape[0]):
        l = np.zeros(num)
        l[y[i]] = 1
        r.append(l)
    return np.array(r)


def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=5e-2)
    return tf.get_variable("weights", shape, initializer=initializer, dtype=tf.float32)


def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


BN_EPSILON = 0.001


def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer)  # , regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''

    # TODO: use the following seven lines to control the usage of batch normalization
    # mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    # beta = tf.get_variable('beta', dimension, tf.float32,
    #                        initializer=tf.constant_initializer(0.0, tf.float32))
    # gamma = tf.get_variable('gamma', dimension, tf.float32,
    #                         initializer=tf.constant_initializer(1.0, tf.float32))
    # bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    # return bn_layer

    return input_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


class ResNet(object):
    def __init__(self, x, y, conf):
        self.x = tf.reshape(x, shape=[-1, 32, 32, 3])
        self.y = y
        self.keep_prob = 1
        self.top_k = 5
        self.NUM_CLASSES = 10
        self.batch_size = conf.batch_size
        self.conf = conf
        self.conv_loss = self.evaluate(self.x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.conv_loss))

        self.softmax = tf.nn.softmax(self.conv_loss)
        self.pred = tf.argmax(self.conv_loss, 1)

        self.correct_prediction = tf.equal(tf.argmax(self.conv_loss, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def evaluate(self, x):
        n = 5
        with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):

            # layers = []
            with tf.variable_scope('conv0'):
                conv0 = conv_bn_relu_layer(x, [3, 3, 3, 16], 1)

            for i in range(n):
                with tf.variable_scope('conv1_%d' % i):
                    if i == 0:
                        conv1 = residual_block(conv0, 16, first_block=True)
                    else:
                        conv1 = residual_block(conv1, 16)
            conv2 = conv1
            for i in range(n):
                with tf.variable_scope('conv2_%d' % i):
                    conv2 = residual_block(conv2, 32)
            conv3 = conv2
            for i in range(n):
                with tf.variable_scope('conv3_%d' % i):
                    conv3 = residual_block(conv3, 64)
                assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

            with tf.variable_scope('fc'):
                # in_channel = conv3.get_shape().as_list()[-1]
                # bn_layer = batch_normalization_layer(layers[-1], in_channel)
                # conv3_drop = tf.nn.dropout(conv3, 0.5)

                relu_layer = tf.nn.relu(conv3)
                global_pool = tf.reduce_mean(relu_layer, [1, 2])

                assert global_pool.get_shape().as_list()[-1:] == [64]
                output = output_layer(global_pool, 10)
        return output

    def loadWeights(self, session, args):
        saveName = getSaveName(args)
        weights_dict = np.load('weights/' + saveName + '.npy', encoding='bytes',
                               allow_pickle=True).item()
        # Loop over all layer names stored in the weights dict
        for v in tf.trainable_variables():
            data = weights_dict[v.name]

            if args.rho > 0:
                if v.name.find('conv') != -1:
                    if v.name.find('conv0')!=-1:
                        data = generateSmoothKernel(data, args.rho)

            session.run(v.assign(data))


def train(args):
    num_classes = 10
    data_path = '../data/CIFAR10/'
    raw_cifar = CIFAR10Data(data_path)

    train_batches_per_epoch = 50000 // args.batch_size

    val_batches_per_epoch = 10000 // 100

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_classes))

    model = ResNet(x, y, args)

    optimizer1 = tf.train.AdamOptimizer(1e-4)
    # optimizer1 = tf.train.GradientDescentOptimizer(1e-2)
    # optimizer1 = tf.train.AdadeltaOptimizer(1e-1)
    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
    first_train_op = optimizer1.minimize(model.loss, var_list=first_train_vars)

    radius = [4, 8, 12, 16]

    trainData = np.load('../data/CIFAR10/train_images.npy')
    if args.frequency != 0:
        if args.high == 1:
            trainData = np.load('../data/CIFAR10/train_data_high_' + str(args.frequency) + '.npy')
        else:
            trainData = np.load('../data/CIFAR10/train_data_low_' + str(args.frequency) + '.npy')

    for i in range(trainData.shape[0]):
        trainData[i] = trainData[i] / np.max(trainData[i]) * 255

    trainDatas = []
    for r in radius:
        trainDatas.append(np.load('../data/CIFAR10/train_data_low_' + str(r) + '.npy'))
        trainDatas.append(np.load('../data/CIFAR10/train_data_high_' + str(r) + '.npy'))

    for td in trainDatas:
        for i in range(td.shape[0]):
            td[i] = td[i] / np.max(td[i]) * 255

    trainLabel = np.load('../data/CIFAR10/train_label.npy')

    if args.shuffle:
        # it's much better to shuffle data once then load it when needed than shuffling every time for replication purpose
        trainLabel = np.load('../data/CIFAR10/train_label_shuffle.npy')

    testDatas = []
    for r in radius:
        testDatas.append(np.load('../data/CIFAR10/test_data_low_' + str(r) + '.npy'))
        testDatas.append(np.load('../data/CIFAR10/test_data_high_' + str(r) + '.npy'))
    testLabel = np.load('../data/CIFAR10/test_label.npy')

    for td in testDatas:
        for i in range(td.shape[0]):
            td[i] = td[i] / np.max(td[i]) * 255

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Starting training')
        best_validate_accuracy = 0
        validation = True
        score = 0
        train_acc = []
        val_acc = []
        for epoch in range(args.epochs):

            begin = time.time()

            train_accuracies = []
            train_losses = []
            adv_losses = []
            adv_loss = 0

            idx = np.array(range(50000))
            np.random.shuffle(idx)
            trainData = trainData[idx, :]
            trainLabel = trainLabel[idx]
            for i in range(len(trainDatas)):
                trainDatas[i] = trainDatas[i][idx, :]

            for i in range(train_batches_per_epoch):
                # batch_x, batch_y = cifar.train_data.get_next_batch(args.batch_size, multiple_passes=True)
                batch_x = trainData[i * args.batch_size:(i + 1) * args.batch_size, :]
                batch_y = trainLabel[i * args.batch_size:(i + 1) * args.batch_size]
                batch_y = oneHotRepresentation(batch_y)
                batch_x = batch_x / 255.0


                # todo: use the following code to control usage of mixup
                # idx = np.array(range(args.batch_size))
                # np.random.shuffle(idx)
                # b = np.random.beta(1, 1)
                # batch_x = batch_x*b + batch_x[idx,:]*(1-b)
                # batch_y = batch_y*b + batch_y[idx,:]*(1-b)

                _, acc, loss = sess.run([first_train_op, model.accuracy, model.loss],
                                        feed_dict={x: batch_x, y: batch_y})

                train_accuracies.append(acc)
                train_losses.append(loss)
                adv_losses.append(adv_loss)

            train_acc_mean = np.mean(train_accuracies)
            train_loss_mean = np.mean(train_losses)
            adv_loss_mean = np.mean(adv_losses)


            if validation:

                val_accuracies = []
                for i in range(val_batches_per_epoch):
                    batch_x, batch_y = raw_cifar.eval_data.get_next_batch(100, multiple_passes=True)
                    batch_y = oneHotRepresentation(batch_y)
                    batch_x = batch_x / 255.0

                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)
                val_acc.append(val_acc_mean)
                # log progress to console
                print(
                    "Epoch %d, time = %ds, train accuracy = %.4f, loss = %.4f, adv accuracy = %.4f,  validation accuracy = %.4f" % (
                        epoch, time.time() - begin, train_acc_mean, train_loss_mean, adv_loss_mean, val_acc_mean))

                for r in range(len(radius)):
                    for j in range(2):
                        train_accuracies = []
                        for i in range(500):
                            batch_x = trainDatas[r * 2 + j][i * 100:(i + 1) * 100, :]
                            batch_y = trainLabel[i * 100:(i + 1) * 100]
                            batch_y = oneHotRepresentation(batch_y)
                            batch_x = batch_x / 255.0

                            acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                            train_accuracies.append(acc)
                        train_acc_mean = np.mean(train_accuracies)
                        train_acc.append(train_acc_mean)
                        # log progress to console
                        if j == 0:
                            print("Radius = %d, low end, train accuracy = %.4f" % (radius[r],
                                                                                   train_acc_mean))
                        else:
                            print("Radius = %d, high end, train accuracy = %.4f" % (radius[r],
                                                                                    train_acc_mean))

                for r in range(len(radius)):
                    for j in range(2):
                        val_accuracies = []
                        for i in range(val_batches_per_epoch):
                            batch_x = testDatas[r * 2 + j][i * 100:(i + 1) * 100, :]
                            batch_y = testLabel[i * 100:(i + 1) * 100]
                            batch_y = oneHotRepresentation(batch_y)
                            batch_x = batch_x / 255.0

                            acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                            val_accuracies.append(acc)
                        val_acc_mean = np.mean(val_accuracies)
                        val_acc.append(val_acc_mean)
                        # log progress to console
                        if j == 0:
                            print("Radius = %d, low end, validation accuracy = %.4f" % (radius[r],
                                                                                        val_acc_mean))
                        else:
                            print("Radius = %d, high end, validation accuracy = %.4f" % (radius[r],
                                                                                         val_acc_mean))

            sys.stdout.flush()

        if args.saveModel == 1:
            weights = {}
            for v in tf.trainable_variables():
                weights[v.name] = v.eval()

            np.save('weights/resnet_' + getSaveName(args), weights)



def trainMadry(args):
    # """ reuse """
    # with tf.variable_scope('model',reuse=tf.AUTO_REUSE ) as scope:
    num_classes = 10
    data_path = '../data/CIFAR10/'
    raw_cifar = CIFAR10Data(data_path)

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_classes))

    model = ResNet(x, y, args)
    # For sanping's server
    train_batches_per_epoch = 50000 // args.batch_size

    val_batches_per_epoch = 10000 // 100
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnn")
    first_train_op = optimizer.minimize(model.loss, var_list=train_vars)

    radius = [4, 8, 12, 16, 20, 24, 28]
    testDatas = []
    for r in radius:
        testDatas.append(np.load('../data/CIFAR10/test_data_low_' + str(r) + '.npy'))
        testDatas.append(np.load('../data/CIFAR10/test_data_high_' + str(r) + '.npy'))
    testLabel = np.load('../data/CIFAR10/test_label.npy')

    attack = LinfPGDAttack(model, args.epsilon, 10, args.epsilon / 10, False, 'loss')


    with tf.Session() as sess:

        print('Starting training')
        sess.run(tf.initialize_all_variables())
        cifar = AugmentedCIFAR10Data(raw_cifar, sess, model)

        validation = True

        train_acc = []
        val_acc = []

        for epoch in range(args.epochs):

            begin = time.time()

            train_accuracies = []
            train_losses = []
            adv_losses = []
            adv_loss = 0
            acc = 0
            adv_acc = 0
            for i in range(train_batches_per_epoch):
                batch_x, batch_y = cifar.train_data.get_next_batch(args.batch_size, multiple_passes=True)
                batch_y = oneHotRepresentation(batch_y)
                batch_x = batch_x / 255.0

                batch_x_adv = attack.perturb(batch_x, batch_y, sess)

                _, adv_acc, loss = sess.run([first_train_op, model.accuracy, model.loss],
                                            feed_dict={x: batch_x_adv, y: batch_y})

                train_accuracies.append(acc)
                train_losses.append(loss)
                adv_losses.append(adv_acc)
            train_acc_mean = np.mean(train_accuracies)
            train_acc.append(train_acc_mean)

            train_loss_mean = np.mean(train_losses)
            adv_loss_mean = np.mean(adv_losses)

            # print ()
            # compute loss over validation data
            if validation:
                val_accuracies = []
                for i in range(val_batches_per_epoch):
                    batch_x, batch_y = raw_cifar.eval_data.get_next_batch(100, multiple_passes=True)
                    batch_y = oneHotRepresentation(batch_y)
                    batch_x = batch_x / 255.0

                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)
                val_acc.append(val_acc_mean)
                # log progress to console
                print(
                    "Epoch %d, time = %ds, train accuracy = %.4f, loss = %.4f, adv accuracy = %.4f,  validation accuracy = %.4f" % (
                        epoch, time.time() - begin, train_acc_mean, train_loss_mean, adv_loss_mean, val_acc_mean))

                for r in range(len(radius)):
                    for j in range(2):
                        val_accuracies = []
                        for i in range(val_batches_per_epoch):
                            batch_x = testDatas[r * 2 + j][i * 100:(i + 1) * 100, :]
                            batch_y = testLabel[i * 100:(i + 1) * 100]
                            batch_y = oneHotRepresentation(batch_y)
                            batch_x = batch_x / 255.0

                            acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                            val_accuracies.append(acc)
                        val_acc_mean = np.mean(val_accuracies)
                        val_acc.append(val_acc_mean)
                        # log progress to console
                        if j == 0:
                            print("Radius = %d, low end, validation accuracy = %.4f" % (radius[r],
                                                                                        val_acc_mean))
                        else:
                            print("Radius = %d, high end, validation accuracy = %.4f" % (radius[r],
                                                                                         val_acc_mean))

            sys.stdout.flush()

        if args.saveModel == 1:
            weights = {}
            for v in tf.trainable_variables():
                weights[v.name] = v.eval()

            np.save('weights/resnet_' + getSaveName(args), weights)


def attackModel(args):
    Xtest, Ytest = loadDataCifar10()

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, 10))

    model = ResNet(x, y, args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadWeights(sess, args)

        Xadv_fgsm, Xadv_pgd = attackAModelCIFAR(x, model.conv_loss, 1.0, Xtest, Ytest)

        saveName = getSaveName(args)

        np.save('advs/fgsm' + saveName, Xadv_fgsm)
        np.save('advs/pgd' + saveName, Xadv_pgd)


def predictAdvDataModelComparisions(args):
    def distance3(a, b):
        a = a.reshape((-1, 1))
        b = b.reshape((-1, 1))
        return np.sum(a != b) / a.shape[0], np.linalg.norm(a - b, ord=2), np.linalg.norm(a - b, ord=np.inf)

    Xtest, Ytest = loadDataCifar10()
    print("Xtest", Xtest.shape[0])

    saveName = getSaveName(args)
    Xadv_fgsm, Xadv_pgd = loadDataCifar10AdvFast(saveName)
    distances = np.zeros([Xtest.shape[0], 3, 3]) # number_of_samples * (attack method + original test) * distances
    correct_predictions = np.zeros([Xtest.shape[0], 3])

    num_classes = 10

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.float32, (None, num_classes))

    model = ResNet(x, y, args)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        model.loadWeights(sess, args)

        test_num_batches = Xtest.shape[0] // 100

        sys.stdout.flush()

        for i in range(test_num_batches):
            batch_x = Xtest[i * args.batch_size:(i + 1) * args.batch_size, :]
            batch_y = Ytest[i * args.batch_size:(i + 1) * args.batch_size, :]
            cp = sess.run(model.correct_prediction, feed_dict={x: batch_x, y: batch_y})
            correct_predictions[i * args.batch_size:(i + 1) * args.batch_size, 0] = cp

        for i in range(test_num_batches):
            batch_x = Xadv_fgsm[i * args.batch_size:(i + 1) * args.batch_size, :]
            batch_y = Ytest[i * args.batch_size:(i + 1) * args.batch_size, :]
            cp = sess.run(model.correct_prediction, feed_dict={x: batch_x, y: batch_y})
            correct_predictions[i * args.batch_size:(i + 1) * args.batch_size, 1] = cp

        for i in range(test_num_batches):
            batch_x = Xadv_pgd[i * args.batch_size:(i + 1) * args.batch_size, :]
            batch_y = Ytest[i * args.batch_size:(i + 1) * args.batch_size, :]
            cp = sess.run(model.correct_prediction, feed_dict={x: batch_x, y: batch_y})
            correct_predictions[i * args.batch_size:(i + 1) * args.batch_size, 2] = cp

    np.save('results/correct_predictions' + saveName, correct_predictions)

    for i in range(Xtest.shape[0]):
        distances[i, 1, :] = distance3(Xadv_fgsm[i], Xtest[i])
        distances[i, 2, :] = distance3(Xadv_pgd[i], Xtest[i])

    np.save('results/distance' + saveName, distances)


def main(args):
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))

    if args.madry == 0:
        train(args)
    else:
        trainMadry(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=100, help='Batch size during training per GPU')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('-s', '--saveModel', type=int, default=1, help='Whether we save this model')
    parser.add_argument('-m', '--madry', type=int, default=0, help='Whether Use Madry (adversarial training) or not')
    parser.add_argument('-l', '--epsilon', type=float, default=0.03, help='epsilon of PGD during training')
    parser.add_argument('-a', '--action', type=int, default=0, help='action to take')
    parser.add_argument('-f', '--shuffle', type=int, default=0, help='whether we shuffle the data')
    parser.add_argument('-q', '--frequency', type=int, default=0,
                        help='frequency of the training data, 0 for original data')
    parser.add_argument('-i', '--high', type=int, default=0, help='high or low frequency, only useful is q is not 0')
    parser.add_argument('-r', '--rho', type=float, default=0, help='the rho of smoothing convolutional kernel, rho=0 for not using the method')

    args = parser.parse_args()

    tf.set_random_seed(100)
    np.random.seed()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.action == 0:
        main(args)
    if args.action == 1:
        attackModel(args)
    if args.action == 2:
        predictAdvDataModelComparisions(args)
