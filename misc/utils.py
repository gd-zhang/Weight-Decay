from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time

import logging
import tensorflow as tf
import numpy as np
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config', default='None', help='The Configuration file')
    argparser.add_argument(
        '-f', '--fig_name', default='tmp', help='The Figure name')
    args = argparser.parse_args()
    return args


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def get_logger(name, logpath, filepath, package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, 'r') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        shapes = map(var_shape, var_list)
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


def flatten(tensors):
    if isinstance(tensors, (tuple, list)):
        return tf.concat(
            tuple(tf.reshape(tensor, [-1]) for tensor in tensors), axis=0)
    else:
        return tf.reshape(tensors, [-1])


class unflatten(object):
    def __init__(self, tensors_template):
        self.tensors_template = tensors_template

    def __call__(self, colvec):
        if isinstance(self.tensors_template, (tuple, list)):
            offset = 0
            tensors = []
            for tensor_template in self.tensors_template:
                sz = np.prod(tensor_template.shape.as_list(), dtype=np.int32)
                tensor = tf.reshape(colvec[offset:(offset + sz)],
                                           tensor_template.shape)
                tensors.append(tensor)
                offset += sz

            tensors = list(tensors)
        else:
            tensors = tf.reshape(colvec, self.tensors_template.shape)

        return tensors


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


def conv2d(inputs, kernel_size, out_channels, is_training, name,
           activation_fn=tf.nn.relu, padding="SAME", strides=(1, 1), use_bias=False,
           batch_norm=False, initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_avg',
                                                                         distribution='uniform')):
    layer = tf.layers.Conv2D(
        out_channels,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=initializer,
        bias_initializer=tf.constant_initializer(0.0),
        padding=padding,
        use_bias=use_bias,
        name=name)
    preactivations = layer(inputs)
    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training, trainable=False)
        activations = activation_fn(bn)
    else:
        activations = activation_fn(preactivations)
    if use_bias:
        return preactivations, activations, (layer.kernel, layer.bias)
    else:
        return preactivations, activations, layer.kernel


def dense(inputs, output_size, is_training, name, batch_norm=False,
          use_bias=False, activation_fn=tf.nn.relu,
          initializer=tf.variance_scaling_initializer(scale=1.0, mode='fan_avg',
                                                      distribution='uniform')):
    layer = tf.layers.Dense(
        output_size,
        kernel_initializer=initializer,
        bias_initializer=tf.constant_initializer(0.0),
        use_bias=use_bias,
        name=name)
    preactivations = layer(inputs)
    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training, trainable=False)
        activations = activation_fn(bn)
    else:
        activations = activation_fn(preactivations)
    if use_bias:
        return preactivations, activations, (layer.kernel, layer.bias)
    else:
        return preactivations, activations, layer.kernel
