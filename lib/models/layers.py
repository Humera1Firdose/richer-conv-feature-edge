#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tensorcv.models.layers import *


@add_arg_scope
def transpose_conv(x,
                   filter_size,
                   out_dim,
                   out_shape=None,
                   use_bias=True,
                   stride=2,
                   padding='SAME',
                   trainable=True,
                   nl=tf.identity,
                   name='dconv'):

    stride = get_shape4D(stride)

    in_dim = x.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    x_shape = tf.shape(x)
    # assume output shape is input_shape*stride
    if out_shape is None:
        out_shape = tf.stack([x_shape[0],
                              tf.multiply(x_shape[1], stride[1]), 
                              tf.multiply(x_shape[2], stride[2]),
                              out_dim])        

    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]

    with tf.variable_scope(name) as scope:
        init_w = tf.keras.initializers.he_normal()
        init_b = tf.zeros_initializer()
        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)
        

        output = tf.nn.conv2d_transpose(x,
                                        weights, 
                                        output_shape=out_shape, 
                                        strides=stride, 
                                        padding=padding, 
                                        name=scope.name)

        output = tf.nn.bias_add(output, biases)
        output.set_shape([None, None, None, out_dim])
        output = nl(output, name='output')
        return output

def upsample_deconv(x,
                    filter_size,
                    up_level,
                    out_dim,
                    trainable=True,
                    nl=tf.identity,
                    name='deconv_upsample'):

    with tf.variable_scope(name):
        cur_input = x
        for level_id in range(0, up_level):
            cur_input = transpose_conv(
                cur_input,
                filter_size=filter_size,
                out_dim=out_dim,
                stride=2,
                padding='SAME',
                trainable=trainable,
                nl=nl,
                name='dconv_{}'.format(level_id))
        return cur_input

# https://github.com/tensorflow/tensorflow/pull/16885
def unpool_2d(pool, 
              ind, 
              stride=[1, 2, 2, 1], 
              scope='unpool_2d'):
  """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """

  with tf.variable_scope(scope):
    ind_shape = tf.shape(ind)
    # pool = pool[:, :ind_shape[1], :ind_shape[2], :]

    input_shape = tf.shape(pool)
    output_shape = [input_shape[0],
                    input_shape[1] * stride[1],
                    input_shape[2] * stride[2],
                    input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0],
                         output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(
        tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
        shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0],
                        set_input_shape[1] * stride[1],
                        set_input_shape[2] * stride[2],
                        set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret


def max_pool(x,
             name='max_pool',
             filter_size=2,
             stride=None,
             padding='VALID',
             switch=False):
    """ 
    Max pooling layer 
    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 2): size of filter
        stride (int or list with length 2): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.
    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)

    if switch == True:
        return tf.nn.max_pool_with_argmax(
            x,
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            Targmax=tf.int64,
            name=name)
    else:
        return tf.nn.max_pool(
            x,
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            name=name), None



