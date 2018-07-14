#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

# from tensorcv.models.layers import *

def get_shape4D(in_val):
    """
    Return a 4D shape
    Args:
        in_val (int or list with length 2)
    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape2D(in_val) + [1]

def get_shape2D(in_val):
    """
    Return a 2D shape 
    Args:
        in_val (int or list with length 2) 
    Returns:
        list with length 2
    """
    in_val = int(in_val)
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))

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
                   init_w = None,
                   init_b = tf.zeros_initializer(),
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
                    # stride=2,
                    trainable=True,
                    nl=tf.identity,
                    init_w=None,
                    init_b=tf.zeros_initializer(),
                    name='deconv_upsample'):
    
    # if stride == 4:
    #     up_level = int(up_level / 2)
    with tf.variable_scope(name):
        cur_input = x
        factor = 2 ** up_level
        
        filter_size = 2 * factor - factor % 2
        print(filter_size)
        cur_input = transpose_conv(
                cur_input,
                filter_size=filter_size,
                out_dim=out_dim,
                stride=factor,
                padding='SAME',
                trainable=trainable,
                init_w=init_w,
                init_b=init_b,
                # nl=tf.nn.relu,
                name='dconv')
        # for level_id in range(0, up_level):
        #     cur_input = transpose_conv(
        #         cur_input,
        #         filter_size=filter_size,
        #         out_dim=out_dim,
        #         stride=2,
        #         padding='SAME',
        #         trainable=trainable,
        #         init_w=init_w,
        #         init_b=init_b,
        #         # nl=tf.nn.relu,
        #         name='dconv_{}'.format(level_id))
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

@add_arg_scope
def conv(filter_size,
         out_dim,
         layer_dict,
         inputs=None,
         pretrained_dict=None,
         stride=1,
         dilations=[1, 1, 1, 1],
         bn=False,
         nl=tf.identity,
         init_w=None,
         init_b=tf.zeros_initializer(),
         padding='SAME',
         pad_type='ZERO',
         trainable=True,
         is_training=None,
         wd=0,
         name='conv'):
    if inputs is None:
        inputs = layer_dict['cur_input']
    stride = get_shape4D(stride)
    in_dim = inputs.get_shape().as_list()[-1]
    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]

    if padding == 'SAME' and pad_type == 'REFLECT':
        pad_size_1 = int((filter_shape[0] - 1) / 2)
        pad_size_2 = int((filter_shape[1] - 1) / 2)
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
            "REFLECT")
        padding = 'VALID'

    with tf.variable_scope(name):
        if wd > 0:
            regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
        else:
            regularizer=None

        if pretrained_dict is not None and name in pretrained_dict:
            try:
                load_w = pretrained_dict[name][0]
                load_b = pretrained_dict[name][1]
            except KeyError:
                load_w = pretrained_dict[name]['weights']
                load_b = pretrained_dict[name]['biases']
            print('Load {} weights!'.format(name))
            print('Load {} biases!'.format(name))

            load_w = np.reshape(load_w, filter_shape)
            init_w = tf.constant_initializer(load_w)

            load_b = np.reshape(load_b, [out_dim])
            init_b = tf.constant_initializer(load_b)

        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable,
                                  regularizer=regularizer)
        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)

        outputs = tf.nn.conv2d(inputs,
                               filter=weights,
                               strides=stride,
                               padding=padding,
                               use_cudnn_on_gpu=True,
                               data_format="NHWC",
                               dilations=dilations,
                               name='conv2d')
        outputs += biases

        # if bn is True:
        #     outputs = layers.batch_norm(outputs, train=is_training, name='bn')

        layer_dict['cur_input'] = nl(outputs)
        layer_dict[name] = layer_dict['cur_input']
        return layer_dict['cur_input']



