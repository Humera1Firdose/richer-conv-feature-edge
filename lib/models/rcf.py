#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rcf.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorcv.models.layers import *

import lib.models.layers as L
from lib.models.vgg import BaseVGG16
from lib.models.loss import class_balanced_cross_entropy_with_logits


class RCF(BaseVGG16):
    def __init__(self, n_channel, vgg_path):
        self._switch = False

        self._n_channel = n_channel
        self._vgg_param = np.load(vgg_path, encoding='latin1').item()

        self._n_vgg_layer = 5
        self._n_vgg_sublayer = 3

    def create_model(self):
        self._create_input()
        self.vgg_layers = {}
        self.layers = {}

        self.image = self._sub_mean(self.raw_image)
        self._creat_vgg_conv(self.image, self.vgg_layers, data_dict=self._vgg_param)
        self._create_richer_feature()

    def get_train_op(self):
        with tf.name_scope('train'):
            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            loss = self.get_loss()
            grads = opt.compute_gradients(loss)
            return opt.apply_gradients(grads, name='train')

    def get_loss(self):
        try:
            return self.loss
        except AttributeError:
            self.loss = self._get_loss()
            return self.loss
    
    def _create_input(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.raw_image = tf.placeholder(tf.float32, name='image',
                            shape=[None, None, None, self._n_channel])
        self.label = tf.placeholder(tf.float32, [None, None, None], 'label')
        self.consensus_label = tf.cast(tf.greater(self.label, 0.2), tf.int32)

    def _create_richer_feature(self):
        def side_layer(layer_id):
            with tf.variable_scope('side_{}'.format(layer_id)):
                init_w = tf.keras.initializers.he_normal()
                init_b = tf.zeros_initializer() 

                down_list = []
                for sub_layer_id in range(1, self._n_vgg_sublayer + 1):
                    layer_name = 'conv{}_{}'.format(layer_id, sub_layer_id)                  
                    try: 
                        down_list.append(
                            conv(self.vgg_layers[layer_name],
                                 filter_size=1,
                                 out_dim=21,
                                 name='down_{}'.format(layer_name),
                                 init_w=init_w,
                                 init_b=init_b,
                                 nl=tf.nn.relu))
                    except KeyError:
                        break
                down_list = tf.reduce_sum(down_list,
                                          axis=0,
                                          name='elesum_{}'.format(layer_id))
                
                side_edge = conv(down_list,
                                filter_size=1,
                                out_dim=1,
                                name='side_edge_{}'.format(layer_name),
                                init_w=init_w,
                                init_b=init_b,
                                nl=tf.nn.relu)

                if layer_id > 1:
                    full_res_side_edge = L.upsample_deconv(
                        x=side_edge,
                        filter_size=4,
                        up_level=(layer_id - 1),
                        out_dim=1,
                        trainable=True,
                        nl=tf.nn.relu,
                        name='upsample_{}'.format(layer_id))
                    self.layers['side_logits_{}'.format(layer_id)] = full_res_side_edge
                    self.layers['edge_fusion'] = tf.concat(
                        [self.layers['edge_fusion'], full_res_side_edge], axis=-1)
                else:
                    self.layers['side_logits_{}'.format(layer_id)] = side_edge
                    self.layers['edge_fusion'] = side_edge

                self.layers['elesum_{}'.format(layer_id)] = down_list
                self.layers['side_edge_{}'.format(layer_id)] = side_edge

            return self.layers['side_logits_{}'.format(layer_id)]

        for layer_id in range(1, self._n_vgg_layer + 1):
            side_layer(layer_id)

        # print(self.layers['edge_fusion'])
        with tf.variable_scope('fusion'):
            init_w = tf.keras.initializers.he_normal()
            init_b = tf.zeros_initializer()
            self.layers['fusion_logists'] = conv(
                self.layers['edge_fusion'],
                filter_size=1,
                out_dim=1,
                name='fusion_conv',
                init_w=init_w,
                init_b=init_b,
                nl=tf.nn.relu)

    def _get_loss(self):
        with tf.variable_scope('loss'):
            label = self.label
            loss_sum = 0
            for layer_id in range(1, self._n_vgg_layer + 1):
                loss_sum += class_balanced_cross_entropy_with_logits(
                    logits=tf.squeeze(self.layers['side_logits_{}'.format(layer_id)], axis=-1),
                    label=label,
                    name='side_loss_{}'.format(layer_id))
            loss_sum += class_balanced_cross_entropy_with_logits(
                logits=tf.squeeze(self.layers['fusion_logists'], axis=-1),
                label=label,
                name='fusion_loss_{}'.format(layer_id))
            return loss_sum
