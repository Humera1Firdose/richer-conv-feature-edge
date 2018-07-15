#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: hed.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorcv.models.layers import *

import src.models.layers as L
from src.models.vgg import RCFVGG16
from src.models.loss import class_balanced_cross_entropy_with_logits


class HED(RCFVGG16):
    def __init__(self, n_channel, vgg_path, vgg_trainable):
        self._switch = False
        self._vgg_trainable = vgg_trainable

        self._n_channel = n_channel
        self._vgg_param = np.load(vgg_path, encoding='latin1').item()

        self._n_vgg_layer = 5
        self._n_vgg_sublayer = 3

        self._vgg_feature = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
        self._stride = [1, 2, 4, 8, 8]

    def _create_hed_feature(self):
        for side_id, (vgg_feature, factor) in enumerate(zip(self._vgg_feature, self._stride)):
            with tf.variable_scope('side_{}'.format(side_id)):
                side_edge = L.conv(
                    filter_size=1,
                    out_dim=1,
                    layer_dict=self.layers,
                    inputs=vgg_feature,
                    init_w=init_conv_stage_w,
                    name='conv')
                full_resolution_side_edge = L.upsampling_deconv(
                        factor=factor,
                        out_dim=1,
                        layer_dict=self.layers,
                        inputs=side_edge,
                        multlayer=True,
                        bilinear_init=True,
                        init_w=init_dconv_w,
                        name='upsample_{}'.format(side_id))

            self.layers['side_edge_{}'.format(side_id)] = side_edge
            self.layers['side_logits_{}'.format(side_id)] = full_resolution_side_edge

        with tf.variable_scope('fusion'):
            self.layers['edge_fusion'] = tf.concat(
                [self.layers['side_logits_{}'.format(side_id)]
                 for side_id in range(0, len(self._vgg_feature))],
                axis=-1)
                init_b = tf.zeros_initializer()
                self.layers['fusion_logists'] = L.conv(
                    filter_size=1,
                    out_dim=1,
                    layer_dict=self.layers,
                    inputs=self.layers['edge_fusion'],
                    init_w=init_conv_fusion_w,
                    use_bias=False,
                    name='fusion_conv')

    def create_model(self):
        self._create_input()
        self.vgg_layers = {}
        self.layers = {}

        self.image = self._sub_mean(self.raw_image)
        self._creat_vgg_conv(self.image, self.vgg_layers, data_dict=self._vgg_param)
        # self._create_richer_feature()

        self.layers['prob'] = tf.nn.sigmoid(self.layers['fusion_logists'])

    def get_train_op(self):
        with tf.name_scope('train'):
            # opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            loss = self.get_loss()
            var_list = tf.trainable_variables()
            grads = tf.gradients(loss, var_list)
            [tf.summary.histogram('gradient/' + var.name, grad, 
             collections = ['train']) for grad, var in zip(grads, var_list)]

            return opt.apply_gradients(zip(grads, var_list))

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

    def _get_loss(self):
        with tf.variable_scope('loss'):
            self.layers['loss_list'] = []
            loss_sum = 0
            for layer_id in range(1, self._n_vgg_layer + 1):
                side_loss = class_balanced_cross_entropy_with_logits(
                    logits=tf.squeeze(self.layers['side_logits_{}'.format(layer_id)], axis=-1),
                    label=self.label,
                    name='side_loss_{}'.format(layer_id))
                side_loss = tf.reduce_mean(side_loss)
                self.layers['loss_list'].append(side_loss)
                loss_sum += side_loss
            fusion_loss = class_balanced_cross_entropy_with_logits(
                logits=tf.squeeze(self.layers['fusion_logists'], axis=-1),
                label=self.label,
                name='fusion_loss')
            fusion_loss = tf.reduce_mean(fusion_loss)
            loss_sum += fusion_loss
            self.layers['loss_list'].append(fusion_loss)

            self.layers['loss_list'] = tf.convert_to_tensor(self.layers['loss_list'])
            return loss_sum

    def get_summary(self, name):
        with tf.name_scope(name):
            tf.summary.image(
                'pred', tf.cast(self.layers['prob'], tf.float32),
                collections=[name])
            tf.summary.image(
                'gt', tf.expand_dims(tf.cast(self.label, tf.float32), -1),
                collections=[name])
            tf.summary.image(
                'input', tf.cast(self.raw_image, tf.float32),
                collections=[name])

            for i in range(1, self._n_vgg_layer + 1):
                tf.summary.image(
                    'side_logit{}'.format(i), tf.cast(tf.nn.sigmoid(self.layers['side_logits_{}'.format(i)]), tf.float32),
                    collections=[name])
                tf.summary.image(
                    'side_{}'.format(i), tf.cast(tf.nn.sigmoid(self.layers['side_edge_{}'.format(i)]), tf.float32),
                    collections=[name])
        return tf.summary.merge_all(key=name)

