#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rcf.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf
from tensorcv.models.layers import *

import src.models.layers as L
from src.models.vgg import RCFVGG16
from src.models.loss import class_balanced_cross_entropy_with_logits

# init_conv_stage_w = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
# init_dconv_w = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
# init_conv_fusion_w = tf.truncated_normal_initializer(mean=0.0, stddev=0.2)

# init_conv_stage_w = tf.contrib.layers.xavier_initializer()
init_dconv_w = tf.contrib.layers.xavier_initializer()

init_conv_stage_w = tf.random_normal_initializer(mean=0.0, stddev=0.01)
# init_dconv_w = tf.random_normal_initializer(mean=0.0, stddev=0.1)
init_conv_fusion_w = tf.random_normal_initializer(mean=0.0, stddev=0.2)

class RCF(RCFVGG16):
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

        self.layers['prob'] = tf.nn.sigmoid(self.layers['fusion_logists'])

    def get_train_op(self):
        with tf.name_scope('train'):
            # opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            loss = self.get_loss()
            # grads = opt.compute_gradients(loss)
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

    def _create_richer_feature(self):
        def side_layer(layer_id):
            with tf.variable_scope('side_{}'.format(layer_id)):
                # init_w = tf.keras.initializers.he_normal()
                init_b = tf.zeros_initializer() 

                down_list = []
                for sub_layer_id in range(1, self._n_vgg_sublayer + 1):
                    layer_name = 'conv{}_{}'.format(layer_id, sub_layer_id)                  
                    try: 
                        down = L.conv(
                            filter_size=1,
                            out_dim=21,
                            layer_dict=self.layers,
                            inputs=self.vgg_layers[layer_name],
                            init_w=init_conv_stage_w,
                            name='down_{}'.format(layer_name))
                        down_list.append(down)
                        # down_list.append(
                        #     conv(self.vgg_layers[layer_name],
                        #          filter_size=1,
                        #          out_dim=21,
                        #          name='down_{}'.format(layer_name),
                        #          init_w=init_conv_stage_w,
                        #          init_b=init_b,
                        #          ))
                    except KeyError:
                        break

                down_list = tf.reduce_sum(down_list,
                                          axis=0,
                                          name='elesum_{}'.format(layer_id))
                # side_edge = conv(down_list,
                #                 filter_size=1,
                #                 out_dim=1,
                #                 name='side_edge_{}'.format(layer_name),
                #                 init_w=init_conv_stage_w,
                #                 init_b=init_b)
                side_edge = L.conv(
                    filter_size=1,
                    out_dim=1,
                    layer_dict=self.layers,
                    inputs=down_list,
                    init_w=init_conv_stage_w,
                    name='side_edge_{}'.format(layer_name))

                if layer_id > 1:
                    full_resolution_side_edge = L.upsample_deconv(
                            x=side_edge,
                            filter_size=4,
                            up_level=np.minimum(layer_id - 1, self._n_vgg_layer-2),
                            out_dim=1,
                            trainable=True,
                            init_w=init_dconv_w,
                            name='upsample_{}'.format(layer_id))

                    self.layers['side_logits_{}'.format(layer_id)] = full_resolution_side_edge
                    self.layers['edge_fusion'] = tf.concat(
                        [self.layers['edge_fusion'], full_resolution_side_edge], axis=-1)
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
            # init_w = tf.keras.initializers.he_normal()
            init_b = tf.zeros_initializer()
            self.layers['fusion_logists'] = conv(
                self.layers['edge_fusion'],
                filter_size=1,
                out_dim=1,
                name='fusion_conv',
                init_w=init_conv_fusion_w,
                init_b=init_b)

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
                if layer_id == self._n_vgg_layer:
                    side_loss = side_loss
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
            # pred_prob = tf.summary.image(
            #     'probability', tf.expand_dims(tf.cast(self.layers['prob'][:,:,:,1], tf.float32), -1),
            #     collections=[name])
            # tf.summary.image(
            #     'prediction', tf.expand_dims(tf.cast(self.layers['pred'], tf.float32), -1),
            #     collections=[name])
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
