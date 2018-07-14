#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loss.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

def  class_balanced_cross_entropy_with_logits(
    logits, label, name='class_balanced_cross_entropy'):
    '''
    original from 'Holistically-Nested Edge Detection (CVPR 15)'
    '''
    with tf.name_scope(name) as scope:
        logits = tf.cast(logits, tf.float32)
        label = tf.cast(label, tf.float32)

        num_pos = tf.reduce_sum(label)
        num_neg = tf.reduce_sum(1.0 - label)
        
        beta = num_neg / (num_neg + num_pos)
        
        pos_weight = beta / (1 - beta)
        check_weight = tf.identity(beta, name = 'check')

        cost = tf.nn.weighted_cross_entropy_with_logits(targets=label, 
                                                        logits=logits, 
                                                        pos_weight=pos_weight)
        # print(cost)
        # t = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, 
        #                                                 logits=logits))
        loss = tf.reduce_mean((1 - beta) * cost)
        # check_weight = tf.identity(t, name = 'check_loss_2')

        return tf.where(tf.equal(beta, 1.0), 0.0, loss)
