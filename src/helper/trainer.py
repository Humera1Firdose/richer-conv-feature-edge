#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import scipy.misc
import numpy as np
import tensorflow as tf


class Trainer(object):
    def __init__(self, model, train_data, init_lr=1e-4):

        self._model = model
        self._train_data = train_data
        self._lr = init_lr
        self._min_lr = init_lr / 1000.

        self._loss_op = model.get_loss()
        self._train_op = model.get_train_op()
        self._loss_list = model.layers['loss_list']

        # self._train_op = model.get_train_op()
        # self._loss_op = model.get_loss()
        # self._accuracy_op = model.get_accuracy()
        # self._sample_loc_op = model.layers['loc_sample']
        # self._pred_op = model.layers['pred']
        self._train_sum_op = model.get_summary('train')

        # self._lr_op = model.cur_lr

        self.global_step = 0
        self.global_epoch = 0

    def train_epoch(self, sess, summary_writer=None):
        self.global_epoch += 1
        self._model.set_is_training(True)
        # cur_lr = self._lr
        # if self.global_epoch % 50 == 0:
        #     self._lr = self._lr * 0.1
        # cur_lr = self._lr
        # self._lr = self._lr * 0.99
        # self._lr = np.maximum(self._lr * 0.985, self._min_lr)
        # self._lr = np.maximum(self._lr * 0.97, 1e-4)

        cur_epoch = self._train_data.epochs_completed

        step = 0
        loss_sum = 0
        acc_sum = 0
        while cur_epoch == self._train_data.epochs_completed:
            self.global_step += 1
            if self.global_step % 10000 == 0:
                self._lr = self._lr * 0.1
                # cur_lr = self._lr

            step += 1

            batch_data = self._train_data.next_batch_dict()
            # im = batch_data['image']
            # label = batch_data['label']
            acc = 0
            _, loss, cur_summary, loss_list = sess.run(
                [self._train_op, self._loss_op, self._train_sum_op, self._loss_list],
                feed_dict={
                           self._model.lr: self._lr,
                           self._model.raw_image: batch_data['image'],
                           self._model.label: batch_data['label'],
                           })
            try:
                loss_list_sum += loss_list
            except UnboundLocalError:
                loss_list_sum = loss_list

            # print(loss_list_sum)

            loss_sum += loss
            acc_sum += acc

            if step % 100 == 0:
                print('step: {}, loss: {:.4f}, accuracy: {:.4f}'
                      .format(self.global_step,
                              loss_sum * 1.0 / step,
                              acc_sum * 1.0 / step))

        print('epoch: {}, loss: {:.4f}, accuracy: {:.4f}, lr:{}'
              .format(cur_epoch,
                      loss_sum * 1.0 / step,
                      acc_sum * 1.0 / step, self._lr))
        if summary_writer is not None:
            s = tf.Summary()
            s.value.add(tag='train/total_loss', simple_value=loss_sum * 1.0 / step)
            s.value.add(tag='train/accuracy', simple_value=acc_sum * 1.0 / step)
            for idx, cur_l in enumerate(loss_list_sum):
                s.value.add(tag='train/side_loss_{}'.format(idx),
                            simple_value=cur_l * 1.0 / step)
            summary_writer.add_summary(s, self.global_step)
            summary_writer.add_summary(cur_summary, self.global_step)

    def valid_epoch(self, sess, dataflow, batch_size):
        # self._model.set_is_training(False)
        dataflow.setup(epoch_val=0, batch_size=batch_size)

        step = 0
        loss_sum = 0
        acc_sum = 0
        while dataflow.epochs_completed == 0:
            step += 1
            batch_data = dataflow.next_batch_dict()
            loss, acc = sess.run(
                [self._loss_op, self._accuracy_op], 
                feed_dict={self._model.image: batch_data['im'],
                           self._model.label: batch_data['label'],
                           })
            loss_sum += loss
            acc_sum += acc
        print('valid loss: {:.4f}, accuracy: {:.4f}'
              .format(loss_sum * 1.0 / step, acc_sum * 1.0 / step))

        # self._model.set_is_training(True)