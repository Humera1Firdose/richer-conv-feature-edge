#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rcfedge.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from lib.models.rcf import RCF
from lib.dataflow.bsd import BSDS500HED

VGG_PATH = '/Users/gq/workspace/Dataset/pretrained/vgg16.npy'
SAVE_PATH = '/Users/gq/workspace/output/rcf/'
DATA_PATH = '/Users/gq/workspace/Dataset/BSR_bsds500/BSR/BSDS500/data/'

def resize_im(im):
    return scipy.misc.imresize(im, [224, 224])

if __name__ == '__main__':
    train_data = BSDS500HED(
        name='train', 
        data_dir=DATA_PATH, 
        shuffle=True, 
        pf=resize_im,
        # is_mask=False,
        # normalize_fnc=identity,
        )
    train_data.setup(epoch_val=0, batch_size=1)
    
    # batch_data = train_data.next_batch_data()
    
    # print(batch_data['label'].shape)
    # plt.figure()
    # plt.imshow(batch_data['image'][0])

    # plt.figure()
    # plt.imshow(batch_data['label'][0])
    # plt.show()
    model = RCF(n_channel=3, vgg_path=VGG_PATH)
    model.create_model()

    loss_op = model.get_loss()
    train_op = model.get_train_op()

    writer = tf.summary.FileWriter(SAVE_PATH)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for i in range(0, 10):
            batch_data = train_data.next_batch_data()
            _, loss = sess.run(
                [train_op, loss_op],
                feed_dict={
                           model.lr: 1e-6,
                           model.raw_image: batch_data['image'],
                           model.label: batch_data['label'],
                           })
            print(loss)

    writer.close()
