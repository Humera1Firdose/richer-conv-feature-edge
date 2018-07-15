#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rcfedge.py
# Author: Qian Ge <geqian1001@gmail.com>

import platform
import argparse
import numpy as np
import scipy.misc
import tensorflow as tf
# import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from src.models.rcf import RCF
from src.helper.trainer import Trainer
from src.dataflow.bsd import BSDS500HED

if platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/dataset/BSR_bsds500/BSR/BSDS500/data/'
    SAVE_PATH = '/home/qge2/workspace/data/out/rich/'
    VGG_PATH = '/home/qge2/workspace/data/pretrain/vgg/vgg16.npy'
elif platform.node() == 'Qians-MacBook-Pro.local':
    VGG_PATH = '/Users/gq/workspace/Dataset/pretrained/vgg16.npy'
    SAVE_PATH = '/Users/gq/workspace/output/rcf/'
    DATA_PATH = '/Users/gq/workspace/Dataset/BSR_bsds500/BSR/BSDS500/data/'
else:
    pass
    # DATA_PATH = 'E:/GITHUB/workspace/topologyseg/synthetic_foram/CNN_sythetic/'
    # # DATA_PATH = 'Q:/My Drive/Foram/Training/CNN_sythetic/'
    # SAVE_PATH = 'E:/tmp/seg/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0005, type=float)

    return parser.parse_args()

def resize_im(im):
    return scipy.misc.imresize(im, [224, 224])

if __name__ == '__main__':
    FLAGS = get_args()
    train_data = BSDS500HED(
        name='train', 
        data_dir=DATA_PATH, 
        shuffle=True, 
        pf_list=[resize_im, resize_im],
        batch_dict_name=['image', 'label']
        # is_mask=False,
        # normalize_fnc=identity,
        )

    train_data.setup(epoch_val=0, batch_size=10)
    
    # batch_data = train_data.next_batch_dict()
    
    # print(batch_data['label'].shape)
    # plt.figure()
    # plt.imshow(batch_data['image'][0])

    # plt.figure()
    # plt.imshow(batch_data['label'][0])
    # plt.show()
    model = RCF(n_channel=3, vgg_path=VGG_PATH, vgg_trainable=True)
    model.create_model()

    # # loss_op = model.get_loss()
    # # train_op = model.get_train_op()
    trainer = Trainer(model, train_data, init_lr=FLAGS.lr)
    writer = tf.summary.FileWriter(SAVE_PATH)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for i in range(0, 5000):
            trainer.train_epoch(sess, summary_writer=writer)
            # batch_data = train_data.next_batch_data()
            # _, loss = sess.run(
            #     [train_op, loss_op],
            #     feed_dict={
            #                model.lr: 1e-6,
            #                model.raw_image: batch_data['image'],
            #                model.label: batch_data['label'],
            #                })
            # print(loss)

        writer.close()
