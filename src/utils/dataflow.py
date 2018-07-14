#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataflow.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import scipy.misc


def identity(inputs):
    return inputs

def load_image(im_path, read_channel=None, pf=identity):
    if read_channel is None:
        im = scipy.misc.imread(im_path)
    elif read_channel == 3:
        im = scipy.misc.imread(im_path, mode='RGB')
    else:
        im = scipy.misc.imread(im_path, flatten=True)

    if len(im.shape) < 3:
        im = pf(im)
        im = np.reshape(im, [im.shape[0], im.shape[1], 1])
    else:
        im = pf(im)

    # if len(im.shape) < 3:
    #     im = pf(im)
    #     im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    # else:
    #     im = pf(im)
    #     im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im