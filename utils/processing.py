#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/23 22:18
# @Author  : chuyu zhang
# @File    : processing.py
# @Software: PyCharm

import numpy as np
import math
from scipy import signal
from tools import read,play
## 语音的预处理函数


def enframe(wavData, frameSize=400, step=160):
    """
    frame the wav data, according to frameSize and overlap
    :param wavData: the input wav data, ndarray
    :return:frameData, shape
    """
    wlen = wavData.shape[0]
    frameNum = math.ceil(wlen / step)
    frameData = np.zeros((frameSize, frameNum))

    window = signal.windows.hamming(frameSize)

    for i in range(frameNum):
        singleFrame = wavData[i * step : min(i * step + frameSize, wlen)]
        frameData[:len(singleFrame), i] = singleFrame
        frameData[:, i] = window*frameData[:, i]

    return frameData


if __name__=='__main__':
    path = '../dataset/ASR/zcy/zcy1.wav'
    play(path)
    samprate, data = read(path)
    framedata = enframe(data[:,0])
    print(framedata.shape)