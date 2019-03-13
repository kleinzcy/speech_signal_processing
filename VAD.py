#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 13:10
# @Author  : chuyu zhang
# @File    : VAD.py
# @Software: PyCharm

import math
import numpy as np
from scipy.io import loadmat
from utils.tools import wave_read, read
import glob
import time
# 计算每一帧的过零率
def ZCR(frameData):
    frameNum = frameData.shape[1]
    frameSize = frameData.shape[0]
    zcr = np.zeros((frameNum, 1))

    for i in range(frameNum):
        singleFrame = frameData[:, i]
        temp = singleFrame[:frameSize-1] * singleFrame[1:frameSize]
        temp = np.sign(temp)
        zcr[i] = np.sum(temp<0)

    return zcr

# 分帧处理函数
# 不加窗
def enframe(wavData, frameSize, overlap):
    # coef = 0.97 # 预加重系数
    wlen = wavData.shape[0]
    step = frameSize - overlap
    frameNum:int = math.ceil(wlen / step)
    frameData = np.zeros((frameSize, frameNum))

    # hamwin = np.hamming(frameSize)

    for i in range(frameNum):
        singleFrame = wavData[np.arange(i * step, min(i * step + frameSize, wlen))]
        # singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coef * singleFrame[1:]) # 预加重
        frameData[:len(singleFrame), i] = singleFrame.reshape(-1, 1)[:, 0]
        # frameData[:, i] = hamwin * frameData[:, i] # 加窗

    return frameData

# 计算每一帧能量
def energy(frameData):
    frameNum = frameData.shape[1]

    frame_energy = np.zeros((frameNum, 1))

    for i in range(frameNum):
        single_frame = frameData[:, i]
        frame_energy[i] = sum(single_frame * single_frame)

    return frame_energy


def feature(waveData):
    print("feature extract !")
    return ZCR(waveData), energy(waveData)


# framesize为帧长，overlap为帧移
def wavdata(wavfile, framesize=256, overlap=0):
    f = wave_read(wavfile)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    # print(type(strData))
    waveData = np.fromstring(strData, dtype=np.int16)
    # print(waveData.shape)
    waveData = waveData/(max(abs(waveData)))
    return enframe(waveData, framesize, overlap)


# 首先判断能量，如果能量低于ampl，则认为是噪音（静音），如果能量高于amph则认为是语音，如果能量处于两者之前则认为是清音。
def VAD_detection(waveData, zcr_gate=10, ampl=2, amph=10):
    # 最短语音帧数
    min_len = 21
    # 一些标记量
    # status = 0
    # speech = 0
    start = 0
    end = 0
    zcr, power = feature(waveData)
    # 结果存储
    res = np.zeros((zcr.shape[0], 1))
    # 修正能量门限

    amph = min(amph, max(power) / 4)
    ampl = min(ampl, max(power) / 8)
    """
    amph = np.mean(power, axis=0)/4
    ampl = amph/4
    """
    for i in range(zcr.shape[0]):
        if power[i] > amph or (power[i] > ampl and zcr[i] > zcr_gate):
            if start == 0:
                start = i
            end = i
        elif power[i] < ampl or ((power[i] > ampl and zcr[i] < zcr_gate)) :
            if end - start + 1 > min_len:
                res[start:end+1] = 1
                start = 0

    return res

# 处理mat文件，统计一个帧数中静音和语音的数量，给这个帧数一个label，具体规则后续完善
def label(mat_file):
    mat = loadmat(mat_file)
    y_label = mat['y_label']
    y_label = enframe(y_label, frameSize=256, overlap=0)
    label_sum = np.sort(y_label.sum(axis=0))
    y_label = np.where(label_sum > 0, 1, 0)

    return y_label


if __name__=='__main__':
    wavfile = glob.glob(r'dataset\VAD\*.wav')
    matfile = glob.glob(r'dataset\VAD\*.mat')

    for wav, mat in zip(wavfile, matfile):
        start = time.time()
        print(wav.split('\\')[-1])
        data = wavdata(wav)
        res = VAD_detection(data)
        y_label = label(mat)
        y_label = y_label.reshape(-1 ,1)
        accuracy = (y_label==res).sum()/y_label.shape[0]
        end = time.time()
        print("accuracy is {:.2%}, for {}, spend {}s".format(accuracy, wav.split('\\')[-1], end - start))

