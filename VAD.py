#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 13:10
# @Author  : chuyu zhang
# @File    : VAD.py
# @Software: PyCharm

import math
import numpy as np
from scipy.io import loadmat
from utils.tools import wave_read, read, plot_confusion_matrix
import glob
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from bayes_opt import BayesianOptimization
import pickle as pkl
from scipy import signal
# 计算每一帧的过零率

frameSize = 256
overlap = 128

# 分帧处理函数
# 不加窗
def enframe(wavData):
    """
    frame the wav data, according to frameSize and overlap
    :param wavData: the input wav data, ndarray
    :return:frameData, shape
    """
    # coef = 0.97 # 预加重系数
    wlen = wavData.shape[0]
    step = frameSize - overlap
    frameNum:int = math.ceil(wlen / step)
    frameData = np.zeros((frameSize, frameNum))

    # hamwin = np.hamming(frameSize)

    for i in range(frameNum):
        singleFrame = wavData[np.arange(i * step, min(i * step + frameSize, wlen))]
        # b, a = signal.butter(8, 1, 'lowpass')
        # filtedData = signal.filtfilt(b, a, data)
        # singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coef * singleFrame[1:]) # 预加重
        frameData[:len(singleFrame), i] = singleFrame.reshape(-1, 1)[:, 0]
        # frameData[:, i] = hamwin * frameData[:, i] # 加窗

    return frameData


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

# 计算每一帧能量
def energy(frameData):
    frameNum = frameData.shape[1]

    frame_energy = np.zeros((frameNum, 1))

    for i in range(frameNum):
        single_frame = frameData[:, i]
        frame_energy[i] = sum(single_frame * single_frame)

    return frame_energy


def stSpectralEntropy(X, n_short_blocks=10, eps=1e-8):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = np.sum(X ** 2)            # total spectral energy

    sub_win_len = int(np.floor(L / n_short_blocks))   # length of sub-frame
    if L != sub_win_len * n_short_blocks:
        X = X[0:sub_win_len * n_short_blocks]

    sub_wins = X.reshape(sub_win_len, n_short_blocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = np.sum(sub_wins ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -np.sum(s*np.log2(s + eps))                                    # compute spectral entropy

    return En


def spectrum_entropy(frameData):
    frameNum = frameData.shape[1]

    frame_spectrum_entropy = np.zeros((frameNum, 1))

    for i in range(frameNum):
        X = np.fft.fft(frameData[:, i])
        frame_spectrum_entropy[i] = stSpectralEntropy(X[:int(frameSize/2)])

    return frame_spectrum_entropy


def feature(waveData):
    # print("feature extract !")
    power = energy(waveData)
    zcr = ZCR(waveData) * (power>0.2)
    spectrumentropy = spectrum_entropy(waveData)
    return zcr, power, spectrumentropy


# framesize为帧长，overlap为帧移
def wavdata(wavfile):
    f = wave_read(wavfile)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    # print(type(strData))
    waveData = np.fromstring(strData, dtype=np.int16)
    # print(waveData.shape)
    waveData = waveData/(max(abs(waveData)))
    return enframe(waveData)


# 首先判断能量，如果能量低于ampl，则认为是噪音（静音），如果能量高于amph则认为是语音，如果能量处于两者之前则认为是清音。
def VAD_detection(zcr, power, zcr_gate=35, ampl=0.5, amph=1.5):
    # 最短语音帧数
    min_len = 16
    # 两段语音间的最短间隔
    min_distance = 10
    # 标记量,status：0为静音状态，1为清音状态，2为浊音状态
    status = 0
    # speech = 0
    start = 0
    end = 0
    last_end = -1

    res = np.zeros((zcr.shape[0], 1))

    for i in range(zcr.shape[0]):
        if power[i] > amph:
            # 此处是浊音状态，记录end即可
            if status != 1:
                start = i

            end = i
            status = 1
            # print(start - end)
        elif end - start + 1 > min_len:

            while(power[start] > ampl or zcr[start] > zcr_gate):
                start -= 1

            start += 1

            while(power[end] > ampl or zcr[end] > zcr_gate):
                end += 1
                if end == power.shape[0]:
                    break

            end -= 1
            # print('ok')
            if last_end > 0 and start - last_end < min_distance:
                res[last_end : end + 1] = 1
                last_end = end
            else:
                res[start: end + 1] = 1

            start = 0
            end = 0
            status = 0

    return res


def VAD_frequency():
    pass


def optimize(X, y):
    zcr, power, spectrumentropy = feature(X)
    """
    sns.distplot(zcr)
    plt.show()
    sns.distplot(power)
    plt.show()
    """
    params ={
        'zcr_gate': (10, 30),
        'ampl': (0.3, 4),
        'amph': (5, 15)
    }
    y = y.reshape(1, -1)

    def cv(zcr_gate, ampl, amph):
        res = VAD_detection(zcr, power, zcr_gate=zcr_gate, amph=amph, ampl=ampl)
        # print((res==0).sum()/res.shape[0])
        res = res.reshape(1, -1)
        # metrics.precision_score(y[0], res[0])
        # accuracy = (y == res).sum() / y.shape[0]
        return metrics.f1_score(y[0], res[0])

    BO = BayesianOptimization(cv, params)

    start_time = time.time()
    BO.maximize(n_iter=30)
    end_time = time.time()
    print("Final result:{}, spend {}s".format(BO.max, end_time - start_time))
    best_params = BO.max['params']

    return best_params

# 处理mat文件，统计一个帧数中静音和语音的数量，给这个帧数一个label，具体规则后续完善
def label(mat_file):
    mat = loadmat(mat_file)
    y_label = mat['y_label']
    y_label = enframe(y_label)

    return np.where(y_label.sum(axis=0) > 0, 1, 0)



def main(wav, mat):
    start = time.time()
    print(wav.split('\\')[-1])
    data = wavdata(wav)
    y_label = label(mat)
    y_label = y_label.reshape(-1, 1)

    param = optimize(data, y_label)
    end = time.time()
    print('spend {}s'.format(end - start))

    return param


if __name__=='__main__':
    wavfile = glob.glob(r'dataset\VAD\*.wav')
    matfile = glob.glob(r'dataset\VAD\*.mat')
    best_params = []


    # print(y_label.sum(), y_label.shape[0])

    for wav, mat in zip(wavfile, matfile):
        best_params.append(main(wav, mat))

    with open('param.pkl', 'wb') as f:
        pkl.dump(best_params, f)


    """
    wav = wavfile[0]
    start = time.time()
    print(wav.split('\\')[-1])
    data = wavdata(wav)
    
    zcr, power = feature(data)
    plt.plot(zcr[:300])
    plt.show()
    plt.plot(power[:300])
    
    plt.show()
    """

    """
    zcr, power = feature(wavdata(wavfile[0]))

    """


    """
    zcr, power = feature(data)
    res = VAD_detection(zcr, power)

    accuracy = (y_label==res).sum()/y_label.shape[0]
    end = time.time()
    print("accuracy is {:.2%}, for {}, spend {}s".format(accuracy, wav.split('\\')[-1], end - start))

    y_label = y_label.reshape(1, -1)
    res = res.reshape(1, -1)
    print((y_label==0).sum()/y_label.shape[1])
    # print(confusion_matrix(y_label[0], res[0]))
    # print(metrics.precision_score(y_label[0], res[0]))
    # print(metrics.recall_score(y_label[0], res[0]))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_label[0].tolist(), res[0].tolist(), classes=['no voices', 'voice'], normalize=False)
    plt.show()
    break
    """
