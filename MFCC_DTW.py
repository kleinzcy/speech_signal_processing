#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 22:08
# @Author  : chuyu zhang
# @File    : MFCC_DTW.py
# @Software: PyCharm

import os
import time
from utils.tools import read
from utils.processing import enframe, stMFCC, mfccInitFilterBanks
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from scipy.fftpack import fft
from scipy.spatial.distance import euclidean
from dtw import dtw
import matplotlib.pyplot as plt
import seaborn as sns

eps = 1e-8
def get_time(start_time=None):
    if start_time == None:
        return time.time()
    else:
        return time.time() - start_time


def load_wav(path='dataset/ASR'):
    """
    load data from dataset/ASR
    :param path: the path of dataset
    :return: x is train data, y_label is the label of x
    """
    start_time = get_time()
    wav_dir = os.listdir(path)
    y_label = []
    x = []
    enc = OrdinalEncoder()
    for _dir in wav_dir:
        for _path in os.listdir(os.path.join(path, _dir)):
            _, data = read(os.path.join(path, _dir, _path))
            x.append(MFCC(data[:, 0]))
            y_label.append(_path.split('.')[0][:-1])
            del data

    y_label = np.array(y_label)
    y_label = enc.fit_transform(y_label.reshape(-1, 1))
    print('loading data and extract mfcc feature spend {}s'.format(get_time(start_time)))
    return x,y_label


def MFCC(raw_signal, fs=16000, frameSize=400, step=160):
    nFFT = int(frameSize / 2)
    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)
    n_mfcc_feats = 13

    signal = enframe(raw_signal, frameSize, step)
    feature = []
    for frame in range(signal.shape[1]):
        x = signal[:, frame]
        X = abs(fft(x))  # get fft magnitude
        X = X[0:nFFT]  # normalize fft
        X = X / len(X)
        feature.append(stMFCC(X, fbank, n_mfcc_feats))

    feature = np.array(feature)
    return feature.flatten()


def distance_dtw(sample_x, sample_y, show=False):
    """
    calculate the distance between sample_x and sample_y using dtw
    :param sample_x: ndarray, mfcc feature for each frame
    :param sample_y: the same as sample_x
    :param show: bool, if true, show the path
    :return: the euclidean distance
    """
    # euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(sample_x, sample_y, dist=euclidean)
    if show:
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.show()
    return d


def distance_all(data):
    """
    calculate the distance of all data
    :param data: input data, list, mfcc feature of all audio
    :return: the distance matrix
    """
    start_time = get_time()
    distance = np.zeros((len(data), len(data)))
    for index, sample_x in enumerate(data):
        col = index + 1
        for sample_y in data[col:]:
            distance[index, col] = distance_dtw(sample_x, sample_y)
            distance[col, index] = distance[index, col]
            col += 1
    print('cost {}s'.format(get_time(start_time)))
    return distance


if __name__=='__main__':
    x,y = load_wav()
    test = [x[0], x[1], x[5], x[6], x[10], x[11], x[15], x[16]]
    distance = distance_all(test)
    np.savetxt('dis.csv', X=distance, delimiter=',')


    # print(distance_dtw(x[0], x[1], show=True))
    # print(distance_dtw(x[0], x[5], show=True))