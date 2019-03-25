#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 22:08
# @Author  : chuyu zhang
# @File    : MFCC_DTW.py
# @Software: PyCharm

import os
import random
from utils.tools import read, get_time
from utils.processing import enframe, stMFCC, mfccInitFilterBanks
import numpy as np
from scipy.fftpack import fft

from scipy.spatial.distance import euclidean,cosine

# dtw is accurate than fastdtw, but it is slower, I will test the speed and acc later
# from dtw import dtw,accelerated_dtw
from fastdtw import fastdtw

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


eps = 1e-8

def MFCC(raw_signal, fs=8000, frameSize=512, step=256):
    nFFT = int(frameSize/2)
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
    # euclidean_norm = lambda x, y: np.abs(x - y)euclidean
    #
    d, path = fastdtw(sample_x, sample_y, dist=euclidean)
    """
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(sample_x, sample_y, dist='cosine')
    if show:
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.show()
    """
    return d


def distance_train(data):
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

def distance_test(x_test, x_train):
    """
    calculate the distance between x_test(one sample) and x_train(many sample)
    :param x_test: a sample
    :param x_train: the whole train dataset
    :return: distance based on dtw
    """
    distance = np.zeros((1, len(x_train)))
    for index in range(len(x_train)):
        distance[0, index] = distance_dtw(x_train[index], x_test)
    return distance


def sample(x, y, sample_num=2):
    index = random.sample(range(5), sample_num)
    sample_x = []
    sample_y = []
    for i in range(4):
        sample_x.append(x[index[0] + 5*i])
        sample_x.append(x[index[1] + 5*i])
        sample_y.append(y[index[0] + 5*i])
        sample_y.append(y[index[1] + 5*i])
    return sample_x, sample_y

def load_wav(path='dataset/ASR/train'):
    """
    load data from dataset/ASR
    :param path: the path of dataset
    :return: x is train data, y_label is the label of x
    """
    start_time = get_time()
    wav_dir = os.listdir(path)
    y_label = []
    x = []
    # enc = OrdinalEncoder()
    for _dir in wav_dir:
        for _path in os.listdir(os.path.join(path, _dir)):
            _, data = read(os.path.join(path, _dir, _path))
            # Some audio has two channel, but some audio has one channel.
            # so, I add "try except" to deal with such problem.
            # downsample the data to 8k
            try:
                x.append(MFCC(data[range(0, data.shape[0], 2), 0]))
            except:
                x.append(MFCC(data[range(0, data.shape[0], 2)]))

            y_label.append(_dir)
            del data

    # y_label = np.array(y_label)
    # y_label = enc.fit_transform(y_label.reshape(-1, 1))
    print('loading data and extract mfcc feature spend {}s'.format(get_time(start_time)))
    return x,y_label


def test(threshold=100):
    x_train,y_train = load_wav(path='dataset/ASR/train')
    x_train,y_train = sample(x_train, y_train)
    x_test,y_test = load_wav(path='dataset/ASR/test')
    y_pred = []
    # x_test = x_test[:5]
    # y_test = y_test[:5]
    for x in tqdm(x_test):
        distance = distance_test(x, x_train)
        y_pred.append(y_train[np.argmin(distance)])
        # when I set threshold to 100, the results is very bad, many sample are classified to other,
        # so, I decide to give up threshold,
        """
        if np.min(distance) < threshold:
            y_pred.append(y_train[np.argmin(distance)])
        else:
            y_pred.append('other')
        """
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    acc = (y_pred==y_test).sum()/y_test.shape[0]
    print("accuracy is {:.2%}".format(acc))


if __name__=='__main__':
    test()
    """
    x_train,y_train = load_wav(path='dataset/ASR/train')
    distance = distance_train(x_train)
    np.savetxt('dis.csv', X=distance, delimiter=',')
    """


    # print(distance_dtw(x[0], x[1], show=True))
    # print(distance_dtw(x[0], x[5], show=True))