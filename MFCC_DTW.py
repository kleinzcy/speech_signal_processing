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

import librosa
# dtw is accurate than fastdtw, but it is slower, I will test the speed and acc later
from scipy.spatial.distance import euclidean
from dtw import dtw,accelerated_dtw
from fastdtw import fastdtw

import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm


eps = 1e-8

def MFCC_lib(raw_signal, n_mfcc=13):
    feature = librosa.feature.mfcc(raw_signal.astype('float32'), n_mfcc=n_mfcc, sr=8000)
    # print(feature.T.shape)
    return feature.T.flatten()

def MFCC(raw_signal, fs=8000, frameSize=512, step=256):
    """
    extract mfcc feature
    :param raw_signal: the original audio signal
    :param fs: sample frequency
    :param frameSize:the size of each frame
    :param step:
    :return: a series of mfcc feature of each frame and flatten to (num, )
    """
    # Signal normalization

    """
    raw_signal = np.double(raw_signal)

    raw_signal = raw_signal / (2.0 ** 15)
    DC = raw_signal.mean()
    MAX = (np.abs(raw_signal)).max()
    raw_signal = (raw_signal - DC) / (MAX + eps)
    """
    nFFT = int(frameSize)
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
    # print(feature.shape)
    return feature.flatten()


def distance_dtw(sample_x, sample_y, show=False, dtw_method=1, dist=euclidean):
    """
    calculate the distance between sample_x and sample_y using dtw
    :param sample_x: ndarray, mfcc feature for each frame
    :param sample_y: the same as sample_x
    :param show: bool, if true, show the path
    :param dtw_method: 1:accelerated_dtw, 2:fastdtw
    :return: the euclidean distance
    """
    # euclidean_norm = lambda x, y: np.abs(x - y)euclidean
    #
    #
    if dtw_method==2:
        d, path = fastdtw(sample_x, sample_y, dist=dist)
    elif dtw_method==1:
        d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(sample_x, sample_y, dist='euclidean')
        if show:
            plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
            plt.plot(path[0], path[1], 'w')
            plt.show()

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


def sample(x, y, sample_num=2, whole_num=8):
    index = random.sample(range(whole_num), sample_num)
    sample_x = []
    sample_y = []
    for i in range(4):
        for _index in index:
            sample_x.append(x[_index + whole_num*i])
            sample_y.append(y[_index + whole_num*i])
    return sample_x, sample_y


def load_train(path='dataset/ASR/train', mfcc_extract=MFCC):
    """
    load data from dataset/ASR
    :param path: the path of dataset
    :return: x is train data, y_label is the label of x
    """
    start_time = get_time()
    # wav_dir is a list, which include four directory in train.
    wav_dir = os.listdir(path)
    y_label = []
    x = []
    print("Generate template according to train set.")
    for _dir in tqdm(wav_dir):
        _x = []
        for _path in os.listdir(os.path.join(path, _dir)):
            _, data = read(os.path.join(path, _dir, _path))
            # Some audio has two channel, but some audio has one channel.
            # so, I add "try except" to deal with such problem.
            # downsample the data to 8k
            try:
                _x.append(mfcc_extract(data[range(0, data.shape[0], 2), 0]))
            except:
                _x.append(mfcc_extract(data[range(0, data.shape[0], 2)]))
            del data
            # print(_x[-1].shape)
        # generate a template of different speaker.
        x.append(generate_template(_x))
        y_label.append(_dir)

    print('Loading train data, extract mfcc feature and generate template spend {}s'.format(get_time(start_time)))
    return x,y_label


def load_test(path='dataset/ASR/test', mfcc_extract=MFCC, template=False):
    """
    load data from dataset/ASR
    :param path: the path of dataset
    :return: x is train data, y_label is the label of x
    """
    start_time = get_time()
    if template:
        # load template directly.
        pass
    # wav_dir is a list, which include four directory in train.
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
                x.append(mfcc_extract(data[range(0, data.shape[0], 2), 0]))
            except:
                x.append(mfcc_extract(data[range(0, data.shape[0], 2)]))
            del data
            y_label.append(_dir)

    print('Loading test data and extract mfcc feature spend {}s'.format(get_time(start_time)))
    return x,y_label


def generate_template(x):
    # max_length is the max length of audio in x.
    max_length = -1

    # max_length_index is the index of max length audio.
    max_length_index = 0
    template = None
    for index, _x in enumerate(x):
        if _x.shape[0] > max_length:
            max_length = _x.shape[0]
            max_length_index = index

    template = x[max_length_index]
    for index, _x in enumerate(x):
        if index != max_length_index:
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(_x, template, dist='euclidean')
            template = (_x[path[0]] + template[path[1]])/2
            # the dimension of template will arise after previous step,
            # so I will decrease the dimension of template, to keep it to be the same as initial.
            pre_road = -1
            ind = []
            for current_road in path[1]:
                if current_road!=pre_road:
                    ind.append(True)
                else:
                    ind.append(False)
                pre_road = current_road

            template = template[ind]

    return template


def vote(label):
    label = np.array(label)
    _dict = {}
    for l in label:
        if l not in _dict:
            _dict[l] = 1
        else:
            _dict[l] += 1

    return sorted(_dict.items(), key=lambda x: x[1], reverse=True)[0][0]


def test(threshold=100):
    x_train,y_train = load_train(path='dataset/ASR/train')
    # x_train,y_train = sample(x_train, y_train)
    x_test,y_test = load_test(path='dataset/ASR/test')
    y_pred = []
    # print(len(x_train))

    for x in tqdm(x_test):
        distance = distance_test(x, x_train)
        # top = np.argsort(distance)
        # print(top)
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
    # distance = distance_dtw(x_train[0], x_train[1])
    print(x_train[0].shape)
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(x_train[0], x_train[1], dist='euclidean')
    print(cost_matrix.shape)
    print(acc_cost_matrix.shape)
    print('*'*50)
    print(path[0].shape)
    print('*'*50)
    print(path[1].shape)
    plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.show()
    """

    # np.savetxt('dis.csv', X=distance, delimiter=',')
    # print(distance_dtw(x[0], x[1], show=True))
    # print(distance_dtw(x[0], x[5], show=True))