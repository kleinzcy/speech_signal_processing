#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/12 22:24
# @Author  : chuyu zhang
# @File    : GMM-UBM.py
# @Software: PyCharm

import os
from utils.tools import read, get_time
from tqdm import tqdm

# from utils.processing import MFCC
import python_speech_features as psf
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


label_encoder = {}


def load_data(path='dataset/ASR_GMM'):
    """
    load audio file.
    :param path: the dir to audio file
    :return: x  type:list,each element is an audio, y type:list,it is the label of x
    """
    start_time = get_time()
    print("Loading data...")
    speaker_list = os.listdir(path)
    y = []
    x = []
    num = 0
    for speaker in tqdm(speaker_list):
        # encoder the speaker to num
        label_encoder[speaker] = num
        path1 = os.path.join(path, speaker)
        for _dir in os.listdir(path1):
            path2 = os.path.join(path1, _dir)
            for _wav in os.listdir(path2):
                samplerate, audio = read(os.path.join(path2, _wav))
                y.append(num)
                # sample rate is 16000, you can down sample it to 8000, but the result will be bad.
                x.append(audio)

        num += 1
    print("Complete! Spend {:.2f}s".format(get_time(start_time)))
    return x,y


def delta(feat, N=2):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    # padded version of feat
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    for t in range(NUMFRAMES):
        # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator
    return delta_feat


def extract_feature(x, y, is_train=False, feature_type='MFCC'):
    """
    extract feature from x
    :param x: type list, each element is audio
    :param y: type list, each element is label of audio in x
    :param filepath: the path to save feature
    :param is_train: if true, generate train_data(type dict, key is lable, value is feature),
                     if false, just extract feature from x
    :return:
    """
    start_time = get_time()
    print("Extract {} feature...".format(feature_type))
    feature = []
    train_data = {}
    for i in tqdm(range(len(x))):
        # extract mfcc feature based on psf, you can look more detail on psf's website.
        # TODO plp feature
        # mfcc feature, we will add plp later
        if feature_type=='MFCC':
            _feature = psf.mfcc(x[i])
            mfcc_delta = delta(_feature)
            _feature = np.hstack((_feature, mfcc_delta))

            _feature = preprocessing.scale(_feature)
        else:
            # feature_type=='PLP'
            _feature = None
            pass

        # append _feature to feature
        feature.append(_feature)

        if is_train:
            if y[i] in train_data:
                train_data[y[i]] = np.vstack((train_data[y[i]], _feature))
            else:
                train_data[y[i]] = _feature

    print("Complete! Spend {:.2f}s".format(get_time(start_time)))


    if is_train:
        return train_data, feature, y
    else:
        return feature, y


def load_extract(test_size=0.3):
    # combination load_data and extract_feature
    x, y = load_data()
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    # extract feature from train
    train_data, x_train, y_train = extract_feature(x=x_train, y=y_train, is_train=True)
    # extract feature from test
    x_test, y_test = extract_feature(x=x_test,y=y_test)

    return train_data, x_train, y_train, x_test, y_test


def GMM(train, x_train, y_train, x_test, y_test, n_components=16, model=False):
    print("Train GMM-UBM model !")
    # split x,y to train and test
    start_time = get_time()
    # if model is True, it will load model from Model/GMM_MFCC_model.pkl.
    # if False,it will train and save model

    if model:
        print("load model from file...")
        with open("Model/GMM_MFCC_model.pkl", 'rb') as f:
            GMM = pkl.load(f)
        with open("Model/UBM_MFCC_model.pkl", 'rb') as f:
            UBM = pkl.load(f)
    else:
        # speaker_list = os.listdir('dataset/ASR_GMM')
        GMM = []
        ubm_train = None
        flag = False
        # UBM = []
        print("Train GMM!")
        for speaker in tqdm(label_encoder.values()):
            # print(type(speaker))
            # speaker = label_encoder[speaker]
            # GMM based on speaker
            gmm = GaussianMixture(n_components = n_components, covariance_type='diag')
            gmm.fit(train[speaker])
            GMM.append(gmm)
            if flag:
                ubm_train = np.vstack((ubm_train, train[speaker]))
            else:
                ubm_train = train[speaker]
                flag = True

        # UBM based on background
        print("Train UBM!")
        UBM = GaussianMixture(n_components = n_components, covariance_type='diag')
        UBM.fit(ubm_train)
        # UBM.append(gmm)

        with open("Model/GMM_MFCC_model.pkl", 'wb') as f:
            pkl.dump(GMM, f)
        with open("Model/UBM_MFCC_model.pkl", 'wb') as f:
            pkl.dump(UBM, f)

    # train accuracy
    valid = np.zeros((len(x_train), len(GMM)))
    for i in range(len(GMM)):
        for j in range(len(x_train)):
            valid[j, i] = GMM[i].score(x_train[j]) - UBM.score(x_train[j])

    valid = valid.argmax(axis=1)
    acc_train = (valid==np.array(y_train)).sum()/len(x_train)

    # test accuracy
    pred = np.zeros((len(x_test), len(GMM)))
    for i in range(len(GMM)):
        for j in range(len(x_test)):
            pred[j, i] = GMM[i].score(x_test[j]) - UBM.score(x_test[j])

    pred = pred.argmax(axis=1)
    acc = (pred==np.array(y_test)).sum()/len(x_test)

    print("spend {:.2f}s, train acc {:.2%}, test acc {:.2%}".format(get_time(start_time), acc_train,acc))


def main():
    train_data, mfcc_x_train, mfcc_y_train, mfcc_x_test, mfcc_y_test = load_extract()
    GMM(train_data, mfcc_x_train, mfcc_y_train, mfcc_x_test, mfcc_y_test, model=False)


if __name__=='__main__':
    main()