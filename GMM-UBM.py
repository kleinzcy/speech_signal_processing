#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/12 22:24
# @Author  : chuyu zhang
# @File    : GMM-UBM.py
# @Software: PyCharm

import os
from utils.tools import read, get_time
from tqdm import tqdm

from utils.processing import MFCC

import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_data(path='dataset/ASR_GMM'):
    start_time = get_time()
    print("Load data...")
    speaker_list = os.listdir(path)
    y = []
    x = []
    for speaker in tqdm(speaker_list):
        path1 = os.path.join(path, speaker)
        for _dir in os.listdir(path1):
            path2 = os.path.join(path1, _dir)
            for _wav in os.listdir(path2):
                # samprate is 16000, downsample audio to 8000
                samprate, audio = read(os.path.join(path2, _wav))
                y.append(speaker)
                x.append(audio[range(0, audio.shape[0], 2)])
    print("Complete! Spend {:.2f}s".format(get_time(start_time)))
    return x,y


def extract_feature(x, y, filepath='feature/MFCC.pkl'):
    start_time = get_time()
    print("Extract MFCC feature...")
    flag = False
    feature_mfcc_x = None
    feature_mfcc_y = None
    train_dict = {}
    for _x,_y in zip(x,y):
        _feature = MFCC(_x)
        # normalize feature
        _feature = preprocessing.scale(_feature)
        label = np.zeros((_feature.shape[0], 1))
        # print(_feature.shape)
        label[:,0] = int(_y[2:])
        if flag:
            feature_mfcc_x = np.vstack((feature_mfcc_x, _feature))
            feature_mfcc_y = np.vstack((feature_mfcc_y, label))
        else:
            feature_mfcc_x = _feature
            feature_mfcc_y = label
            flag = True

    with open(filepath, 'wb') as f:
        pkl.dump(np.hstack((feature_mfcc_x, feature_mfcc_y)), f)
    print("Complete! Spend {:.2f}s".format(get_time(start_time)))

    return feature_mfcc_x, feature_mfcc_y


def load_extract(file=False):
    if file:
        start_time = get_time()
        print("Load MFCC feature from file...")
        with open('feature/MFCC.pkl', 'rb') as f:
            train = pkl.load(f)
            mfcc_x = train[:, :-1]
            mfcc_y = train[:, -1]
            del train
        print("spend {:.2f}s".format(get_time(start_time)))
    else:
        x, y = load_data()
        mfcc_x, mfcc_y = extract_feature(x=x, y=y)

    return mfcc_x,mfcc_y


def GMM(x, y, test_size=0.3, n_components=256, model=False):
    # split x,y to train and test
    print("Train GMM model !")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    start_time = get_time()
    if model:
        print("load model from file...")
        with open("Model/GMM_MFCC_model.pkl", 'rb') as f:
            GMM = pkl.load(f)
        with open("Model/UBM_MFCC_model.pkl", 'rb') as f:
            UBM = pkl.load(f)
    else:
        speaker_list = os.listdir('dataset/ASR_GMM')
        GMM = []
        UBM = []
        for speaker in tqdm(speaker_list):
            speaker = int(speaker[2:])
            # GMM based on speaker
            gmm = GaussianMixture(n_components = n_components, max_iter = 200, covariance_type='diag',n_init = 3)
            gmm.fit(x_train[y_train==speaker])
            GMM.append(gmm)
            # UBM based on background
            gmm = GaussianMixture(n_components = n_components, max_iter = 200, covariance_type='diag',n_init = 3)
            gmm.fit(x_train[y_train!=speaker])
            UBM.append(gmm)

        with open("Model/GMM_MFCC_model.pkl", 'wb') as f:
            pkl.dump(GMM, f)
        with open("Model/UBM_MFCC_model.pkl", 'wb') as f:
            pkl.dump(UBM, f)
    # train accuracy
    valid = np.zeros((x_train.shape[0], len(GMM)))
    for i in range(len(GMM)):
        valid[:, i] = GMM[i].score_samples(x_train) - UBM[i].score_samples(x_train)

    valid = valid.argmax(axis=1)+10270
    acc_train = (valid==y_train).sum()/y_train.shape[0]

    # test accuracy
    pred = np.zeros((x_test.shape[0], len(GMM)))
    for i in range(len(GMM)):
        pred[:, i] = GMM[i].score_samples(x_test) - UBM[i].score_samples(x_test)

    pred = pred.argmax(axis=1)+10270
    acc = (pred==y_test).sum()/y_test.shape[0]

    print("spend {:.2f}s, train acc {:.2%}, test acc {:.2%}".format(get_time(start_time), acc_train,acc))


if __name__=='__main__':
    mfcc_x, mfcc_y = load_extract(file=True)
    GMM(mfcc_x, mfcc_y, model=False)

    # print(len(x), len(y))

