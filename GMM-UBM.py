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
import python_speech_features as psf
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_data(path='dataset/ASR_GMM'):
    """
    load audio file.
    :param path: the dir to audio file
    :return: x  type:list,each element is an audio, y type:list,it is the label of x
    """
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
                samplerate, audio = read(os.path.join(path2, _wav))
                # speaker is id10270,id10271 etc.
                y.append(speaker)
                # sample rate is 16000, you can down sample it to 8000, but the result will be bad.
                x.append(audio)
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


def extract_feature(x, y, filepath='feature/MFCC.pkl'):
    start_time = get_time()
    print("Extract MFCC feature...")
    flag = False
    feature_mfcc_x = None
    feature_mfcc_y = None
    for i in tqdm(range(len(x))):
        # extract mfcc feature based on psf, you can look more detail on psf's website.
        # TODO plp feature
        # mfcc feature, we will add plp later
        _feature = psf.mfcc(x[i])
        mfcc_delta = delta(_feature)
        _feature = np.hstack((_feature, mfcc_delta))

        # print(mfcc_delta.shape)
        # normalize feature based on preprocessing scale
        _feature = preprocessing.scale(_feature)
        label = np.zeros((_feature.shape[0], 1))
        # _y is speaker id, like id10270, transform it to 10270
        label[:,0] = int(y[i][2:])

        if flag:
            feature_mfcc_x = np.vstack((feature_mfcc_x, _feature))
            feature_mfcc_y = np.vstack((feature_mfcc_y, label))
        else:
            feature_mfcc_x = _feature
            feature_mfcc_y = label
            flag = True

    # save feature in filepath
    with open(filepath, 'wb') as f:
        pkl.dump(np.hstack((feature_mfcc_x, feature_mfcc_y)), f)

    print("Complete! Spend {:.2f}s".format(get_time(start_time)))

    return feature_mfcc_x, feature_mfcc_y.reshape(-1,)


def load_extract(file=False):
    # combination load_data and extract_feature
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


def GMM(x, y, test_size=0.3, n_components=16, model=False):
    print("Train GMM model !")
    # split x,y to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
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
    mfcc_x, mfcc_y = load_extract(file=False)
    GMM(mfcc_x, mfcc_y, model=False)