#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 22:08
# @Author  : chuyu zhang
# @File    : MFCC_DTW.py
# @Software: PyCharm

import os
from utils.tools import read
from utils.processing import enframe
from sklearn.preprocessing import OrdinalEncoder

import numpy as np
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct


eps = 1e-8
def load_wav(path='dataset/ASR'):
    """
    load data from dataset/ASR
    :param path: the path of dataset
    :return: x is train data, y_label is the label of x
    """
    wav_dir = os.listdir(path)
    y_label = []
    x = []
    enc = OrdinalEncoder()
    for _dir in wav_dir:
        for _path in os.listdir(os.path.join(path, _dir)):
            _, data = read(os.path.join(path, _dir, _path))
            x.append(data[:, 0])
            y_label.append(_path.split('.')[0][:-1])
            del data

    y_label = np.array(y_label)
    y_label = enc.fit_transform(y_label.reshape(-1, 1))
    return x,y_label

def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """
    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = np.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** np.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nFiltTotal, nfft))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1,
                           np.floor(cenTrFreq * nfft / fs) + 1,
                                       dtype=np.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1,
                                       np.floor(highTrFreq * nfft / fs) + 1,
                                       dtype=np.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def stMFCC(X, fbank, n_mfcc_feats):
    """
    Computes the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the
             scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more
         compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = np.log10(np.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:n_mfcc_feats]
    return ceps

def MFCC(signal, fs, win, step):
    nFFT = int(win / 2)
    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)
    n_mfcc_feats = 13

    feature = []
    for x in signal:
        X = abs(fft(x))  # get fft magnitude
        X = X[0:nFFT]  # normalize fft
        X = X / len(X)
        feature.append(stMFCC(X, fbank, n_mfcc_feats))

if __name__=='__main__':
    x,y = load_wav()
    print(len(x), x[0].shape, y)