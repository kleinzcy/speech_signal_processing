#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/23 22:18
# @Author  : chuyu zhang
# @File    : processing.py
# @Software: PyCharm

import numpy as np
import math
from scipy import signal
from scipy.io import wavfile
# import read,play
from scipy.fftpack.realtransforms import dct
from scipy.fftpack import fft
import matplotlib.pyplot as plt

eps = 1e-8
## 语音的预处理函数
def enframe(wavData, frameSize=400, step=160):
    """
    frame the wav data, according to frameSize and overlap
    :param wavData: the input wav data, ndarray
    :return:frameData, shape
    """
    coef = 0.97
    wlen = wavData.shape[0]
    frameNum = math.ceil(wlen / step)
    frameData = np.zeros((frameSize, frameNum))

    window = signal.windows.hamming(frameSize)

    for i in range(frameNum):
        singleFrame = wavData[i * step : min(i * step + frameSize, wlen)]
        # singleFrame[1:] = singleFrame[:-1] - coef * singleFrame[1:]
        frameData[:len(singleFrame), i] = singleFrame
        frameData[:, i] = window*frameData[:, i]

    return frameData


# frequency domain feature
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

def test():
    path = '../dataset/ASR/zcy/zcy1.wav'
    frameSize = 400
    nFFT = int(frameSize/2)
    fs, audio = wavfile.read(path)
    audio = audio[:, 0]

    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)
    n_mfcc_feats = 13
    x = audio[4000:4000+frameSize]
    X = abs(fft(x))  # get fft magnitude
    X = X[0:nFFT]  # normalize fft
    X = X / len(X)
    feature = stMFCC(X, fbank, n_mfcc_feats)
    plt.figure()
    plt.subplot(121)
    plt.plot(X)
    plt.subplot(122)
    plt.plot(feature)
    plt.show()


if __name__=='__main__':
    test()

    """
    path = '../dataset/ASR/zcy/zcy1.wav'
    play(path)
    samprate, data = read(path)
    framedata = enframe(data[:,0])
    print(framedata.shape)
    """
