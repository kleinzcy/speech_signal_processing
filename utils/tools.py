#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 15:46
# @Author  : chuyu zhang
# @File    : read.py
# @Software: PyCharm

import wave
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
# import math
# from scipy import signal
import numpy as np
from pyaudio import PyAudio, paInt16
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# import sounddevice as sd
import simpleaudio as sa

from utils.processing import enframe

"""
framerate=8000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=2
"""

def get_time(start_time=None):
    if start_time == None:
        return time.time()
    else:
        return time.time() - start_time

## 开发计划，后续添加，play(data),data是read函数的输出
# read .wav file through wave, return a wave object
def wave_read(filename='test.wav'):
    audio = wave.open(filename, mode='rb')
    return audio


# read .wav file through scipy
def read(filename='test.wav'):
    sampling_freq, audio = wavfile.read(filename)
    return sampling_freq, audio


# 根据给定的参数保存.wav文件
def save_wave_file(filename, data, channels=1, sampwidth=2, framerate=8000):
    '''save the data to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)#声道
    wf.setsampwidth(sampwidth)#采样字节 1 or 2
    wf.setframerate(framerate)#采样频率 8000 or 16000
    wf.writeframes(b"".join(data))
    wf.close()


# num_samples这个参数的意义？？
def record(filename="test.wav",seconds=10,
           framerate=8000, format=paInt16, channels=1, num_samples=2000):
    p=PyAudio()
    stream=p.open(format = format, channels=channels, rate=framerate,
                  input=True, frames_per_buffer=num_samples)
    my_buf=[]
    # 控制录音时间
    print("start the recording !")
    start = time.time()
    while time.time() - start < seconds:
        # 一次性录音采样字节大小
        string_audio_data = stream.read(num_samples)
        my_buf.append(string_audio_data)

    save_wave_file(filename, my_buf)
    stream.close()
    print("{} seconds record has completed.".format(seconds))


def play(audio=None, sampling_freq=8000, filename=None, chunk=1024):
    start_time = get_time()
    if filename==None:
        # pass
        play_obj = sa.play_buffer(audio, 1, 2, sampling_freq)
        play_obj.wait_done()

    else:
        wf=wave.open(filename,'rb')
        p=PyAudio()
        stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),
                      channels=wf.getnchannels(),rate=wf.getframerate(),
                      output=True)
        while True:
            data=wf.readframes(chunk)
            # char b is absolutely necessary. It represents that the str is byte.
            # For more detail, please refer to 3,4
            if data == b"":
                break
            stream.write(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
    print("{} seconds".format(get_time(start_time)))


def playback(filename='test.wav', silent=False):
    rate, audio = read(filename)
    new_filename = 'reverse_' + filename
    wavfile.write(new_filename, rate, audio[::-1])
    if not silent:
        play(filename=new_filename)
    print("complete!")


def change_rate(filename='test.wav', new_rate=4000, silent=False):
    rate, audio = read(filename)
    print("the original frequent rate is {}".format(rate))
    new_filename = str(new_rate) + "_" + filename
    wavfile.write(new_filename, new_rate, audio)
    if not silent:
        play(filename=new_filename)
    print("complete !")

def change_volume(filename='test.wav', volume_rate=1, silent=False):
    rate, audio = read(filename)
    print("change volume to {}".format(volume_rate))
    new_filename = str(volume_rate) + "_" + filename
    new_audio = (audio*volume_rate).astype('int16')
    # print(audio.dtype, new_audio.dtype)
    wavfile.write(new_filename, rate, new_audio)
    if not silent:
        play(filename=new_filename)
    print("complete !")

# 定义合成音调
def Synthetic_tone(freq, duration=2, amp=1000, sampling_freq=44100):
    # 建立时间轴
    # scaling_factor = pow(2, 15) - 1  # 转换为16位整型数
    t = np.linspace(0, duration, duration * sampling_freq)
    # 构建音频信号
    audio = amp * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.int16)


def simple_music(tone='A', duration=2, amplitude=10000, sampling_freq=44100):
    tone_freq_map = {'A': 440, 'Asharp': 466, 'B': 494, 'C': 523, 'Csharp': 554,
                    'D': 587, 'Dsharp': 622, 'E': 659, 'F': 698, 'Fsharp': 740,
                    'G': 784, 'Gsharp': 831}

    synthesized_tone = Synthetic_tone(tone_freq_map[tone], duration, amplitude, sampling_freq)
    wavfile.write('{}.wav'.format(tone), sampling_freq, synthesized_tone)
    play('{}.wav'.format(tone))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black")
    fig.tight_layout()
    return ax


def test(filename):
    framerate, audio = read(filename)
    play(audio, framerate)
    #　print(audio.shape)
    audio_frame = enframe(audio[:,0], frameSize=512, step=256)
    audio_frame_new = np.zeros_like(audio_frame)
    for frame in range(audio_frame.shape[1]):
        # print(audio_frame[:, frame])
        audio_fft = fft(audio_frame[:, frame])
        # audio_fft_abs = abs(audio_fft)
        # angle = audio_fft.real/audio_fft_abs
        audio_fft_new = np.sqrt(0.8)*audio_fft
        audio_new = ifft(audio_fft_new)
        # print(audio_new)
        audio_frame_new[:, frame] = audio_new.real
        print((audio_new-audio_frame[:, frame]).sum())
        # break
    audio_new = audio_frame_new[:256, :].flatten('F')[:audio.shape[0]].reshape(-1, 1)
    audio_new = np.concatenate([audio_new, audio_new], axis=1)
    save_wave_file('zero_trans.wav', audio_new)
    print(audio_new.shape)
    # print(audio_new.astype(audio.dtype))
    # audio_fft_abs = np.abs(audio_fft)
    # angle = audio_fft.real()/audio_fft_abs
    # audio_fft_abs_new = audio_fft_abs*0.8
    # au
    play(audio_new.astype(audio.dtype), framerate)
    plt.figure()
    plt.plot(audio[:,0])
    plt.figure()
    plt.plot(audio_new[:,0])
    plt.show()




if __name__=="__main__":
    test(filename='../dataset/ASR/test/zcy/zcy1.wav')
    """
    wave_read = wave.open('../dataset/ASR/test/zcy/zcy1.wav', 'rb')
    audio_data = wave_read.readframes(wave_read.getnframes())
    num_channels = wave_read.getnchannels()
    bytes_per_sample = wave_read.getsampwidth()
    sample_rate = wave_read.getframerate()
    print(type(audio_data))
    play_obj = sa.play_buffer(audio_data, num_channels, bytes_per_sample, sample_rate)
    """

    # record()
    # playback(filename='test.wav')
    # rate, audio = read(filename='01.wav')
    # print(rate, audio)
    # print(audio[::-1])
    # change_volume(volume_rate=1)
    """
    duration = 4
    music = 0.9*Synthetic_tone(freq=440, duration=duration) + \
            0.75*Synthetic_tone(freq=880, duration=duration)
    wavfile.write('music.wav', 44100, music.astype('int16'))
    play(filename='music.wav')

    """
    # change_rate('../dataset/ASR/train/hyy/hyy1.wav', new_rate=8000)
    # simple_music(tone='Gsharp')
    # framerate, audio = read('../dataset/ASR/train/hyy/hyy1.wav')
    # downsample = audio[range(0, audio.shape[0], 2), 0]
    # save_wave_file('test.wav', downsample)
    # play(filename='test.wav')
    # play(downsample, 8000)
