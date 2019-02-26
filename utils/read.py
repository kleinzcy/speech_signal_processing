#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 15:46
# @Author  : chuyu zhang
# @File    : read.py
# @Software: PyCharm

import wave
from scipy.io import wavfile
import numpy as np
from pyaudio import PyAudio, paInt16
import time

# https://www.cnblogs.com/LXP-Never/p/10078200.html
"""
framerate=8000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=2
"""


def wave_read(filename=None):
    audio = wave.open(filename, mode='rb')
    return audio


def read(filename=None):
    sampling_freq, audio = wavfile.read(filename)
    return sampling_freq, audio


def save_wave_file(filename, data, channels=1, sampwidth=2, framerate=8000):
    '''save the data to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)#声道
    wf.setsampwidth(sampwidth)#采样字节 1 or 2
    wf.setframerate(framerate)#采样频率 8000 or 16000
    wf.writeframes(b"".join(data))
    wf.close()


def record(filename="01.wav" ,framerate=8000, format=paInt16, channels=1, num_samples=2000, seconds=2):
    p=PyAudio()
    stream=p.open(format = format,channels=channels,
                   rate=framerate,input=True,
                   frames_per_buffer=num_samples)
    my_buf=[]
    # 控制录音时间
    start = time.time()
    while time.time() - start < seconds:
        # 一次性录音采样字节大小
        string_audio_data = stream.read(num_samples)
        my_buf.append(string_audio_data)

    save_wave_file(filename, my_buf)
    stream.close()
    print("{} seconds record has completed.".format(seconds))


def play(filename="01.wav", chunk=1024):
    wf=wave.open(filename,'rb')
    p=PyAudio()
    stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=
    wf.getnchannels(),rate=wf.getframerate(),output=True)
    start = time.time()
    while True:
        data=wf.readframes(chunk)
        if data=="":
            break
        stream.write(data)
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()
    print("{} seconds".format(time.time() - start))


if __name__=="__main__":
    # play()
    rate, audio = read(filename='01.wav')
    print(rate, audio)
    play()
