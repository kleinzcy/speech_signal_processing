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

"""
framerate=8000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=2
"""
def wave_read(filename='test.wav'):
    audio = wave.open(filename, mode='rb')
    return audio


def read(filename='test.wav'):
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


def play(filename="test.wav", chunk=1024):
    wf=wave.open(filename,'rb')
    p=PyAudio()
    stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),
                  channels=wf.getnchannels(),rate=wf.getframerate(),
                  output=True)
    start = time.time()
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
    print("{} seconds".format(time.time() - start))


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


if __name__=="__main__":
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
    simple_music(tone='Gsharp')
