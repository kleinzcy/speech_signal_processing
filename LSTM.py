#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time     : 2018/9/13 12:59
# @Author   : klein
# @File     : stock_pre.py
# @Software : PyCharm
import numpy as np
import tensorflow as tf
import warnings
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import time
from sidekit.frontend.features import plp,mfcc
import os
from utils.tools import read, get_time
from tqdm import tqdm
warnings.filterwarnings("ignore")


param = {'epoch': 100, 'lr': 0.001, 'num_feature': 28, 'pre_step': 15,
         'num_units': 32, 'batch_size': 100, 'time_step': 28, 'num_output': 10}

label_encoder = {}


class LSTM:
    """
    many to one model.
    """
    def __init__(self):
        self.lr = param['lr']
        self.bz = param['batch_size']
        self.time_step = param['time_step']

    def inference(self, x):
        """
        build the graph
        :param x: input shape:[batch_size, time_step, num_feature]
        :return: pred according to x
        """
        w_out = self.weight_variable([param['num_units'], param['num_output']])
        b_out = self.bias_variable(shape=[param['num_output'], ])

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_feature)
        # x = tf.unstack(x, param['time_step'], 1)

        cell = tf.nn.rnn_cell.LSTMCell(num_units=param['num_units'])
        init_state = cell.zero_state(param['batch_size'], dtype=tf.float32)
        # output_rnn是记录lstm每个输出节点的结果，shape: [batch_size, time_step, num_unit]
        # final_states是最后一个cell的结果
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, dtype=tf.float32)

        # 取最后一个unit作为最后输出
        output = output_rnn[:,-1,:]
        pred = tf.matmul(output, w_out)+b_out

        return pred

    def train(self, x_train, y_train, x_test, y_test):
        """
        train lstm
        :param x_train: type: ndarray,shape: [n, 28, 28]
        :param y_train: type: ndarray,shape: [n, 10]
        :param x_test: type: ndarray,shape: [n, 28, 28]
        :param y_test: type: ndarray,shape: [n, 10]
        :return:
        """
        x = tf.placeholder(tf.float32, shape=[None, self.time_step, param['num_feature']])
        y = tf.placeholder(tf.float32, shape=[None, param['num_output']])

        # 衰减率
        decay_rate = 0.995
        # 衰减次数
        decay_steps = 10
        # define
        global_ = tf.Variable(tf.constant(0))
        lr = tf.train.exponential_decay(param['lr'], global_, decay_steps, decay_rate, staircase=False)

        self.x = x
        # forward
        pred = self.inference(x)

        # why not put it in inference?
        prediction = tf.nn.softmax(pred)
        self.pred = prediction

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()
        # 开始会话
        sess = tf.Session()
        sess.run(init)

        batch_size = param['batch_size']
        for epoch in range(1,param['epoch']+1):
            sess.run(lr, feed_dict={global_: epoch})

            # train
            start = time.time()
            for j in range(x_train.shape[0]//batch_size):
                sess.run(optimizer, feed_dict={x: x_train[j*batch_size:(j+1)*batch_size],
                                              y: y_train[j*batch_size:(j+1)*batch_size]})
            end = time.time() - start
            print("epoch:{}, time:{}".format(epoch, end))

            # eval
            if epoch % 10 == 0:
                # eval train
                train_accuracyes = []
                start = time.time()
                for j in range(x_train.shape[0]//batch_size):
                    train_accuracy = sess.run(
                            fetches=accuracy,
                            feed_dict={
                                    x: x_train[j*batch_size:(j+1)*batch_size],
                                    y: y_train[j*batch_size:(j+1)*batch_size]
                                }
                    )
                    train_accuracyes.append(train_accuracy)
                end = time.time() - start
                print("epoch:{}, average training accuracy:{:.3%}, time:{}".
                      format(epoch, sum(train_accuracyes) / len(train_accuracyes), end))

                # eval test
                test_accuracyes = []
                start = time.time()
                for j in range(x_test.shape[0]//batch_size):
                    test_accuracy = sess.run(
                            fetches=accuracy,
                            feed_dict={
                                    x: x_test[j*batch_size:(j+1)*batch_size],
                                    y: y_test[j*batch_size:(j+1)*batch_size]
                                }
                    )
                    test_accuracyes.append(test_accuracy)
                end = time.time() - start
                print("epoch:{}, average training accuracy:{:.3%}, time:{}".
                      format(epoch, sum(test_accuracyes) / len(test_accuracyes), end))

            self.sess = sess

    def predict(self, x_pred):
        """
        predict
        :param x_pred: type: ndarray, shape: [n, time_step, num_feature]
        :return: y: type:ndarray, shape: [n, num_output]
        """
        sess = self.sess
        batch_size = self.bz
        y = []
        for i in range(x_pred.shape[0]//self.bz):
            _y = sess.run(self.pred, feed_dict={self.x: x_pred[i*batch_size:(i+1)*batch_size]})
            y.extend(_y)

        y = np.array(y)
        self.close()
        return y

    def close(self):
        self.sess.close()

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
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
        return x, y

    @staticmethod
    def delta(feat, N=2):
        """Compute delta features from a feature vector sequence.
        :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        :param N: For each frame, calculate delta features based on preceding and following N frames
        :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
        """
        if N < 1:
            raise ValueError('N must be an integer >= 1')
        NUMFRAMES = len(feat)
        denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
        delta_feat = np.empty_like(feat)
        # padded version of feat
        padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
        for t in range(NUMFRAMES):
            # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
            delta_feat[t] = np.dot(np.arange(-N, N + 1), padded[t: t + 2 * N + 1]) / denominator
        return delta_feat

    def extract_feature(self, x, y, is_train=False, feature_type='MFCC'):
        """
        extract feature from x
        :param x: type list, each element is audio
        :param y: type list, each element is label of audio in x
        :param filepath: the path to save feature
        :param is_train: if true, generate train_data(type dict, key is lable, value is feature),
                         if false, just extract feature from x
        :return:
        """
        # TODO 以秒为单位提取特征。
        start_time = get_time()
        print("Extract {} feature...".format(feature_type))
        feature = []
        train_data = {}
        for i in tqdm(range(len(x))):
            # extract mfcc feature based on psf, you can look more detail on psf's website.
            if feature_type == 'MFCC':
                _feature = mfcc(x[i])
                mfcc_delta = self.delta(_feature)
                _feature = np.hstack((_feature, mfcc_delta))

                _feature = preprocessing.scale(_feature)
            elif feature_type == 'PLP':
                _feature = plp(x[i])
                mfcc_delta = self.delta(_feature)
                _feature = np.hstack((_feature, mfcc_delta))

                _feature = preprocessing.scale(_feature)
            else:
                raise NameError

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


if __name__ == '__main__':
    pass

