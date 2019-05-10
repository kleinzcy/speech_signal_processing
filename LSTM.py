#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Time     : 2018/9/13 12:59
# @Author   : klein
# @File     : lstm.py
# @Software : PyCharm

import os
import pickle as pkl
from utils.tools import read, get_time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
from sidekit.frontend.features import plp,mfcc
from sidekit.frontend import vad
import warnings
warnings.filterwarnings("ignore")


label_encoder = {}

class LSTM:
    """
    many to one model.
    """
    def __init__(self, param, path='dataset/ASR_GMM_big'):
        self.lr = param['lr']
        self.bz = param['batch_size']
        self.time_step = param['time_step']
        self.path = path

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
        decay_rate = 0.99
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
        # TODO 为什么训练结果不变
        for epoch in range(1,param['epoch']+1):
            sess.run(lr, feed_dict={global_: epoch})

            # train
            start = time.time()
            acc = []
            for j in range(x_train.shape[0]//batch_size):
                _, _acc = sess.run([optimizer,accuracy], feed_dict={x: x_train[j*batch_size:(j+1)*batch_size],
                                              y: y_train[j*batch_size:(j+1)*batch_size]})
                acc.append(_acc)
            end = time.time() - start
            print("epoch:{}, time:{:.2f}s, acc:{:.2%}".format(epoch, end, sum(acc) / len(acc)))

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
                print("epoch:{}, average training accuracy:{:.3%}, time:{:.2f}s".
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
                print("epoch:{}, average test accuracy:{:.3%}, time:{:.2f}s".
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

    def load_data(self):
        """
        load audio file.
        :param path: the dir to audio file
        :return: x  type:list,each element is an audio, y type:list,it is the label of x
        """
        start_time = get_time()
        path = self.path
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

    def extract_feature(self, feature_type='MFCC'):
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
        if not os.path.exists('feature'):
            os.mkdir('feature')

        if not os.path.exists('feature/{}_feature.pkl'.format(feature_type)):
            x, y = self.load_data()
            print("Extract {} feature...".format(feature_type))
            feature = []
            label = []
            for i in tqdm(range(len(x))):
                # 这里MFCC和PLP默认是16000Hz，注意修改
                # mfcc 25ms窗长，10ms重叠
                if feature_type == 'MFCC':
                    _feature = mfcc(x[i])[0]
                elif feature_type == 'PLP':
                    _feature = plp(x[i])[0]
                else:
                    raise NameError

                # _feature = self.delta(_feature)
                # TODO 兼容i-vector 和 d-vector
                # _feature = preprocessing.scale(_feature)
                num = 100
                for j in range(_feature.shape[0]//num):
                    feature.append(_feature[j*num:(j+1)*num])
                    label.append(y[i])
            print(len(feature), feature[0].shape)
            self.save(feature, '{}_feature'.format(feature_type))
            self.save(label, '{}_label'.format(feature_type))

        else:
            feature = self.load('{}_feature'.format(feature_type))
            label = self.load('{}_label'.format(feature_type))

        print("Complete! Spend {:.2f}s".format(get_time(start_time)))
        return feature, label

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def save(data, file_name):
        with open('feature/{}.pkl'.format(file_name), 'wb') as f:
            pkl.dump(data, f)

    @staticmethod
    def load(file_name):
        with open('feature/{}.pkl'.format(file_name), 'rb') as f:
            return pkl.load(f)


if __name__ == '__main__':
    param = {'epoch': 100, 'lr': 1e-1, 'num_feature': 13, 'pre_step': 15,
             'num_units': 16, 'batch_size': 50, 'time_step': 100, 'num_output': 10}
    model = LSTM(param=param)
    feature, label = model.extract_feature()
    feature = np.array(feature)
    label = np.array(label).reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    # 这里如果不toarray的话，得到的是一个csr矩阵
    label = enc.fit_transform(label).toarray()
    X_train, X_test, y_train, y_test = train_test_split(feature, label, shuffle=True, test_size=0.3)
    print(len(X_train), X_train[0].shape)
    model.train(X_train, y_train, X_test, y_test)

