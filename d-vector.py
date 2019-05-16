#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 22:56
# @Author  : chuyu zhang
# @File    : d-vector.py
# @Software: PyCharm

#TODO 明天完成d-vector的基本结构。
import os
import pickle as pkl
import numpy as np
from utils.tools import read, get_time
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sidekit.frontend.features import plp,mfcc

from keras.layers import Dense, Activation, Dropout, Input
from keras.optimizers import Adam
from keras.models import Model,Sequential,load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,CSVLogger
from keras import regularizers


label_encoder = {}
class Data_gen:
    # 生成数据
    def __init__(self, path='dataset/ASR_GMM_big'):
        self.path = path

    def _load(self):
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

    def extract_feature(self, sample_rate=16000, feature_type='MFCC'):
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
        if not os.path.exists('feature'):
            os.mkdir('feature')

        if not os.path.exists('feature/{}_feature.pkl'.format(feature_type)):
            x, y = self._load()
            print("Extract {} feature...".format(feature_type))
            feature = []
            label = []
            for i in tqdm(range(len(x))):
                # 这里MFCC和PLP默认是16000Hz，注意修改
                # mfcc 25ms窗长，10ms重叠
                if feature_type == 'MFCC':
                    _feature = mfcc(x[i], fs=sample_rate)[0]
                elif feature_type == 'PLP':
                    _feature = plp(x[i], fs=sample_rate)[0]
                else:
                    raise NameError
                # 特征出了问题，需要特殊处理才行
                if np.isnan(_feature).sum()>0:
                    continue
                # _feature = np.concatenate([_feature,self.delta(_feature)],axis=1)
                # _feature = preprocessing.scale(_feature)
                # _feature = preprocessing.StandardScaler().fit_transform(_feature)
                # 每2*num为一个输入，并且重叠num
                num = 20
                for j in range(_feature.shape[0]//num-1):
                    feature.append(_feature[j*num:j*num+2*num])
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
    def save(data, file_name):
        with open('feature/{}.pkl'.format(file_name), 'wb') as f:
            pkl.dump(data, f)

    @staticmethod
    def load(file_name):
        with open('feature/{}.pkl'.format(file_name), 'rb') as f:
            return pkl.load(f)

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


class nn_model:
    def __init__(self, n_class=10):
        self.n_class = n_class

    def model(self, X_train, Y_train, X_val, Y_val):
        #需要修改input_shape等一些参数
        print("Training model")
        model = Sequential()

        model.add(Dense(256, input_shape=(X_train.shape[1],), name="dense1",
                        kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
        model.add(Activation('relu', name="activation1"))
        model.add(Dropout(0.5, name="drop1"))

        model.add(Dense(256, name="dense2",
                        kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
        model.add(Activation('relu', name="activation2"))
        model.add(Dropout(0.5, name="drop2"))

        model.add(Dense(256, name="dense3"))

        modelInput = Input(shape=(X_train.shape[1],))
        features = model(modelInput)
        spkModel = Model(inputs=modelInput, outputs=features)

        model1 = Activation('relu')(features)
        model1 = Dropout(0.5)(model1)
        model1 = Dense(self.n_class, activation='softmax')(model1)

        spk = Model(inputs=modelInput, outputs=model1)


        sgd = Adam(lr=1e-4)
        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
        csv_logger = CSVLogger('feature/d_vector/nn_training.log')

        spk.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

        spk.fit(X_train, Y_train, batch_size = 128, epochs=2, validation_data = (X_val, Y_val),
                            callbacks=[early_stopping, reduce_lr, csv_logger])

        if not os.path.exists('feature/d_vector'):
            os.mkdir('feature/d_vector')
        # spk.save('spk.h5')
        spkModel.save('feature/d_vector/d_vector_nn.h5')

        X_train = spkModel.predict(X_train)
        X_val = spkModel.predict(X_val)
        self.data = [X_train, Y_train, X_val, Y_val]

    def test(self):
        # model = load_model('feature/d_vector/d_vector_nn.h5')
        X_train, Y_train, X_val, Y_val = self.data
        # feature = model.predict(X_train)

        return feature


if __name__=="__main__":
    data_gen = Data_gen()
    feature, label = data_gen.extract_feature()
    feature = np.array(feature)
    # 由于是全连接,故需要reshape
    feature = feature.reshape(feature.shape[0], -1)
    label = np.array(label).reshape(-1, 1)
    enc = preprocessing.OneHotEncoder()
    label = enc.fit_transform(label).toarray()

    X_train, X_val, y_train, y_val = train_test_split(feature, label, shuffle=True, test_size=0.3)
    model = nn_model()
    model.model(X_train, y_train, X_val, y_val)
    model.test()