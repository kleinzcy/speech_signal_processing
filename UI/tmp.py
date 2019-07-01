# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'final.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from d_vector import nn_model
from keras.models import load_model
import pickle as pkl
import numpy as np
from scipy.spatial.distance import cosine
from sidekit.frontend.features import plp,mfcc
from sklearn import preprocessing
from utils.tools import record,read
from tqdm import tqdm
import os
import python_speech_features as psf
#TODO 测试录音数据
#TODO 实验报告
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        #TODO 播放条
        #TODO 上下拉升问题
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(739, 583)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        MainWindow.setAnimated(True)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        # verticalLayout_1这是groupbox和pushbutton垂直排列
        self.verticalLayout_1 = QtWidgets.QVBoxLayout()
        self.verticalLayout_1.setSpacing(40)
        self.verticalLayout_1.setObjectName("verticalLayout_1")

        # verticalLayout_5是三个groupbox横向排列
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")

        # groupbox_3
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_7.setObjectName("radioButton_7")
        self.horizontalLayout_2.addWidget(self.radioButton_7)
        self.radioButton_8 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_8.setObjectName("radioButton_8")
        self.horizontalLayout_2.addWidget(self.radioButton_8)

        self.verticalLayout_5.addWidget(self.groupBox_3)

        # horizontalLayout这是两个groupbox横向排列
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, -1, 0, -1)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # groupbox
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.groupBox.setObjectName("groupBox")

        # verticalLayout_3这是groupbox内垂直排列
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_3.setSpacing(10)

        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout_3.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout_3.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setObjectName("radioButton_3")
        self.verticalLayout_3.addWidget(self.radioButton_3)
        self.horizontalLayout.addWidget(self.groupBox)

        # groupbox_2
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.groupBox_2.setObjectName("groupBox_2")

        # verticalLayout_2这是groupbox_2内垂直排列
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_2.setSpacing(10)

        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setObjectName("radioButton_4")
        self.verticalLayout_2.addWidget(self.radioButton_4)
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_5.setObjectName("radioButton_5")
        self.verticalLayout_2.addWidget(self.radioButton_5)
        self.radioButton_6 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_6.setObjectName("radioButton_6")
        self.verticalLayout_2.addWidget(self.radioButton_6)
        self.horizontalLayout.addWidget(self.groupBox_2)

        self.verticalLayout_5.addLayout(self.horizontalLayout)

        self.verticalLayout_1.addLayout(self.verticalLayout_5)

        # verticalLayout这是三个pushbutton垂直排列
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setSpacing(50)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)

        self.verticalLayout_1.addLayout(self.verticalLayout)

        self.verticalLayout_1.setStretch(1, 4)

        self.gridLayout.addLayout(self.verticalLayout_1, 0, 0, 1, 1)

        # verticalLayout_4这是三个文本框的垂直排列
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setContentsMargins(-1, 0, -1, -1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(1080, 960))
        self.label.setSizeIncrement(QtCore.QSize(1, 1))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("img/1.jpg"))
        # self.label.setPixmap(QtGui.QPixmap(":/newPrefix/1.jpg"))
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        
        self.verticalLayout_4.addWidget(self.label)


        self.playButton = QtWidgets.QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(MainWindow.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        self.positionSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.positionSlider.setRange(0, 0)

        controlLayout = QtWidgets.QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        self.verticalLayout_4.addLayout(controlLayout)

        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        # self.textBrowser.setReadOnly(True)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_4.addWidget(self.textBrowser)

        self.gridLayout.addLayout(self.verticalLayout_4, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 739, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.signal(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "说护者识别"))
        self.groupBox.setTitle(_translate("MainWindow", "说话人识别"))
        self.radioButton.setText(_translate("MainWindow", "NN"))
        self.radioButton_2.setText(_translate("MainWindow", "GRU"))
        self.radioButton_3.setText(_translate("MainWindow", "LSTM"))
        self.groupBox_2.setTitle(_translate("MainWindow", "语种识别"))
        self.radioButton_4.setText(_translate("MainWindow", "MFCC"))
        self.radioButton_5.setText(_translate("MainWindow", "PLP"))
        self.radioButton_6.setText(_translate("MainWindow", "MFCC+PLP"))
        self.groupBox_3.setTitle(_translate("MainWindow", "mode"))
        self.radioButton_7.setText(_translate("MainWindow", "说话人识别"))
        self.radioButton_8.setText(_translate("MainWindow", "语种识别"))
        self.pushButton.setText(_translate("MainWindow", "Enroll"))
        self.pushButton_2.setText(_translate("MainWindow", "Record"))
        self.pushButton_3.setText(_translate("MainWindow", "Test"))
        self.textBrowser.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'Adobe 宋体 Std L\'; font-size:15pt; font-weight:400; font-style:normal;\">\n"))

    def signal(self, MainWindow):
        self.radioButton.clicked.connect(MainWindow.nn_model)
        self.radioButton_2.clicked.connect(MainWindow.gru_model)
        self.radioButton_3.clicked.connect(MainWindow.lstm_model)
        self.radioButton_4.clicked.connect(MainWindow.mfcc_fea)
        self.radioButton_5.clicked.connect(MainWindow.plp_fea)
        self.radioButton_6.clicked.connect(MainWindow.mfcc_plp_fea)
        self.radioButton_7.clicked.connect(MainWindow.speaker_model)
        self.radioButton_8.clicked.connect(MainWindow.language_model)

        self.pushButton.clicked.connect(MainWindow.enroll)
        self.pushButton_2.clicked.connect(MainWindow.records)
        self.pushButton_3.clicked.connect(MainWindow.test)

        # self.playButton.clicked.connect(MainWindow.play)
        # self.positionSlider.sliderMoved.connect(MainWindow.setPosition)

    def speaker_model(self):
        self.textBrowser.clear()
        self.textBrowser.append('Speaker model')
        self.state_speaker = True
        self.state_language = False

    def language_model(self):
        self.textBrowser.clear()
        self.textBrowser.append('Language model')
        self.state_speaker = False
        self.state_language = True

    def nn_model(self):
        self._load_speaker_model(model_type='nn')

    def gru_model(self):
        self._load_speaker_model(model_type='gru')

    def lstm_model(self):
        self._load_speaker_model(model_type='lstm')

    def _load_speaker_model(self, model_type='lstm'):
        if self.state_speaker:
            self.textBrowser.clear()
            self.textBrowser.append('Loading {} d-vector model'.format(model_type))
            self.model_type = model_type
            self.model = load_model('feature/d_vector/d_vector_{}.h5'.format(model_type))
            self.textBrowser.append('finished!')

    def mfcc_fea(self):
        self._load_language_model(feature_type='MFCC')

    def plp_fea(self):
        self._load_language_model(feature_type='PLP')

    def mfcc_plp_fea(self):
        self._load_language_model(feature_type='MFCC_PLP')

    def _load_language_model(self, feature_type='MFCC'):
        if self.state_language:
            self.textBrowser.clear()
            self.textBrowser.append('Loading GMM-UBM {} model'.format(feature_type))
            self.feature_type = feature_type
            with open("feature/language/GMM_8_" + feature_type + "_model.pkl", 'rb') as f:
                self.GMM = pkl.load(f)
            with open("feature/language/UBM_8_" + feature_type + "_model.pkl", 'rb') as f:
                self.UBM = pkl.load(f)

            self.textBrowser.append('finished!')

    def test(self):
        if self.state_language:
            # 语种识别
            self._GMM_test()
        elif self.state_speaker:
            # 说话人识别
            self._dvector_test()

    def records(self):
        seconds, ok = QtWidgets.QInputDialog.getText(self, "records",
                                                   "please input how much seconds:", QtWidgets.QLineEdit.Normal)
        self.textBrowser.append('{}s'.format(seconds))
        record(seconds=int(seconds))
        self.textBrowser.append('{} seconds record has completed.'.format(seconds))
        sample, audio = read(filename='test.wav')
        # print("the length of audio:", str(len(audio)))
        testone = []
        testone_feature = []
        sample = 16000
        for i in range(len(audio) // sample):
            testone.append(audio[i * sample:(i + 1) * sample])
        if self.state_language:
            for i in tqdm(range(len(testone))):
                try:
                    _feature = None
                    if self.feature_type == 'MFCC':
                        _feature = mfcc(testone[i])[0]
                    elif self.feature_type == 'PLP':
                        _feature = plp(testone[i])[0]
                    elif self.feature_type == 'MFCC_PLP':
                        _feature1 = mfcc(testone[i])[0]
                        _feature2 = plp(testone[i])[0]
                        _feature = np.hstack((_feature1, _feature2))

                    _feature = preprocessing.scale(_feature)
                except ValueError:
                    continue
                testone_feature.append(_feature)
            os.remove('test.wav')
            self.gmm_feature = testone_feature
        elif self.state_speaker:
            for i in tqdm(range(len(testone))):
                try:
                    # fs=8000会出错
                    _feature = mfcc(testone[i])[0]
                except :
                    continue
                testone_feature.append(_feature)
            self.d_vector_feature = testone_feature

    def _GMM_test(self):
        testone = self.gmm_feature
        pred = np.zeros((len(testone), len(self.GMM)))
        for i in range(len(self.GMM)):
            for j in range(len(testone)):
                pred[j, i] = self.GMM[i].score(testone[j]) - self.UBM.score(testone[j])

        prob = np.exp(pred.max(axis=1))/np.exp(pred).sum(axis=1)
        prob_str = []
        for p in prob:
            prob_str.append('{:.2%}'.format(p))
        print(prob)
        pred = pred.argmax(axis=1)
        res = []
        for i in range(pred.shape[0]):
            if pred[i]==0:
                res.append('Chinese')
            elif pred[i]==1:
                res.append('English')
            else:
                res.append('Japanese')
        self.textBrowser.append('The result is: ')
        self.textBrowser.append(' '.join(res))
        self.textBrowser.append(' '.join(prob_str))

    def _dvector_test(self):
        # d_vector_feature是一个列表，每一个元素存储一秒语音的特征。
        with open('feature/d_vector/d_vector.pkl', 'rb') as f:
            d_vector = pkl.load(f)

        pred = []
        target = np.array(self.d_vector_feature)
        for i in range(len(self.d_vector_feature)):
            # 根据对应的模型，调整输入格式
            if self.model_type=='lstm':
                pass
            elif self.model_type=='nn':
                target = target.reshape(target.shape[0], -1)
            else:
                target = target[:,:,:,np.newaxis]

        target = self.model.predict(target)

        # TODO 后续增加概率计算
        prob = []
        for i in range(target.shape[0]):
            min_distance = 1
            target_name = None
            distance_list = []
            for name in d_vector.keys():
                distance_list.append(cosine(target[i,:], d_vector[name]))
                if min_distance > distance_list[-1]:
                    min_distance = distance_list[-1]
                    target_name = name
            pred.append(target_name)
            distance = -np.array(distance_list)
            prob.append('{:.2%}'.format(np.exp(distance.max())/np.exp(distance).sum()))
        self.textBrowser.append('The result is :')
        self.textBrowser.append(' '.join(pred))
        self.textBrowser.append(' '.join(prob))

    def enroll(self):
        """
        注册一个陌生人到库中，以字典形式保存
        :param X_train: 样本语音
        :param name: 该样本语音的人名，唯一标识，不可重复。
        :param model_name: 使用模型的名字，nn，lstm，gru
        :return: none
        """
        name, ok = QtWidgets.QInputDialog.getText(self, "Enroll",
                                                   "please input your name:", QtWidgets.QLineEdit.Normal)
        self.textBrowser.append('your name is {}'.format(name))
        self.records()
        try:
            with open('feature/d_vector/d_vector.pkl', 'rb') as f:
                d_vector = pkl.load(f)
        except:
            d_vector = {}

        target = np.array(self.d_vector_feature)
        for i in range(len(self.d_vector_feature)):
            # 根据对应的模型，调整输入格式
            if self.model_type=='lstm':
                pass
            elif self.model_type=='nn':
                target = target.reshape(target.shape[0], -1)
            else:
                target = target[:,:,:,np.newaxis]

        target = self.model.predict(target)
        print(target.shape)
        if name in d_vector:
            self.textBrowser.append('your name is already exist')
            d_vector[name] = (d_vector[name] + target.mean(axis=0))/2
        else:
            d_vector[name] = target.mean(axis=0)

        with open('feature/d_vector/d_vector.pkl', 'wb') as f:
            pkl.dump(d_vector, f)

        self.textBrowser.append('Finished')