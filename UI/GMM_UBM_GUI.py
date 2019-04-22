# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GMM_UBM_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import pickle as pkl
import os
from playsound import playsound
from GMM_UBM import delta
from utils.tools import record,read
import numpy as np
from sidekit.frontend.features import plp,mfcc
from sklearn import preprocessing
from aip import AipSpeech

import warnings
warnings.filterwarnings("ignore")

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(300, 343)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(70, 20, 160, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioButton_2 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout.addWidget(self.radioButton_2)
        self.radioButton = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout.addWidget(self.radioButton)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(100, 120, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(100, 180, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(20, 220, 256, 111))
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Form)
        self.radioButton_2.clicked.connect(Form.load_plp)
        self.radioButton.clicked.connect(Form.load_mfcc)
        self.pushButton.clicked.connect(Form.record)
        self.pushButton_2.clicked.connect(Form.test)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.radioButton_2.setText(_translate("Form", "PLP"))
        self.radioButton.setText(_translate("Form", "MFCC"))
        self.pushButton.setText(_translate("Form", "Record"))
        self.pushButton_2.setText(_translate("Form", "Test"))

    def load_plp(self):
        self.textBrowser.clear()
        self.textBrowser.append('Loading PLP feature model')
        self.feature_type = 'PLP'
        self.load()

    def load_mfcc(self):
        self.textBrowser.clear()
        self.textBrowser.append('Loading MFCC feature model')
        self.feature_type = 'MFCC'
        self.load()

    def load(self):
        with open("Model/GMM_{}_model.pkl".format(self.feature_type), 'rb') as f:
            self.GMM = pkl.load(f)
        with open("Model/UBM_{}_model.pkl".format(self.feature_type), 'rb') as f:
            self.UBM = pkl.load(f)

        self.textBrowser.append('complete')


    def record(self):
        self.textBrowser.append('Start the recording !')
        record(seconds=3)
        self.textBrowser.append('3 seconds record has completed.')
        _, audio = read(filename='test.wav')
        if self.feature_type=='MFCC':
            feature = mfcc(audio)[0]
        else:
            feature = plp(audio)[0]

        _delta = delta(feature)
        feature = np.hstack((feature, _delta))

        feature = preprocessing.scale(feature)
        self.feature = feature
        os.remove('test.wav')

    def test(self):
        prob = np.zeros((1, len(self.GMM)))
        for i in range(len(self.GMM)):
            prob[0,i] = self.GMM[i].score(self.feature) - self.UBM.score(self.feature)

        res = prob.argmax(axis=1)

        num2name = ['班富景', '郭佳怡', '黄心羿', '居慧敏', '廖楚楚', '刘山', '任蕴菡', '阮煜文', '苏林林', '万之颖',
                '陈斌', '陈泓宇', '陈军栋', '蔡晓明', '邓刚刚', '董俊虎', '代旭辉', '高威', '龚兵庆', '姜宇伦',
                '靳子涵', '李恩', '罗远哲', '罗伟宇', '李想', '李晓波', '李彦能', '刘乙灼', '刘志航', '李忠亚']

        prob = np.exp(prob)/np.exp(prob).sum(axis=1)
        # print(prob)
        output = str(num2name[res[0]]) + ':' + str(prob[0,res[0]])
        self.textBrowser.append(output)
        APP_ID = '11719204'
        API_KEY = 'g7SpqGrkSKgTEBti3pfDsprD'
        SECRET_KEY = 'Tn5CS7EE26rDH34H8z7GV3p0DYYpsksZ'

        client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        result = client.synthesis('{:.2%}的可能是{}'.format(prob[0,res[0]],num2name[res[0]]), 'zh', 0, {
            'vol': 5,
        })
        if not isinstance(result, dict):
            with open('result.mp3', 'wb') as f:
                f.write(result)

        playsound("result.mp3")
        os.remove('result.mp3')
