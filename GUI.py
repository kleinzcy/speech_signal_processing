#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 22:49
# @Author  : chuyu zhang
# @File    : GUI.py
# @Software: PyCharm

# 继承至界面文件的主窗口类
import sys
from PyQt5 import QtWidgets
from UI.GMM_UBM_GUI import Ui_Form

class MyPyQT_Form(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(MyPyQT_Form,self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())
