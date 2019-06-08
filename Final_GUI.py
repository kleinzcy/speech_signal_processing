#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/30 10:12
# @Author  : chuyu zhang
# @File    : Final_GUI.py
# @Software: PyCharm

import sys
from PyQt5 import QtWidgets
from UI.tmp import Ui_MainWindow

class MyPyQT_Form(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyPyQT_Form,self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())
