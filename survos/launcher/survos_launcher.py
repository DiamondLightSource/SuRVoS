#!/usr/bin/env python

import sys
import os
import logging
import numpy as np
import h5py as h5

import argparse
def main():
    survos_title = "SuRVoS: Super-Region Volume Segmentation workbench"

    parser = argparse.ArgumentParser(description=survos_title)
    parser.add_argument('--workspace', type=str, default=None, help='Load workspace')
    parser.add_argument('--gpu', type=int, default=-1, help='Select GPU')
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    import sip
    sip.setapi("QString", 2)

    import matplotlib as mpl
    mpl.use('Qt5Agg')
    logging.getLogger('matplotlib.font_manager').disabled = True

    from matplotlib import style
    style.use( u'seaborn-ticks')

    from survos.qt_compat import QtGui, QtCore, QtWidgets
    from survos.widgets import MainWindow

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    root.addHandler(ch)

    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("DLS")
    app.setApplicationName("SuRVoS")

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'survos.qcs')) as f:
        css = f.read()

    window = MainWindow(survos_title)
    window.setMinimumSize(1024, 768)
    window.setStyleSheet(css)
    window.showMaximized()


    if args.gpu >= 0:
        window.setGPU(args.gpu)

    if args.workspace is not None and os.path.isdir(args.workspace):
        window.load_workspace(args.workspace)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()