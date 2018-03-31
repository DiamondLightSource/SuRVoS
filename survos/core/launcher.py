import six
from ..qt_compat import QtGui, QtCore, QtWidgets

from .singleton import Singleton
from .model import DataModel

import logging as log

from six.moves.queue import Queue

import sys
import traceback


@Singleton
class Launcher(QtCore.QObject):

    pre = QtCore.pyqtSignal(str)
    post = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(object)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.result.connect(self.on_result)
        self.current = None
        self.actionQueue = Queue(maxsize=1)
        self.actionQueue.put(True)
        self.forced = False
        self.cb = None

    def run(self, fn, cb=None, caption='Loading...', *args, **kwargs):
        self.actionQueue.get()

        self.cb = cb
        self.setup(caption)

        self.current = Action(fn, cb, *args, **kwargs)
        self.current.dummy.done.connect(self.on_result)
        self.current.dummy.error.connect(self.on_error)
        self.current.start()

    def setup(self, caption):
        log.info('\n### {} ###'.format(caption))
        self.pre.emit(caption)
        QtWidgets.qApp.processEvents()

    def cleanup(self):
        self.post.emit()
        QtWidgets.qApp.processEvents()

    def on_result(self, res=None):
        if self.cb is not None:
            self.cb(res)
        self.clear()

    def show_error(self, err):
        if type(err) == str:
            title = '[Error]'
        else:
            title = '[{}]'.format(type(err).__name__)

        try:
            traceback.print_last()
        except Exception as e:
            pass

        self.error.emit(title + '\n' + str(err))
        log.error(title + '\n' + str(err))

    def on_error(self, err):
        self.show_error(err)
        self.clear()

    def clear(self):
        self.current.dummy.done.disconnect()
        self.current.dummy.error.disconnect()
        del self.current
        self.actionQueue.put(True)
        self.cleanup()

    def terminate(self):
        pass

class Dummy(QtCore.QObject):
    done = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(object)

class Action(QtCore.QThread):

    dummy = Dummy()

    def __init__(self, fn, cb=None, *args, **kwargs):
        QtCore.QThread.__init__(self)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        log.info('\n### {} ###'.format(self.fn.__name__))
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.dummy.done.emit(res)
        except Exception as e:
            self.dummy.error.emit(e)
