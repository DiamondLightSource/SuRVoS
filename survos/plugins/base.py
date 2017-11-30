
from ..qt_compat import QtGui, QtCore, QtWidgets


class Plugin(QtWidgets.QScrollArea):

    name = "Plugin"
    header = None

    Widget = 'widget'
    Plugin = 'plugin'

    completed = QtCore.pyqtSignal(bool)

    def __init__(self, ptype=Plugin, height=0, width=300, dock='left', parent=None):

        super(Plugin, self).__init__(parent=parent)

        self.ptype = ptype
        self.container = QtWidgets.QWidget()
        self.container.setWindowTitle(self.name)
        self.setWidget(self.container)
        self.setWidgetResizable(True)

        self.layout = QtWidgets.QGridLayout(self.container)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        if ptype == self.Plugin:
            self.layout.setContentsMargins(0,10,0,0)
        self.container.setLayout(self.layout)
        self.row = 0
        self.tab_idx = 0

    def setup(self):
        self.header.setEnabled(True)
        self.header.setVisible(True)

    def addWidget(self, widget, align=QtCore.Qt.AlignTop):
        self.layout.addWidget(widget, self.row, 0, align)
        self.row += 1

    def __add__(self, widget):
        self.addWidget(widget)
        return self

    def toggle(self):
        self.setVisible(not self.isVisible())

    def value(self):
        pass

    def on_focus(self):
        pass

    def on_tab_changed(self, idx):
        pass
