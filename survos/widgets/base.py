
from ..qt_compat import QtGui, QtCore

import os
from collections import defaultdict
from ..core import DataModel, LabelManager

class SectionCombo(QtGui.QToolButton):

	currentIndexChanged = QtCore.pyqtSignal(int)

	def __init__(self, text='Select', parent=None):
		super(SectionCombo, self).__init__(parent)
		self.setMaximumHeight(30)
		self.setStyleSheet('QToolButton {background-color: #D0EAFF; padding-right: 20px;}'
						   'QMenu {background-color: #A5D1F3;}')
		self.setText(text)
		self.toolmenu = QtGui.QMenu(self)
		self.toolmenu.setContentsMargins(3,3,3,3)
		self.setMenu(self.toolmenu)
		self.setPopupMode(QtGui.QToolButton.InstantPopup)
		self.setMinimumWidth(100)
		self._names = []
		self._actions = []
		self.selected_idx = 0
		self.item_count = 0

	def setMinimumWidth(self, val):
		super(SectionCombo, self).setMinimumWidth(val)
		self.toolmenu.setMinimumWidth(val)

	def addItem(self, item, section=False):
		curr_idx = self.item_count
		widget = QtGui.QWidgetAction(self.toolmenu)
		if section:
			lbl = QtGui.QLabel(item)
			lbl.setStyleSheet('background-color: #fefefe; color: #000;'
							  'padding-top: 4px; padding-bottom: 4px;'
							  'padding-left: 10px; padding-right: 0px;'
							  'font-weight: bold;')
			widget.setEnabled(False)
		else:
			lbl = QtGui.QPushButton(item)
			lbl.setStyleSheet('border-radius: 0px; border: 0px;'
							  'text-align: left; padding-left: 15px;')
			lbl.clicked.connect(lambda: self.on_clicked(curr_idx))
		widget.setDefaultWidget(lbl)
		self._names.append(item)
		self._actions.append(widget)
		self.toolmenu.addAction(widget)
		self.item_count += 1

	def sizeHint(self):
		size = self.toolmenu.sizeHint()
		size.setWidth(size.width() + 20)
		return size

	def on_clicked(self, idx):
		self.selected_idx = idx
		self.setText(self._names[idx])
		self.toolmenu.close()
		self.currentIndexChanged.emit(idx)

	def setCurrentIndex(self, idx):
		self.selected_idx = idx
		self.setText(self._names[idx])

	def currentIndex(self):
		return self.selected_idx

	def removeItem(self, idx):
		action = self._actions[idx]
		del self._names[idx]
		del self._actions[idx]
		self.toolmenu.removeAction(action)
		if idx == self.selected_idx:
			self.currentIndexChanged.emit(0)
			self.setCurrentIndex(0)

	def clear(self):
		self._names = []
		self._actions = []
		self.toolmenu.clear()


class FileWidget(QtGui.QWidget):

	def __init__(self, extensions='*.h5', folder=False, save=True, parent=None):
		super(FileWidget, self).__init__(parent)
		hbox = QtGui.QHBoxLayout()
		hbox.setContentsMargins(0,0,0,0)
		self.setLayout(hbox)

		self.extensions = extensions
		self.folder = folder
		self.save = save

		home = os.path.expanduser('~')
		self.path = QtGui.QLineEdit(home)
		self.path.setReadOnly(True)
		self.path.mousePressEvent = self.find_path

		hbox.addWidget(self.path)

	def find_path(self, ev):
		if ev.button() != 1:
			return
		if self.folder:
			flags = QtGui.QFileDialog.ShowDirsOnly
			message = 'Select Output Folder'
			path = QtGui.QFileDialog.getExistingDirectory(self, message,
														  self.path.text(), flags)
			if path is not None and len(path) > 0 and os.path.isdir(path):
				self.path.setText(path)
		else:
			if self.save:
				path = QtGui.QFileDialog.getSaveFileName(self, "Select input source",
														 filter='*.rec *.npy *.h5')
			else:
				path = QtGui.QFileDialog.getOpenFileName(self, "Select input source",
														 filter='*.rec *.npy *.h5')
			if path is not None and len(path) > 0:
				self.path.setText(path)

	def value(self):
		return self.path.text()


class ActionButton(QtGui.QPushButton):
	def __init__(self, text, parent=None):
		super(ActionButton, self).__init__(text, parent=parent)
		self.setMinimumWidth(70)


class ActionButton(QtGui.QPushButton):
	def __init__(self, text, parent=None):
		super(ActionButton, self).__init__(text, parent=parent)
		self.setMinimumWidth(70)


class CheckableCombo(QtGui.QToolButton):

	def __init__(self, text='Select', parent=None):
		super(CheckableCombo, self).__init__(parent)
		self.setMaximumHeight(30)
		self.setText(text)
		self.toolmenu = QtGui.QMenu(self)
		self.setMenu(self.toolmenu)
		self.setPopupMode(QtGui.QToolButton.InstantPopup)
		self.setStyleSheet('QToolButton {'
						   '  color: #0056b3; background-color: #D0EAFF;'
						   '  padding-right: 25px; border-radius: 4px;'
						   '  border: 1px solid #fefefe;'
						   '  margin: 0px;'
						   '}'
						   'QMenu {'
						   '  background: #D0EAFF; padding: 10px;'
						   '}'
						   'QCheckBox {color: #0056b3;}'
						   'QAction{ padding: 5px;}')
		self.setMinimumWidth(100)
		self._names = []
		self._data = []

	def sizeHint(self):
		size = self.toolmenu.sizeHint()
		size.setWidth(size.width() + 20)
		return size

	def addItem(self, item):
		chk = QtGui.QCheckBox(item + '    ')
		widget = QtGui.QWidgetAction(self.toolmenu)
		widget.setDefaultWidget(chk)
		self.toolmenu.addAction(widget)
		self._names.append(item)
		self._data.append((widget, chk))
		self.toolmenu.repaint()
		self.repaint()

	def setItemText(self, item, text):
		self._data[item][1].setText(text + '    ')
		self.toolmenu.repaint()
		self.repaint()

	def removeItem(self, idx):
		self.toolmenu.removeAction(self._data[idx][0])
		del self._data[idx]
		del self._names[idx]

	def clear(self):
		self._names = []
		self._data = []
		self.toolmenu.clear()
		self.toolmenu.repaint()
		self.repaint()

	def getSelectedIndexes(self):
		return [i for i in range(len(self._data))
				if self._data[i][1].isChecked()]

class CheckableLabels(CheckableCombo):
	def __init__(self, restrict_level=None, parent=None):
		super(CheckableLabels, self).__init__('Select Labels', parent)
		self.DM = DataModel.instance()
		self.LBLM = LabelManager.instance()

		self.restrict_level = restrict_level
		self._labels = []
		self.LBLM.labelAdded.connect(self.on_label_added)
		self.LBLM.labelLoaded.connect(self.on_label_added)
		self.LBLM.labelRemoved.connect(self.on_label_removed)
		self.LBLM.labelNameChanged.connect(self.on_label_name_changed)

	def selectLevel(self, level):
		if level >= 0:
			self.restrict_level = level
			self._labels = []
			self.clear()
			for lbl in self.LBLM.labels(level):
				self.on_label_added(level, '', lbl.idx, lbl.name)

	def on_label_added(self, level, dataset, label, name, *args):
		if self.restrict_level is None:
			self._labels.append((level, label))
			self.addItem('Level {}/{}'.format(level, name))
		elif level == self.restrict_level:
			self._labels.append((level, label))
			self.addItem(name)

	def on_label_removed(self, level, dataset, labelobj):
		if self.restrict_level is None or level == self.restrict_level:
			if (level, labelobj.idx) in self._labels:
				idx = self._labels.index((level, labelobj.idx))
				self.removeItem(idx)
				del self._labels[idx]

	def on_label_name_changed(self, level, dataset, label, newname):
		if self.restrict_level is None:
			idx = self._labels.index((level, label))
			self.setItemText(idx, 'Level {}/{}'.format(level, newname))
		elif level == self.restrict_level:
			idx = self._labels.index((level, label))
			self.setItemText(idx, newname)

	def getSelectedLabels(self):
		return [self._labels[i][1] for i in self.getSelectedIndexes()]


class MultiSourceCombo(CheckableCombo):

	def __init__(self, parent=None):
		super(MultiSourceCombo, self).__init__('Select Sources', parent)
		self.DM = DataModel.instance()
		self.setMinimumWidth(200)
		self.repopulate()

		self.DM.channel_computed.connect(self.on_computed)
		self.DM.channel_removed.connect(self.on_removed)

	def repopulate(self):
		sources = sorted(self.DM.available_channels(return_names=True))
		self.names = ['Data'] + [p['feature_name'] for _, _, p in sources]
		self.sources = ['/data'] + [p for _, p, _ in sources]
		self.clear()
		for source in self.names:
			self.addItem(source)

	def on_update_channel(self, name):
		self.setCurrentIndex(self.sources.index(name))

	def on_computed(self, name, ndict):
		if not name in self.sources:
			self.names.append(ndict['feature_name'])
			self.sources.append(name)
			self.addItem(ndict['feature_name'])

	def on_removed(self, name):
		if name in self.sources:
			idx = self.sources.index(name)
			del self.sources[idx]
			del self.names[idx]
			self.removeItem(idx)

	def value(self):
		return [self.sources[i] for i in self.getSelectedIndexes()]

class SourceCombo(SectionCombo):

	def __init__(self, parent=None, listen_DM=False, listen_self=False,
				 ignore=None):
		super(SourceCombo, self).__init__(parent)
		self.DM = DataModel.instance()
		self.ignore_name = 'channels/{}'.format(ignore) if ignore is not None else None
		self.repopulate()

		self.currentIndexChanged.connect(self.on_selected)
		self.DM.channel_computed.connect(self.on_computed)
		self.DM.channel_removed.connect(self.on_removed)
		if listen_DM:
			self.DM.update_channel.connect(self.on_update_channel)
		if listen_self:
			self.DM.select_channel.connect(self.other_selection)

		#self.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContents)

	def repopulate(self):
		sources = sorted(self.DM.available_channels(return_names=True))
		self.names = ['Data'] + [p['feature_name'] for _, _, p in sources]
		self.sources = ['/data'] + [p for _, p, _ in sources]
		if self.ignore_name is not None:
			if self.ignore_name in self.sources:
				index = self.sources.index(self.ignore_name)
				del self.names[index]
				del self.sources[index]
		self.clear()
		for source in self.names:
			self.addItem(source)
		self.setCurrentIndex(0)

	def other_selection(self, name):
		self.blockSignals(True)
		self.setCurrentIndex(self.sources.index(name))
		self.blockSignals(False)

	def on_selected(self):
		self.DM.select_channel.emit(self.value())

	def on_update_channel(self, name):
		if name in self.sources:
			self.setCurrentIndex(self.sources.index(name))

	def on_computed(self, name, ndict):
		if not name in self.sources and name != self.ignore_name:
			self.names.append(ndict['feature_name'])
			self.sources.append(name)
			self.addItem(ndict['feature_name'])

	def setIgnore(self, name):
		self.ignore_name = 'channels/{}'.format(name)

	def on_removed(self, name):
		if name == self.ignore_name or name not in self.sources:
			return
		idx = self.sources.index(name)
		del self.sources[idx]
		del self.names[idx]
		self.removeItem(idx)

	def value(self):
		return self.sources[self.currentIndex()]

	def setSource(self, source):
		self.on_update_channel(source)

	def loadSource(self, source):
		self.blockSignals(True)
		self.on_update_channel(source)
		self.blockSignals(False)

class SComboBox(QtGui.QComboBox):
	def __init__(self, *args):
		super(SComboBox, self).__init__(*args)

class ComboDialog(QtGui.QDialog):
	def __init__(self, options=None, parent=None):
		super(ComboDialog, self).__init__(parent=parent)

		layout = QtGui.QVBoxLayout(self)

		self.combo = SComboBox()
		for option in options:
			self.combo.addItem(option)
		layout.addWidget(self.combo)

		# OK and Cancel buttons
		buttons = QtGui.QDialogButtonBox(
			QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
			QtCore.Qt.Horizontal, self)
		buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)
		layout.addWidget(buttons)

	@staticmethod
	def getOption(options, parent=None):
		dialog = ComboDialog(options, parent=parent)
		result = dialog.exec_()
		option = dialog.combo.currentText()
		return (option, result == QtGui.QDialog.Accepted)

	@staticmethod
	def getOptionIdx(options, parent=None):
		dialog = ComboDialog(options, parent=parent)
		result = dialog.exec_()
		option = dialog.combo.currentIndex()
		return (option, result == QtGui.QDialog.Accepted)

class RoundedWidget(QtGui.QWidget):

	def __init__(self, parent=None, color=(168, 34, 3), bg=None, width=2, radius=3):
		super(RoundedWidget, self).__init__(parent=parent)
		self.border_color = color
		self.border_width = width
		self.background = bg
		self.radius = radius
		self.margin = int(width / 2.0)

	def paintEvent(self, e):
		qp = QtGui.QPainter()
		qp.begin(self)

		pen = None
		if type(self.border_color) is tuple:
			pen = QtGui.QPen(QtGui.QColor(*self.border_color))
		elif self.border_color is not None:
			pen = QtGui.QPen(QtGui.QColor(self.border_color))

		brush = None
		if type(self.background) is tuple:
			brush = QtGui.QColor(*self.background)
		elif self.background is not None:
			brush = QtGui.QColor(self.background)

		if pen is not None:
			pen.setWidth(self.border_width)
			qp.setPen(pen)
		else:
			qp.setPen(QtCore.Qt.NoPen)

		if brush is not None:
			qp.setBrush(brush)

		qp.drawRoundedRect(self.margin, self.margin,
						   self.width()-2*self.margin-1,
						   self.height()-2*self.margin-1,
						   self.radius, self.radius);
		qp.end()


class PicButton(QtGui.QToolButton):
	def __init__(self, pixmap, parent=None):
		super(PicButton, self).__init__(parent)
		self.icon = QtGui.QIcon(pixmap)
		self.setIcon(self.icon)

	def sizeHint(self):
		return QtCore.QSize(50, 30)


class HeaderLabel(QtGui.QLabel):

	def __init__(self, text, parent=None, height=30,
				 bgcolor='#6194BC', color='#fefefe', bradius=0,
				 bsides=0, fontsize=15):
		super(HeaderLabel, self).__init__(text, parent=parent)
		self.setMinimumHeight(height)
		self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
		self.setStyleSheet('background-color: {}; color: {};'
						   'font-weight: bold; font-size: {}px;'
						   'border-radius: {}px;'
						   .format(bgcolor, color, fontsize,
								   bsides, bsides, bradius))

class SubHeaderLabel(HeaderLabel):

	def __init__(self, text, parent=None, height=22, fontsize=14, color='#fefefe',
				 bgcolor='#009999'):
		super(SubHeaderLabel, self).__init__(text, parent=parent, height=height,
											 bgcolor=bgcolor, color=color,
											 bradius=3, bsides=2,
											 fontsize=fontsize)


class HeaderButton(object):

	def __init__(self, parent, index, enabled=True):
		self.parent = parent
		self.index = index
		self.setEnabled(enabled)

	def setEnabled(self, bool):
		self.parent.setTabEnabled(self.index, bool)

	def setDisabled(self, bool):
		self.parent.setTabDisabled(self.index, bool)

	def setVisible(self, bool):
		#self.parent.setCurrentIndex(self.index)
		pass

class MHBoxLayout(QtGui.QHBoxLayout):
	def __init__(self, *widgets, **kwargs):
		super(MHBoxLayout, self).__init__()
		self.setContentsMargins(0,0,0,0)

class MVBoxLayout(QtGui.QVBoxLayout):
	def __init__(self, *widgets, **kwargs):
		super(MVBoxLayout, self).__init__()
		self.setContentsMargins(0,0,0,0)

class HWidgets(QtGui.QWidget):

	def __init__(self, *widgets, **kwargs):
		kwargs.setdefault('parent', None)
		super(HWidgets, self).__init__(kwargs['parent'])

		s = [0]*len(widgets) if not 'stretch' in kwargs else kwargs['stretch']
		if len(s) < len(widgets):
			s += [0] * (len(widgets) - len(s))

		hbox = QtGui.QHBoxLayout(self)
		self.setLayout(hbox)
		hbox.setContentsMargins(0,0,0,0)
		for i, widget in enumerate(widgets):
			if widget is None:
				widget = QtGui.QWidget()
			elif type(widget) in [str, unicode]:
				widget = QtGui.QLabel(widget)
			hbox.addWidget(widget, s[i])
		self.widgets = widgets

	def setStyle(self, style):
		for widget in self.widgets:
			widget.setStyleSheet(style)
		self.setStyleSheet(style)

class BLabel(QtGui.QLabel):

	def __init__(self, *args, **kwargs):
		super(BLabel, self).__init__(*args, **kwargs)
		self.setStyleSheet('font-weight: bold;')

class RLabel(QtGui.QLabel):

	def __init__(self, *args, **kwargs):
		super(RLabel, self).__init__(*args, **kwargs)
		self.setAlignment(QtCore.Qt.AlignRight)

class ULabel(QtGui.QLabel):

	def __init__(self, *args, **kwargs):
		super(ULabel, self).__init__(*args, **kwargs)
		self.setStyleSheet('color: #6194BC;')

class TComboBox(QtGui.QWidget):

	def __init__(self, label, options, selected=None, parse=str, *args, **kwargs):
		super(TComboBox, self).__init__(*args, **kwargs)
		hbox = MHBoxLayout()
		if label is not None:
			hbox.addWidget(QtGui.QLabel(label))
		self.combo = SComboBox()
		for option in options:
			self.combo.addItem(str(option))
		if selected:
			self.combo.setCurrentIndex(selected)
		hbox.addWidget(self.combo, 1)
		self.setLayout(hbox)
		self.currentIndexChanged = self.combo.currentIndexChanged
		self.parse = parse
		self.findText = self.combo.findText

	def clear(self):
		self.combo.clear()

	def currentIndex(self):
		return self.combo.currentIndex()

	def currentText(self):
		return str(self.combo.currentText())

	def setCurrentIndex(self, idx):
		self.combo.setCurrentIndex(idx)

	def addItem(self, item):
		self.combo.addItem(item)

	def removeItem(self, item):
		self.combo.removeItem(item)

	def updateItems(self, items):
		self.combo.clear()
		for option in items:
			self.combo.addItem(option)

	def count(self):
		return self.combo.count()

	def itemText(self, i):
		return self.combo.itemText(i)

	def setItemText(self, i, txt):
		self.combo.setItemText(i, txt)

	def value(self):
		try:
			return self.parse(self.combo.currentText())
		except:
			return self.parse(0)

class PLineEdit(QtGui.QLineEdit):

	def __init__(self, default=0, parse=int, fontsize=None, *args, **kwargs):
		super(PLineEdit, self).__init__(*args, **kwargs)
		self.default = default
		self.parse = parse
		self.setPlaceholderText(str(default))

		if fontsize is not None:
			self.setStyleSheet('font-size: {}px;'.format(fontsize))

	def value(self):
		val = self.default
		try:
			val = self.parse(self.text())
		except:
			pass
		return val

	def setDefault(self, val):
		self.default = val
		self.setPlaceholderText(str(val))


class RCheckBox(QtGui.QWidget):

	def __init__(self, text, fontsize=None, **kwargs):
		super(RCheckBox, self).__init__(**kwargs)

		hbox = MHBoxLayout()
		self.setLayout(hbox)

		self.label = RClickableLabel(text)
		self.check = QtGui.QCheckBox()
		self.check.setChecked(True)
		hbox.addWidget(self.label, 1)
		hbox.addWidget(self.check)
		self.label.clicked.connect(self.on_label)
		self.stateChanged = self.check.stateChanged

		if fontsize is not None:
			self.label.setStyleSheet('font-size: {}px;'.format(fontsize))

	def on_label(self):
		self.check.toggle()

	def setChecked(self, bol):
		self.check.setChecked(bol)

	def setCheckState(self, st):
		self.check.setCheckState(st)

	def isChecked(self):
		return self.check.isChecked()

	def setMaximumHeight(self, h):
		self.label.setMaximumHeight(h)
		self.check.setMaximumHeight(h)

	def setCheckStyle(self, style):
		self.check.setStyleSheet(style)

class RClickableLabel(RLabel):
	clicked = QtCore.pyqtSignal()
	def mousePressEvent(self, ev):
		self.clicked.emit()

class ClickableLabel(QtGui.QLabel):
	clicked = QtCore.pyqtSignal()
	def mousePressEvent(self, ev):
		self.clicked.emit()

class LCheckBox(QtGui.QWidget):

	def __init__(self, text, fontsize=None, **kwargs):
		super(LCheckBox, self).__init__(**kwargs)

		hbox = MHBoxLayout()
		self.setLayout(hbox)

		self.label = ClickableLabel(text)
		self.check = QtGui.QCheckBox()
		self.check.setMaximumWidth(20)
		self.check.setChecked(True)
		hbox.addWidget(self.check)
		hbox.addWidget(self.label, 1)
		self.label.clicked.connect(self.on_label)
		self.stateChanged = self.check.stateChanged

		if fontsize is not None:
			self.label.setStyleSheet('font-size: {}px;'.format(fontsize))

	def on_label(self):
		self.check.toggle()

	def setChecked(self, bol):
		self.check.setChecked(bol)

	def setCheckState(self, st):
		self.check.setCheckState(st)

	def isChecked(self):
		return self.check.isChecked()

	def setMaximumHeight(self, h):
		self.label.setMaximumHeight(h)
		self.check.setMaximumHeight(h)

	def setCheckStyle(self, style):
		self.check.setStyleSheet(style)

class HLayer(QtGui.QWidget):

	valueChanged = QtCore.pyqtSignal(str, str, int)
	toggled  = QtCore.pyqtSignal(str, str, bool)
	export = QtCore.pyqtSignal(str, str)

	def __init__(self, name, level, current=None, visible=True, **kwargs):
		super(HLayer, self).__init__(**kwargs)
		hbox = MHBoxLayout()
		self.setLayout(hbox)

		self.name = name
		self.level = level

		self.label = QtGui.QLabel(name + ':')
		self.label.setFixedWidth(90)
		self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
		self.slider = CSlider(self.name, current=current)
		self.slider.lbl_current.setFixedWidth(25)
		self.check = QtGui.QCheckBox()
		self.check.setChecked(visible)
		#self.btn_tiff = QtGui.QPushButton('.rec')
		#self.btn_tiff.setFixedWidth(40)

		hbox.addWidget(self.label)
		hbox.addWidget(self.slider, 1)
		hbox.addWidget(self.check)
		#hbox.addWidget(self.btn_tiff)

		self.slider.valueChanged.connect(self.on_value)
		self.check.stateChanged.connect(self.on_toggled)
		#self.btn_tiff.clicked.connect(self.on_tiff_clicked)

	def on_value(self, val):
		self.valueChanged.emit(self.name, self.level, val)

	def on_toggled(self, val):
		self.toggled.emit(self.name, self.level, val)

	def on_tiff_clicked(self):
		self.export.emit(self.name, self.level)

	def setChecked(self, value):
		self.check.setChecked(value)


class FLineEdit(QtGui.QLineEdit):

	strChanged = QtCore.pyqtSignal(str)

	def __init__(self, *args, **kwargs):
		super(FLineEdit, self).__init__(*args, **kwargs)
		self.returnPressed.connect(self.clearFocus)

	def focusOutEvent(self, ev):
		super(FLineEdit, self).focusOutEvent(ev)
		self.strChanged.emit(self.text())

class ColorButton(QtGui.QPushButton):

	colorChanged = QtCore.pyqtSignal(str)

	def __init__(self, color='#000000', clickable=True, *args, **kwargs):
		super(ColorButton, self).__init__(*args, **kwargs)
		self.setColor(color)
		if clickable:
			self.clicked.connect(self.on_click)

	def c2h(self, color):
		return str(color.name())

	def setColor(self, color):
		if color is None:
			self.setStyleSheet('QPushButton, QPushButton:hover'
							   '{background-color:'
							   'qlineargradient(x1:0, y1:0, x2:1, y2:1,'
							   'stop: 0 white, stop: 0.15 white,'
							   'stop: 0.2 red,'
							   'stop: 0.25 white, stop: 0.45 white,'
							   'stop: 0.5 red,'
							   'stop: 0.55 white, stop: 0.75 white,'
							   'stop: 0.8 red, stop: 0.85 white);}')
		else:
			self.setStyleSheet('QPushButton {background-color: %s;}'
							   'QPushButton:hover {background-color:'
							   'qlineargradient(x1:0, y1:0, x2:0.5, y2:1,'
							   'stop: 0 white, stop: 1 %s);}'
							   % (color, color))
		self.color = color

	def on_click(self):
		c = QtGui.QColorDialog.getColor(QtGui.QColor(self.color), self.parent())
		if not c.isValid():
			return
		self.setColor(self.c2h(c))
		self.colorChanged.emit(self.color)

class HLabelParent(QtGui.QWidget):

	def __init__(self, name, **kwargs):
		super(HLabelParent, self).__init__(**kwargs)
		hbox = MHBoxLayout()
		self.setLayout(hbox)

		self.txt_label_name = QtGui.QLabel(name)
		self.combo_parent = SComboBox()
		self.combo_parent.addItem('None')
		dummy = HWidgets(self.txt_label_name, self.combo_parent,
						 stretch=[0,0,0])
		hbox.addWidget(dummy)

	def setColor(self, color):
		self.txt_label_name.setStyleSheet('background-color: %s;' % color)

	def setText(self, name):
		self.txt_label_name.setText(name)

	def addParent(self, name):
		self.combo_parent.addItem(name)

	def getParent(self):
		return str(self.combo_parent.currentText())


class HEditLabel(QtGui.QWidget):

	selected = QtCore.pyqtSignal(int)
	removed = QtCore.pyqtSignal(int)
	nameChanged = QtCore.pyqtSignal(int, str)
	colorChanged = QtCore.pyqtSignal(int, str)
	parentRequested = QtCore.pyqtSignal(int)
	visibilityChanged = QtCore.pyqtSignal(int, bool)

	def __init__(self, name, idx, visible=True, **kwargs):
		super(HEditLabel, self).__init__(**kwargs)
		hbox = MHBoxLayout()
		self.setLayout(hbox)

		self.idx = idx

		self.btn_remove = QtGui.QPushButton('X')
		self.btn_remove.setContentsMargins(0,0,0,0)
		self.btn_remove.setMaximumWidth(25)
		self.btn_remove.setCheckable(True)

		self.label = FLineEdit(name)
		self.label.setMinimumWidth(100)
		self.label.setMaximumWidth(100)

		self.lbutton = ColorButton()
		self.pbutton = ColorButton(color=None)
		self.pbutton.clicked.disconnect()

		self.sbutton = QtGui.QPushButton(u'\u2713')
		self.sbutton.setContentsMargins(0,0,0,0)
		self.sbutton.setMaximumWidth(25)
		self.sbutton.setCheckable(True)

		self.check = QtGui.QCheckBox()
		self.check.setChecked(True)

		hbox.addWidget(self.btn_remove)
		hbox.addWidget(self.label)
		hbox.addWidget(self.lbutton, 1)
		hbox.addWidget(self.pbutton, 1)
		hbox.addWidget(self.sbutton)
		#hbox.addWidget(self.check)

		self.sbutton.clicked.connect(self.on_select)
		self.label.strChanged.connect(self.on_name)
		self.lbutton.colorChanged.connect(self.on_color)
		self.pbutton.clicked.connect(self.on_parent_clicked)
		self.check.stateChanged.connect(self.on_check)
		self.btn_remove.clicked.connect(self.on_remove)

	def on_name(self, name):
		self.nameChanged.emit(self.idx, name)

	def on_color(self, color):
		self.colorChanged.emit(self.idx, color)

	def on_select(self):
		self.selected.emit(self.idx)

	def on_check(self, val):
		self.visibilityChanged.emit(self.idx, val)

	def on_remove(self, val):
		self.removed.emit(self.idx)

	def setName(self, value):
		self.label.setText(value)

	def setColor(self, value):
		self.lbutton.setColor(value)

	def setChecked(self, bol):
		self.check.setChecked(bol)

	def setSelected(self, bol):
		self.btn_remove.setChecked(bol)
		self.sbutton.setChecked(bol)

	def on_parent_clicked(self):
		self.parentRequested.emit(self.idx)

	def setParentColor(self, color):
		self.pbutton.setColor(color)


class CSlider(QtGui.QWidget):

	valueChanged = QtCore.pyqtSignal(int)

	def __init__(self, name, vmin=0, vmax=100, current=None,
				 interval=1, lblwidth=50, **kwargs):
		super(CSlider, self).__init__(**kwargs)

		vbox = MVBoxLayout()
		self.setLayout(vbox)

		self.current = current if current is not None else vmax
		self.vmin = vmin
		self.vmax = vmax
		self.interval = interval
		self.initial = self.current

		# Header
		self.lbl_current = QtGui.QLabel("%d" % self.current)
		self.lbl_current.setMinimumWidth(lblwidth)
		self.lbl_current.setStyleSheet('color: #6194BC;')

		# Slider
		self.sld_val = QtGui.QSlider(1)
		self.sld_val.setMinimum(vmin)
		self.sld_val.setMaximum(vmax)
		self.sld_val.setValue(self.current)
		self.sld_val.setTickInterval(interval)
		self.sld_val.setSingleStep(interval)
		self.sld_val.setTracking(True)

		hbox = QtGui.QHBoxLayout()
		hbox.addWidget(self.sld_val, 1)
		hbox.addWidget(self.lbl_current)
		vbox.addLayout(hbox)

		# Slots
		self.sld_val.valueChanged.connect(self.on_value_changed)
		self.sld_val.wheelEvent = self.wheelEvent

	def on_value_changed(self, val):
		self.current = val
		self.lbl_current.setText("%d" % self.current)
		self.valueChanged.emit(self.current)

	def value(self):
		return self.current

	def setValue(self, val):
		self.sld_val.setValue(val)

	def maximum(self):
		return self.sld_val.maximum()

	def minimum(self):
		return self.sld_val.minimum()

	def wheelEvent(self, e):
		if e.delta() > 0 and self.value() < self.maximum():
			self.setValue(self.value()+self.interval)
		elif e.delta() < 0 and self.value() > self.minimum():
			self.setValue(self.value()-self.interval)


class OddSlider(CSlider):

	def __init__(self, *args, **kwargs):
		kwargs.setdefault('interval', 2)
		super(OddSlider, self).__init__(*args, **kwargs)

	def on_value_changed(self, val):
		if val%2 == 0:
			if val < self.maximum():
				val = val + 1
			else:
				val = val - 1
		self.current = val
		self.lbl_current.setText("%d" % self.current)
		self.valueChanged.emit(self.current)


class WSlider(QtGui.QSlider):

	def __init__(self, maximum, minimum=0, orient=1, parent=None):
		super(WSlider, self).__init__(orient, parent=parent)

		self.setMinimum(minimum)
		self.setMaximum(maximum)
		self.setValue(maximum//2)
		self.setSingleStep(1)

	def wheelEvent(self, e):
		if e.delta() > 0 and self.value() < self.maximum():
			self.setValue(self.value()+1)
		elif e.delta() < 0 and self.value() > self.minimum():
			self.setValue(self.value()-1)

class WSliderPane(QtGui.QWidget):

	def __init__(self,  maximum, minimum=0, orient=1, parent=None):
		super(WSliderPane, self).__init__(parent=parent)

		hbox = QtGui.QHBoxLayout()
		self.setLayout(hbox)
		self.slider = WSlider(maximum, minimum=minimum, orient=orient)

		hbox.addWidget(self.slider, 1)

		self.valueChanged = self.slider.valueChanged
		self.setMinimum = self.slider.setMinimum
		self.setMaximum = self.slider.setMaximum
		self.setValue = self.slider.setValue

		self.wheelEvent = self.slider.wheelEvent
		self.value = self.slider.value
		self.setValue = self.slider.setValue
		self.maximum = self.slider.maximum
		self.minimum = self.slider.minimum
		self.setTracking = self.slider.setTracking

class HSlider(QtGui.QWidget):

	valueChanged = QtCore.pyqtSignal(float)

	def __init__(self, name, vmin, vmax, current=None, **kwargs):
		super(HSlider, self).__init__(**kwargs)

		vbox = MVBoxLayout()
		self.setLayout(vbox)

		self.current = current
		if self.current is None:
			self.current = (vmax - vmin) / 2.
		self.vmin = vmin
		self.vmax = vmax

		# Header
		hbox = MHBoxLayout()
		self.lbl_name = QtGui.QLabel(name)
		self.lbl_name.setStyleSheet('font-weight: bold;')
		self.lbl_current = QtGui.QLabel("%.2f" % self.current)
		self.lbl_current.setStyleSheet('color: #6194BC;')

		hbox.addWidget(self.lbl_name)
		hbox.addWidget(self.lbl_current)
		vbox.addLayout(hbox)

		# Slider
		self.lbl_min = QtGui.QLabel('%.2f' % vmin)
		self.lbl_min.setMinimumWidth(40)
		self.lbl_min.setAlignment(QtCore.Qt.AlignCenter)
		self.lbl_max = QtGui.QLabel('%.2f' % vmax)
		self.lbl_max.setAlignment(QtCore.Qt.AlignCenter)
		self.lbl_max.setMinimumWidth(40)
		self.sld_val = QtGui.QSlider(1)
		self.sld_val.setMinimum(0)
		self.sld_val.setMaximum(1000)
		self.sld_val.setValue(self.val2slider(self.current))

		hbox = QtGui.QHBoxLayout()
		hbox.addWidget(self.lbl_min)
		hbox.addWidget(self.sld_val)
		hbox.addWidget(self.lbl_max)
		vbox.addLayout(hbox)

		# Slots
		self.sld_val.valueChanged.connect(self.on_value_changed)

	def val2slider(self, val):
		return (val - self.vmin) / float(self.vmax - float(self.vmin)) * 1000

	def slider2val(self, val):
		return val * (self.vmax - float(self.vmin)) / 1000. + self.vmin

	def on_value_changed(self, val):
		self.current = self.slider2val(val)
		self.lbl_current.setText("%.2f" % self.current)
		self.valueChanged.emit(self.current)

	def value(self):
		return self.current

	def setMinimum(self, val):
		self.vmin = val
		self.lbl_min.setText('%.2f' % val)

	def setMaximum(self, val):
		self.vmax = val
		self.lbl_max.setText('%.2f' % val)

	def setValue(self, val):
		self.sld_val.setValue(int(self.val2slider(float(val))))

class HSize3D(QtGui.QWidget):

	def __init__(self, name, default=(0, 0, 0), txtwidth=None,
				 parse=int, coordwidth=None):
		super(HSize3D, self).__init__()

		self.default = default
		self.parse = parse

		hbox = MHBoxLayout()
		self.setLayout(hbox)

		self.zbox = QtGui.QLineEdit()
		self.ybox = QtGui.QLineEdit()
		self.xbox = QtGui.QLineEdit()

		if coordwidth is not None:
			self.zbox.setMaximumWidth(coordwidth)
			self.ybox.setMaximumWidth(coordwidth)
			self.xbox.setMaximumWidth(coordwidth)

		self.zbox.setPlaceholderText(str(default[0]))
		self.ybox.setPlaceholderText(str(default[1]))
		self.xbox.setPlaceholderText(str(default[2]))

		zlbl = QtGui.QLabel('z')
		zlbl.setStyleSheet('color: #6194BC;')
		ylbl = QtGui.QLabel('y')
		ylbl.setStyleSheet('color: #6194BC;')
		xlbl = QtGui.QLabel('x')
		xlbl.setStyleSheet('color: #6194BC;')

		title = QtGui.QLabel(name + ':')

		if txtwidth is not None:
			title.setMinimumWidth(txtwidth)
			title.setMaximumWidth(txtwidth)

		hbox.addWidget(title)
		hbox.addWidget(zlbl)
		hbox.addWidget(self.zbox, 1)
		hbox.addWidget(ylbl)
		hbox.addWidget(self.ybox, 1)
		hbox.addWidget(xlbl)
		hbox.addWidget(self.xbox, 1)

	def value(self):
		z, y, x = self.default
		try:
			z = self.parse(self.zbox.text())
		except:
			pass
		try:
			y = self.parse(self.ybox.text())
		except:
			pass
		try:
			x = self.parse(self.xbox.text())
		except:
			pass
		return (z, y, x)

	def setValue(self, value):
		self.zbox.setText(str(value[0]))
		self.ybox.setText(str(value[1]))
		self.xbox.setText(str(value[2]))
