#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from . import Object3DQt as qt
from .HorizontalSpacer import HorizontalSpacer
from .VerticalSpacer import VerticalSpacer

COLORS_LIST = ['white', 'pink', 'red', 'brown', 'orange', 'yellow',
               'green', 'blue', 'violet', 'black', 'auto']

COLORS = {}
if 0:
    COLORS['blue']     = qt.QColor(0, 0, 0xFF, 0xFF)
    COLORS['red']      = qt.QColor(0xFF, 0, 0, 0xFF)
    COLORS['yellow']   = qt.QColor(0xFF, 0xFF, 0, 0xFF)
    COLORS['black']    = qt.QColor(0, 0, 0, 0xFF)
    COLORS['green']    = qt.QColor(0, 0xFF, 0, 0xFF)
    COLORS['white']    = qt.QColor(0xFF, 0xFF, 0xFF, 0xFF)
    # added colors #
    COLORS['pink']     = qt.QColor(255,  20, 147, 0xFF)
    COLORS['brown']    = qt.QColor(165,  42,  42, 0xFF)
    COLORS['orange']   = qt.QColor(255, 165,   0, 0xFF)
    COLORS['violet']   = qt.QColor(148,   0, 211, 0xFF)
else:
    COLORS['blue']     = (0, 0, 0xFF, 0xFF)
    COLORS['red']      = (0xFF, 0, 0, 0xFF)
    COLORS['yellow']   = (0xFF, 0xFF, 0, 0xFF)
    COLORS['black']    = (0, 0, 0, 0xFF)
    COLORS['green']    = (0, 0xFF, 0, 0xFF)
    COLORS['white']    = (0xFF, 0xFF, 0xFF, 0xFF)
    # added colors #
    COLORS['pink']     = (255,  20, 147, 0xFF)
    COLORS['brown']    = (165,  42,  42, 0xFF)
    COLORS['orange']   = (255, 165,   0, 0xFF)
    COLORS['violet']   = (148,   0, 211, 0xFF)
    COLORS['auto']     = (None, None, None, 0xFF)

class ColorLabel(qt.QLabel):
    def __init__(self,parent=None, bold=True, color= qt.Qt.red):
        qt.QLabel.__init__(self,parent)
        palette = self.palette()
        role = self.foregroundRole()
        palette.setColor(role,color)
        self.setPalette(palette)
        self.font().setBold(bold)

class InfoLabel(qt.QGroupBox):
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Info')
        self.mainLayout = qt.QHBoxLayout(self)
        self.mainLayout.addWidget(HorizontalSpacer(self))
        self.infoLabel = qt.QLabel(self)
        self.infoLabel.setText("Private widget configuration")
        self.mainLayout.addWidget(self.infoLabel)
        self.mainLayout.addWidget(HorizontalSpacer(self))

    def setParameters(self, ddict=None):
        if ddict is None:
            ddict = {}
        key = 'infolabel'
        if key in ddict:
            self.infoLabel.setText(ddict[key])

    def getParameters(self):
        ddict = {}
        key = 'infolabel'
        ddict[key] = str(self.infoLabel.text())
        return ddict


class ColorFilter(qt.QGroupBox):
    sigColorFilterSignal = qt.pyqtSignal(object)
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Color Filter')
        self.mainLayout = qt.QVBoxLayout(self)
        self.buttonGroup = qt.QButtonGroup(self)
        self.__options = ['None', 'MinMax']
        for j in range(len(self.__options)):
            text = self.__options[j]
            rButton = qt.QRadioButton(self)
            rButton.setText(text)
            if j == 0:
                rButton.setChecked(True)
            self.mainLayout.addWidget(rButton)
            #self.mainLayout.setAlignment(rButton, qt.Qt.AlignHCenter)
            self.buttonGroup.addButton(rButton)
            self.buttonGroup.setId(rButton, j)
        self.mainLayout.addWidget(VerticalSpacer(self))
        self.buttonGroup.buttonPressed.connect(self._slot)

    def _slot(self, button):
        button.setChecked(True)
        ddict = {}
        ddict['event']  = 'ColorFilterUpdated'
        ddict['colorfilter'] = self.__options.index(button.text())
        self.sigColorFilterSignal.emit(ddict)

    def setParameters(self, ddict=None):
        if ddict is None:
            ddict = {}
        key = 'colorfilter'
        if key in ddict:
            idx = int(ddict[key])
            self.buttonGroup.button(idx).setChecked(True)

    def getParameters(self):
        ddict = {}
        key = 'colorfilter'
        ddict[key] = self.buttonGroup.checkedId()
        return ddict

class ValueFilter(qt.QGroupBox):
    def __init__(self, parent = None):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Value Filter')
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)
        self.mainLayout.setSpacing(4)
        i = 0
        for text in ["Use", "Min Value", "Max Value"]:
            label = qt.QLabel(self)
            label.setText(text)
            self.mainLayout.addWidget(label, 0, i)
            i += 1

        self.useCheckBox = qt.QCheckBox(self)
        self.minLineEdit = qt.QLineEdit(self)
        w = self.minLineEdit.fontMetrics().width("#####.####")
        self.minLineEdit.setFixedWidth(w)
        self.minLineEdit.setText("0.0")
        self.minLineEdit.validator = qt.CLocaleQDoubleValidator(self.minLineEdit)
        self.minLineEdit.setValidator(self.minLineEdit.validator)
        self.maxLineEdit = qt.QLineEdit(self)
        self.maxLineEdit.setFixedWidth(w)
        self.maxLineEdit.setText("0.0")
        self.maxLineEdit.validator = qt.CLocaleQDoubleValidator(self.maxLineEdit)
        self.maxLineEdit.setValidator(self.maxLineEdit.validator)
        self.mainLayout.addWidget(self.useCheckBox, 1, 0)
        self.mainLayout.addWidget(self.minLineEdit, 1, 1)
        self.mainLayout.addWidget(self.maxLineEdit, 1, 2)
        self.mainLayout.addWidget(VerticalSpacer(self), 2, 0)

    def setParameters(self, ddict=None):
        if ddict is None:
            ddict = {}
        key = 'useminmax'
        if key in ddict:
            if ddict[key][0]:
                self.useCheckBox.setChecked(True)
            else:
                self.useCheckBox.setChecked(False)
            self.minLineEdit.setText("%g" % ddict[key][1])
            self.maxLineEdit.setText("%g" % ddict[key][2])

    def getParameters(self):
        ddict = {}
        key = 'useminmax'
        if self.useCheckBox.isChecked():
            cb = 1
        else:
            cb = 0
        minValue = float(str(self.minLineEdit.text()))
        maxValue = float(str(self.maxLineEdit.text()))
        ddict[key] = [cb, minValue, maxValue]
        return ddict

class Isosurfaces(qt.QGroupBox):
    def __init__(self, parent = None, niso=5):
        qt.QGroupBox.__init__(self, parent)
        self.setTitle('Isosurfaces')
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(4, 4, 4, 4)
        self.mainLayout.setSpacing(4)
        self.__nIsosurfaces = niso
        i = 0
        for text in ["Use", "Value", "Color"]:
            label = qt.QLabel(self)
            label.setText(text)
            self.mainLayout.addWidget(label, 0, i)
            i += 1
        self.__isosurfaceList = []
        colorOptions = ["White",
                        "Red",
                        "Yellow",
                        "Green",
                        "Cyan",
                        "Blue",
                        "Black"]
        for i in range(self.__nIsosurfaces):
            cb = qt.QCheckBox(self)
            le = qt.QLineEdit(self)
            le.setText("0.0")
            le.validator = qt.CLocaleQDoubleValidator(le)
            le.setValidator(le.validator)
            color = qt.QComboBox(self)
            color.insertItems(0, COLORS_LIST)
            color.setCurrentIndex(i+1)
            self.mainLayout.addWidget(cb, i+1, 0)
            self.mainLayout.addWidget(le, i+1, 1)
            self.mainLayout.addWidget(color, i+1, 2)
            self.__isosurfaceList.append((cb, le, color))

    def setParameters(self, ddict=None):
        global COLORS_LIST
        if ddict is None:
            ddict = {}
        key = 'isosurfaces'
        if key in ddict:
            if type(ddict[key][0]) not in [type((1,)), type([])]:
                ddict[key] = [ddict[key] * 1]
            for i in range(len(ddict[key])):
                if i >= self.__nIsosurfaces:
                    break
                cb, le, label, r, g, b, a = ddict[key][i]
                color = (r, g, b, a)
                if cb in [0, 'False', '0']:
                    cb = False
                if cb:
                    self.__isosurfaceList[i][0].setChecked(True)
                else:
                    self.__isosurfaceList[i][0].setChecked(False)
                self.__isosurfaceList[i][1].setText("%g" % le)
                if label not in COLORS_LIST:
                    COLORS_LIST.append(label)
                    COLORS[label] = color
                    self.__isosurfaceList[i][2].addItem(label)
                    continue
                for c in COLORS_LIST:
                    if COLORS[c] == color:
                        idx = COLORS_LIST.index(c)
                        self.__isosurfaceList[i][2].setCurrentIndex(idx)
                        break

    def getParameters(self):
        ddict = {}
        key = 'isosurfaces'
        ddict[key] = []
        for i in range(self.__nIsosurfaces):
            if self.__isosurfaceList[i][0].isChecked():
                cb = 1
            else:
                cb = 0
            le = float(self.__isosurfaceList[i][1].text())
            label = str(self.__isosurfaceList[i][2].currentText())
            r, g, b, a = COLORS[label]
            ddict[key].append([cb, le, label, r, g, b, a])
        return ddict

if __name__ == "__main__":
    import sys
    app = qt.QApplication(sys.argv)
    def mySlot(ddict):
        print(ddict)
    w = Isosurfaces()
    #w.sigColorFilterSignal.connect(mySlot)
    w.show()
    app.exec_()
