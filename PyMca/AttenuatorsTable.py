#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
__revision__ = "$Revision: 1.26 $"
import sys
from PyMca import PyMcaQt as qt
QTVERSION = qt.qVersion()

if QTVERSION < '4.0.0':
    import qttable

if QTVERSION > '4.0.0':
    qt.QLabel.AlignRight = qt.Qt.AlignRight
    qt.QLabel.AlignCenter = qt.Qt.AlignCenter
    qt.QLabel.AlignVCenter = qt.Qt.AlignVCenter

    class Q3GridLayout(qt.QGridLayout):
        def addMultiCellWidget(self, w, r0, r1, c0, c1, *var):
            self.addWidget(w, r0, c0, 1 + r1 - r0, 1 + c1 - c0)
from PyMca import Elements
from PyMca import MaterialEditor
from PyMca import MatrixEditor
import re
DEBUG = 0


class MyQLabel(qt.QLabel):
    def __init__(self, parent=None, name=None, fl=0, bold=True,
                 color= qt.Qt.red):
        qt.QLabel.__init__(self, parent)
        if QTVERSION <'4.0.0':
            self.color = color
            self.bold = bold
        else:
            palette = self.palette()
            role = self.foregroundRole()
            palette.setColor(role, color)
            self.setPalette(palette)
            self.font().setBold(bold)


    if QTVERSION < '4.0.0':
        def drawContents(self, painter):
            painter.font().setBold(self.bold)
            pal = self.palette()
            pal.setColor(qt.QColorGroup.Foreground, self.color)
            self.setPalette(pal)
            qt.QLabel.drawContents(self, painter)
            painter.font().setBold(0)

class AttenuatorsTab(qt.QWidget):
    def __init__(self, parent=None, name="Attenuators Tab",
                 attenuators=None, graph=None):
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        if QTVERSION > '4.0.0':
            maxheight = qt.QDesktopWidget().height()
            if maxheight < 800:
                layout.setMargin(0)
                layout.setSpacing(2)
        self.table  = AttenuatorsTableWidget(self, name, attenuators,
                                             funnyfilters=True)
        layout.addWidget(self.table)
        spacer = MaterialEditor.VerticalSpacer(self)
        layout.addWidget(spacer)
        if QTVERSION < '4.0.0':
            self.mainTab = qt.QTabWidget(self, "mainTab")
            layout.addWidget(self.mainTab)
            self.editor = MaterialEditor.MaterialEditor(self.mainTab,
                                                        "tabEditor")
            self.mainTab.insertTab(self.editor, str("Material Editor"))
            self.table.setMinimumHeight(1.1 * self.table.sizeHint().height())
            self.table.setMaximumHeight(1.1*self.table.sizeHint().height())
        else:
            self.mainTab = qt.QTabWidget(self)
            layout.addWidget(self.mainTab)
            rheight = self.table.horizontalHeader().sizeHint().height()
            if maxheight < 800:
                self.editor = MaterialEditor.MaterialEditor(height=5,
                                                            graph=graph)
                self.table.setMinimumHeight(7 * rheight)
                self.table.setMaximumHeight(13 * rheight)
            else:
                self.editor = MaterialEditor.MaterialEditor(graph=graph)
                self.table.setMinimumHeight(13*rheight)
                self.table.setMaximumHeight(13*rheight)
            self.mainTab.addTab(self.editor, "Material Editor")

class MultilayerTab(qt.QWidget):
    def __init__(self,parent=None, name="Multilayer Tab", matrixlayers=None):
        if matrixlayers is None:
            matrixlayers=["Layer0", "Layer1", "Layer2", "Layer3",
                          "Layer4", "Layer5", "Layer6", "Layer7",
                          "Layer8", "Layer9"]
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)

        self.matrixGeometry = MatrixEditor.MatrixEditor(self, "tabMatrix",
                                   table=False, orientation="horizontal",
                                   density=False, thickness=False,
                                   size="image2")
        layout.addWidget(self.matrixGeometry)

        text = "This matrix definition will only be "
        text += "considered if Matrix is selected and material is set to "
        text += "MULTILAYER in the ATTENUATORS tab.\n  "
        self.matrixInfo  = qt.QLabel(self)
        layout.addWidget(self.matrixInfo)
        self.matrixInfo.setText(text)
        self.matrixTable = AttenuatorsTableWidget(self, name,
                                                  attenuators=matrixlayers,
                                                  matrixmode=True)
        layout.addWidget(self.matrixTable)

class CompoundFittingTab(qt.QWidget):
    def __init__(self, parent=None, name="Compound Tab",
                    layerlist=None):
        qt.QWidget.__init__(self, parent)
        if layerlist is None:
            self.nlayers = 5
        else:
            self.nlayers = len(layerlist)
        layout = qt.QVBoxLayout(self)
        hbox = qt.QWidget(self)
        hboxlayout  = qt.QHBoxLayout(hbox)
        #hboxlayout.addWidget(qt.HorizontalSpacer(hbox))
        self._compoundFittingLabel = MyQLabel(hbox, color=qt.Qt.red)
        self._compoundFittingLabel.setText("Compound Fitting Mode is OFF")
        self._compoundFittingLabel.setAlignment(qt.QLabel.AlignCenter)
        hboxlayout.addWidget(self._compoundFittingLabel)
        #hboxlayout.addWidget(qt.HorizontalSpacer(hbox))
        layout.addWidget(hbox)

        grid = qt.QWidget(self)
        if QTVERSION < '4.0.0':
            glt = qt.QGridLayout(grid, 1, 1, 11, 2, "gridlayout")
        else:
            glt = Q3GridLayout(grid)
            glt.setMargin(11)
            glt.setSpacing(2)

        self._layerFlagWidgetList = []
        options = ["FREE", "FIXED", "IGNORED"]
        for i in range(self.nlayers):
            r = int(i / 5)
            c = 3 * (i % 5)
            label = qt.QLabel(grid)
            label.setText("Layer%d" % i)
            cbox = qt.QComboBox(grid)
            if QTVERSION < '4.0.0':
                cbox.insertStrList(options)
                if i == 0:
                    cbox.setCurrentItem(0)
                else:
                    cbox.setCurrentItem(1)
            else:
                for item in options:
                    cbox.addItem(item)
                if i == 0:
                    cbox.setCurrentIndex(0)
                else:
                    cbox.setCurrentIndex(1)
            glt.addWidget(label, r, c)
            glt.addWidget(cbox, r, c + 1)
            glt.addWidget(qt.QWidget(grid), r, c + 2)
        
        layout.addWidget(grid)
        if QTVERSION < '4.0.0':
            self.mainTab = qt.QTabWidget(self, "mainTab")
            layout.addWidget(self.mainTab)
            self._editorList = []
            for i in range(self.nlayers):
                editor = CompoundFittingTab0(self.mainTab, layerindex=i)
                self.mainTab.insertTab(editor, str("Layer%d" % i))
                self._editorList.append(editor)
        else:
            self.mainTab = qt.QTabWidget(self)
            layout.addWidget(self.mainTab)
            self._editorList = []
            for i in range(self.nlayers):
                editor = CompoundFittingTab0(layerindex=i)
                self.mainTab.addTab(editor, "layer Editor")
                self._editorList.append(editor)


class CompoundFittingTab0(qt.QWidget):
    def __init__(self, parent=None, name="Compound Tab",
                    layerindex=None, compoundlist=None):
        if layerindex is None:
            layerindex = 0
        if compoundlist is None:
            compoundlist = []
            for i in range(10):
                compoundlist.append("Compound%d%d" % (layerindex, i))
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)

        grid = qt.QWidget(self)
        if QTVERSION < '4.0.0':
            gl = qt.QGridLayout(grid, 1, 1, 11, 2, "gridlayout")
        else:
            gl = Q3GridLayout(grid)
            gl.setMargin(11)
            gl.setSpacing(2)

        # Layer name
        nameLabel = qt.QLabel(grid)
        nameLabel.setText("Name")

        self.nameLine = qt.QLineEdit(grid)
        self.nameLine.setText("Compound fitting layer %d" % layerindex)

        gl.addWidget(nameLabel, 0, 0)
        gl.addMultiCellWidget(self.nameLine, 0, 0, 1, 5)

        Line = qt.QFrame(grid)
        Line.setFrameShape(qt.QFrame.HLine)
        Line.setFrameShadow(qt.QFrame.Sunken)
        Line.setFrameShape(qt.QFrame.HLine)
        gl.addMultiCellWidget(Line, 1, 1, 0, 5)

        #labels
        fixedLabel = qt.QLabel(grid)
        fixedLabel_font = qt.QFont(fixedLabel.font())
        fixedLabel_font.setItalic(1)
        fixedLabel.setFont(fixedLabel_font)
        fixedLabel.setText(str("Fixed"))
        if QTVERSION < '4.0.0':
            fixedLabel.setAlignment(qt.QLabel.AlignVCenter)
            fixedLabel.setAlignment(qt.QLabel.AlignHCenter)
        else:
            fixedLabel.setAlignment(qt.Qt.AlignVCenter)

        valueLabel = qt.QLabel(grid)
        valueLabel_font = qt.QFont(valueLabel.font())
        valueLabel_font.setItalic(1)
        valueLabel.setFont(valueLabel_font)
        valueLabel.setText(str("Value"))
        valueLabel.setAlignment(qt.QLabel.AlignCenter)


        errorLabel = qt.QLabel(grid)
        errorLabel_font = qt.QFont(errorLabel.font())
        errorLabel_font.setItalic(1)
        errorLabel.setFont(errorLabel_font)
        errorLabel.setText(str("Error"))
        errorLabel.setAlignment(qt.QLabel.AlignCenter)

        gl.addWidget(fixedLabel, 2, 2)
        gl.addWidget(valueLabel, 2, 3)
        gl.addWidget(errorLabel, 2, 5)
        
        #density
        densityLabel = qt.QLabel(grid)
        densityLabel.setText("Density")
        
        self.densityCheck = qt.QCheckBox(grid)
        self.densityCheck.setText(str(""))

        self.densityValue = qt.QLineEdit(grid)
        densitySepLabel = qt.QLabel(grid)
        densitySepLabel_font = qt.QFont(densitySepLabel.font())
        densitySepLabel_font.setBold(1)
        densitySepLabel.setFont(densitySepLabel_font)
        densitySepLabel.setText(str("+/-"))

        self.densityError = qt.QLineEdit(grid)

        gl.addWidget(densityLabel, 3, 0)
        gl.addWidget(qt.HorizontalSpacer(grid), 3, 1)
        gl.addWidget(self.densityCheck, 3, 2)
        gl.addWidget(self.densityValue, 3, 3)
        gl.addWidget(densitySepLabel, 3, 4)
        gl.addWidget(self.densityError, 3, 5)
        
        #thickness
        thicknessLabel = qt.QLabel(grid)
        thicknessLabel.setText("Thickness")
        
        self.thicknessCheck = qt.QCheckBox(grid)
        self.thicknessCheck.setText(str(""))

        self.thicknessValue = qt.QLineEdit(grid)
        thicknessSepLabel = qt.QLabel(grid)
        thicknessSepLabel_font = qt.QFont(thicknessSepLabel.font())
        thicknessSepLabel_font.setBold(1)
        thicknessSepLabel.setFont(thicknessSepLabel_font)
        thicknessSepLabel.setText(str("+/-"))

        self.thicknessError = qt.QLineEdit(grid)

        gl.addWidget(thicknessLabel, 4, 0)
        gl.addWidget(self.thicknessCheck, 4, 2)
        gl.addWidget(self.thicknessValue, 4, 3)
        gl.addWidget(thicknessSepLabel, 4, 4)
        gl.addWidget(self.thicknessError, 4, 5)
        
        Line = qt.QFrame(grid)
        Line.setFrameShape(qt.QFrame.HLine)
        Line.setFrameShadow(qt.QFrame.Sunken)
        Line.setFrameShape(qt.QFrame.HLine)
        gl.addMultiCellWidget(Line, 5, 5, 0, 5)
        
        layout.addWidget(grid)
        """
        self.matrixGeometry = MatrixEditor.MatrixEditor(self,"tabMatrix",
                                   table=False, orientation="horizontal",
                                   density=False, thickness=False,
                                   size="image2")
        layout.addWidget(self.matrixGeometry)
        
        text  ="This matrix definition will only be "
        text +="considered if Matrix is selected and material is set to "
        text +="MULTILAYER in the ATTENUATORS tab.\n  "
        self.matrixInfo  = qt.QLabel(self)
        layout.addWidget(self.matrixInfo)
        self.matrixInfo.setText(text)
        """ 
        self.matrixTable = AttenuatorsTableWidget(self, name,
                                                  attenuators=compoundlist,
                                                  matrixmode=False,
                                                  compoundmode=True,
                                                  layerindex=layerindex)
        layout.addWidget(self.matrixTable)


if QTVERSION < '4.0.0':
    class QTable(qttable.QTable):
        def __init__(self, parent=None, name=""):
            qttable.QTable.__init__(self, parent, name)
            self.rowCount = self.numRows
            self.columnCount = self.numCols
            self.setRowCount = self.setNumRows
            self.setColumnCount = self.setNumCols
        
else:
    QTable = qt.QTableWidget


class AttenuatorsTableWidget(QTable):
    def __init__(self, parent=None, name="Attenuators Table",
                 attenuators=None, matrixmode=None, compoundmode=None,
                 layerindex=0, funnyfilters=False):
        attenuators0 = ["Atmosphere", "Air", "Window", "Contact", "DeadLayer",
                       "Filter5", "Filter6", "Filter7", "BeamFilter1",
                       "BeamFilter2", "Detector", "Matrix"]

        if QTVERSION < '4.0.0':
            QTable.__init__(self, parent, name)
            self.setCaption(name)
        else:
            QTable.__init__(self, parent)
            self.setWindowTitle(name)

        if attenuators is None:
            attenuators = attenuators0
        if matrixmode is None:
            matrixmode = False
        if matrixmode:
            self.compoundMode = False
        elif compoundmode is None:
            self.compoundMode = False
        else:
            self.compoundMode = compoundmode
        if funnyfilters is None:
            funnyfilters = False
        self.funnyFiltersMode = funnyfilters
        if self.compoundMode:
            self.funnyFiltersMode = False
            labels = ["Compound", "Name", "Material", "Initial Amount"]        
        else:
            if self.funnyFiltersMode:
                labels = ["Attenuator", "Name", "Material",
                          "Density (g/cm3)", "Thickness (cm)", "Funny Factor"]
            else:
                labels = ["Attenuator", "Name", "Material",
                          "Density (g/cm3)", "Thickness (cm)"]
        self.layerindex = layerindex
        self.matrixMode = matrixmode
        self.attenuators = attenuators
        self.verticalHeader().hide()
        if QTVERSION < '4.0.0':
            self.setLeftMargin(0)
            self.setFrameShape(qttable.QTable.NoFrame)
            #self.setFrameShadow(qttable.QTable.Sunken)
            self.setSelectionMode(qttable.QTable.Single)
            self.setFocusStyle(qttable.QTable.FollowStyle)
            self.setNumCols(len(labels))
            for label in labels:
                self.horizontalHeader().setLabel(labels.index(label), label)
        else:
            if DEBUG:
                print("margin to adjust")
                print("focus style")
            self.setFrameShape(qt.QTableWidget.NoFrame)
            self.setSelectionMode(qt.QTableWidget.NoSelection)
            self.setColumnCount(len(labels))
            for i in range(len(labels)):
                item = self.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(labels[i],
                                               qt.QTableWidgetItem.Type)
                item.setText(labels[i])
                self.setHorizontalHeaderItem(i,item)
        if self.matrixMode:
            self.__build(len(attenuators))
        elif self.compoundMode:
            self.__build(len(attenuators))
        else:
            self.__build(len(attenuators0))
            #self.adjustColumn(0)
        if self.matrixMode:
            if QTVERSION < '4.0.0':
                self.horizontalHeader().setLabel(0, 'Layer')
            else:
                item = self.horizontalHeaderItem(0)
                item.setText('Layer')
                self.setHorizontalHeaderItem(0, item)
        if self.compoundMode:
            if QTVERSION < '4.0.0':
                self.adjustColumn(0)
                self.adjustColumn(1)
            else:
                self.resizeColumnToContents(0)
                self.resizeColumnToContents(1)
                                           
        self.connect(self, qt.SIGNAL("valueChanged(int,int)"), self.mySlot)

    def __build(self, nfilters=12):
        n = 0
        if (not self.matrixMode) and (not self.compoundMode):
            n = 4
            #self.setNumRows(nfilters+n)
            if QTVERSION < '4.0.0':
                self.setNumRows(12)
            else:
                self.setRowCount(12)
        else:
            if QTVERSION < '4.0.0':
                self.setNumRows(nfilters)
            else:
                self.setRowCount(nfilters)
        if QTVERSION > '4.0.0':
            rheight = self.horizontalHeader().sizeHint().height()
            for idx in range(self.rowCount()):
                self.setRowHeight(idx, rheight)

        self.comboList = []
        matlist = list(Elements.Material.keys())
        matlist.sort()
        if self.matrixMode or self.compoundMode:
            if self.matrixMode:
                roottext = "Layer"
            else:
                roottext = "Compound%d" % self.layerindex
            a = []
            #a.append('')
            for key in matlist:
                a.append(key)
            if QTVERSION < '4.0.0':
                for idx in range(self.numRows()):
                    item= qttable.QCheckTableItem(self, roottext + "%d" % idx)
                    self.setItem(idx, 0, item)
                    item.setText(roottext + "%d" % idx)
                    #item= qttable.QTableItem(self,
                    #                         qttable.QTableItem.OnTyping,
                    item= qttable.QTableItem(self, qttable.QTableItem.Never,
                                             self.attenuators[idx])
                    self.setItem(idx, 1, item)
                    combo = MyQComboBox(options=a)
                    combo.setEditable(True)
                    self.setCellWidget(idx, 2, combo)
                    qt.QObject.connect(combo,
                                       qt.PYSIGNAL("MaterialComboBoxSignal"),
                                       self._comboSlot)
            else:
                for idx in range(self.rowCount()):
                    item= qt.QCheckBox(self)
                    self.setCellWidget(idx, 0, item)
                    text = roottext+"%d" % idx
                    item.setText(text)
                    item = self.item(idx, 1)
                    if item is None:
                        item = qt.QTableWidgetItem(text,
                                                   qt.QTableWidgetItem.Type)
                        self.setItem(idx, 1, item)
                    else:
                        item.setText(text)
                    item.setFlags(qt.Qt.ItemIsSelectable|
                                  qt.Qt.ItemIsEnabled)
                    combo = MyQComboBox(self, options=a, row = idx, col = 2)
                    combo.setEditable(True)
                    self.setCellWidget(idx, 2, combo)
                    qt.QObject.connect(combo,
                                       qt.SIGNAL("MaterialComboBoxSignal"),
                                       self._comboSlot)
            return
        if QTVERSION < '4.0.0':
            selfnumRows = self.numRows()
        else:
            selfnumRows = self.rowCount()
            
        for idx in range(selfnumRows - n):
            text = "Filter% 2d" % idx
            if QTVERSION < '4.0.0':
                item = qttable.QCheckTableItem(self, text)
                self.setItem(idx, 0, item)
                item.setText(text)
                if idx < len(self.attenuators):
                    self.setText(idx, 1,self.attenuators[idx])
                else:
                    self.setText(idx, 1,text)
            else:
                item = qt.QCheckBox(self)
                self.setCellWidget(idx, 0, item)
                item.setText(text)
                if idx < len(self.attenuators):
                    text = self.attenuators[idx]

                item = self.item(idx, 1)
                if item is None:
                    item = qt.QTableWidgetItem(text,
                                               qt.QTableWidgetItem.Type)
                    self.setItem(idx, 1, item)
                else:
                    item.setText(text)

            #a = qt.QStringList()
            a = []
            #a.append('')            
            for key in matlist:
                a.append(key)
            combo = MyQComboBox(self, options=a, row=idx, col = 2)
            combo.setEditable(True)
            self.setCellWidget(idx, 2, combo)
            #self.setItem(idx,2,combo)
            if QTVERSION < '4.0.0':
                qt.QObject.connect(combo,
                                   qt.PYSIGNAL("MaterialComboBoxSignal"),
                                   self._comboSlot)
            else:
                qt.QObject.connect(combo,
                                   qt.SIGNAL("MaterialComboBoxSignal"),
                                   self._comboSlot)

        for i in range(2):
            #BeamFilter(i)
            if QTVERSION < '4.0.0':
                item = qttable.QCheckTableItem(self, "BeamFilter%d" % i)
                #,color=qt.Qt.red)
                idx = self.numRows() - (4 - i)
                self.setItem(idx, 0, item)
                item.setText("BeamFilter%d" % i)
                item= qttable.QTableItem(self, qttable.QTableItem.Never,
                                         "BeamFilter%d" % i)
                self.setItem(idx, 1, item)
                item= qttable.QTableItem(self, qttable.QTableItem.Never,
                                         "1.0")
                self.setItem(idx, 5,item)
            else:
                item = qt.QCheckBox(self)
                idx = self.rowCount() - (4 - i)
                self.setCellWidget(idx, 0, item)
                text = "BeamFilter%d" % i
                item.setText(text)

                item = self.item(idx,1)
                if item is None:
                    item = qt.QTableWidgetItem(text,
                                               qt.QTableWidgetItem.Type)
                    self.setItem(idx, 1, item)
                else:
                    item.setText(text)
                item.setFlags(qt.Qt.ItemIsSelectable|
                              qt.Qt.ItemIsEnabled)

                text = "1.0"
                item = self.item(idx, 5)
                if item is None:
                    item = qt.QTableWidgetItem(text,
                                               qt.QTableWidgetItem.Type)
                    self.setItem(idx, 5, item)
                else:
                    item.setText(text)
                item.setFlags(qt.Qt.ItemIsSelectable|
                              qt.Qt.ItemIsEnabled)

            combo = MyQComboBox(self, options=a, row=idx, col=2)
            combo.setEditable(True)
            self.setCellWidget(idx, 2, combo)
            if QTVERSION < '4.0.0':
                qt.QObject.connect(combo,
                                   qt.PYSIGNAL("MaterialComboBoxSignal"),
                                   self._comboSlot)
            else:
                qt.QObject.connect(combo,
                                   qt.SIGNAL("MaterialComboBoxSignal"),
                                   self._comboSlot)
            
        if QTVERSION < '4.0.0':
            #Detector
            item= qttable.QCheckTableItem(self, "Detector")
            #,color=qt.Qt.red)
            idx = self.numRows()-2
            self.setItem(idx, 0, item)
            item.setText("Detector")
            item= qttable.QTableItem(self,
                                     qttable.QTableItem.Never,
                                     "Detector")
            self.setItem(idx, 1, item)
            item= qttable.QTableItem(self,
                                     qttable.QTableItem.Never,
                                     "1.0")
            self.setItem(idx, 5, item)
        else:
            item = qt.QCheckBox(self)
            idx = self.rowCount() - 2
            self.setCellWidget(idx, 0, item)
            text = "Detector"
            item.setText(text)

            item = self.item(idx,1)
            if item is None:
                item = qt.QTableWidgetItem(text,
                                           qt.QTableWidgetItem.Type)
                self.setItem(idx, 1, item)
            else:
                item.setText(text)
            item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)

            text = "1.0"
            item = self.item(idx, 5)
            if item is None:
                item = qt.QTableWidgetItem(text,
                                           qt.QTableWidgetItem.Type)
                self.setItem(idx, 5, item)
            else:
                item.setText(text)
            item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
            
        combo = MyQComboBox(self, options=a, row=idx, col=2)
        combo.setEditable(True)
        self.setCellWidget(idx, 2, combo)
        if QTVERSION < '4.0.0':
            #Matrix            
            item= qttable.QCheckTableItem(self, "Matrix")
            #,color=qt.Qt.red)
            idx = self.numRows() - 1
            self.setItem(idx, 0, item)
            item.setText("Matrix")
            item= qttable.QTableItem(self,
                                     qttable.QTableItem.Never,
                                     "Matrix")
            self.setItem(idx, 1, item)
            item= qttable.QTableItem(self,
                                     qttable.QTableItem.Never,
                                     "1.0")
            self.setItem(idx, 5, item)
        else:
            item = qt.QCheckBox(self)
            idx = self.rowCount() - 1
            self.setCellWidget(idx, 0, item)
            text = "Matrix"
            item.setText(text)
            item = self.item(idx, 1)
            if item is None:
                item = qt.QTableWidgetItem(text,
                                           qt.QTableWidgetItem.Type)
                self.setItem(idx, 1, item)
            else:
                item.setText(text)
            item.setFlags(qt.Qt.ItemIsSelectable |qt.Qt.ItemIsEnabled)

            text = "1.0"
            item = self.item(idx, 5)
            if item is None:
                item = qt.QTableWidgetItem(text,
                                           qt.QTableWidgetItem.Type)
                self.setItem(idx, 5, item)
            else:
                item.setText(text)
            item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)

        if QTVERSION < '4.0.0':
            qt.QObject.connect(combo,
                               qt.PYSIGNAL("MaterialComboBoxSignal"),
                               self._comboSlot)
        else:
            qt.QObject.connect(combo,
                               qt.SIGNAL("MaterialComboBoxSignal"),
                               self._comboSlot)

        #a = qt.QStringList()
        a = []
        #a.append('')
        for key in matlist:
            a.append(key)
        #combo = qttable.QComboTableItem(self,a)
        self.combo = MyQComboBox(self, options=a, row=idx, col=2)
        self.setCellWidget(idx, 2, self.combo)
        if QTVERSION < '4.0.0':
            self.connect(self.combo,
                         qt.PYSIGNAL("MaterialComboBoxSignal"),
                         self._comboSlot)
        else:
            self.connect(self.combo,
                         qt.SIGNAL("MaterialComboBoxSignal"),
                         self._comboSlot)
            
    def mySlot(self,row,col):
        if DEBUG:
            print("Value changed row = %d cole = &d" % (row, col))
            print("Text = %s" % self.text(row, col))

    def _comboSlot(self, ddict):
        if DEBUG:
            print("_comboSlot", ddict)
        row = ddict['row']
        col = ddict['col']
        text = ddict['text']
        self.setCurrentCell(row, col)
        self._checkDensityThickness(text, row)
        if QTVERSION < '4.0.0':
            self.emit(qt.SIGNAL("valueChanged(int,int)"), (row, col))
        else:
            self.emit(qt.SIGNAL("valueChanged(int,int)"), row, col)

    def text(self, row, col):
        if col == 2:
            return self.cellWidget(row, col).currentText()
        else:
            if QTVERSION < '4.0.0':
                return qttable.QTable.text(self, row, col)
            else:
                if col not in [1, 3, 4, 5]:
                    print("row, col = %d, %d" % (row, col))
                    print("I should not be here")
                else:
                    item = self.item(row, col)
                    return item.text()

    def setText(self, row, col, text):
        if QTVERSION < "4.0.0":
            QTable.setText(self, row, col, text)
        else:
            if col == 0:
                self.cellWidget(row, 0).setText(text)
                return
            if col not in [1, 3, 4, 5]:
                print("only compatible columns 1, 3 and 4")
                raise ValueError("method for column > 2")
            item = self.item(row, col)
            if item is None:
                item = qt.QTableWidgetItem(text,
                                           qt.QTableWidgetItem.Type)
                self.setItem(row, col, item)
            else:
                item.setText(text)

    def setCellWidget(self, row, col, w):
        if QTVERSION < '4.0.0':
            w.row = row
            w.col = col
        QTable.setCellWidget(self, row, col, w)

    def _checkDensityThickness(self, text, row):
        try:
            currentDensity = float(str(self.text(row, 3)))
        except:
            currentDensity = 0.0
        try:
            currentThickness = float(str(self.text(row, 4)))
        except:
            currentThickness = 0.0
        defaultDensity = -1.0
        defaultThickness = -0.1
        #check if default density is there
        if Elements.isValidFormula(text):
            #check if single element
            if text in Elements.Element.keys():
                defaultDensity = Elements.Element[text]['density']
            else: 
                elts = [ w for w in re.split('[0-9]', text) if w != '']
                nbs = [ int(w) for w in re.split('[a-zA-Z]', text) if w != '']
                if len(elts) == 1 and len(nbs) == 1:
                    defaultDensity = Elements.Element[elts[0]]['density']
        elif Elements.isValidMaterial(text):
            key = Elements.getMaterialKey(text)
            if key is not None:
                if 'Density' in Elements.Material[key]:
                    defaultDensity = Elements.Material[key]['Density']
                if 'Thickness' in Elements.Material[key]:
                    defaultThickness = Elements.Material[key]['Thickness'] 
        if defaultDensity >= 0.0:
            self.setText(row, 3, "%g" % defaultDensity)
        elif currentDensity <= 0:
            # should not be better to raise an exception if the
            # entered density or thickness were negative?
            self.setText(row, 3, "%g" % 1.0)
        if defaultThickness >= 0.0:
            self.setText(row, 4, "%g" % defaultThickness)
        elif currentThickness <= 0.0:
            # should not be better to raise an exception if the
            # entered density or thickness were negative?
            self.setText(row, 4, "%g" % 0.1)

class MyQComboBox(MaterialEditor.MaterialComboBox):
    def _mySignal(self, qstring0):
        qstring = qstring0
        (result, index) = self.ownValidator.validate(qstring, 0)
        if result != self.ownValidator.Valid:
            qstring = self.ownValidator.fixup(qstring)
            (result, index) = self.ownValidator.validate(qstring,0)
        if result != self.ownValidator.Valid:
            text = str(qstring)
            if text.upper() != "MULTILAYER":
                qt.QMessageBox.critical(self, "Invalid Material '%s'" % text,
                                        "The material '%s' is not a valid Formula " \
                                        "nor a valid Material.\n" \
                                        "Please define the material %s or correct the formula\n" % \
                                        (text, text))
                if QTVERSION < '4.0.0':
                    self.setCurrentItem(0)
                else:
                    self.setCurrentIndex(0)
                for i in range(self.count()):
                    if QTVERSION < '4.0.0':
                        selftext = self.text(i)
                    else:
                        selftext = self.itemText(i)
                    if selftext == qstring0:
                        self.removeItem(i)
                        break
                return
        text = str(qstring)
        self.setCurrentText(text)
        ddict = {}
        ddict['event'] = 'activated'
        ddict['row'] = self.row
        ddict['col'] = self.col
        ddict['text'] = text
        if qstring0 != qstring:
            self.removeItem(self.count() - 1)
        insert = True
        for i in range(self.count()):
            if QTVERSION < '4.0.0':
                selftext = self.text(i)
            else:
                selftext = self.itemText(i)
            if qstring == selftext:
                insert = False
        if insert:
            self.insertItem(-1, qstring)

        if self.lineEdit() is not None:
            if QTVERSION < '4.0.0':
                self.lineEdit().setPaletteBackgroundColor(qt.QColor("white"))
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('MaterialComboBoxSignal'), (ddict,))
        else:
            self.emit(qt.SIGNAL('MaterialComboBoxSignal'), ddict)

def main(args):
    app = qt.QApplication(args)
    #tab = AttenuatorsTableWidget(None)
    if len(args) < 2:
        tab = AttenuatorsTab(None)
    elif len(args) > 3:
        tab = CompoundFittingTab(None)
    else:
        tab = MultilayerTab(None)
    if QTVERSION < '4.0.0':
        tab.show()
        app.setMainWidget(tab)
        app.exec_loop()
    else:
        tab.show()
        app.exec_()


if __name__=="__main__":
    main(sys.argv)

