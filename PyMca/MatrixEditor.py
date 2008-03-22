#/*##########################################################################
# Copyright (C) 2004-2008 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
__revision__ = "$Revision: 1.9 $"
from MaterialEditor import qt
import MaterialEditor
import MatrixImage

QTVERSION = qt.qVersion()

class MatrixEditor(qt.QWidget):
    def __init__(self, parent=None, name="Matrix Editor",current=None, 
                    table=True,orientation="vertical",thickness=True,
                   density=True, size=None):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self, parent, name)
            self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
            self.setWindowTitle(name)
            
        self._current={'Density':     1.0,
                       'Thickness':   1.0,
                       'AlphaIn':    45.0,
                       'AlphaOut':   45.0,
                       'AlphaScatteringFlag':False,
                       'AlphaScattering': 90,
                       'Material':  "Water"}
        if current is not None: self._current.update(current)
        self.build(table,orientation, thickness, density, size)
        self._update()
        
    def build(self, table,orientation, thickness, density, size=None):
        if size is None: size="medium"
        layout = qt.QHBoxLayout(self)
        
        if table:
            #the material definition
            matBox = qt.QWidget(self)
            layout.addWidget(matBox)
            matBoxLayout = qt.QVBoxLayout(matBox)
            self.materialEditor =  MaterialEditor.MaterialEditor(matBox,
                                                         comments=False, height=7)
            matBoxLayout.addWidget(self.materialEditor)
            matBoxLayout.addWidget(VerticalSpacer(matBox))
        else:
            self.materialEditor = None
        
        #the matrix definition
        sampleBox = qt.QWidget(self)
        layout.addWidget(sampleBox)
        if orientation == "vertical":
            sampleBoxLayout = qt.QVBoxLayout(sampleBox)
        else:
            sampleBoxLayout = qt.QHBoxLayout(sampleBox)            
        
        #the image
        if orientation == "vertical":
            labelHBox = qt.QWidget(sampleBox)
            sampleBoxLayout.addWidget(labelHBox)
            labelHBoxLayout = qt.QHBoxLayout(labelHBox)
            labelHBoxLayout.addWidget(HorizontalSpacer(labelHBox))
            label = MatrixImage.MatrixImage(labelHBox,size=size)
            labelHBoxLayout.addWidget(label)
            labelHBoxLayout.addWidget(HorizontalSpacer(labelHBox))
        else:
            labelHBox = qt.QWidget(sampleBox)
            sampleBoxLayout.addWidget(labelHBox)
            labelHBoxLayout = qt.QVBoxLayout(labelHBox)
            labelHBoxLayout.setMargin(11)
            labelHBoxLayout.setSpacing(4)
            label = MatrixImage.MatrixImage(labelHBox,size=size)
            labelHBoxLayout.addWidget(label)
        if orientation != "vertical":
            labelHBoxLayout.addWidget(VerticalSpacer(labelHBox))
        self.imageLabel = label
        #the input fields container
        self.__gridSampleBox = qt.QWidget(sampleBox)
        grid = self.__gridSampleBox
        sampleBoxLayout.addWidget(grid)
        if QTVERSION < '4.0.0':
            gridLayout=qt.QGridLayout(grid,5,2,11,4)
        else:
            gridLayout = qt.QGridLayout(grid)
            gridLayout.setMargin(11)
            gridLayout.setSpacing(4)            
        
        #the angles
        angle1Label  = qt.QLabel(grid)
        angle1Label.setText("Incoming Angle (deg.):")
        self.__angle1Line  = MyQLineEdit(grid)
        self.__angle1Line.setReadOnly(False)

        angle2Label  = qt.QLabel(grid)
        angle2Label.setText("Outgoing Angle (deg.):")
        self.__angle2Line  = MyQLineEdit(grid)
        self.__angle2Line.setReadOnly(False)

        self.__angle3Label  = qt.QCheckBox(grid)
        self.__angle3Label.setText("Scattering Angle (deg.):")
        self.__angle3Line  = MyQLineEdit(grid)
        self.__angle3Line.setReadOnly(False)
        self.__angle3Line.setDisabled(True)
        if QTVERSION < '4.0.0':
            angle1Label.setAlignment(qt.QLabel.WordBreak | \
                                     qt.QLabel.AlignVCenter)
            angle2Label.setAlignment(qt.QLabel.WordBreak | \
                                     qt.QLabel.AlignVCenter)
            self.__angle3Label.setAlignment(qt.QLabel.WordBreak | \
                                     qt.QLabel.AlignVCenter)
        else:
            angle1Label.setAlignment(qt.Qt.AlignVCenter)
            angle2Label.setAlignment(qt.Qt.AlignVCenter)

        self.__angle3Label.setChecked(0)

        gridLayout.addWidget(angle1Label, 0, 0)
        gridLayout.addWidget(self.__angle1Line, 0, 1)
        gridLayout.addWidget(angle2Label, 1, 0)
        gridLayout.addWidget(self.__angle2Line, 1, 1)
        gridLayout.addWidget(self.__angle3Label, 2, 0)
        gridLayout.addWidget(self.__angle3Line, 2, 1)

        rowoffset = 3
        #thickness and density
        if density:
            densityLabel  = qt.QLabel(grid)
            densityLabel.setText("Sample Density (g/cm3):")
            if QTVERSION < '4.0.0':
                densityLabel.setAlignment(qt.QLabel.WordBreak | \
                                          qt.QLabel.AlignVCenter)
            else:
                densityLabel.setAlignment(qt.Qt.AlignVCenter)
            self.__densityLine  = MyQLineEdit(grid)
            self.__densityLine.setReadOnly(False)
            gridLayout.addWidget(densityLabel, rowoffset, 0)
            gridLayout.addWidget(self.__densityLine, rowoffset, 1)
            rowoffset = rowoffset + 1
        else:
            self.__densityLine = None

        if thickness:
            thicknessLabel  = qt.QLabel(grid)
            thicknessLabel.setText("Sample Thickness   (cm):")
            if QTVERSION < '4.0.0':
                thicknessLabel.setAlignment(qt.QLabel.WordBreak | \
                                            qt.QLabel.AlignVCenter)
            else:
                thicknessLabel.setAlignment(qt.Qt.AlignVCenter)
            self.__thicknessLine  = MyQLineEdit(grid)
            self.__thicknessLine.setReadOnly(False)
            gridLayout.addWidget(thicknessLabel, rowoffset, 0)
            gridLayout.addWidget(self.__thicknessLine, rowoffset, 1)
            rowoffset = rowoffset + 1
        else:
            self.__thicknessLine  = None

        if QTVERSION < '4.0.0':
            self.connect(self.__angle1Line,qt.PYSIGNAL('MyQLineEditSignal'),
                         self.__angle1Slot)
            self.connect(self.__angle2Line, qt.PYSIGNAL('MyQLineEditSignal'),
                         self.__angle2Slot)
            self.connect(self.__angle3Line, qt.PYSIGNAL('MyQLineEditSignal'),
                         self.__angle3Slot)
            if self.__densityLine is not None:
                self.connect(self.__densityLine, qt.PYSIGNAL('MyQLineEditSignal'),
                         self.__densitySlot)
            if self.__thicknessLine is not None:
                self.connect(self.__thicknessLine,qt.PYSIGNAL('MyQLineEditSignal'),
                         self.__thicknessSlot)
        else:
            self.connect(self.__angle1Line,qt.SIGNAL('MyQLineEditSignal'),
                         self.__angle1Slot)
            self.connect(self.__angle2Line, qt.SIGNAL('MyQLineEditSignal'),
                         self.__angle2Slot)
            self.connect(self.__angle3Line, qt.SIGNAL('MyQLineEditSignal'),
                         self.__angle3Slot)
            if self.__densityLine is not None:
                self.connect(self.__densityLine, qt.SIGNAL('MyQLineEditSignal'),
                         self.__densitySlot)
            if self.__thicknessLine is not None:
                self.connect(self.__thicknessLine,qt.SIGNAL('MyQLineEditSignal'),
                         self.__thicknessSlot)
        self.connect(self.__angle3Label, qt.SIGNAL('clicked()'),
                       self.__angle3LabelSlot)

        if orientation == "vertical":
            sampleBoxLayout.addWidget(VerticalSpacer(sampleBox))
    
    def setParameters(self, param):
        for key in param.keys():
            self._current[key] = param[key]   
        self._update()    
        
    def getParameters(self, param = None):
        if param is None:
            return copy(self._current)
        elif param in self._current.keys():
            return self._current[param]
        else:
            raise "KeyError", "%s" % param
            return

    def _update(self):
        if self.materialEditor is not None:
            self.materialEditor.materialGUI.setCurrent(self._current['Material'])
        self.__angle1Line.setText("%.5g" % self._current['AlphaIn'])
        self.__angle2Line.setText("%.5g" % self._current['AlphaOut'])
        if self._current['AlphaScatteringFlag']:
            self.__angle3Label.setChecked(1)
            self.__angle3Line.setEnabled(True)
            self.__angle3Line.setText("%.5g" % self._current['AlphaScattering'])
        else:
            self.__angle3Label.setChecked(False)
            self.__angle3LabelSlot()
            
        if self.__densityLine is not None:
            self.__densityLine.setText("%.5g" % self._current['Density'])
        if self.__thicknessLine is not None:
            self.__thicknessLine.setText("%.5g" % self._current['Thickness'])
    

    def __angle1Slot(self, ddict):
        if (ddict['value'] < -90.) or (ddict['value'] > 90.):
            msg=qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Incident beam has to be in the range [-90, 90]")
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            self.__angle1Line.setFocus()
            return
        
        doit = False
        if self._current['AlphaIn'] > 0:
            if ddict['value'] < 0:
                doit = True
        elif self._current['AlphaIn'] < 0:
            if ddict['value'] > 0:
                doit = True
         
        self._current['AlphaIn'] = ddict['value']
        if doit:
            self.__updateImage()
        self.__updateScattering()

    def __updateImage(self):
        if self._current['AlphaIn'] < 0:
            self.imageLabel.setPixmap("image2trans")
        else:
            self.imageLabel.setPixmap("image2")
   
    def __angle2Slot(self, ddict):
        if (ddict['value'] <= 0.0) or (ddict['value'] > 180.):
            msg=qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Fluorescent beam has to be in the range ]0, 180]")
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            self.__angle2Line.setFocus()
            return
        
        self._current['AlphaOut'] = ddict['value']
        self.__updateScattering()

    def __angle3Slot(self, ddict):
        self._current['AlphaScattering'] = ddict['value']

    def __angle3LabelSlot(self):
        if self.__angle3Label.isChecked():
            self._current['AlphaScatteringFlag'] = 1
            self.__angle3Line.setEnabled(True)
        else:
            self._current['AlphaScatteringFlag'] = 0
            self.__angle3Line.setEnabled(False)
            self.__updateScattering()

    def __updateScattering(self):
        if not self.__angle3Label.isChecked():
            self._current['AlphaScattering'] = self._current['AlphaIn'] +\
                                               self._current['AlphaOut']
            self.__angle3Line.setText("%.5g" % self._current['AlphaScattering'])

    def __thicknessSlot(self, ddict):
        self._current['Thickness'] = ddict['value']

    def __densitySlot(self, ddict):
        self._current['Density'] = ddict['value']

class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=None):
        qt.QLineEdit.__init__(self,parent)
        if QTVERSION < '4.0.0':
            self.connect(self, qt.SIGNAL("returnPressed()"), self.__mySlot)
        else:
            self.connect(self, qt.SIGNAL("editingFinished()"), self.__mySlot)

    def focusInEvent(self,event):
        if QTVERSION < '3.0.0':
            pass
        else:
            if QTVERSION < '4.0.0':
                self.backgroundcolor = self.paletteBackgroundColor()
                self.setPaletteBackgroundColor(qt.QColor('yellow'))

    def focusOutEvent(self,event):
        if QTVERSION < '3.0.0':
            pass
        else:
            self.setPaletteBackgroundColor(qt.QColor('white'))
        self.__mySlot()

    def setPaletteBackgroundColor(self, color):
        if QTVERSION < '3.0.0':
            pass
        elif QTVERSION < '4.0.0':
            qt.QLineEdit.setPaletteBackgroundColor(self, color)
        else:
            pass

    def __mySlot(self):
        qstring = self.text()
        text = str(qstring)
        try:
            if len(text):
                value = float(str(qstring))
                ddict={}
                ddict['event']   = 'returnPressed'
                ddict['value']   = value
                ddict['text']    = text
                ddict['qstring'] = qstring
                if QTVERSION < '4.0.0':
                    self.emit(qt.PYSIGNAL('MyQLineEditSignal'),(ddict,))
                else:
                    self.emit(qt.SIGNAL('MyQLineEditSignal'), ddict)
        except:
            msg=qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Invalid Float")
            if QTVERSION < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            self.setFocus()

class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed))

    
class VerticalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,qt.QSizePolicy.Expanding))
        
if __name__ == "__main__":
    app = qt.QApplication([])
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),app,qt.SLOT("quit()"))
    #demo = MatrixEditor(table=False, orientation="horizontal")
    demo = MatrixEditor(table=True, orientation="vertical")
    if QTVERSION < '4.0.0':
        app.setMainWidget(demo)
        demo.show()
        app.exec_loop()
    else:
        demo.show()
        app.exec_()
