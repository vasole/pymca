#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
import sys
import os

from PyMca import PyMcaQt as qt
QTVERSION = qt.qVersion()
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str

from PyMca.widgets import McaROIWidget
from PyMca import PyMcaDirs

DEBUG = 0
class McaControlGUI(qt.QWidget):
    sigMcaROIWidgetSignal = qt.pyqtSignal(object)
    sigMcaControlGUISignal = qt.pyqtSignal(object)
    
    def __init__(self, parent=None, name=""):
        qt.QWidget.__init__(self, parent)
        if name is not None:
            self.setWindowTitle(name)
        self.roiList = ['ICR']
        self.roiDict = {}
        self.roiDict['ICR'] = {'type':'Default',
                               'from':0,
                               'to':-1}
        self.lastInputDir = None
        self.build()
        self.connections()

    def build(self):
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.calbox =    None
        self.calbut =    None

        calibration  = McaCalControlLine(self)
        self.calbox  = calibration.calbox
        self.calbut  = calibration.calbut
        self.calinfo = McaCalInfoLine(self)

        self.calmenu = qt.QMenu()
        self.calmenu.addAction(QString("Edit"),    self._copysignal)
        self.calmenu.addAction(QString("Compute") ,self._computesignal)
        self.calmenu.addSeparator()
        self.calmenu.addAction(QString("Load") ,   self._loadsignal)
        self.calmenu.addAction(QString("Save") ,   self._savesignal)

        layout.addWidget(calibration)
        layout.addWidget(self.calinfo)

        # ROI
        #roibox = qt.QHGroupBox(self)
        #roibox.setTitle(' ROI ')
        roibox = qt.QWidget(self)
        roiboxlayout = qt.QHBoxLayout(roibox)
        layout.setContentsMargins(0, 0, 0, 0)
        roiboxlayout.setSpacing(0)
        
        #roibox.setAlignment(qt.Qt.AlignHCenter)
        self.roiWidget = McaROIWidget.McaROIWidget(roibox)
        self.roiWidget.fillFromROIDict(roilist=self.roiList,
                                       roidict=self.roiDict)
        self.fillFromROIDict   = self.roiWidget.fillFromROIDict
        self.getROIListAndDict = self.roiWidget.getROIListAndDict
        self.addROI            = self.roiWidget.addROI
        self.setHeader            = self.roiWidget.setHeader

        roiboxlayout.addWidget(self.roiWidget)
        layout.addWidget(roibox)
        layout.setStretchFactor(roibox, 1)
        
    def connections(self):
        #QObject.connect(a,SIGNAL("lastWindowClosed()"),a,SLOT("quit()"
        #selection changed
        self.connect(self.calbox,qt.SIGNAL("activated(const QString &)"),
                    self._calboxactivated)
        self.connect(self.calbut,qt.SIGNAL("clicked()"),self._calbuttonclicked)
        self.roiWidget.sigMcaROIWidgetSignal.connect(self._forward)
    
    def addROI(self,xmin,xmax):
        if [xmin,xmax] not in self.roiList:
            self.roiList.append([xmin,xmax])
        self._updateroibox()
        
    def delroi(self,number):
        if number > 0:
            if number < len(self.roiList):
                del self.roiList[number]
        self._updateroibox()
            
    def _updateroibox(self):
        options = []
        for i in range(len(self.roiList)):
            options.append("%d" % i)
        options.append('Add')
        options.append('Del')
        self.roibox.setoptions(options)
        self._roiboxactivated(self,'0')
    
    def resetroilist(self):
        self.roiList = [[0,-1]]
        self.roiList.append(None,None)
        self.roibox.setoptions(['0','Add','Del'])
        self.roibox.setCurrentItem(0)

    def getroilist(self):
        return self.roiList

    def _calboxactivated(self, item):
        self._calboxactivated(item)
        
    def _calboxactivated(self,item):
        item = qt.safe_str(item)
        if DEBUG:
            print("Calibration box activated %s" % item)
        comboitem,combotext = self.calbox.getCurrent()
        self._emitpysignal(box=[comboitem,combotext],boxname='Calibration',
                            event='activated')

    def _forward(self, ddict):
        self.sigMcaROIWidgetSignal.emit(ddict)

        
    def _calbuttonclicked(self):
        if DEBUG:
            print("Calibration button clicked")
        if qt.qVersion() < '4.0.0':
            self.calmenu.exec_loop(self.cursor().pos())
        else:
            self.calmenu.exec_(self.cursor().pos())
        
    def _copysignal(self):
        comboitem,combotext = self.calbox.getCurrent()
        self._emitpysignal(button="CalibrationCopy",
                            box=[comboitem,combotext],event='clicked')
                
    def _computesignal(self):
        comboitem,combotext = self.calbox.getCurrent()
        self._emitpysignal(button="Calibration",
                            box=[comboitem,combotext],event='clicked')
                
    def _loadsignal(self):
        if self.lastInputDir is None:
            self.lastInputDir = PyMcaDirs.inputDir            
        if self.lastInputDir is not None:
            if not os.path.exists(self.lastInputDir):
                self.lastInputDir = None
        self.lastInputFilter = "Calibration files (*.calib)\n"
        if sys.platform == "win32":
            windir = self.lastInputDir
            if windir is None:
                windir = os.getcwd()
            filename= qt.safe_str(qt.QFileDialog.getOpenFileName(self,
                          "Load existing calibration file",
                          windir,
                          self.lastInputFilter))                
        else:
            windir = self.lastInputDir
            if windir is None:windir = os.getcwd()
            filename = qt.QFileDialog(self)
            filename.setWindowTitle("Load existing calibration file")
            filename.setModal(1)
            if hasattr(qt, "QStringList"):
                strlist = qt.QStringList()
            else:
                strlist = []
            tmp = [self.lastInputFilter.replace("\n","")] 
            for filetype in tmp:
                strlist.append(filetype.replace("(","").replace(")",""))
            filename.setFilters(strlist)
            filename.setFileMode(qt.QFileDialog.ExistingFile)
            filename.setDirectory(windir)
            ret = filename.exec_()
            if ret:
                if len(filename.selectedFiles()):
                    filename = qt.safe_str(filename.selectedFiles()[0])
                else:
                    return
            else:
                return
        if not len(filename):
            return
        if len(filename) < 6:
            filename = filename + ".calib"
        elif filename[-6:] != ".calib":
            filename = filename + ".calib"        
        self.lastInputDir = os.path.dirname(filename)
        comboitem,combotext = self.calbox.getCurrent()
        self._emitpysignal(button="CalibrationLoad",
                            box=[comboitem,combotext],
                            line_edit = filename,
                            event='clicked')
                
    def _savesignal(self):
        if self.lastInputDir is None:
            self.lastInputDir = PyMcaDirs.outputDir
        if self.lastInputDir is not None:
            if not os.path.exists(self.lastInputDir):
                self.lastInputDir = None
        self.lastInputFilter = "Calibration files (*.calib)\n"
        if sys.platform == "win32":
            windir = self.lastInputDir
            if windir is None:
                windir = ""
            filename= qt.safe_str(qt.QFileDialog.getSaveFileName(self,
                          "Save a new calibration file",
                          windir,
                          self.lastInputFilter))                
        else:
            windir = self.lastInputDir
            if windir is None:windir = os.getcwd()
            filename = qt.QFileDialog(self)
            filename.setWindowTitle("Save a new calibration file")
            filename.setModal(1)
            if hasattr(qt, "QStringList"):
                strlist = qt.QStringList()
            else:
                strlist = []
            tmp = [self.lastInputFilter.replace("\n","")] 
            for filetype in tmp:
                strlist.append(filetype.replace("(","").replace(")",""))
            filename.setFilters(strlist)
            filename.setFileMode(qt.QFileDialog.AnyFile)
            filename.setDirectory(windir)
            ret = filename.exec_()
            if ret:
                if len(filename.selectedFiles()):
                    filename = qt.safe_str(filename.selectedFiles()[0])
                else:
                    return
            else:
                return

        if not len(filename):
            return
        if len(filename) < 6:
            filename = filename + ".calib"
        elif filename[-6:] != ".calib":
            filename = filename + ".calib"        
        self.lastInputDir = os.path.dirname(filename)
        PyMcaDirs.outputDir = os.path.dirname(filename)
        comboitem,combotext = self.calbox.getCurrent()
        self._emitpysignal(button="CalibrationSave",
                            box=[comboitem,combotext],
                            line_edit = filename,
                            event='clicked')

    def _roiresetbuttonclicked(self):
        if DEBUG:
            print("ROI reset button clicked")
        comboitem,combotext = self.roibox.getCurrent()
        self._emitpysignal(button="Roi reset",
                            box=[comboitem,combotext],
                            event='clicked')

    def _emitpysignal(self,button=None,
                            box=None,
                            boxname=None,
                            checkbox=None,
                            line_edit=None,
                            event=None):
        if DEBUG:
            print("_emitpysignal called ",button,box)
        data={}
        data['button']        = button
        data['box']           = box
        data['checkbox']      = checkbox
        data['line_edit']     = line_edit
        data['event']         = event
        data['boxname']       = boxname
        self.sigMcaControlGUISignal.emit(data)

class McaCalControlLine(qt.QWidget):
    def __init__(self, parent=None, name=None, calname="",
                 caldict = None):
        if caldict is None:
            caldict = {}
        qt.QWidget.__init__(self, parent)
        if name is not None:
            self.setWindowTitle(name)
        self.l = qt.QHBoxLayout(self)
        self.l.setContentsMargins(0, 0, 0, 0)
        self.l.setSpacing(0)
        self.build()
    
    def build(self):
        widget = self
        callabel    = qt.QLabel(widget)
        callabel.setText(str("<b>%s</b>" % 'Calibration'))
        self.calbox = SimpleComboBox(widget,
                                     options=['None',
                                              'Original (from Source)',
                                              'Internal (from Source OR PyMca)'])
        self.calbox.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                                 qt.QSizePolicy.Fixed))
        self.calbut = qt.QPushButton(widget)
        self.calbut.setText('Calibrate')
        self.calbut.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,
                                                 qt.QSizePolicy.Fixed))

        self.l.addWidget(callabel)
        self.l.addWidget(self.calbox)
        self.l.addWidget(self.calbut)


class McaCalInfoLine(qt.QWidget):
    def __init__(self, parent=None, name=None, calname="",
                 caldict = None):
        if caldict is None:
            caldict = {}
        qt.QWidget.__init__(self, parent)
        if name is not None:self.setWindowTitle(name)
        self.caldict=caldict
        if calname not in self.caldict.keys():
            self.caldict[calname] = {}
            self.caldict[calname]['order'] = 1
            self.caldict[calname]['A'] = 0.0
            self.caldict[calname]['B'] = 1.0
            self.caldict[calname]['C'] = 0.0
        self.currentcal = calname
        self.build()
        self.setParameters(self.caldict[calname])
    
    def build(self):
        layout= qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        parw = self

        self.lab= qt.QLabel("<nobr><b>Active Curve Uses</b></nobr>", parw)
        layout.addWidget(self.lab)

        lab= qt.QLabel("A:", parw)
        layout.addWidget(lab)

        self.AText= qt.QLineEdit(parw)
        self.AText.setReadOnly(1)
        layout.addWidget(self.AText)

        lab= qt.QLabel("B:", parw)
        layout.addWidget(lab)

        self.BText= qt.QLineEdit(parw)
        self.BText.setReadOnly(1)
        layout.addWidget(self.BText)

        lab= qt.QLabel("C:", parw)
        layout.addWidget(lab)

        self.CText= qt.QLineEdit(parw)
        self.CText.setReadOnly(1)
        layout.addWidget(self.CText)

    def setParameters(self, pars, name = None):
        if name is not None:
            self.currentcal = name
        if 'order' in pars:
            order = pars['order']
        elif pars["C"] != 0.0:
            order = 2
        else:
            order = 1            
        self.AText.setText("%.8g" % pars["A"])
        self.BText.setText("%.8g" % pars["B"])
        self.CText.setText("%.8g" % pars["C"])
        """
        if pars['order'] > 1:
            self.orderbox.setCurrentItem(1)
            self.CText.setReadOnly(0)
        else:
            self.orderbox.setCurrentItem(0)
            self.CText.setReadOnly(1)
        """
        self.caldict[self.currentcal]["A"] = pars["A"]
        self.caldict[self.currentcal]["B"] = pars["B"]
        self.caldict[self.currentcal]["C"] = pars["C"]
        self.caldict[self.currentcal]["order"] = order


class SimpleComboBox(qt.QComboBox):
        def __init__(self, parent=None, name=None, options=['1','2','3']):
            qt.QComboBox.__init__(self,parent)
            self.setOptions(options) 
            
        def setOptions(self,options=['1','2','3']):
            self.clear()
            for item in options:
                self.addItem(item)

        def getCurrent(self):
            return   self.currentIndex(), qt.safe_str(self.currentText())
             

if __name__ == '__main__':
    app = qt.QApplication(sys.argv)
    demo = McaControlGUI()
    demo.show()
    app.exec_()            
