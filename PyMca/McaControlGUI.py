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
import sys
import os

from PyMca import PyMcaQt as qt
QTVERSION = qt.qVersion()
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = qt.safe_str

from PyMca import McaROIWidget
from PyMca import PyMcaDirs

DEBUG = 0
class McaControlGUI(qt.QWidget):
    def __init__(self, parent=None, name="",fl=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, name,fl)
            if name is not None:self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
            if name is not None:self.setWindowTitle(name)
        self.roilist = ['ICR']
        self.roidict = {}
        self.roidict['ICR'] = {'type':'Default',
                               'from':0,'to':-1}
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

        if qt.qVersion() < '4.0.0':
            self.calmenu = qt.QPopupMenu()
            self.calmenu.insertItem(QString("Edit"),    self.__copysignal)
            self.calmenu.insertItem(QString("Compute") ,self.__computesignal)
            self.calmenu.insertSeparator()
            self.calmenu.insertItem(QString("Load") ,   self.__loadsignal)
            self.calmenu.insertItem(QString("Save") ,   self.__savesignal)
        else:
            self.calmenu = qt.QMenu()
            self.calmenu.addAction(QString("Edit"),    self.__copysignal)
            self.calmenu.addAction(QString("Compute") ,self.__computesignal)
            self.calmenu.addSeparator()
            self.calmenu.addAction(QString("Load") ,   self.__loadsignal)
            self.calmenu.addAction(QString("Save") ,   self.__savesignal)

        layout.addWidget(calibration)
        layout.addWidget(self.calinfo)

        # ROI
        #roibox = qt.QHGroupBox(self)
        #roibox.setTitle(' ROI ')
        roibox = qt.QWidget(self)
        roiboxlayout = qt.QHBoxLayout(roibox)
        roiboxlayout.setContentsMargins(0, 0, 0, 0)
        roiboxlayout.setSpacing(0)
        
        #roibox.setAlignment(qt.Qt.AlignHCenter)
        self.roiwidget = McaROIWidget.McaROIWidget(roibox)
        self.roiwidget.fillfromroidict(roilist=self.roilist,
                                       roidict=self.roidict)
        self.fillfromroidict = self.roiwidget.fillfromroidict
        self.addroi          = self.roiwidget.addroi

        roiboxlayout.addWidget(self.roiwidget)
        layout.addWidget(roibox)
        if qt.qVersion() > '4.0.0':
            layout.setStretchFactor(roibox, 1)
        
    def connections(self):
        #QObject.connect(a,SIGNAL("lastWindowClosed()"),a,SLOT("quit()"
        #selection changed
        self.connect(self.calbox,qt.SIGNAL("activated(const QString &)"),
                    self.__calboxactivated)
        self.connect(self.calbut,qt.SIGNAL("clicked()"),self.__calbuttonclicked)
        if qt.qVersion() < '4.0.0':
            self.connect(self.roiwidget,qt.PYSIGNAL("McaROIWidgetSignal"),
                    self.__forward)
        else:
            self.connect(self.roiwidget,qt.SIGNAL("McaROIWidgetSignal"),
                    self.__forward)

    
    def addroi(self,xmin,xmax):
        if [xmin,xmax] not in self.roilist:
            self.roilist.append([xmin,xmax])
        self.__updateroibox()
        
    def delroi(self,number):
        if number > 0:
            if number < len(self.roilist):
                del self.roilist[number]
        self.__updateroibox()
            
    def __updateroibox(self):
        options = []
        for i in range(len(self.roilist)):
            options.append("%d" % i)
        options.append('Add')
        options.append('Del')
        self.roibox.setoptions(options)
        self.__roiboxactivated(self,'0')
    
    def resetroilist(self):
        self.roilist = [[0,-1]]
        self.roilist.append(None,None)
        self.roibox.setoptions(['0','Add','Del'])
        self.roibox.setCurrentItem(0)

    def getroilist(self):
        return self.roilist

    def _calboxactivated(self, item):
        self.__calboxactivated(item)
        
    def __calboxactivated(self,item):
        item = qt.safe_str(item)
        if DEBUG:
            print("Calibration box activated %s" % item)
        comboitem,combotext = self.calbox.getcurrent()
        self.__emitpysignal(box=[comboitem,combotext],boxname='Calibration',
                            event='activated')

    def __forward(self, ddict):
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("McaControlGUISignal"), (ddict,))    
        else:
            self.emit(qt.SIGNAL("McaControlGUISignal"), ddict)  

        
    def __calbuttonclicked(self):
        if DEBUG:
            print("Calibration button clicked")
        if qt.qVersion() < '4.0.0':
            self.calmenu.exec_loop(self.cursor().pos())
        else:
            self.calmenu.exec_(self.cursor().pos())
        
    def __copysignal(self):
        comboitem,combotext = self.calbox.getcurrent()
        self.__emitpysignal(button="CalibrationCopy",
                            box=[comboitem,combotext],event='clicked')
                
    def __computesignal(self):
        comboitem,combotext = self.calbox.getcurrent()
        self.__emitpysignal(button="Calibration",
                            box=[comboitem,combotext],event='clicked')
                
    def __loadsignal(self):
        if self.lastInputDir is None:
            self.lastInputDir = PyMcaDirs.inputDir            
        if self.lastInputDir is not None:
            if not os.path.exists(self.lastInputDir):
                self.lastInputDir = None
        self.lastInputFilter = "Calibration files (*.calib)\n"
        if sys.platform == "win32":
            windir = self.lastInputDir
            if windir is None:windir = os.getcwd()
            if qt.qVersion() < '4.0.0':
                filename= qt.safe_str(qt.QFileDialog.getOpenFileName(windir,
                             self.lastInputFilter,
                             self,
                            "Load File", "Load existing calibration file"))
            else:
                filename= qt.safe_str(qt.QFileDialog.getOpenFileName(self,
                              "Load existing calibration file",
                              windir,
                              self.lastInputFilter))                
        else:
            if QTVERSION < '4.0.0':
                filename = qt.QFileDialog(self, "Load existing calibration file", 1)
                filename.setFilters(self.lastInputFilter)
                if self.lastInputDir is not None:
                    filename.setDir(self.lastInputDir)
                filename.setMode(qt.QFileDialog.ExistingFile)
                if filename.exec_loop() == qt.QDialog.Accepted:
                    #selectedfilter = qt.safe_str(filename.selectedFilter())
                    filename= qt.safe_str(filename.selectedFile())
                    #print selectedfilter
                else:
                    return
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
        if not len(filename):    return
        if len(filename) < 6:
            filename = filename + ".calib"
        elif filename[-6:] != ".calib":
            filename = filename + ".calib"        
        self.lastInputDir = os.path.dirname(filename)
        comboitem,combotext = self.calbox.getcurrent()
        self.__emitpysignal(button="CalibrationLoad",
                            box=[comboitem,combotext],
                            line_edit = filename,
                            event='clicked')
                
    def __savesignal(self):
        if self.lastInputDir is None:
            self.lastInputDir = PyMcaDirs.outputDir
        if self.lastInputDir is not None:
            if not os.path.exists(self.lastInputDir):
                self.lastInputDir = None
        self.lastInputFilter = "Calibration files (*.calib)\n"
        if sys.platform == "win32":
            windir = self.lastInputDir
            if windir is None:windir = ""
            if qt.qVersion() < '4.0.0':
                filename= qt.safe_str(qt.QFileDialog.getSaveFileName(windir,
                             self.lastInputFilter,
                             self,
                            "Save File", "Save a new calibration file"))
            else:
                filename= qt.safe_str(qt.QFileDialog.getSaveFileName(self,
                              "Save a new calibration file",
                              windir,
                              self.lastInputFilter))                
        else:
            if QTVERSION < '4.0.0':
                filename = qt.QFileDialog(self, "Save a new calibration file", 1)
                filename.setFilters(self.lastInputFilter)
                if self.lastInputDir is not None:
                    filename.setDir(self.lastInputDir)
                filename.setMode(qt.QFileDialog.AnyFile)
                if filename.exec_loop() == qt.QDialog.Accepted:
                    #selectedfilter = qt.safe_str(filename.selectedFilter())
                    filename= qt.safe_str(filename.selectedFile())
                    #print selectedfilter
                else:
                    return
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

        if not len(filename):    return
        if len(filename) < 6:
            filename = filename + ".calib"
        elif filename[-6:] != ".calib":
            filename = filename + ".calib"        
        self.lastInputDir = os.path.dirname(filename)
        PyMcaDirs.outputDir = os.path.dirname(filename)
        comboitem,combotext = self.calbox.getcurrent()
        self.__emitpysignal(button="CalibrationSave",
                            box=[comboitem,combotext],
                            line_edit = filename,
                            event='clicked')

    def __roiresetbuttonclicked(self):
        if DEBUG:
            print("ROI reset button clicked")
        comboitem,combotext = self.roibox.getcurrent()
        self.__emitpysignal(button="Roi reset",box=[comboitem,combotext],event='clicked')

    def __emitpysignal(self,button=None,
                            box=None,
                            boxname=None,
                            checkbox=None,
                            line_edit=None,
                            event=None):
        if DEBUG:
            print("__emitpysignal called ",button,box)
        data={}
        data['button']        = button
        data['box']           = box
        data['checkbox']      = checkbox
        data['line_edit']     = line_edit
        data['event']         = event
        data['boxname']       = boxname
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL("McaControlGUISignal"),(data,))
        else:
            self.emit(qt.SIGNAL("McaControlGUISignal"), data)

class McaCalControlLine(qt.QWidget):
    def __init__(self, parent=None, name=None, calname="",
                 caldict = {},fl=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, name,fl)
            if name is not None:self.setCaption(name)
        else:
            qt.QWidget.__init__(self, parent)
            if name is not None:self.setWindowTitle(name)
        self.l = qt.QHBoxLayout(self)
        self.l.setContentsMargins(0, 0, 0, 0)
        self.l.setSpacing(0)
        self.build()
    
    def build(self):
        widget = self
        callabel    = qt.QLabel(widget)
        callabel.setText(str("<b>%s</b>" % 'Calibration'))
        self.calbox = SimpleComboBox(widget,
                                     options=['None','Original (from Source)','Internal (from Source OR PyMca)'])
        self.calbox.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,qt.QSizePolicy.Fixed))
        self.calbut = qt.QPushButton(widget)
        self.calbut.setText('Calibrate')
        self.calbut.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Fixed,qt.QSizePolicy.Fixed))

        self.l.addWidget(callabel)
        self.l.addWidget(self.calbox)
        self.l.addWidget(self.calbut)


class McaCalInfoLine(qt.QWidget):
    def __init__(self, parent=None, name=None, calname="",
                 caldict = {},fl=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self, parent, name,fl)
            if name is not None:self.setCaption(name)
        else:
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
        def __init__(self,parent = None,name = None,fl = 0,options=['1','2','3']):
            qt.QComboBox.__init__(self,parent)
            self.setoptions(options) 
            
        def setoptions(self,options=['1','2','3']):
            self.clear()
            if qt.qVersion() < '4.0.0':
                self.insertStrList(options)
            else:
                for item in options:
                    self.addItem(item)
                    
            
        def getcurrent(self):
            if qt.qVersion() < '4.0.0':
                return   self.currentItem(),qt.safe_str(self.currentText())
            else:
                return   self.currentIndex(),qt.safe_str(self.currentText())
             

if __name__ == '__main__':
    if 1:
        app = qt.QApplication(sys.argv)
        #demo = make()
        demo = McaControlGUI()
        if qt.qVersion() < '4.0.0':
            app.setMainWidget(demo)
            demo.show()
            app.exec_loop()
        else:
            demo.show()
            app.exec_()            
    else:
        app = qt.QApplication(sys.argv)
        #demo = make()
        demo = qt.QVBox()
        control = McaCalControlLine(demo)
        info    = McaCalInfoLine(demo)
        app.setMainWidget(demo)
        demo.show()
        app.exec_loop()
    

