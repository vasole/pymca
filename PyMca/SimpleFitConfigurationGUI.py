#/*##########################################################################
# Copyright (C) 2004-2010 European Synchrotron Radiation Facility
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
import sys
import SimpleFitControlWidget
qt = SimpleFitControlWidget.qt
HorizontalSpacer = SimpleFitControlWidget.HorizontalSpacer
VerticalSpacer = SimpleFitControlWidget.VerticalSpacer
import ConfigDict
import PyMca_Icons as Icons
import os.path
import PyMcaDirs
DEBUG = 0

class SimpleFitConfigurationGUI(qt.QDialog):
    def __init__(self, parent = None, specfit=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("PyMca - Simple Fit Configuration")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(Icons.IconDict["gioconda16"])))
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)
        self.fitControlWidget = SimpleFitControlWidget.SimpleFitControlWidget(self)
        self.mainLayout.addWidget(self.fitControlWidget)
        self.buildAndConnectActions()
        self.mainLayout.addWidget(VerticalSpacer(self))

        #input output directory
        self.initDir = None

    def buildAndConnectActions(self):
        buts= qt.QGroupBox(self)
        buts.layout = qt.QHBoxLayout(buts)
        load= qt.QPushButton(buts)
        load.setAutoDefault(False)
        load.setText("Load")
        save= qt.QPushButton(buts)
        save.setAutoDefault(False)
        save.setText("Save")
        reject= qt.QPushButton(buts)
        reject.setAutoDefault(False)
        reject.setText("Cancel")
        accept= qt.QPushButton(buts)
        accept.setAutoDefault(False)
        accept.setText("OK")
        buts.layout.addWidget(load)
        buts.layout.addWidget(save)
        buts.layout.addWidget(reject)
        buts.layout.addWidget(accept)
        self.mainLayout.addWidget(buts)

        self.connect(load, qt.SIGNAL("clicked()"), self.load)
        self.connect(save, qt.SIGNAL("clicked()"), self.save)
        self.connect(reject, qt.SIGNAL("clicked()"), self.reject)
        self.connect(accept, qt.SIGNAL("clicked()"), self.accept)

    def setConfiguration(self, ddict):
        if ddict.has_key('fit'):
            self.fitControlWidget.setConfiguration(ddict['fit'])

    def getConfiguration(self):
        ddict = {}
        for name in ['fit']:
            ddict[name] = self.fitControlWidget.getConfiguration()
        return ddict

    def __getConfiguration(self, name):
        if name in ['fit', 'FIT']:
            return self.fitControlWidget.getConfiguration()

    def load(self):
        if PyMcaDirs.nativeFileDialogs:
            filedialog = qt.QFileDialog(self)
            filedialog.setFileMode(filedialog.ExistingFiles)
            filedialog.setWindowIcon(qt.QIcon(qt.QPixmap(Icons.IconDict["gioconda16"])))
            initdir = os.path.curdir
            if self.initDir is not None:
                if os.path.isdir(self.initDir):
                    initdir = self.initDir
            filename = filedialog.getOpenFileName(
                        self,
                        "Choose fit configuration file",
                        initdir,
                        "Fit configuration files (*.cfg)\nAll Files (*)")
            filename = str(filename)
            if len(filename):
                self.loadConfiguration(filename)
                self.initDir = os.path.dirname(filename)
        else:
            filedialog = qt.QFileDialog(self)
            filedialog.setFileMode(filedialog.ExistingFiles)
            filedialog.setWindowIcon(qt.QIcon(qt.QPixmap(Icons.IconDict["gioconda16"])))
            initdir = os.path.curdir
            if self.initDir is not None:
                if os.path.isdir(self.initDir):
                    initdir = self.initDir
            filename = filedialog.getOpenFileName(
                        self,
                        "Choose fit configuration file",
                        initdir,
                        "Fit configuration files (*.cfg)\nAll Files (*)")
            filename = str(filename)
            if len(filename):
                self.loadConfiguration(filename)
                self.initDir = os.path.dirname(filename)
        
    def save(self):
        if self.initDir is None:
            self.initDir = PyMcaDirs.outputDir
        if PyMcaDirs.nativeFileDialogs:
            filedialog = qt.QFileDialog(self)
            filedialog.setFileMode(filedialog.AnyFile)
            filedialog.setWindowIcon(qt.QIcon(qt.QPixmap(Icons.IconDict["gioconda16"])))
            initdir = os.path.curdir
            if self.initDir is not None:
                if os.path.isdir(self.initDir):
                    initdir = self.initDir
            filename = filedialog.getSaveFileName(
                        self,
                        "Enter output fit configuration file",
                        initdir,
                        "Fit configuration files (*.cfg)\nAll Files (*)")
            filename = str(filename)
            if len(filename):
                if len(filename) < 4:
                    filename = filename+".cfg"
                elif filename[-4:] != ".cfg":
                    filename = filename+".cfg"
                self.saveConfiguration(filename)
                self.initDir = os.path.dirname(filename)
        else:
            filedialog = qt.QFileDialog(self)
            filedialog.setFileMode(filedialog.AnyFile)
            filedialog.setWindowIcon(qt.QIcon(qt.QPixmap(Icons.IconDict["gioconda16"])))
            initdir = os.path.curdir
            if self.initDir is not None:
                if os.path.isdir(self.initDir):
                    initdir = self.initDir
            filename = filedialog.getSaveFileName(
                        self,
                        "Enter output fit configuration file",
                        initdir,
                        "Fit configuration files (*.cfg)\nAll Files (*)")
            filename = str(filename)
            if len(filename):
                if len(filename) < 4:
                    filename = filename+".cfg"
                elif filename[-4:] != ".cfg":
                    filename = filename+".cfg"
                self.saveConfiguration(filename)
                self.initDir = os.path.dirname(filename)
                PyMcaDirs.outputDir = os.path.dirname(filename)

    def loadConfiguration(self, filename):
        cfg= ConfigDict.ConfigDict()
        if DEBUG:
            cfg.read(filename)
            self.initDir = os.path.dirname(filename)
            self.setConfiguration(cfg)
        else:
            try:
                cfg.read(filename)
                self.initDir = os.path.dirname(filename)
                self.setConfiguration(cfg)
            except:
                qt.QMessageBox.critical(self, "Load Parameters",
                    "ERROR while loading parameters from\n%s"%filename, 
                    qt.QMessageBox.Ok, qt.QMessageBox.NoButton, qt.QMessageBox.NoButton)

        
    def saveConfiguration(self, filename):
        cfg= ConfigDict.ConfigDict(self.getConfiguration())
        if DEBUG:
            cfg.write(filename)
            self.initDir = os.path.dirname(filename)
        else:
            try:
                cfg.write(filename)
                self.initDir = os.path.dirname(filename)
            except:
                qt.QMessageBox.critical(self, "Save Parameters", 
                    "ERROR while saving parameters to\n%s"%filename,
                    qt.QMessageBox.Ok, qt.QMessageBox.NoButton, qt.QMessageBox.NoButton)
        

def test():
    app = qt.QApplication(sys.argv)
    app.connect(app, qt.SIGNAL("lastWindowClosed()"), app.quit)
    wid = SimpleFitConfigurationGUI()
    ddict = {}
    ddict['fit'] = {}
    ddict['fit']['use_limits'] = 1
    ddict['fit']['xmin'] = 1
    ddict['fit']['xmax'] = 1024
    wid.setConfiguration(ddict)
    wid.exec_()
    print wid.getConfiguration()
    sys.exit()

if __name__=="__main__":
    DEBUG = 1
    test()
