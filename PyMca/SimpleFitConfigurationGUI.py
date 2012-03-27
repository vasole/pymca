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
import os.path
import numpy
from PyMca import SimpleFitControlWidget
qt = SimpleFitControlWidget.qt
HorizontalSpacer = SimpleFitControlWidget.HorizontalSpacer
VerticalSpacer = SimpleFitControlWidget.VerticalSpacer
from PyMca import ConfigDict
from PyMca import PyMca_Icons as Icons
from PyMca import PyMcaDirs

#strip background handling
from PyMca import StripBackgroundWidget
from PyMca import Parameters

DEBUG = 0

class DummyWidget(qt.QWidget):
    def __init__(self, parent=None, text="Automatically estimated function"):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.label = qt.QLabel(self)
        self.label.setAlignment(qt.Qt.AlignHCenter)
        self.label.setText(text)
        self.mainLayout.addWidget(VerticalSpacer(self))
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(VerticalSpacer(self))
        self._configuration = {}

    def setConfiguration(self, ddict):
        self._configuration = ddict

    def getConfiguration(self):
        return self._configuration

    def configure(self, ddict=None):
        if ddict is None:
            return self.getConfiguration()
        else:
            return self.setConfiguration(ddict)

class DefaultParametersWidget(qt.QWidget):
    def __init__(self, parent=None, fit=None, background=False):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(0)
        self.mainLayout.setSpacing(0)
        self.parametersWidget = Parameters.Parameters(self)
        self.mainLayout.addWidget(self.parametersWidget)
        self.simpleFitInstance = fit
        self.background = background
        self._buffer = {}

    def setConfiguration(self, ddict):
        if self.simpleFitInstance is None:
            self._buffer = ddict
            return
        if ddict['configuration']['estimation'] is None:
            #initialize with the default parameters
            parameters = ddict['parameters']
            if type(parameters) == type(""):
                parameters = [parameters]
            xmin = self.simpleFitInstance._x.min()
            xmax = self.simpleFitInstance._x.max()
            if self.background:
                group = 0
            else:
                group = 1
            paramlist = []
            for i in range(len(parameters)):
                pname = parameters[i]+"_1"
                paramdict = {'name':pname,
                             'estimation':0,
                             'group':group,
                             'code':'FREE',
                             'cons1':0,
                             'cons2':0,
                             'fitresult':0.0,
                             'sigma':0.0,
                             'xmin':xmin,
                             'xmax':xmax}               
                paramlist.append(paramdict)
        else:
            parameters = ddict['configuration']['estimation']['parameters']
            if type(parameters) == type(""):
                parameters = [parameters]
            paramlist = []
            for parameter in parameters:
                paramdict = ddict['configuration']['estimation'][parameter]              
                paramlist.append(paramdict)
        self.parametersWidget.fillTableFromFit(paramlist)
            
    def getConfiguration(self):
        if self.simpleFitInstance is None:
            return  self._buffer
        paramlist = self.parametersWidget.fillFitFromTable()
        ddict = {}
        ddict['configuration']={}
        ddict['configuration']['estimation'] = {}
        ddict['configuration']['estimation']['parameters'] = []
        for param in paramlist:
            name = param['name']
            ddict['configuration']['estimation']['parameters'].append(name)
            ddict['configuration']['estimation'][name] = {}
            for key in param.keys():
                if key in ['xmax', 'xmin']:
                    ddict['configuration']['estimation'][name][key] = float(param[key])
                else:
                    ddict['configuration']['estimation'][name][key] = param[key]
        return ddict        

    def configure(self, ddict=None):
        if ddict is None:
            return self.getConfiguration()
        else:
            return self.setConfiguration(ddict)

class SimpleFitConfigurationGUI(qt.QDialog):
    def __init__(self, parent = None, fit=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("PyMca - Simple Fit Configuration")
        self.setWindowIcon(qt.QIcon(qt.QPixmap(Icons.IconDict["gioconda16"])))
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setMargin(2)
        self.mainLayout.setSpacing(2)
        if 0:
            self.fitControlWidget = SimpleFitControlWidget.SimpleFitControlWidget(self)
            self.mainLayout.addWidget(self.fitControlWidget)
            self.connect(self.fitControlWidget,
                         qt.SIGNAL("FitControlSignal"),
                         self._fitControlSlot)
            self._stripDialog = None
        else:            
            self.tabWidget = qt.QTabWidget(self)
            self.fitControlWidget = SimpleFitControlWidget.SimpleFitControlWidget(self)
            self.connect(self.fitControlWidget,
                         qt.SIGNAL("FitControlSignal"),
                         self._fitControlSlot)
            self.tabWidget.insertTab(0, self.fitControlWidget, "FIT")
            self.fitFunctionWidgetStack = qt.QWidget(self)
            self.fitFunctionWidgetStack.mainLayout = qt.QStackedLayout(self.fitFunctionWidgetStack)
            self.fitFunctionWidgetStack.mainLayout.setMargin(0)
            self.fitFunctionWidgetStack.mainLayout.setSpacing(0)            
            self.tabWidget.insertTab(1, self.fitFunctionWidgetStack, "FUNCTION")
            self.backgroundWidgetStack = qt.QWidget(self)
            self.backgroundWidgetStack.mainLayout = qt.QStackedLayout(self.backgroundWidgetStack)
            self.backgroundWidgetStack.mainLayout.setMargin(0)
            self.backgroundWidgetStack.mainLayout.setSpacing(0)            
            self.tabWidget.insertTab(2, self.backgroundWidgetStack, "BACKGROUND")
            self.mainLayout.addWidget(self.tabWidget)
            self._stripDialog = None
        self.buildAndConnectActions()
        self.mainLayout.addWidget(VerticalSpacer(self))
        self._fitFunctionWidgets = {}
        self._backgroundWidgets = {}
        self.setSimpleFitInstance(fit)
        #input output directory
        self.initDir = None

    def _fitControlSlot(self, ddict):
        if DEBUG:
            print("FitControlSignal", ddict)
        event = ddict['event']
        if event == "stripSetupCalled":
            if self._stripDialog is None:
                self._stripDialog = StripBackgroundWidget.StripBackgroundDialog()
                self._stripDialog.setWindowIcon(qt.QIcon(\
                                    qt.QPixmap(Icons.IconDict["gioconda16"])))
            pars = self.__getConfiguration("FIT")
            if self.simpleFitInstance is None:
                return
            xmin = pars['xmin']
            xmax = pars['xmax']
            idx = (self.simpleFitInstance._x0 >= xmin) & (self.simpleFitInstance._x0 <= xmax)
            x = self.simpleFitInstance._x0[idx] * 1
            y = self.simpleFitInstance._y0[idx] * 1
            self._stripDialog.setParameters(pars)
            self._stripDialog.setData(x, y)
            ret = self._stripDialog.exec_()
            if not ret:
                return
            pars = self._stripDialog.getParameters()
            self.fitControlWidget.setConfiguration(pars)
            
        if event == "fitFunctionChanged":
            functionName = ddict['fit_function']
            if functionName in [None, "None", "NONE"]:
                functionName = "None"
                instance = self._fitFunctionWidgets.get(functionName, None)
                if instance is None:
                    instance = qt.QWidget(self.fitFunctionWidgetStack)
                    self.fitFunctionWidgetStack.mainLayout.addWidget(instance)
                    self._fitFunctionWidgets[functionName] = instance
                self.fitFunctionWidgetStack.mainLayout.setCurrentWidget(instance)
                return
            fun = self.simpleFitInstance._fitConfiguration['functions'][functionName]
            instance = self._fitFunctionWidgets.get(functionName, None)
            if instance is None:
                widget = fun.get('widget', None)
                if widget is None:
                    instance = self._buildDefaultWidget(functionName, background=False)
                else:
                    instance = widget(self.fitFunctionWidgetStack)
                    self.fitFunctionWidgetStack.mainLayout.addWidget(instance)
                self._fitFunctionWidgets[functionName] = instance
            if hasattr(instance, 'configure'):
                configureMethod = fun['configure']
                if configureMethod is not None:
                    #make sure it is up-to-date
                    fun['configuration'].update(configureMethod())
                    instance.configure(fun)
            self.fitFunctionWidgetStack.mainLayout.setCurrentWidget(instance)
            
        if event == "backgroundFunctionChanged":
            functionName = ddict['background_function']
            if functionName in [None, "None", "NONE"]:
                functionName = "None"
                instance = self._backgroundWidgets.get(functionName, None)
                if instance is None:
                    instance = qt.QWidget(self.backgroundWidgetStack)
                    self.backgroundWidgetStack.mainLayout.addWidget(instance)
                    self._backgroundWidgets[functionName] = instance
                self.backgroundWidgetStack.mainLayout.setCurrentWidget(instance)
                return
            fun = self.simpleFitInstance._fitConfiguration['functions'][functionName]
            instance = self._backgroundWidgets.get(functionName, None)
            if instance is None:
                widget = fun.get('widget', None)
                if widget is None:
                    instance = self._buildDefaultWidget(functionName, background=True)
                else:
                    instance = widget(self.backgroundWidgetStack)
                    self.backgroundWidgetStack.mainLayout.addWidget(instance)
                self._backgroundWidgets[functionName] = instance
            if hasattr(instance, 'configure'):
                configureMethod = fun['configure']
                if configureMethod is not None:
                    #make sure it is up-to-date
                    fun['configuration'].update(configureMethod())
                    instance.configure(fun)
            self.backgroundWidgetStack.mainLayout.setCurrentWidget(instance)

    def _buildDefaultWidget(self, functionName, background=False):
        functionDescription = self.simpleFitInstance._fitConfiguration['functions']\
                                                                  [functionName]

        #if we here that means the function does not provide a widget
        #if the function does not provide an authomatic estimate
        #the user has to fill the default parameters in the default table
        estimate   = functionDescription['estimate']
        if estimate is None:
            if background:
                widget = DefaultParametersWidget(self.backgroundWidgetStack,
                                                 self.simpleFitInstance, background=background)
                widget.setConfiguration(functionDescription)
                self.backgroundWidgetStack.mainLayout.addWidget(widget)
            else:
                widget = DefaultParametersWidget(self.fitFunctionWidgetStack,
                                                 self.simpleFitInstance,
                                                 background=background)
                widget.setConfiguration(functionDescription)
                self.fitFunctionWidgetStack.mainLayout.addWidget(widget)
        else:
            text = "%s is automatically configured and estimated" % functionName
            if background:
                widget = DummyWidget(self.backgroundWidgetStack, text=text)
                self.backgroundWidgetStack.mainLayout.addWidget(widget)
            else:
                widget = DummyWidget(self.fitFunctionWidgetStack, text=text)
                self.fitFunctionWidgetStack.mainLayout.addWidget(widget)
        return widget

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

    def setSimpleFitInstance(self, fitInstance):
        self.simpleFitInstance = fitInstance
        if self.simpleFitInstance is not None:
            self.setConfiguration(self.simpleFitInstance.getConfiguration())

    def setConfiguration(self, ddict):
        currentConfig = self.simpleFitInstance.getConfiguration()
        currentFiles  = []
        for functionName in currentConfig['functions'].keys():
            fname = currentConfig['functions'][functionName]['file']
            if fname not in currentFiles:
                currentFiles.append(fname)

        if 'functions' in ddict:
            #make sure new modules are imported
            for functionName in ddict['functions'].keys():
                fileName = ddict['functions'][functionName]['file']
                if fileName not in currentFiles:
                    try:
                        if DEBUG:
                            print("Adding file %s" % fileName)
                        self.simpleFitInstance.importFunctions(fileName)
                        currentFiles.append(fileName)
                    except:
                        print("Cannot import file %s" % fileName)
                        print(sys.exc_info()[1])
            
        if 'fit' in ddict:
            self.fitControlWidget.setConfiguration(ddict['fit'])
            fitFunction = ddict['fit']['fit_function']
            background = ddict['fit']['background_function']
            if fitFunction not in self._fitFunctionWidgets.keys():
                self._fitControlSlot({'event':'fitFunctionChanged',
                                      'fit_function':fitFunction})

            if background not in self._backgroundWidgets.keys():
                self._fitControlSlot({'event':'backgroundFunctionChanged',
                                      'background_function':background})
            #fit function
            fname = ddict['fit']['fit_function']
            widget = self._fitFunctionWidgets[fname]
            if fname not in [None, "None", "NONE"]:
                if fname in ddict['functions']:
                    #if currentConfig['functions'][fname]['widget'] is not None:
                        widget.setConfiguration(ddict['functions'][fname])
                        self.fitFunctionWidgetStack.mainLayout.setCurrentWidget(widget)
            
            #background function
            fname = ddict['fit']['background_function']
            widget = self._backgroundWidgets[fname]
            if fname not in [None, "None", "NONE"]:
                if fname in ddict['functions']:
                    #if currentConfig['functions'][fname]['widget'] is not None:
                        widget.setConfiguration(ddict['functions'][fname])
                        self.backgroundWidgetStack.mainLayout.setCurrentWidget(widget)

    def getConfiguration(self):
        oldConfiguration = self.simpleFitInstance.getConfiguration()
        ddict = {}
        for name in ['fit']:
            ddict[name] = self.__getConfiguration(name)

        #fit function
        fname = ddict['fit']['fit_function']
        ddict['functions'] = {}
        widget = self._fitFunctionWidgets[fname]
        if fname not in [None, "None", "NONE"]:
            ddict['functions'][fname]={}
            ddict['functions'][fname]['file'] = \
                oldConfiguration['functions'][fname]['file']                
            ddict['functions'][fname]['configuration'] =\
                oldConfiguration['functions'][fname]['configuration']
            newConfig = widget.getConfiguration()
            if 'configuration' in newConfig:
                ddict['functions'][fname]['configuration'].update(\
                                        newConfig['configuration'])
            else:
                ddict['functions'][fname]['configuration'].update(newConfig)

        #background function
        fname = ddict['fit']['background_function']
        widget = self._backgroundWidgets[fname]
        if fname not in [None, "None", "NONE"]:
            ddict['functions'][fname]={}
            ddict['functions'][fname]['file'] = \
                oldConfiguration['functions'][fname]['file']                
            ddict['functions'][fname]['configuration'] =\
                oldConfiguration['functions'][fname]['configuration']
            newConfig = widget.getConfiguration()
            if 'configuration' in newConfig:
                ddict['functions'][fname]['configuration'].update(\
                                        newConfig['configuration'])
            else:
                ddict['functions'][fname]['configuration'].update(newConfig)
                    
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
        try:
            cfg.read(filename)
            self.initDir = os.path.dirname(filename)
            self.setConfiguration(cfg)
        except:
            if DEBUG:
                raise
            qt.QMessageBox.critical(self, "Load Parameters",
                "ERROR while loading parameters from\n%s"%filename, 
                qt.QMessageBox.Ok,
                qt.QMessageBox.NoButton,
                qt.QMessageBox.NoButton)
        
    def saveConfiguration(self, filename):
        cfg = ConfigDict.ConfigDict(self.getConfiguration())
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
    print(wid.getConfiguration())
    sys.exit()

if __name__=="__main__":
    DEBUG = 1
    test()
