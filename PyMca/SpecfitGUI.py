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
from PyMca import EventHandler
from PyMca import Specfit
from PyMca import PyMcaQt as qt
    
QTVERSION = qt.qVersion()
from PyMca import FitConfigGUI
from PyMca import MultiParameters
from PyMca import FitActionsGUI
from PyMca import FitStatusGUI
from PyMca import EventHandler
from PyMca import QScriptOption
DEBUG = 0

class SpecfitGUI(qt.QWidget):
    def __init__(self,parent = None,name = None,fl = 0, specfit = None,
                 config = 0, status = 0, buttons = 0, eh = None):
        if QTVERSION < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)
            if name == None:
                self.setName("SpecfitGUI")
        else:
            qt.QWidget.__init__(self, parent)
        layout= qt.QVBoxLayout(self)
        #layout.setAutoAdd(1)
        if eh == None:
            self.eh = EventHandler.EventHandler()
        else:
            self.eh = eh
        if specfit is None:
            self.specfit = Specfit.Specfit(eh=self.eh)
        else:
            self.specfit = specfit

        #initialize the default fitting functions in case
        #none is present
        if not len(self.specfit.theorylist):
            funsFile = "SpecfitFunctions.py"
            if not os.path.exists(funsFile):
                funsFile = os.path.join(os.path.dirname(Specfit.__file__),\
                                funsFile)
            self.specfit.importfun(funsFile)
        
        #copy specfit configure method for direct access
        self.configure=self.specfit.configure
        self.fitconfig=self.specfit.fitconfig

        self.setdata=self.specfit.setdata
        self.guiconfig=None
        if config:
            self.guiconfig = FitConfigGUI.FitConfigGUI(self)
            self.guiconfig.connect(self.guiconfig.MCACheckBox,
                                   qt.SIGNAL("stateChanged(int)"),self.mcaevent) 
            self.guiconfig.connect(self.guiconfig.WeightCheckBox,
                                   qt.SIGNAL("stateChanged(int)"),self.weightevent) 
            self.guiconfig.connect(self.guiconfig.AutoFWHMCheckBox,
                                   qt.SIGNAL("stateChanged(int)"),self.autofwhmevent) 
            self.guiconfig.connect(self.guiconfig.AutoScalingCheckBox,
                                   qt.SIGNAL("stateChanged(int)"),self.autoscaleevent) 
            self.guiconfig.connect(self.guiconfig.ConfigureButton,
                                   qt.SIGNAL("clicked()"),self.__configureGUI) 
            self.guiconfig.connect(self.guiconfig.PrintPushButton,
                                   qt.SIGNAL("clicked()"),self.printps) 
            self.guiconfig.connect(self.guiconfig.BkgComBox,
                                qt.SIGNAL("activated(const QString &)"),self.bkgevent)
            self.guiconfig.connect(self.guiconfig.FunComBox,
                                qt.SIGNAL("activated(const QString &)"),self.funevent)
            layout.addWidget(self.guiconfig)

        self.guiparameters = MultiParameters.ParametersTab(self)
        layout.addWidget(self.guiparameters)
        if QTVERSION < '4.0.0':
            self.connect(self.guiparameters,qt.PYSIGNAL('MultiParametersSignal'),
                self.__forward)
        else:
            self.connect(self.guiparameters,
                         qt.SIGNAL('MultiParametersSignal'),
                         self.__forward)
        if config:
            if QTVERSION < '4.0.0':
                for key in self.specfit.bkgdict.keys():
                    self.guiconfig.BkgComBox.insertItem(str(key))
                for key in self.specfit.theorylist:            
                    self.guiconfig.FunComBox.insertItem(str(key))
            else:
                for key in self.specfit.bkgdict.keys():
                    self.guiconfig.BkgComBox.addItem(str(key))
                for key in self.specfit.theorylist:            
                    self.guiconfig.FunComBox.addItem(str(key))
            configuration={}
            if specfit is not None:
                configuration = specfit.configure()
                if configuration['fittheory'] is None:
                    if QTVERSION < '4.0.0':
                        self.guiconfig.FunComBox.setCurrentItem(1)
                    else:
                        self.guiconfig.FunComBox.setCurrentIndex(1)
                    self.funevent(self.specfit.theorylist[0])
                else:
                    self.funevent(configuration['fittheory'])
                if configuration['fitbkg']    is None:
                    if QTVERSION < '4.0.0':
                        self.guiconfig.BkgComBox.setCurrentItem(1)
                    else:
                        self.guiconfig.BkgComBox.setCurrentIndex(1)
                    self.bkgevent(list(self.specfit.bkgdict.keys())[0])
                else:
                    self.bkgevent(configuration['fitbkg'])
            else:
                if QTVERSION < '4.0.0':
                    self.guiconfig.BkgComBox.setCurrentItem(1)
                    self.guiconfig.FunComBox.setCurrentItem(1)
                else:
                    self.guiconfig.BkgComBox.setCurrentIndex(1)
                    self.guiconfig.FunComBox.setCurrentIndex(1)
                self.funevent(self.specfit.theorylist[0])
                self.bkgevent(list(self.specfit.bkgdict.keys())[0])
            configuration.update(self.configure())
            if configuration['McaMode']:            
                self.guiconfig.MCACheckBox.setChecked(1)
            else:
                self.guiconfig.MCACheckBox.setChecked(0)
            if configuration['WeightFlag']:            
                self.guiconfig.WeightCheckBox.setChecked(1)
            else:
                self.guiconfig.WeightCheckBox.setChecked(0)
            if configuration['AutoFwhm']:            
                self.guiconfig.AutoFWHMCheckBox.setChecked(1)
            else:
                self.guiconfig.AutoFWHMCheckBox.setChecked(0)
            if configuration['AutoScaling']:            
                self.guiconfig.AutoScalingCheckBox.setChecked(1)
            else:
                self.guiconfig.AutoScalingCheckBox.setChecked(0)

        if status:
            self.guistatus =  FitStatusGUI.FitStatusGUI(self)
            self.eh.register('FitStatusChanged',self.fitstatus)
            layout.addWidget(self.guistatus)
        if buttons:
            self.guibuttons = FitActionsGUI.FitActionsGUI(self)
            self.guibuttons.connect(self.guibuttons.EstimateButton,
                                    qt.SIGNAL("clicked()"),self.estimate)
            self.guibuttons.connect(self.guibuttons.StartfitButton,
                                    qt.SIGNAL("clicked()"),self.startfit)
            self.guibuttons.connect(self.guibuttons.DismissButton,
                                    qt.SIGNAL("clicked()"),self.dismiss)
            layout.addWidget(self.guibuttons)

    def updateGUI(self,configuration=None):
        self.__configureGUI(configuration)

    def __configureGUI(self,newconfiguration=None):
        if self.guiconfig is not None:
            #get current dictionary
            #print "before ",self.specfit.fitconfig['fitbkg']
            configuration=self.configure()
            #get new dictionary
            if newconfiguration is None:
                newconfiguration=self.configureGUI(configuration)
            #update configuration
            configuration.update(self.configure(**newconfiguration))
            #print "after =",self.specfit.fitconfig['fitbkg']
            #update GUI
            #current function
            #self.funevent(self.specfit.theorylist[0])
            try:
                i=1+self.specfit.theorylist.index(self.specfit.fitconfig['fittheory'])
                if QTVERSION < '4.0.0':
                    self.guiconfig.FunComBox.setCurrentItem(i)
                else:
                    self.guiconfig.FunComBox.setCurrentIndex(i)
                self.funevent(self.specfit.fitconfig['fittheory'])
            except:
                print("Function not in list %s" %\
                      self.specfit.fitconfig['fittheory'])
                self.funevent(self.specfit.theorylist[0])
            #current background
            try:
                #the list conversion is needed in python 3.
                i=1+list(self.specfit.bkgdict.keys()).index(self.specfit.fitconfig['fitbkg'])
                if QTVERSION < '4.0.0':
                    self.guiconfig.BkgComBox.setCurrentItem(i)
                else:
                    self.guiconfig.BkgComBox.setCurrentIndex(i)
            except:
                print("Background not in list %s" %\
                      self.specfit.fitconfig['fitbkg'])
                self.bkgevent(list(self.specfit.bkgdict.keys())[0])
            #and all the rest
            if configuration['McaMode']:            
                self.guiconfig.MCACheckBox.setChecked(1)
            else:
                self.guiconfig.MCACheckBox.setChecked(0)
            if configuration['WeightFlag']:            
                self.guiconfig.WeightCheckBox.setChecked(1)
            else:
                self.guiconfig.WeightCheckBox.setChecked(0)
            if configuration['AutoFwhm']:            
                self.guiconfig.AutoFWHMCheckBox.setChecked(1)
            else:
                self.guiconfig.AutoFWHMCheckBox.setChecked(0)
            if configuration['AutoScaling']:            
                self.guiconfig.AutoScalingCheckBox.setChecked(1)
            else:
                self.guiconfig.AutoScalingCheckBox.setChecked(0)
            #update the GUI
            self.__initialparameters() 

    
    def configureGUI(self,oldconfiguration):
        #this method can be overwritten for custom
        #it should give back a new dictionary
        newconfiguration={}
        newconfiguration.update(oldconfiguration)
        if (0):
        #example to force a given default configuration
            newconfiguration['FitTheory']="Pseudo-Voigt Line"
            newconfiguration['AutoFwhm']=1
            newconfiguration['AutoScaling']=1
        
        #example script options like
        if (1):
            sheet1={'notetitle':'Restrains',
                'fields':(["CheckField",'HeightAreaFlag','Force positive Height/Area'],
                          ["CheckField",'PositionFlag','Force position in interval'],
                          ["CheckField",'PosFwhmFlag','Force positive FWHM'],
                          ["CheckField",'SameFwhmFlag','Force same FWHM'],
                          ["CheckField",'EtaFlag','Force Eta between 0 and 1'],
                          ["CheckField",'NoConstrainsFlag','Ignore Restrains'])}
 
            sheet2={'notetitle':'Search',
                'fields':(["EntryField",'FwhmPoints', 'Fwhm Points: '],
                          ["EntryField",'Sensitivity','Sensitivity: '],
                          ["EntryField",'Yscaling',   'Y Factor   : '],
                          ["CheckField",'ForcePeakPresence',   'Force peak presence '])}
            w=QScriptOption.QScriptOption(self,name='Fit Configuration',
                            sheets=(sheet1,sheet2),
                            default=oldconfiguration)
            
            w.show()
            if QTVERSION < '4.0.0': w.exec_loop()
            else:  w.exec_()
            if w.result():
                newconfiguration.update(w.output)
            #we do not need the dialog any longer
            del w
            newconfiguration['FwhmPoints']=int(float(newconfiguration['FwhmPoints']))
            newconfiguration['Sensitivity']=float(newconfiguration['Sensitivity'])
            newconfiguration['Yscaling']=float(newconfiguration['Yscaling'])
        return newconfiguration

    def estimate(self):
        if self.specfit.fitconfig['McaMode']:
            try:
                mcaresult=self.specfit.mcafit()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error on mcafit")
                if QTVERSION < '4.0.0':msg.exec_loop()
                else: msg.exec_()
                ddict={}
                ddict['event'] = 'FitError'
                if QTVERSION < '4.0.0':
                    self.emit(qt.PYSIGNAL('SpecfitGUISignal'),(ddict,))
                else:
                    self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
                if DEBUG:
                    raise
                return
            self.guiparameters.fillfrommca(mcaresult)
            ddict={}
            ddict['event'] = 'McaFitFinished'
            ddict['data']  = mcaresult
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('SpecfitGUISignal'),(ddict,))
            else:
                self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
            #self.guiparameters.removeallviews(keep='Region 1')
        else:
            try:
                if self.specfit.theorydict[self.specfit.fitconfig['fittheory']][2] is not None:
                    self.specfit.estimate()
                else:
                    msg = qt.QMessageBox(self)
                    msg.setIcon(qt.QMessageBox.Information)
                    text  = "Function does not define a way to estimate\n"
                    text += "the initial parameters. Please, fill them\n"
                    text += "yourself in the table and press Start Fit\n"
                    msg.setText(text)
                    if QTVERSION < '4.0.0':
                        msg.exec_loop()
                    else:
                        msg.setWindowTitle('SpecfitGUI Message')
                        msg.exec_()
                    return
            except:
                if DEBUG:
                    raise
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error on estimate: %s" % sys.exc_info()[1])
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                return
            self.guiparameters.fillfromfit(self.specfit.paramlist,current='Fit')
            self.guiparameters.removeallviews(keep='Fit')
            ddict={}
            ddict['event'] = 'EstimateFinished'
            ddict['data']  = self.specfit.paramlist
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('SpecfitGUISignal'),(ddict,))
            else:
                self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
            
        return

    def __forward(self,ddict):
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('SpecfitGUISignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
    
    def startfit(self):
        if self.specfit.fitconfig['McaMode']:
            try:
                mcaresult=self.specfit.mcafit()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error on mcafit: %s" % sys.exc_info()[1])
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                if DEBUG:
                    raise
                return
            self.guiparameters.fillfrommca(mcaresult)
            ddict={}
            ddict['event'] = 'McaFitFinished'
            ddict['data']  = mcaresult
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('SpecfitGUISignal'),(ddict,))
            else:
                self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
            #self.guiparameters.removeview(view='Fit')
        else:
            #for param in self.specfit.paramlist:
            #    print param['name'],param['group'],param['estimation']
            self.specfit.paramlist=self.guiparameters.fillfitfromtable()
            if DEBUG:
                for param in self.specfit.paramlist:
                    print(param['name'],param['group'],param['estimation'])
                print("TESTING")
                self.specfit.startfit()
            try:
                self.specfit.startfit()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error on Fit")
                if QTVERSION < '4.0.0':
                    msg.exec_loop()
                else:
                    msg.exec_()
                if DEBUG:
                    raise
                return
            self.guiparameters.fillfromfit(self.specfit.paramlist,current='Fit')
            self.guiparameters.removeallviews(keep='Fit')
            ddict={}
            ddict['event'] = 'FitFinished'
            ddict['data']  = self.specfit.paramlist
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('SpecfitGUISignal'),(ddict,))
            else:
                self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
        return
    
    
    def printps(self,**kw):
        text = self.guiparameters.gettext(**kw)
        if __name__ == "__main__":
            self.__printps(text)
        else:
            ddict={}
            ddict['event'] = 'print'
            ddict['text']  = text
            if QTVERSION < '4.0.0':
                self.emit(qt.PYSIGNAL('SpecfitGUISignal'), (ddict,))
            else:
                self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
        return

    if QTVERSION < '4.0.0':        
        def __printps(self,text):
            printer = qt.QPrinter()
            if printer.setup(self):
                painter = qt.QPainter()
                if not(painter.begin(printer)):
                    return 0
            metrics = qt.QPaintDeviceMetrics(printer)
            dpiy    = metrics.logicalDpiY()
            margin  = int((2/2.54) * dpiy) #2cm margin
            body = qt.QRect(0.5*margin, margin, metrics.width()- 1 * margin, metrics.height() - 2 * margin)
            #text = self.mcatable.gettext()
            #html output -> print text
            richtext = qt.QSimpleRichText(text, qt.QFont(),
                                                qt.QString(""),
                                                #0,
                                                qt.QStyleSheet.defaultSheet(),
                                                qt.QMimeSourceFactory.defaultFactory(),
                                                body.height())
            view = qt.QRect(body)
            richtext.setWidth(painter,view.width())
            page = 1                
            while(1):
                richtext.draw(painter,body.left(),body.top(),
                              view,qt.QColorGroup())
                view.moveBy(0, body.height())
                painter.translate(0, -body.height())
                painter.drawText(view.right()  - painter.fontMetrics().width(qt.QString.number(page)),
                                 view.bottom() - painter.fontMetrics().ascent() + 5,qt.QString.number(page))
                if view.top() >= richtext.height():
                    break
                printer.newPage()
                page += 1

            #painter.flush()
            painter.end()
    else:
        def __printps(self, text):
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Sorry, Qt4 printing not implemented yet")
            msg.exec_()            
    
    def mcaevent(self,item):
        if int(item):
            self.configure(McaMode=1)
            mode = 1
        else:
            self.configure(McaMode=0)
            mode = 0
        self.__initialparameters() 
        ddict={}
        ddict['event'] = 'McaModeChanged'
        ddict['data']  = mode 
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('SpecfitGUISignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('SpecfitGUISignal'), ddict)
        return

    def weightevent(self,item):
        if int(item):
            self.configure(WeightFlag=1)
        else:
            self.configure(WeightFlag=0)
        return

    def autofwhmevent(self,item):
        if int(item):
            self.configure(AutoFwhm=1)
        else:
            self.configure(AutoFwhm=0)
        return

    def autoscaleevent(self,item):
        if int(item):
            self.configure(AutoScaling=1)
        else:
            self.configure(AutoScaling=0)
        return
    
    def bkgevent(self,item):
        item=str(item)
        if item in self.specfit.bkgdict.keys():
            self.specfit.setbackground(item)
        else:
            qt.QMessageBox.information(self, "Info", "Function not implemented")
            return
            i=1+self.specfit.bkgdict.keys().index(self.specfit.fitconfig['fitbkg'])
            if QTVERSION < '4.0.0':
                self.guiconfig.BkgComBox.setCurrentItem(i)
            else:
                self.guiconfig.BkgComBox.setCurrentIndex(i)
        self.__initialparameters()
        return

    def funevent(self,item):
        item=str(item)
        if item in self.specfit.theorylist:
            self.specfit.settheory(item)
        else:
            # TODO why this strange condition
            if 1:
                fn = qt.QFileDialog.getOpenFileName()
            else:
                dlg=qt.QFileDialog(qt.QString.null,qt.QString.null,self,None,1)
                dlg.show()            
                fn=dlg.selectedFile()
            if fn.isEmpty():
                functionsfile = ""
            else:
                functionsfile="%s" % fn
            if len(functionsfile):
                try:
                    if self.specfit.importfun(functionsfile):
                        qt.QMessageBox.critical(self, "ERROR",
                                                "Function not imported")
                        return
                    else:
                        #empty the ComboBox
                        n=self.guiconfig.FunComBox.count()
                        while(self.guiconfig.FunComBox.count()>1):
                          self.guiconfig.FunComBox.removeItem(1)
                        #and fill it again
                        for key in self.specfit.theorylist:
                            if QTVERSION < '4.0.0':
                                self.guiconfig.FunComBox.insertItem(str(key))
                            else:
                                self.guiconfig.FunComBox.addItem(str(key))
                except:
                    qt.QMessageBox.critical(self, "ERROR",
                                            "Function not imported")
            i=1+self.specfit.theorylist.index(self.specfit.fitconfig['fittheory'])
            if QTVERSION < '4.0.0':
                self.guiconfig.FunComBox.setCurrentItem(i)
            else:
                self.guiconfig.FunComBox.setCurrentIndex(i)
        self.__initialparameters()
        return
    
    def __initialparameters(self):
        self.specfit.final_theory=[]
        self.specfit.paramlist=[]
        for pname in self.specfit.bkgdict[self.specfit.fitconfig['fitbkg']][1]:
            self.specfit.final_theory.append(pname)
            self.specfit.paramlist.append({'name':pname,
                                       'estimation':0,
                                       'group':0,
                                       'code':'FREE',
                                       'cons1':0,
                                       'cons2':0,
                                       'fitresult':0.0,
                                       'sigma':0.0,
                                       'xmin':None,
                                       'xmax':None})
        if self.specfit.fitconfig['fittheory'] is not None:
          for pname in self.specfit.theorydict[self.specfit.fitconfig['fittheory']][1]:
            self.specfit.final_theory.append(pname+"1")
            self.specfit.paramlist.append({'name':pname+"1",
                                       'estimation':0,
                                       'group':1,
                                       'code':'FREE',
                                       'cons1':0,
                                       'cons2':0,
                                       'fitresult':0.0,
                                       'sigma':0.0,
                                       'xmin':None,
                                       'xmax':None})
        if self.specfit.fitconfig['McaMode']:
            self.guiparameters.fillfromfit(self.specfit.paramlist,current='Region 1')
            self.guiparameters.removeallviews(keep='Region 1')
        else:
            self.guiparameters.fillfromfit(self.specfit.paramlist,current='Fit')
            self.guiparameters.removeallviews(keep='Fit')
        return

    def fitstatus(self,data):
        if 'chisq' in data:
            if data['chisq'] is None:
                self.guistatus.ChisqLine.setText(" ")
            else:
                chisq=data['chisq']
                self.guistatus.ChisqLine.setText("%6.2f" % chisq)
            
        if 'status' in data:
            status=data['status']
            self.guistatus.StatusLine.setText(str(status))
        return


    
    def dismiss(self):
        self.close()
        return
        
if __name__ == "__main__":
    import numpy
    from PyMca import SpecfitFunctions
    a=SpecfitFunctions.SpecfitFunctions()
    x = numpy.arange(2000).astype(numpy.float)
    p1 = numpy.array([1500,100.,30.0])
    p2 = numpy.array([1500,300.,30.0])
    p3 = numpy.array([1500,500.,30.0])
    p4 = numpy.array([1500,700.,30.0])
    p5 = numpy.array([1500,900.,30.0])
    p6 = numpy.array([1500,1100.,30.0])
    p7 = numpy.array([1500,1300.,30.0])
    p8 = numpy.array([1500,1500.,30.0])
    p9 = numpy.array([1500,1700.,30.0])
    p10 = numpy.array([1500,1900.,30.0])
    y = a.gauss(p1,x)+1
    y = y + a.gauss(p2,x)
    y = y + a.gauss(p3,x)
    y = y + a.gauss(p4,x)
    y = y + a.gauss(p5,x)
    #y = y + a.gauss(p6,x)
    #y = y + a.gauss(p7,x)
    #y = y + a.gauss(p8,x)
    #y = y + a.gauss(p9,x)
    #y = y + a.gauss(p10,x)
    y=y/1000.0
    a = qt.QApplication(sys.argv)
    qt.QObject.connect(a,qt.SIGNAL("lastWindowClosed()"),a,qt.SLOT("quit()"))
    w = SpecfitGUI(config=1, status=1, buttons=1)
    w.setdata(x=x,y=y)
    if QTVERSION < '4.0.0':
        a.setMainWidget(w)
        w.show()
        a.exec_loop()
    else:
        w.show()
        a.exec_()
