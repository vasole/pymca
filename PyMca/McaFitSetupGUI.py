#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
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
# is a problem to you.
#############################################################################*/
import qt
DEBUG = 0
class McaFitSetupGUI(qt.QWidget):
    def __init__(self, parent=None, name=None,
                 sourcewidget = 0,
                 calwidget = 0,fl=0):
        qt.QWidget.__init__(self, parent, name, fl)
        self.sourcewidget = sourcewidget
        self.calwidget    = calwidget
        self.build()
        self.connections()
        
    def build(self):
        self.layout = qt.QGridLayout(self)
        self.layout.setSpacing(5)
        offset = 0
        if self.sourcewidget:
            # Data Source
            sourcelabel     = qt.QLabel(self,"label")
            sourcelabel.setText(str("<b>%s</b>" % 'Data Source'))
            self.sourcebox = SimpleComboBox(self,
                                                options=['Active Graph','SpecFile','EDF File','SPS'])
            self.sourcebut = qt.QPushButton(self)
            self.sourcebut.setText('Get Data')
            self.layout.addWidget(sourcelabel,0,0)
            self.layout.addWidget(self.sourcebox,0,1)
            self.layout.addWidget(self.sourcebut,0,2)
            offset = 1
        if self.calwidget:
            # Calibration Source
            callabel    = qt.QLabel(self)
            callabel.setText(str("<b>%s</b>" % 'Calibration'))
            self.calbox = SimpleComboBox(self,
                                                options=['None','Internal','Fit'])
            self.calbut = qt.QPushButton(self)
            self.calbut.setText('Calibrate')
            self.layout.addWidget(callabel,offset,0)                                      
            self.layout.addWidget(self.calbox,offset,1)        
            self.layout.addWidget(self.calbut,offset,2)
            offset = offset + 1       
        # Detector type
        detlabel    = qt.QLabel(self)
        detlabel.setText(str("<b>%s</b>" % 'Detector'))
        self.detbox = SimpleComboBox(self,
                                            options=['None','Si','Ge'])
        self.detbut = qt.QPushButton(self)
        self.detbut.setText('Get Det')
        self.layout.addWidget(detlabel,offset,0)              
        self.layout.addWidget(self.detbox,offset,1)              
        self.layout.addWidget(self.detbut,offset,2)        
        offset += 1
        # Search type
        searchlabel    = qt.QLabel(self)
        searchlabel.setText(str("<b>%s</b>" % 'Peak Search'))
        self.searchbox = SimpleComboBox(self,
                                            options=['Auto','User Defined'])
        self.searchbut = qt.QPushButton(self)
        self.searchbut.setText('Configure')
        self.layout.addWidget(searchlabel,offset,0)              
        self.layout.addWidget(self.searchbox,offset,1)              
        self.layout.addWidget(self.searchbut,offset,2)        

        offset += 1
        # Background
        bkglabel     = qt.QLabel(self)
        bkglabel.setText(str("<b>%s</b>" % 'Background'))
        self.bkgbox = SimpleComboBox(self,
                                            options=['None','Constant','Linear'])
        self.layout.addWidget(bkglabel,offset,0)
        self.layout.addWidget(self.bkgbox,offset,1)

        # Detector
        paroffset = 1 + offset
        parlabel = qt.QLabel(self)
        parlabel.setText(str("<b>%s</b>" % 'Parameter'))
        vallabel = qt.QLabel(self)
        vallabel.setText(str("<b>%s</b>" % 'Value'))
        deltalabel = qt.QLabel(self)
        deltalabel.setText(str("<b>%s</b>" % '+/-  Delta'))
        self.layout.addWidget(parlabel,paroffset,0)  
        self.layout.addWidget(vallabel,paroffset,1)              
        self.layout.addWidget(deltalabel,paroffset,2)        
        # global fit parameters
        zerolabel =qt.QLabel(self)
        zerolabel.setText(str("<b>%s</b>" % 'Zero')) 
        gainlabel =qt.QLabel(self)
        gainlabel.setText(str("<b>%s</b>" % 'Gain')) 
        fanolabel =qt.QLabel(self)
        fanolabel.setText(str("<b>%s</b>" % 'Fano')) 
        noiselabel =qt.QLabel(self)
        noiselabel.setText(str("<b>%s</b>" % 'Noise')) 
        self.zeroline=qt.QLineEdit(self)
        self.zeroline.setText('0.0')
        self.deltazeroline=qt.QLineEdit(self)        
        self.deltazeroline.setText('0.1')
        self.gainline=qt.QLineEdit(self)
        self.gainline.setText('1.0')
        self.deltagainline=qt.QLineEdit(self)        
        self.deltagainline.setText('1.0')

        self.fanoline=qt.QLineEdit(self)
        self.fanoline.setText('0.114')
        self.deltafanoline=qt.QLineEdit(self)        
        self.deltafanoline.setText('0.05')
        self.noiseline=qt.QLineEdit(self)
        self.noiseline.setText('0.15')
        self.deltanoiseline=qt.QLineEdit(self)        
        self.deltanoiseline.setText('0.05')

        sumlabel =qt.QLabel(self)
        sumlabel.setText(str("<b>%s</b>" % 'Sum')) 
        stalabel =qt.QLabel(self)
        stalabel.setText(str("<b>%s</b>" % 'ST Area  Ratio')) 
        stslabel =qt.QLabel(self)
        stslabel.setText(str("<b>%s</b>" % 'ST Slope Ratio')) 
        ltalabel =qt.QLabel(self)
        ltalabel.setText(str("<b>%s</b>" % 'LT Area  Ratio')) 
        ltslabel =qt.QLabel(self)
        ltslabel.setText(str("<b>%s</b>" % 'LT Slope Ratio')) 
        sthlabel =qt.QLabel(self)
        sthlabel.setText(str("<b>%s</b>" % 'Step H   Ratio')) 
        
        self.sumline=qt.QLineEdit(self)
        self.sumline.setText('1.0E-10')
        self.deltasumline=qt.QLineEdit(self)
        self.deltasumline.setText('1.0E-10')

        self.staline=qt.QLineEdit(self)
        self.staline.setText('0.0')
        self.deltastaline=qt.QLineEdit(self)
        self.deltastaline.setText('0.0')

        self.stsline=qt.QLineEdit(self)
        self.stsline.setText('0.0')
        self.deltastsline=qt.QLineEdit(self)
        self.deltastsline.setText('0.0')

        self.ltaline=qt.QLineEdit(self)
        self.ltaline.setText('0.0')
        self.deltaltaline=qt.QLineEdit(self)
        self.deltaltaline.setText('0.0')
        
        self.ltsline=qt.QLineEdit(self)
        self.ltsline.setText('0.0')
        self.deltaltsline=qt.QLineEdit(self)
        self.deltaltsline.setText('0.0')

        self.sthline=qt.QLineEdit(self)
        self.sthline.setText('0.0')
        self.deltasthline=qt.QLineEdit(self)
        self.deltasthline.setText('0.0')


        self.layout.addWidget(zerolabel,paroffset+1,0)
        self.layout.addWidget(self.zeroline,paroffset+1,1)
        self.layout.addWidget(self.deltazeroline,paroffset+1,2)
        self.layout.addWidget(gainlabel,paroffset+2,0)
        self.layout.addWidget(self.gainline,paroffset+2,1)
        self.layout.addWidget(self.deltagainline,paroffset+2,2)
        self.layout.addWidget(fanolabel,paroffset+3,0)
        self.layout.addWidget(self.fanoline,paroffset+3,1)
        self.layout.addWidget(self.deltafanoline,paroffset+3,2)
        self.layout.addWidget(noiselabel,paroffset+4,0)
        self.layout.addWidget(self.noiseline,paroffset+4,1)
        self.layout.addWidget(self.deltanoiseline,paroffset+4,2)

        self.layout.addWidget(sumlabel,paroffset+5,0)
        self.layout.addWidget(self.sumline,paroffset+5,1)
        self.layout.addWidget(self.deltasumline,paroffset+5,2)
        
        #HYPERMET
        if 0:
            oldparoffset = paroffset * 1
            paroffset = -5
            coffset = 4
            lab = qt.QLabel(self)
            lab.setText(str("<b>%s</b>" % 'HYPERMET'))
            self.layout.addMultiCellWidget(lab,0,0,coffset +0,coffset +3,qt.Qt.AlignCenter)
        else:
            coffset = 0
        self.layout.addWidget(stalabel,paroffset+6,coffset +0)
        self.layout.addWidget(self.staline,paroffset+6,coffset +1)
        self.layout.addWidget(self.deltastaline,paroffset+6,coffset +2)

        self.layout.addWidget(stslabel,paroffset+7,coffset +0)
        self.layout.addWidget(self.stsline,paroffset+7,coffset +1)
        self.layout.addWidget(self.deltastsline,paroffset+7,coffset +2)

        self.layout.addWidget(ltalabel,paroffset+8,coffset +0)
        self.layout.addWidget(self.ltaline,paroffset+8,coffset +1)
        self.layout.addWidget(self.deltaltaline,paroffset+8,coffset +2)

        self.layout.addWidget(ltslabel,paroffset+9,coffset +0)
        self.layout.addWidget(self.ltsline,paroffset+9,coffset +1)
        self.layout.addWidget(self.deltaltsline,paroffset+9,coffset +2)

        self.layout.addWidget(sthlabel,paroffset+10,coffset +0)
        self.layout.addWidget(self.sthline,paroffset+10,coffset +1)
        self.layout.addWidget(self.deltasthline,paroffset+10,coffset +2)
        

        # Flags
        flagsoffset = 10 + paroffset + 1
        hyplabel = qt.QLabel(self)
        hyplabel.setText(str("<b>%s</b>" % 'FLAGS'))
        self.stbox = qt.QCheckBox(self)
        self.stbox.setText('Short Tail')        
        self.ltbox = qt.QCheckBox(self)
        self.ltbox.setText('Long Tail')        
        self.stepbox = qt.QCheckBox(self)
        self.stepbox.setText('Step Tail')        
        self.escapebox = qt.QCheckBox(self)
        self.escapebox.setText('Escape')        
        self.sumbox = qt.QCheckBox(self)
        self.sumbox.setText('Summing')        
        self.stripbox = qt.QCheckBox(self)
        self.stripbox.setText('Strip Back.')        
        self.layout.addWidget(hyplabel,flagsoffset,coffset +1) 
        self.layout.addWidget(self.stbox,flagsoffset+1,coffset +0)              
        self.layout.addWidget(self.ltbox,flagsoffset+1,coffset +1)              
        self.layout.addWidget(self.stepbox,flagsoffset+1,coffset +2)        
        self.layout.addWidget(self.escapebox,flagsoffset+2,coffset +0)              
        self.layout.addWidget(self.sumbox,flagsoffset+2,coffset +1)              
        self.layout.addWidget(self.stripbox,flagsoffset+2,coffset +2)        
        
        # Background
        
        spacer = qt.QSpacerItem(40,20,qt.QSizePolicy.Minimum,qt.QSizePolicy.Expanding)
        self.layout.addItem(spacer)

    def connections(self):
        #QObject.connect(a,SIGNAL("lastWindowClosed()"),a,SLOT("quit()"
        #selection changed
        if self.sourcewidget:
            self.connect(self.sourcebox,qt.SIGNAL("activated(const QString &)"),
                    self.__sourceboxactivated)
            self.connect(self.sourcebut,qt.SIGNAL("clicked()"),self.__sourcebuttonclicked)
        if self.calwidget:
            self.connect(self.calbox,qt.SIGNAL("activated(const QString &)"),
                    self.__calboxactivated)
            self.connect(self.calbut,qt.SIGNAL("clicked()"),self.__calbuttonclicked)
        self.connect(self.searchbox,qt.SIGNAL("activated(const QString &)"),
                    self.__searchboxactivated)
        self.connect(self.detbox,qt.SIGNAL("activated(const QString &)"),
                    self.__detboxactivated)
        #push button pressed
        self.connect(self.searchbut,qt.SIGNAL("clicked()"),self.__searchbuttonclicked)
        self.connect(self.detbut,qt.SIGNAL("clicked()"),   self.__detbuttonclicked)
        #            lambda this=self,but=self.sourcebut,box=self.sourcebox:McaFitSetupGUI.__buttonclicked(this,but,box,label='Source')) 
    
    def __sourceboxactivated(self,item):
        item = str(item)
        if DEBUG:
            print "Source box activated ",item
        comboitem,combotext = self.sourcebox.getcurrent()
        self.__emitpysignal(box=[comboitem,combotext],boxname='Source',
                           event='activated')

    def __calboxactivated(self,item):
        item = str(item)
        if DEBUG:
            print "Calibration box activated ",item
        comboitem,combotext = self.calbox.getcurrent()
        self.__emitpysignal(box=[comboitem,combotext],boxname='Calibration',
                            event='activated')

    def __searchboxactivated(self,item):
        item = str(item)
        if DEBUG:
            print "Search box activated ",item
        comboitem,combotext = self.searchbox.getcurrent()
        self.__emitpysignal(box=[comboitem,combotext],boxname='Search',
                            event='activated')

    def __detboxactivated(self,item):
        item = str(item)
        if DEBUG:
            print "Detector box activated ",item
        comboitem,combotext = self.detbox.getcurrent()
        self.__emitpysignal(box=[comboitem,combotext],boxname='Detector',
                            event='activated')

    def __sourcebuttonclicked(self):
        if DEBUG:
            print "Source button clicked"
        comboitem,combotext = self.sourcebox.getcurrent()
        self.__emitpysignal(button="Source",box=[comboitem,combotext],event='clicked')
        

    def __calbuttonclicked(self):
        if DEBUG:
            print "Calibration button clicked"
        comboitem,combotext = self.calbox.getcurrent()
        self.__emitpysignal(button="Calibration",box=[comboitem,combotext],event='clicked')

    def __detbuttonclicked(self):
        if DEBUG:
            print "Detector button clicked"
        comboitem,combotext = self.detbox.getcurrent()
        self.__emitpysignal(button="Detector",box=[comboitem,combotext],event='clicked')

    def __searchbuttonclicked(self):
        if DEBUG:
            print "Search button clicked"
        comboitem,combotext = self.searchbox.getcurrent()
        self.__emitpysignal(button="Search",box=[comboitem,combotext],event='clicked')
        
    def __emitpysignal(self,button=None,
                            box=None,
                            boxname=None,
                            checkbox=None,
                            line_edit=None,
                            event=None):
        if DEBUG:
            print "__emitpysignal called ",button,box
        data={}
        data['button']        = button
        data['box']           = box
        data['checkbox']      = checkbox
        data['line_edit']     = line_edit
        data['event']         = event
        data['boxname']       = boxname        
        self.emit(qt.PYSIGNAL("McaFitSetupGUISignal"),(data,))
        
    def getconfig(self):
        pass
    
    def setconfig(self,config):
        pass

    
class SimpleComboBox(qt.QComboBox):
        def __init__(self,parent = None,name = None,fl = 0,options=['1','2','3']):
            qt.QComboBox.__init__(self,parent)
            self.setoptions(options) 
            
        def setoptions(self,options=['1','2','3']):
            self.clear()    
            self.insertStrList(options)
            
        def getcurrent(self):
            return   self.currentItem(),str(self.currentText())
             
def main(args):
    app = qt.QApplication(args)
    #demo = make()
    demo = McaFitSetupGUI()
    app.setMainWidget(demo)
    demo.show()
    app.exec_loop()

if __name__ == '__main__':
    import sys
    main(sys.argv)

