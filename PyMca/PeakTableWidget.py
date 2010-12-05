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
__revision__ = "$Revision: 1.8 $"
__author__="V.A. Sole - ESRF BLISS Group"
import sys
import PyMcaQt as qt
if qt.qVersion() < '3.0.0':
    import Myqttable as qttable
elif qt.qVersion() < '4.0.0':
    import qttable
import string
import Elements

DEBUG=0

if qt.qVersion() < '4.0.0':
    class QTable(qttable.QTable):
        def __init__(self, parent=None, name=""):
            qttable.QTable.__init__(self, parent, name)
            self.rowCount    = self.numRows
            self.columnCount = self.numCols
            self.setRowCount = self.setNumRows
            self.setColumnCount = self.setNumCols
    QComboTableItem = qttable.QComboTableItem
else:
    QTable = qt.QTableWidget
    class QComboTableItem(qt.QComboBox):
        def __init__(self, parent=None, row = None, col = None):
            self._row = row
            self._col = col
            qt.QComboBox.__init__(self,parent)
            self.connect(self, qt.SIGNAL('activated(int)'), self._cellChanged)

        def _cellChanged(self, idx):
            if DEBUG:
                print("cell changed",idx)
            self.emit(qt.SIGNAL('cellChanged(int,int)'), self._row, self._col)

    class QCheckBoxItem(qt.QCheckBox):
        def __init__(self, parent=None, row = None, col = None):
            self._row = row
            self._col = col
            qt.QCheckBox.__init__(self,parent)
            self.connect(self, qt.SIGNAL('clicked()'), self._cellChanged)

        def _cellChanged(self):
            self.emit(qt.SIGNAL('cellChanged(int, int)'), self._row, self._col)


class PeakTableWidget(QTable):
    def __init__(self, *args,**kw):
        QTable.__init__(self, *args)
        self.setRowCount(0)
        self.labels=['Peak','Channel','Element','Line',
                     'Energy','Use','Calc. Energy']
        self.setColumnCount(len(self.labels))
        if 'labels' in kw:
            self.labels = kw['labels']
        if qt.qVersion() < '4.0.0':
            i=0
            for label in self.labels:
                qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                i=i+1
        else:
            for i in range(len(self.labels)):
                item = self.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(self.labels[i],
                                               qt.QTableWidgetItem.Type)
                item.setText(self.labels[i])
                self.setHorizontalHeaderItem(i,item)
                
        self.peaks={}
        self.peaklist=[]
        if 'peaklist' in kw:
            self.peaklist = kw['peaklist']
        self.build()
        if qt.qVersion() < '4.0.0':
            self.connect(self,qt.SIGNAL("valueChanged(int,int)"),self.myslot)
        else:
            self.connect(self,qt.SIGNAL("cellChanged(int,int)"),self.myslot)

        if qt.qVersion() > '4.0.0':
            rheight = self.horizontalHeader().sizeHint().height()
            for idx in range(self.rowCount()):
                self.setRowHeight(idx, rheight)
        

    def build(self):
        line = 1
        oldlist=list(self.peaklist)
        self.peaklist=[]
        for peak in oldlist:
            self.newpeakline(peak,line)
            line=line+1
        if qt.qVersion() < '4.0.0':
            self.adjustColumn(0)    
            self.adjustColumn(1)    
            self.adjustColumn(2)    
            self.adjustColumn(5)
        else:
            self.resizeColumnToContents(0)
            #self.resizeColumnToContents(1)
            #self.resizeColumnToContents(2)
            self.resizeColumnToContents(5)

    def clearPeaks(self):
        self.peaks = {}
        self.peaklist = []
        self.setRowCount(0)
        
    def newpeakline(self,peak,line):
        #get current number of lines
        nlines=self.rowCount()
        #if the number of lines is smaller than line resize table
        if (line > nlines):
            self.setRowCount(line)
        linew=line-1
        self.peaks[peak]={ 'line':linew,
                           'fields':['number',
                                    'channel',
                                    'element',
                                    'elementline',
                                    'setenergy',
                                    'use',
                                    'calenergy'],
                          'number':     qt.QString('1'),          
                          'channel':    qt.QString('0'),
                          'element':    qt.QString('-'),
                          'elementline':qt.QString('-'),
                          'setenergy':  qt.QString('0'),
                          'use':        0,
                          'calenergy':  qt.QString()}
        self.peaklist.append(peak)
        self.setReadWrite(peak,'setenergy')
        self.setReadWrite(peak,'channel')
        self.setReadOnly (peak,['number','line','calenergy'])
        col = self.peaks[peak]['fields'].index('element')
        self.peaks[peak]['element_item']=QPeriodicComboTableItem(self,
                                        row = linew, col= col)
        if qt.qVersion() < '4.0.0':
            self.setItem(linew,
                         col,
                         self.peaks[peak]['element_item'])
        else:
            self.setCellWidget(linew,
                               col,
                               self.peaks[peak]['element_item'])
            self.connect(self.peaks[peak]['element_item'],
                         qt.SIGNAL('cellChanged(int,int)'), self.myslot)
                         #qt.SIGNAL('activated(int)'), self.myslot)
        try:
            a = qt.QStringList()
        except AttributeError:
            a = []
        a.append('-')
        col = self.peaks[peak]['fields'].index('elementline')
        if qt.qVersion() < '4.0.0':
            self.peaks[peak]['elementline_item']= QComboTableItem(self,a)
            self.setItem(linew,
                         col,
                         self.peaks[peak]['elementline_item'])
        else:
           self.peaks[peak]['elementline_item']= QComboTableItem(self,
                                                                 row = linew,
                                                                 col = col)
           self.peaks[peak]['elementline_item'].addItems(a)
           self.setCellWidget(linew,
                         col,
                         self.peaks[peak]['elementline_item'])
           self.connect(self.peaks[peak]['elementline_item'],
                         qt.SIGNAL('cellChanged(int,int)'), self.myslot)
        
        col = self.peaks[peak]['fields'].index('use')
        if qt.qVersion() < '4.0.0':
            self.peaks[peak]['use_item']    = qttable.QCheckTableItem(self,"")
            self.setItem(linew, col,
                     self.peaks[peak]['use_item'])
        else:
            self.peaks[peak]['use_item']    = QCheckBoxItem(self,
                                                            row = linew,
                                                            col = col)
            self.peaks[peak]['use_item'].setText("")
            self.setCellWidget(linew, col,
                     self.peaks[peak]['use_item'])
            self.connect(self.peaks[peak]['use_item'],
                         qt.SIGNAL('cellChanged(int,int)'), self.myslot)

        self.peaks[peak]['use_item'].setChecked(self.peaks[peak]['use'])
        #Not supported below 3.0
        #self.setColumnReadOnly(self.parameters[param]['fields'].index('name'),1)
        #self.setColumnReadOnly(self.parameters[param]['fields'].index('fitresult'),1)
        #self.setColumnReadOnly(self.parameters[param]['fields'].index('sigma'),1)

    def myslot(self,row,col):
        if DEBUG:
            print("Passing by myslot",
                  self.peaks[self.peaklist[row]]['fields'][col])
        peak=self.peaklist[row]
        field=self.peaks[peak]['fields'][col]
        if (field == "element") or (field == "elementline"):
            key = field+"_item"
            newvalue=self.peaks[peak][key].currentText()
        elif field == "use":
            pass
        else:
            if qt.qVersion() < '4.0.0':
                newvalue=qt.QString(self.text(row,col))
            else:
                newvalue = self.item(row, col).text()
        if field == "element":
            if str(newvalue) == '-':
                #no element
                #set line to -
                try:
                    options  = qt.QStringList()
                except AttributeError:
                    options = []
                options.append('-')
                if qt.qVersion() < '4.0.0':
                    self.peaks[peak]["elementline_item"].setStringList(options)
                    self.peaks[peak]["elementline_item"].setCurrentItem(0)
                else:
                    self.peaks[peak]["elementline_item"].insertItems(0, options)
                    self.peaks[peak]["elementline_item"].setCurrentIndex(0)
            else:
                #get the emission energies
                ele = str(newvalue).split()[0]
                try:
                    options  = qt.QStringList()
                    energies = qt.QStringList()
                except AttributeError:
                    options  = []
                    energies = []                    
                options.append('-') 
                energies.append('0.000')
                emax = 0.0
                for rays in Elements.Element[ele]['rays']:
                    for transition in Elements.Element[ele][rays]:
                        options.append("%s (%.5f)" % (transition,
                                    Elements.Element[ele][transition]['rate']))
                        energies.append("%.5f " % (Elements.Element[ele][transition]['energy']))
                        emax = max(emax,Elements.Element[ele][transition]['rate'])
                energies[0] = "%.5f " % emax
                #lineitem=qttable.QComboTableItem(self,options)
                if qt.qVersion() < '4.0.0':
                    self.peaks[peak]["elementline_item"].setStringList(options)
                    self.peaks[peak]["elementline_item"].setCurrentItem(0)
                else:
                    self.peaks[peak]["elementline_item"].insertItems(0, options)
                    self.peaks[peak]["elementline_item"].setCurrentIndex(0)
                #self.setItem(row,
                #             col+1,
                #             lineitem)
            self.peaks[peak][field] = newvalue
        if field == "elementline":
            if str(newvalue) == '-':
                #no element
                #set energy to rw
                self.setReadWrite(peak,'setenergy')
            else:
                #get the element energy
                #newvalue=qt.QString(self.text(row,col-1))
                elevalue=self.peaks[peak]["element_item"].currentText()
                ele = str(elevalue).split()[0]
                energy = "0.0"
                for rays in Elements.Element[ele]['rays']:
                    for transition in Elements.Element[ele][rays]:
                        option = qt.QString("%s (%.5f)" % (transition,
                                    Elements.Element[ele][transition]['rate']))
                        if option == newvalue:
                            energy = "%.5f " % (Elements.Element[ele][transition]['energy'])
                            break
                if energy == "0.0":
                    print("Something is wrong")
                else:
                    self.configure(name=peak,setenergy=energy)
                self.setReadOnly(peak,'setenergy')
            self.peaks[peak][field] = newvalue
        if field == "setenergy":
            oldvalue = self.peaks[peak]["setenergy"]
            try:
                value = float(str(newvalue))
            except:
                print("taking old value")
                if qt.qVersion() < '4.0.0':
                    self.setText(row,col,oldvalue)
                else:
                    item = self.item(row, col)
                    item.setText("%s" % oldvalue)
                value = float(str(oldvalue))
            self.peaks[peak][field] = value 
            ddict={}
            ddict['event'] = 'use'
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL('PeakTableWidgetSignal'),(ddict,))
            else:
                self.emit(qt.SIGNAL('PeakTableWidgetSignal'), (ddict))

        if field == "channel":
            oldvalue = self.peaks[peak]["channel"]
            try:
                value = float(str(newvalue))
            except:
                print("taking old value")
                if qt.qVersion() < '4.0.0':
                    self.setText(row,col,oldvalue)
                else:
                    item = self.item(row, col)
                    item.setText("%s" % oldvalue)
                value = float(str(oldvalue))
            self.peaks[peak][field] = value 
            ddict={}
            ddict['event'] = 'use'
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL('PeakTableWidgetSignal'),(ddict,))
            else:
                self.emit(qt.SIGNAL('PeakTableWidgetSignal'), (ddict))

        if field == "use":
            if self.peaks[peak][field+"_item"].isChecked():
                self.peaks[peak][field] = 1
            else:
                self.peaks[peak][field] = 0
            ddict={}
            ddict['event'] = 'use'
            if qt.qVersion() < '4.0.0':
                self.emit(qt.PYSIGNAL('PeakTableWidgetSignal'),(ddict,))
            else:
                self.emit(qt.SIGNAL('PeakTableWidgetSignal'), (ddict))

    def setReadOnly(self,parameter,fields):
        if DEBUG:
            print("peak ",parameter,"fields = ",fields,"asked to be read only")
        if qt.qVersion() < '4.0.0':
            self.setfield(parameter,fields,qttable.QTableItem.Never)
        else:
            self.setfield(parameter, fields,
                          qt.Qt.ItemIsSelectable|qt.Qt.ItemIsEnabled)

        
    def setReadWrite(self,parameter,fields):
        if DEBUG:
            print("peak ",parameter,"fields = ",fields,"asked to be read write")
        if qt.qVersion() < '4.0.0':
            self.setfield(parameter,fields,qttable.QTableItem.OnTyping)
        else:
            self.setfield(parameter, fields,
                          qt.Qt.ItemIsEditable|qt.Qt.ItemIsSelectable|qt.Qt.ItemIsEnabled)
                            
    def setfield(self,peak,fields,EditType):
        if DEBUG:
            print("setfield. peak =",peak,"fields = ",fields)
        if type(peak) == type (()) or \
           type(peak) == type ([]):
            peaklist=peak
        else:
            peaklist=[peak]
        if type(fields) == type (()) or \
           type(fields) == type ([]):
            fieldlist=fields
        else:
            fieldlist=[fields]
        for peak in peaklist:
            if peak in self.peaklist:            
                try:
                    row=self.peaklist.index(peak)
                except ValueError:
                    row=-1
                if row >= 0:
                    for field in fieldlist:
                        if field in self.peaks[peak]['fields']:
                            col=self.peaks[peak]['fields'].index(field)
                            if (field != 'element') and (field != 'elementline'):
                                key=field+"_item"
                                if qt.qVersion() < '4.0.0':
                                    self.peaks[peak][key]=qttable.QTableItem(self,
                                                            EditType,
                                                        self.peaks[peak][field])
                                    self.setItem(row,col,self.peaks[peak][key])
                                else:
                                    item = self.item(row, col)
                                    text = "%s" % self.peaks[peak][field]
                                    if item is None:
                                        item = qt.QTableWidgetItem(text,
                                                   qt.QTableWidgetItem.Type)
                                        self.setItem(row, col, item)
                                    else:                                        
                                        item.setText(str(text))
                                    item.setFlags(EditType)
                                    

    def configure(self,*vars,**kw):
        if DEBUG:
            print("configure called with **kw = ",kw)
            print("configure called with *vars = ",vars)
        name=None
        error=0
        if 'name' in kw:
            name=kw['name']        
        elif 'number' in kw:
            name=kw['number']
        else:
            return 1
        
        keylist = []
        if "channel" in kw:
            keylist=["channel"]
        for key in kw.keys():
            if key != "setenergy":
                if key not in keylist:
                    keylist.append(key)
        if "setenergy"  in kw.keys():
            keylist.append("setenergy")    

        if name in self.peaks:
            row=self.peaks[name]['line']
            for key in keylist:
              if key is not 'name':
                if key in self.peaks[name]['fields']:
                    col=self.peaks[name]['fields'].index(key)
                    oldvalue=self.peaks[name][key]
                    if key is 'code':
                        newvalue = qt.QString(str(kw[key]))
                    elif key is 'element':
                        newvalue = str(kw[key]).split()[0]
                        if newvalue == "-":
                            if qt.qVersion() < '4.0.0':
                                self.peaks[name][key+"_item"].setCurrentItem(0)
                            else:
                                self.peaks[name][key+"_item"].setCurrentIndex(0)
                        else:
                            self.peaks[name][key+"_item"].setSelection(newvalue)
                        try:
                            self.myslot(row,col)
                        except:
                            print("Error setting element") 
                    elif key is 'elementline':
                        try:
                            if qt.qVersion() < '4.0.0':
                                self.peaks[name][key+"_item"].setCurrentItem(kw[key])
                            else:
                                iv = self.peaks[name][key+"_item"].findText(qt.QString(kw[key]))
                                self.peaks[name][key+"_item"].setCurrentIndex(iv)
                        except:
                            print("Error setting elementline")
                    elif key is 'use':
                        if kw[key]:
                            self.peaks[name][key] = 1
                        else:
                            self.peaks[name][key] = 0
                        self.peaks[name][key+"_item"].setChecked(self.peaks[name][key])  
                    elif key == 'number':
                        if len(str(kw[key])):
                            newvalue=float(str(kw[key]))
                            newvalue= qt.QString("%3d" % newvalue)
                            self.peaks[name][key]=newvalue
                        else:
                            self.peaks[name][key]=oldvalue
                        if qt.qVersion() < '4.0.0':
                            self.setText(row,col,self.peaks[name][key])
                        else:
                            text = self.peaks[name][key]
                            item = self.item(row, col)
                            if item is None:
                                item = qt.QTableWidgetItem(text,
                                                qt.QTableWidgetItem.Type)
                                self.setItem(row, col, item)
                            else:
                                item.setText(text)
                    elif key == 'channel':
                        if DEBUG:
                            print("setting channel in configure")
                        if len(str(kw[key])):
                            newvalue=float(str(kw[key]))
                            newvalue= qt.QString("%.3f" % newvalue)
                            self.peaks[name][key]=newvalue
                        else:
                            self.peaks[name][key]=oldvalue
                        if qt.qVersion() < '4.0.0':
                            self.setText(row,col,self.peaks[name][key])
                        else:
                            text = self.peaks[name][key]
                            item = self.item(row, col)
                            if item is None:
                                item = qt.QTableWidgetItem(text,
                                                qt.QTableWidgetItem.Type)
                                self.setItem(row, col, item)
                            else:
                                item.setText(text)
                    elif (key == 'setenergy') or (key == 'calenergy'):
                        if len(str(kw[key])):
                            newvalue=float(str(kw[key]))
                            newvalue= qt.QString("%.4f" % newvalue)
                            self.peaks[name][key]=newvalue
                        else:
                            self.peaks[name][key]=oldvalue
                        if qt.qVersion() < '4.0.0':
                            self.setText(row,col,self.peaks[name][key])
                        else:
                            text = self.peaks[name][key]
                            item = self.item(row, col)
                            if item is None:
                                item = qt.QTableWidgetItem(text,
                                                qt.QTableWidgetItem.Type)
                                self.setItem(row, col, item)
                            else:
                                item.setText(text)
                        #self.myslot(row,col)
                    else:
                        if len(str(kw[key])):
                            newvalue=float(str(kw[key]))
                            if key is 'sigma':
                                newvalue= "%6.3g" % newvalue
                            else:
                                newvalue= "%8g" % newvalue
                        else:
                            newvalue=""
                        newvalue=qt.QString(newvalue)
        return error


    def validate(self,name,key,oldvalue,newvalue):
        if (key == 'setenergy') or (key == 'number') or (key == 'calcenergy'):
            try:
                float(str(newvalue))
            except:
                return 0
        return 1

    def getdict(self,*var):
        dict={}
        if len(var) == 0:
            #asked for the dict of dicts
            for peak in self.peaks.keys():
                dict[peak] = {}
                dict[peak]['number']      = float(str(self.peaks[peak]['number']))
                dict[peak]['channel']     = float(str(self.peaks[peak]['channel']))
                dict[peak]['element']     = str(self.peaks[peak]['element'])
                dict[peak]['elementline'] = str(self.peaks[peak]['elementline'])
                dict[peak]['setenergy']   = float(str(self.peaks[peak]['setenergy']))
                dict[peak]['use']         = self.peaks[peak]['use']
                if len(str(self.peaks[peak]['calenergy'])):
                    dict[peak]['calenergy']   = float(str(self.peaks[peak]['calenergy']))
                else:
                    dict[peak]['calenergy']   = ""
        else:
            peak=var[0]              
            if peak in self.peaks.keys():        
                dict['number']      = float(str(self.peaks[peak]['number']))
                dict['channel']     = float(str(self.peaks[peak]['channel']))
                dict['element']     = str(self.peaks[peak]['element'])
                dict['elementline'] = str(self.peaks[peak]['elementline'])
                dict['setenergy']   = float(str(self.peaks[peak]['setenergy']))
                dict['use']         = self.peaks[peak]['use']
                if len(str(self.peaks[peak]['calenergy'])):
                    dict['calenergy']   = float(str(self.peaks[peak]['calenergy']))
                else:
                    dict['calenergy']   = ""
        return dict
        
class QPeriodicComboTableItem(QComboTableItem):
    """ Periodic Table Combo List to be used in a QTable
        Init options:
            table (mandatory)= parent QTable
            addnone= 1 (default) add "-" in the list to provide possibility
                        to select no specific element.
                 0 only element list.
            detailed= 1 (default) display element symbol, Z and name
                  0 display only element symbol and Z
        Public methods:
            setSelection(eltsymbol):
                Set the element selected given its symbol
            getSelection():
                Return symbol of element selected

        Signals:
            No specific signals in Qt3. Use signals from QTable
            SIGNAL("valueChanged(int,int)") for example.
    """
    def __init__(self, table=None, addnone=1, detailed=0, row=None, col=None):
        try:
            strlist = qt.QStringList()
        except AttributeError:
            strlist = []
        self.addnone= (addnone==1)
        if self.addnone: strlist.append("-")
        for (symbol, Z, x, y, name, mass, density) in Elements.ElementsInfo:
            if detailed:    txt= "%2s (%d) - %s"%(symbol, Z, name)
            else:       txt= "%2s (%d)"%(symbol, Z)
            strlist.append(txt)
        if row is None: row = 0
        if col is None: col = 0
        self._row = row
        self._col = col
        if qt.qVersion() < '4.0.0':
            qttable.QComboTableItem.__init__(self, table, strlist)
        else:
            qt.QComboBox.__init__(self)
            self.addItems(strlist)
            self.connect(self, qt.SIGNAL('activated(int)'), self._cellChanged)

    if qt.qVersion() > "4.0.0":
        def _cellChanged(self, idx):
            self.emit(qt.SIGNAL('cellChanged(int, int)'), self._row, self._col)

    def setSelection(self, symbol=None):
        if symbol is None:
            if self.addnone:
                if qt.qVersion() < '4.0.0':
                    self.setCurrentItem(0)
                else:
                    self.setCurrentIndex(0)
        else:
            idx= self.addnone+Elements.getz(symbol)-1
            if qt.qVersion() < '4.0.0':
                self.setCurrentItem(idx)
            else:
                self.setCurrentIndex(idx)

    def getSelection(self):
        if qt.qVersion() < '4.0.0':
            id = self.currentItem()
        else:
            id = self.currentIndex()
        if self.addnone and not id: return None
        else: return Elements.ElementList[id-self.addnone]

def main(args):
    app=qt.QApplication(args)
    win=qt.QMainWindow()
    #tab = Parameters(labels=['Parameter','Estimation','Fit Value','Sigma',
    #                    'Restrains','Min/Parame','Max/Factor/Delta/'],
    #               paramlist=['Height','Position','FWHM'])
    tab = PeakTableWidget(labels= ['Peak','Channel','Element','Line','Set Energy','Use',
                            'Cal. Energy'],
                            peaklist=['1'])
    tab.showGrid()
    tab.configure(name='1',number=24,channel='1234',use=1,
                  setenergy=12.5,calenergy=24.0)    
    tab.show()
    if qt.qVersion() < '4.0.0':
        app.setMainWidget( tab )
        app.exec_loop()
    else:
        app.exec_()
        

                            

if __name__=="__main__":
    main(sys.argv)
