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
from PyMca import PyMcaQt as qt
if not hasattr(qt, "QString"):
    QString = str
else:
    QString = qt.QString

if not hasattr(qt, "QStringList"):
    QStringList = list
else:
    QStringList = qt.QStringList
    
QTVERSION = qt.qVersion()
if QTVERSION < '4.0.0':
    import qttable

if QTVERSION < '4.0.0':
    class QTable(qttable.QTable):
        def __init__(self, parent=None, name=""):
            qttable.QTable.__init__(self, parent, name)
            self.rowCount    = self.numRows
            self.columnCount = self.numCols
            self.setRowCount = self.setNumRows
            self.setColumnCount = self.setNumCols
            self.resizeColumnToContents = self.adjustColumn

    #Class added for compatibility with previous Qt versions
    class QComboTableItem(qttable.QTableItem):
        def __init__(self, table,list):
            qttable.QTableItem.__init__(self,table,qttable.QTableItem.Always,"")
            self.setReplaceable(0)
            self.table=table
            self.list=list
            
        def setCurrentItem(self,cur):
            for i in range (len(self.list)):
                if str(cur)==str(self.list[i]):
                    self.cb.setCurrentItem(i)
                    return

     
        def createEditor(self):
            self.cb=qt.QComboBox(self.table.viewport())
            self.cb.insertStringList(self.list)
            self.cb.connect(self.cb,SIGNAL("activated(int)"),self.mySlot)
            return self.cb

        def mySlot (self,index):
            self.table.setText(self.row(),self.col(),self.cb.currentText())
            self.table.myslot(self.row(),self.col())
        
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
            self.emit(qt.SIGNAL('cellChanged'), self._row, self._col)

    class QCheckBoxItem(qt.QCheckBox):
        def __init__(self, parent=None, row = None, col = None):
            self._row = row
            self._col = col
            qt.QCheckBox.__init__(self,parent)
            self.connect(self, qt.SIGNAL('clicked()'), self._cellChanged)

        def _cellChanged(self):
            self.emit(qt.SIGNAL('cellChanged'), self._row, self._col)

DEBUG=0

class Parameters(QTable):
    def __init__(self, *args,**kw):
        QTable.__init__(self, *args)
        if QTVERSION < '4.0.0':
            self.setNumRows(1)
            self.setNumCols(1)
        else:
            self.setRowCount(1)
            self.setColumnCount(1)
        self.labels=['Parameter','Estimation','Fit Value','Sigma',
                     'Constraints','Min/Parame','Max/Factor/Delta/']
        if DEBUG:
            self.code_options=["FREE","POSITIVE","QUOTED",
                 "FIXED","FACTOR","DELTA","SUM","IGNORE","ADD","SHOW"]
        else:
            self.code_options=["FREE","POSITIVE","QUOTED",
                 "FIXED","FACTOR","DELTA","SUM","IGNORE","ADD"]
        self.__configuring = False
        self.setColumnCount(len(self.labels))
        i=0
        if 'labels' in kw:
            if QTVERSION < '4.0.0':
                for label in kw['labels']:
                    qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                    i=i+1
            else:
                for label in kw['labels']:
                    item = self.horizontalHeaderItem(i)
                    if item is None:
                        item = qt.QTableWidgetItem(self.labels[i],
                                                   qt.QTableWidgetItem.Type)
                        self.setHorizontalHeaderItem(i,item)
                    item.setText(self.labels[i])
                    i = i+1
                
        else:
            if QTVERSION < '4.0.0':
                for label in self.labels:
                    qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                    i=i+1
            else:
                for label in self.labels:
                    item = self.horizontalHeaderItem(i)
                    if item is None:
                        item = qt.QTableWidgetItem(self.labels[i],
                                                   qt.QTableWidgetItem.Type)
                        self.setHorizontalHeaderItem(i,item)
                    item.setText(self.labels[i])
                    i = i+1
        if QTVERSION < '4.0.0':
            self.adjustColumn(self.labels.index('Parameter'))
            self.adjustColumn(1)
            self.adjustColumn(3)
            self.adjustColumn(len(self.labels)-1)
            self.adjustColumn(len(self.labels)-2)
        else:
            self.resizeColumnToContents(self.labels.index('Parameter'))
            self.resizeColumnToContents(1)
            self.resizeColumnToContents(3)
            self.resizeColumnToContents(len(self.labels)-1)
            self.resizeColumnToContents(len(self.labels)-2)
            
        self.parameters={}
        self.paramlist=[]
        if 'paramlist' in kw:
            self.paramlist = kw['paramlist']
        self.build()
        if QTVERSION < '4.0.0':
            self.connect(self,qt.SIGNAL("valueChanged(int,int)"),self.myslot)
        else:
            self.connect(self,qt.SIGNAL("cellChanged(int,int)"),self.myslot)

    def build(self):
        line = 1
        oldlist=list(self.paramlist)
        self.paramlist=[]
        for param in oldlist:
            self.newparameterline(param,line)
            line=line+1

    def newparameterline(self,param,line):
        #get current number of lines
        if QTVERSION < '4.0.0':
            nlines=self.numRows()
        else:
            nlines=self.rowCount()
            self.__configuring = True
        if (line > nlines):
            if QTVERSION < '4.0.0':
                self.setNumRows(line)
            else:
                self.setRowCount(line)
        linew=line-1
        self.parameters[param]={'line':linew,
                                'fields':['name',
                                          'estimation',
                                          'fitresult',
                                          'sigma',
                                          'code',
                                          'val1',
                                          'val2'],
                                'estimation':QString('0'),          
                                'fitresult':QString(),
                                'sigma':QString(),
                                'code':QString('FREE'),
                                'val1':QString(),
                                'val2':QString(),
                                'cons1':0,
                                'cons2':0,
                                'vmin':QString('0'),
                                'vmax':QString('1'),
                                'relatedto':QString(),
                                'factor':QString('1.0'),
                                'delta':QString('0.0'),
                                'sum':QString('0.0'),
                                'group':QString(),
                                'name':QString(param),
                                'xmin':None,
                                'xmax':None}
        self.paramlist.append(param)
        #Aleixandre request:self.setText(linew,0,QString(param))
        self.setReadWrite(param,'estimation')
        self.setReadOnly(param,['name','fitresult','sigma','val1','val2'])

        #the code
        a=QStringList()
        for option in self.code_options:
            a.append(option)
        if QTVERSION < '4.0.0':
            self.parameters[param]['code_item']=qttable.QComboTableItem(self,a)
            self.setItem(linew,
                         self.parameters[param]['fields'].index('code'),
                         self.parameters[param]['code_item'])
        else:
            cellWidget = self.cellWidget(linew,
                         self.parameters[param]['fields'].index('code'))
            if cellWidget is None:
                col = self.parameters[param]['fields'].index('code')
                cellWidget = QComboTableItem(self,
                                             row = linew,
                                             col = col)
                cellWidget.addItems(a)
                self.setCellWidget(linew, col, cellWidget)
                self.connect(cellWidget, qt.SIGNAL("cellChanged"),
                             self.myslot)
            self.parameters[param]['code_item']= cellWidget
        self.parameters[param]['relatedto_item']=None
        self.__configuring = False

    def fillTableFromFit(self, fitparameterslist):
        return self.fillfromfit(fitparameterslist)

    def fillfromfit(self,fitparameterslist):
        if QTVERSION < '4.0.0':
            self.setNumRows(len(fitparameterslist))
        else:
            self.setRowCount(len(fitparameterslist))
        self.parameters={}
        self.paramlist=[]
        line=1
        for param in fitparameterslist:
            self.newparameterline(param['name'],line)
            line=line+1
        for param in fitparameterslist:
            name=param['name']
            code=str(param['code'])
            if code not in self.code_options:
                code=self.code_options[int(code)]
            val1=param['cons1']
            val2=param['cons2']
            estimation=param['estimation']
            group=param['group']
            sigma=param['sigma']
            fitresult=param['fitresult']
            if 'xmin' in param:
                xmin=param['xmin']
            else:
                xmin=None
            if 'xmax' in param:
                xmax=param['xmax']
            else:
                xmax=None
            self.configure(name=name,
                           code=code,
                           val1=val1,val2=val2,
                           estimation=estimation,
                           fitresult=fitresult,
                           sigma=sigma,
                           group=group,
                           xmin=xmin,
                           xmax=xmax)

    def fillFitFromTable(self):
        return self.fillfitfromtable()

    def getConfiguration(self):
        ddict = {}
        ddict['parameters'] = self.fillFitFromTable()
        return ddict

    def setConfiguration(self, ddict):
        self.fillTableFromFit(ddict['parameters'])

    def fillfitfromtable(self):
        fitparameterslist=[]
        for param in self.paramlist:
            fitparam={}
            #name=self.parameters['name']
            name=param
            estimation,[code,cons1,cons2]=self.cget(name)
            buf=str(self.parameters[param]['fitresult'])
            xmin=self.parameters[param]['xmin']
            xmax=self.parameters[param]['xmax']            
            if len(buf):
                fitresult=float(buf)
            else:
                fitresult=0.0
            buf=str(self.parameters[param]['sigma'])
            if len(buf):
                sigma=float(buf)
            else:
                sigma=0.0
            buf=str(self.parameters[param]['group'])
            if len(buf):
                group=float(buf)
            else:
                group=0
            fitparam['name']=name
            fitparam['estimation']=estimation
            fitparam['fitresult']=fitresult
            fitparam['sigma']=sigma
            fitparam['group']=group
            fitparam['code']=code
            fitparam['cons1']=cons1
            fitparam['cons2']=cons2
            fitparam['xmin']=xmin
            fitparam['xmax']=xmax
            fitparameterslist.append(fitparam)
        return fitparameterslist
        
    def myslot(self,row,col):
        if DEBUG:
            print("Passing by myslot(%d, %d)" % (row, col))
            print("current(%d, %d)" % (self.currentRow(), self.currentColumn()))
        if QTVERSION > '4.0.0':
            if (col != 4) and (col != -1):
                if row != self.currentRow():return
                if col != self.currentColumn():return
            if self.__configuring:return
        param=self.paramlist[row]
        field=self.parameters[param]['fields'][col]
        oldvalue=QString(self.parameters[param][field])
        if QTVERSION < '4.0.0':
            newvalue=QString(self.text(row,col))
        else:
            if col != 4:
                item = self.item(row, col)
                if item is not None:
                    newvalue = item.text()
                else:
                    newvalue = QString()
            else:
                #this is the combobox
                widget   = self.cellWidget(row, col)
                newvalue = widget.currentText()  
        if DEBUG:
            print("old value = ",oldvalue)
            print("new value = ",newvalue)
        if self.validate(param,field,oldvalue,newvalue):
            #self.parameters[param][field]=newvalue
            if DEBUG:
                print("Change is valid")
            exec("self.configure(name=param,%s=newvalue)" % field)
        else:
            if DEBUG:
                print("Change is not valid")
                print("oldvalue ", oldvalue)
            if field == 'code':
                if QTVERSION < '4.0.0':
                    self.parameters[param]['code_item'].setCurrentItem(oldvalue)
                else:
                    index = self.code_options.index(oldvalue)
                    self.__configuring = True
                    try:
                        self.parameters[param]['code_item'].setCurrentIndex(index)
                    finally:
                        self.__configuring = False
            else:
                exec("self.configure(name=param,%s=oldvalue)" % field)

    def validate(self,param,field,oldvalue,newvalue):
        if field == 'code':
            pass
            return self.setcodevalue(param,field,oldvalue,newvalue)
        if ((str(self.parameters[param]['code']) == 'DELTA') or\
            (str(self.parameters[param]['code']) == 'FACTOR') or\
            (str(self.parameters[param]['code']) == 'SUM')) and \
            (field == 'val1'):
                best,candidates=self.getrelatedcandidates(param)
                if str(newvalue) in candidates:
                    return 1
                else:
                    return 0
        else:
            try:
                float(str(newvalue))
            except:
                return 0
        return 1

    def setcodevalue(self,workparam,field,oldvalue,newvalue):
        if   str(newvalue) == 'FREE':
            self.configure(name=workparam,
                           code=newvalue)
                           #,
                           #cons1=0,
                           #cons2=0,
                           #val1='',
                           #val2='')                          
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'POSITIVE':
            self.configure(name=workparam,
                           code=newvalue)
                           #,
                           #cons1=0,
                           #cons2=0,
                           #val1='',
                           #val2='')
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'QUOTED':
            #I have to get the limits
            self.configure(name=workparam,
                           code=newvalue)
                           #,
                           #cons1=self.parameters[workparam]['vmin'],
                           #cons2=self.parameters[workparam]['vmax'])
                           #,
                           #val1=self.parameters[workparam]['vmin'],
                           #val2=self.parameters[workparam]['vmax'])
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'FIXED':
            self.configure(name=workparam,
                           code=newvalue)
                           #,
                           #cons1=0,
                           #cons2=0,
                           #val1='',
                           #val2='')
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)                
            return 1
        elif str(newvalue) == 'FACTOR':
            #I should check here that some parameter is set
            best,candidates=self.getrelatedcandidates(workparam)
            if len(candidates) == 0:
                return 0
            self.configure(name=workparam,
                           code=newvalue,
                           relatedto=best)
                           #,
                           #cons1=0,
                           #cons2=0,
                           #val1='',
                           #val2='')
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)    
            return 1
        elif str(newvalue) == 'DELTA':
            #I should check here that some parameter is set
            best,candidates=self.getrelatedcandidates(workparam)
            if len(candidates) == 0:
                return 0
            self.configure(name=workparam,
                           code=newvalue,
                           relatedto=best)
                           #,
                           #cons1=0,
                           #cons2=0,
                           #val1='',
                           #val2='')
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)                
            return 1
        elif str(newvalue) == 'SUM':
            #I should check here that some parameter is set
            best,candidates=self.getrelatedcandidates(workparam)
            if len(candidates) == 0:
                return 0
            self.configure(name=workparam,
                           code=newvalue,
                           relatedto=best)
                           #,
                           #cons1=0,
                           #cons2=0,
                           #val1='',
                           #val2='')
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)                
            return 1
        elif str(newvalue) == 'IGNORE':
            # I should check if the group can be ignored
            # for the time being I just fix all of them to ignore
            group=int(float(str(self.parameters[workparam]['group'])))
            candidates = []
            for param in self.parameters.keys():
                if group == int(float(str(self.parameters[param] ['group']))):
                    candidates.append(param)
            #print candidates 
            #I should check here if there is any relation to them
            for param in candidates:
                self.configure(name=param,
                               code=newvalue)
                               #,
                               #cons1=0,
                               #cons2=0,
                               #val1='',
                               #val2='')
            return 1
        elif str(newvalue) == 'ADD':
            group=int(float(str(self.parameters[workparam]['group'])))
            if group == 0:
                #One cannot add a background group
                return 0
            i=0
            for param in self.paramlist:
                if i <=  int(float(str(self.parameters [param] ['group']))):
                    i = i + 1
            self.addgroup(i,group)
            return 0

        elif str(newvalue) == 'SHOW':
            print(self.cget(workparam))
            return 0

        else:
            print("None of the others!")

    def addgroup(self,newg,gtype):
        line = 0
        newparam=[]
        oldparamlist=list(self.paramlist)
        for param in oldparamlist:
            line = line + 1
            paramgroup = int(float(str(self.parameters [param] ['group'])))
            if paramgroup  == gtype:
                #Try to construct an appropriate name
                #I have to remove any possible trailing number
                #and append the group index
                xmin=self.parameters[param]['xmin']
                xmax=self.parameters[param]['xmax']
                j=len(param)-1
                while ('0' < param[j]) & (param[j] < '9'):
                    j=j-1
                    if j == -1:
                        break
                if j >= 0 :
                    newparam.append(param[0:j+1] + "%d" % newg)
                else:
                    newparam.append("%d" % newg)
        for param in newparam:
            line=line+1
            self.newparameterline(param,line)
        for param in newparam:
            self.configure(name=param,group=newg,xmin=xmin,xmax=xmax)

    def freerestofgroup(self,workparam):
        if workparam in self.parameters.keys():
            group=int(float(str(self.parameters[workparam]['group'])))
            for param in self.parameters.keys():
                if param != workparam:
                    if group == int(float(str(self.parameters[param]['group']))):
                        self.configure(name=param,
                                       code='FREE',
                                       cons1=0,
                                       cons2=0,
                                       val1='',
                                       val2='')
                                       
    def getrelatedcandidates(self,workparam):
        best=None
        candidates = []
        for param in self.paramlist:
            if param != workparam:
                if str(self.parameters[param]['code']) != 'IGNORE' and \
                   str(self.parameters[param]['code']) != 'FACTOR' and \
                   str(self.parameters[param]['code']) != 'DELTA'  and \
                   str(self.parameters[param]['code']) != 'SUM' :
                        candidates.append(param)
        #Now get the best from the list
        if candidates == None:
            return best,candidates
        #take the previous one if possible
        if str(self.parameters[workparam]['relatedto']) in candidates:
            best=str(self.parameters[workparam]['relatedto'])
            return best,candidates
        #take the first with similar name
        for param in candidates:
                j=len(param)-1
                while ('0' <= param[j]) & (param[j] < '9'):
                    j=j-1
                    if j == -1:
                        break
                if j >= 0 :
                    try:
                        pos=workparam.index(param[0:j+1])
                        if pos == 0:
                            best=param
                            return best,candidates
                    except:
                        pass
        #take the first
        return candidates[0],candidates             

    def setReadOnly(self,parameter,fields):
        if DEBUG:
            print("parameter ",parameter)
            print("fields = ",fields)
            print("asked to be read only")
        if QTVERSION < '4.0.0':
            self.setfield(parameter,fields,qttable.QTableItem.Never)
        else:
            editflags = qt.Qt.ItemIsSelectable|qt.Qt.ItemIsEnabled
            self.setfield(parameter, fields, editflags)
            
        
    def setReadWrite(self,parameter,fields):
        if DEBUG:
            print("parameter ",parameter)
            print("fields = ",fields)
            print("asked to be read write")
        if QTVERSION < '4.0.0':
            self.setfield(parameter,fields,qttable.QTableItem.OnTyping)
        else:
            editflags = qt.Qt.ItemIsSelectable|qt.Qt.ItemIsEnabled|qt.Qt.ItemIsEditable
            self.setfield(parameter, fields, editflags)
                            
    def setfield(self,parameter,fields,EditType):
        if DEBUG:
            print("setfield. parameter =",parameter)
            print("fields = ",fields)
        if type(parameter) == type (()) or \
           type(parameter) == type ([]):
            paramlist=parameter
        else:
            paramlist=[parameter]
        if type(fields) == type (()) or \
           type(fields) == type ([]):
            fieldlist=fields
        else:
            fieldlist=[fields]
        _oldvalue = self.__configuring
        self.__configuring = True
        for param in paramlist:
            if param in self.paramlist:            
                try:
                    row=self.paramlist.index(param)
                except ValueError:
                    row=-1
                if row >= 0:
                    for field in fieldlist:
                        if field in self.parameters[param]['fields']:
                            col=self.parameters[param]['fields'].index(field)
                        if field != 'code':
                            key=field+"_item"
                            if QTVERSION < '4.0.0':
                                self.parameters[param][key]=qttable.QTableItem(self,EditType,
                                        self.parameters[param][field])
                                self.setItem(row,col,self.parameters[param][key])
                            else:
                                item = self.item(row, col)
                                if item is None:
                                    item = qt.QTableWidgetItem()
                                    item.setText(self.parameters[param][field])
                                    self.setItem(row, col, item)
                                else:
                                    item.setText(self.parameters[param][field])
                                self.parameters[param][key] = item
                                item.setFlags(EditType)
        self.__configuring = _oldvalue
        
    def configure(self,*vars,**kw):
        if DEBUG:
            print("configure called with **kw = ",kw)
        name=None
        error=0
        if 'name' in kw:
            name=kw['name']
        else:
            return 1
        if name in self.parameters:
            row=self.parameters[name]['line']
            for key in kw.keys():
              if key is not 'name':
                if key in self.parameters[name]['fields']:
                    col=self.parameters[name]['fields'].index(key)
                    oldvalue=self.parameters[name][key]
                    if key is 'code':
                        newvalue=QString(str(kw[key]))
                    else:
                        if len(str(kw[key])):
                            keyDone = False
                            if key == "val1":
                                if str(self.parameters[name]['code']) in\
                                       ['DELTA', 'FACTOR', 'SUM']:
                                    newvalue = str(kw[key])
                                    keyDone = True
                            if not keyDone:
                                newvalue=float(str(kw[key]))
                                if key is 'sigma':
                                    newvalue= "%6.3g" % newvalue
                                else:
                                    newvalue= "%8g" % newvalue
                        else:
                            newvalue=""
                        newvalue=QString(newvalue)
                    #avoid endless recursivity
                    if key is not 'code':
                        if self.validate(name,key,oldvalue,newvalue):
                            #self.setText(row,col,newvalue)
                            self.parameters[name][key]=newvalue
                        else:
                            #self.setText(row,col,oldvalue)
                            self.parameters[name][key]=oldvalue
                            error=1
                    #else:
                    #    self.parameters[name][key]=newvalue
                    #    self.parameters[name]['code_item'].setCurrentItem(newvalue)
                elif key in self.parameters[name].keys():
                    newvalue=QString(str(kw[key]))
                    self.parameters[name][key]=newvalue
                    #if key == 'relatedto':
                    #    self.parameters[name]['relatedto_item'].setCurrentItem(newvalue)    
            if DEBUG:
                print("error = ",error)
            #if error == 0:
            if 1:
                if 'code' in kw.keys():
                    newvalue=QString(kw['code'])
                    self.parameters[name]['code']=newvalue
                    if QTVERSION < '4.0.0':
                        self.parameters[name]['code_item'].setCurrentItem(newvalue)
                    else:
                        done = 0
                        for i in range(self.parameters[name]['code_item'].count()):
                            if str(newvalue) == str(self.parameters[name]['code_item'].itemText(i)):
                                self.parameters[name]['code_item'].setCurrentIndex(i)
                                done = 1
                                break
                    if str(self.parameters[name]['code']) == 'QUOTED':
                      if 'val1'in kw.keys():
                        self.parameters[name]['vmin']=self.parameters[name]['val1']
                      if 'val2'in kw.keys():
                        self.parameters[name]['vmax']=self.parameters[name]['val2']
                        #print "vmin =",str(self.parameters[name]['vmin'])
                    if str(self.parameters[name]['code']) == 'DELTA':
                      if 'val1'in kw.keys():
                          if kw['val1'] in self.paramlist:
                            self.parameters[name]['relatedto']=kw['val1']
                          else:
                            self.parameters[name]['relatedto']=\
                                     self.paramlist[int(float(str(kw['val1'])))]
                      if 'val2'in kw.keys():
                        self.parameters[name]['delta']=self.parameters[name]['val2']
                    if str(self.parameters[name]['code']) == 'SUM':
                      if 'val1'in kw.keys():
                          if kw['val1'] in self.paramlist:
                            self.parameters[name]['relatedto']=kw['val1']
                          else:
                            self.parameters[name]['relatedto']=\
                                     self.paramlist[int(float(str(kw['val1'])))]
                      if 'val2'in kw.keys():
                        self.parameters[name]['sum']=self.parameters[name]['val2']
                    if str(self.parameters[name]['code']) == 'FACTOR':
                      if 'val1'in kw.keys():
                          if kw['val1'] in self.paramlist:
                            self.parameters[name]['relatedto']=kw['val1']
                          else:
                            self.parameters[name]['relatedto']=\
                                     self.paramlist[int(float(str(kw['val1'])))]
                      if 'val2'in kw.keys():
                        self.parameters[name]['factor']=self.parameters[name]['val2']
                else:
                    #Update the proper parameter in case of change in val1 and val2
                    if str(self.parameters[name]['code']) == 'QUOTED':
                        self.parameters[name]['vmin']=self.parameters[name]['val1']
                        self.parameters[name]['vmax']=self.parameters[name]['val2']
                        #print "vmin =",str(self.parameters[name]['vmin'])
                    if str(self.parameters[name]['code']) == 'DELTA':
                        self.parameters[name]['relatedto']=self.parameters[name]['val1']
                        self.parameters[name]['delta']=self.parameters[name]['val2']
                    if str(self.parameters[name]['code']) == 'SUM':
                        self.parameters[name]['relatedto']=self.parameters[name]['val1']
                        self.parameters[name]['sum']=self.parameters[name]['val2']
                    if str(self.parameters[name]['code']) == 'FACTOR':
                        self.parameters[name]['relatedto']=self.parameters[name]['val1']
                        self.parameters[name]['factor']=self.parameters[name]['val2']
         
            if(1):
                #Update val1 and val2 according to the parameters
                #and Update the table
                if str(self.parameters[name]['code']) == 'FREE' or \
                   str(self.parameters[name]['code']) == 'POSITIVE' or \
                   str(self.parameters[name]['code']) == 'IGNORE' or\
                   str(self.parameters[name]['code']) == 'FIXED' :
                    self.parameters[name]['val1']=QString()
                    self.parameters[name]['val2']=QString()                   
                    self.parameters[name]['cons1']=0                   
                    self.parameters[name]['cons2']=0
                    self.setReadWrite(name,'estimation')
                    self.setReadOnly(name,['fitresult','sigma','val1','val2'])
                elif str(self.parameters[name]['code']) == 'QUOTED':
                    self.parameters[name]['val1']=self.parameters[name]['vmin']
                    self.parameters[name]['val2']=self.parameters[name]['vmax']
                    try:
                        self.parameters[name]['cons1']=\
                            float(str(self.parameters[name]['val1']))
                    except:
                        self.parameters[name]['cons1']=0
                    try:
                        self.parameters[name]['cons2']=\
                            float(str(self.parameters[name]['val2']))
                    except:
                        self.parameters[name]['cons2']=0
                    if self.parameters[name]['cons1'] > self.parameters[name]['cons2']:
                        buf=self.parameters[name]['cons1']
                        self.parameters[name]['cons1']=self.parameters[name]['cons2']
                        self.parameters[name]['cons2']=buf
                    self.setReadWrite(name,['estimation','val1','val2'])
                    self.setReadOnly(name,['fitresult','sigma'])
                elif str(self.parameters[name]['code']) == 'FACTOR':
                    self.parameters[name]['val1']=self.parameters[name]['relatedto']
                    self.parameters[name]['val2']=self.parameters[name]['factor']
                    self.parameters[name]['cons1']=\
                        self.paramlist.index(str(self.parameters[name]['val1']))
                    try:
                        self.parameters[name]['cons2']=\
                            float(str(self.parameters[name]['val2']))
                    except:
                        error=1
                        print("Forcing factor to 1") 
                        self.parameters[name]['cons2']=1.0
                    self.setReadWrite(name,['estimation','val1','val2'])
                    self.setReadOnly(name,['fitresult','sigma'])
                elif str(self.parameters[name]['code']) == 'DELTA':
                    self.parameters[name]['val1']=self.parameters[name]['relatedto']
                    self.parameters[name]['val2']=self.parameters[name]['delta']
                    self.parameters[name]['cons1']=\
                        self.paramlist.index(str(self.parameters[name]['val1']))
                    try:
                        self.parameters[name]['cons2']=\
                            float(str(self.parameters[name]['val2']))
                    except:
                        error=1
                        print("Forcing delta to 0") 
                        self.parameters[name]['cons2']=0.0
                    self.setReadWrite(name,['estimation','val1','val2'])
                    self.setReadOnly(name,['fitresult','sigma'])
                elif str(self.parameters[name]['code']) == 'SUM':
                    self.parameters[name]['val1']=self.parameters[name]['relatedto']
                    self.parameters[name]['val2']=self.parameters[name]['sum']
                    self.parameters[name]['cons1']=\
                        self.paramlist.index(str(self.parameters[name]['val1']))
                    try:
                        self.parameters[name]['cons2']=\
                            float(str(self.parameters[name]['val2']))
                    except:
                        error=1
                        print("Forcing sum to 0") 
                        self.parameters[name]['cons2']=0.0
                    self.setReadWrite(name,['estimation','val1','val2'])
                    self.setReadOnly(name,['fitresult','sigma'])
                else:
                    self.setReadWrite(name,['estimation','val1','val2'])
                    self.setReadOnly(name,['fitresult','sigma'])                                 
        return error
        
    def cget(self,param):
        """
        Return tuple estimation,constraints where estimation is the
        value in the estimate field and constraints are the relevant
        constraints according to the active code
        """
        estimation=None
        costraints=None
        if param in self.parameters.keys():
            buf=str(self.parameters[param]['estimation'])
            if len(buf):
                estimation=float(buf)
            else:
                estimation=0
            self.parameters[param]['code_item']
            if str(self.parameters[param]['code']) in self.code_options:
                code = self.code_options.index(str(self.parameters[param]['code']))
            else:
                code = str(self.parameters[param]['code'])
            cons1=self.parameters[param]['cons1']
            cons2=self.parameters[param]['cons2']
            constraints = [code,cons1,cons2]
        return estimation,constraints 
        
                    
def main(args):
    app=qt.QApplication(args)
    win=qt.QMainWindow()
    tab=Parameters(labels=['Parameter','Estimation','Fit Value','Sigma',
                        'Restrains','Min/Parame','Max/Factor/Delta/'],
                   paramlist=['Height','Position','FWHM'])
    tab.showGrid()
    if (0):
        tab.setText(0,3,QString('Hello World!'))
        tab.setItem(0,1,qttable.QTableItem(tab,1,QString('a try')))
        check=qttable.QCheckTableItem(tab,QString("Check me!"))
        tab.setItem(0,2,check)
        a=QStringList()
        a.append("1")
        a.append("2")    
        checklist=qttable.QComboTableItem(tab,a)
        tab.setItem(0,0,checklist)
    tab.configure(name='Height',estimation='1234',group=0)    
    tab.configure(name='Position',code='FIXED',group=1)
    tab.configure(name='FWHM',group=1)
    #for item,value in tab.parameters['Position'].items():
    #    print item,value
    if 1:
        import specfile
        import Specfit
        from numpy.oldnumeric import sqrt,equal,array,Float,concatenate,arange,take,nonzero
        sf=specfile.Specfile('02021201.dat')
        scan=sf.select('14')
        #sf=specfile.Specfile('02022101.dat')
        #scan=sf.select('11')
        nbmca=scan.nbmca()
        mcadata=scan.mca(1)
        y=array(mcadata)
        #x=arange(len(y))
        x=arange(len(y))*0.0200511-0.003186
        fit=Specfit.Specfit()
        fit.setdata(x=x,y=y)
        fit.importfun("SpecfitFunctions.py")
        fit.settheory('Hypermet')
        fit.configure(Yscaling=1.,
                      WeightFlag=1,
                      PosFwhmFlag=1,
                      HeightAreaFlag=1,
                      FwhmPoints=50,
                      PositionFlag=1,
                      HypermetTails=1)        
        fit.setbackground('Linear')
        if 0:
            mcaresult=fit.mcafit(x=x,xmin=x[70],xmax=x[500])
        else:
            fit.estimate()
            fit.startfit()
        tab.fillfromfit(fit.paramlist)
    tab.show()
    #button = QPushButton('bla',None)
    #button.show()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()")
                                 , app
                                 , qt.SLOT("quit()")
                                 )
    if QTVERSION < '4.0.0':
        app.setMainWidget( tab )
        app.exec_loop()
    else:
        app.exec_()
if __name__=="__main__":
    main(sys.argv)	
