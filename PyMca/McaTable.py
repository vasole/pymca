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
import PyMcaQt as qt
    
QTVERSION = qt.qVersion()
if QTVERSION < '3.0.0':
    import Myqttable as qttable
elif QTVERSION < '4.0.0':
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

    class MyQTableItem(qttable.QTableItem):
         def __init__(self, table, edittype, text,color=qt.Qt.white):
                 qttable.QTableItem.__init__(self, table, edittype, text)
                 self.color = color
         def paint(self, painter, colorgroup, rect, selected):
                 cg = qt.QColorGroup(colorgroup)
                 cg.setColor(qt.QColorGroup.Base, self.color)
                 qttable.QTableItem.paint(self,painter, cg, rect, selected)
else:
    QTable = qt.QTableWidget
    

import string

DEBUG=0

class McaTable(QTable):
    def __init__(self, *args,**kw):
        apply(QTable.__init__, (self, ) + args)
        if QTVERSION < '4.0.0':
            self.setNumRows(1)
            self.setNumCols(1)
        else:
            self.setRowCount(1)
            self.setColumnCount(1)            
        self.labels=['Parameter','Estimation','Fit Value','Sigma',
                     'Restrains','Min/Parame','Max/Factor/Delta/']
        self.code_options=["FREE","POSITIVE","QUOTED",
                 "FIXED","FACTOR","DELTA","SUM","IGNORE","ADD","SHOW"]

        i=0
        if kw.has_key('labels'):
            self.labels=[]
            for label in kw['labels']:
                self.labels.append(label)
        else:
            self.labels=['Position','Fit Area','MCA Area','Sigma','Fwhm','Chisq',
                         'Region','XBegin','XEnd']
        if QTVERSION < '4.0.0':
            self.setNumCols(len(self.labels))        
            for label in self.labels:
                qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                self.adjustColumn(i)
                i=i+1
        else:
            self.setColumnCount(len(self.labels))
            for label in self.labels:
                item = self.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(self.labels[i],
                                               qt.QTableWidgetItem.Type)
                    self.setHorizontalHeaderItem(i,item)
                item.setText(self.labels[i])
                self.resizeColumnToContents(i)
                i=i+1
                
        self.regionlist=[]
        self.regiondict={}
        if QTVERSION < '4.0.0':
            self.verticalHeader().setClickEnabled(1)
            self.connect(self.verticalHeader(),qt.SIGNAL('clicked(int)'),self.__myslot)
        else:
            if DEBUG:
                print("MCATABLE click on vertical header items?")
            self.connect(self,
                         qt.SIGNAL('cellClicked(int, int)'),
                         self.__myslot)
        self.connect(self,qt.SIGNAL("selectionChanged()"),self.__myslot)

                
    def fillfrommca(self,mcaresult,diag=1):
        line0=0
        region=0
        alreadyforced = 0
        for result in mcaresult:
            region=region+1
            if result['chisq'] is not None:
                chisq=qt.QString("%6.2f" % (result['chisq']))
            else:
                chisq=qt.QString("Fit Error")
            if 1:
                xbegin=qt.QString("%6g" % (result['xbegin']))
                xend=qt.QString("%6g" % (result['xend']))
                fitlabel,fitpars, fitsigmas = self.__getfitpar(result)
                if QTVERSION < '4.0.0':
                    qt.QHeader.setLabel(self.horizontalHeader(),1,"Fit "+fitlabel)
                else:
                    item = self.horizontalHeaderItem(1)
                    item.setText("Fit "+fitlabel)
                i = 0
                for (pos,area,sigma,fwhm) in result['mca_areas']:
                    line0=line0+1
                    if QTVERSION < '4.0.0':
                        nlines=self.numRows()
                        if (line0 > nlines):
                            self.setNumRows(line0)
                    else:
                        nlines=self.rowCount()
                        if (line0 > nlines):
                            self.setRowCount(line0)
                    line=line0-1
                    #pos=QString(str(pos))
                    #area=QString(str(area))
                    #sigma=QString(str(sigma))
                    #fwhm=QString(str(fwhm))
                    tregion=qt.QString(str(region))
                    pos=qt.QString("%6g" % (pos))
                    fitpar = qt.QString("%6g" % (fitpars[i]))
                    if fitlabel == 'Area':
                        sigma = max(sigma,fitsigmas[i])
                    areastr=qt.QString("%6g" % (area))
                    sigmastr=qt.QString("%6.3g" % (sigma))
                    fwhm=qt.QString("%6g" % (fwhm))
                    tregion=qt.QString("%6g" % (region))
                    fields=[pos,fitpar,areastr,sigmastr,fwhm,chisq,tregion,xbegin,xend]
                    col=0
                    recolor = 0
                    if fitlabel == 'Area':
                        if diag:
                            if abs(fitpars[i]-area) > (3.0 * sigma):
                                color = qt.QColor(255,182,193)
                                recolor = 1
                    for field in fields:
                        if QTVERSION < '4.0.0':
                            if recolor:
                                key=MyQTableItem(self,qttable.QTableItem.Never,field,color=color)
                            else:
                                key=qttable.QTableItem(self,qttable.QTableItem.Never,field)
                            self.setItem(line,col,key)
                        else:
                            key = self.item(line, col)
                            if key is None:
                                key = qt.QTableWidgetItem(field)
                                self.setItem(line, col, key)
                            else:
                                item.setText(field)
                            if recolor:
                                #function introduced in Qt 4.2.0
                                if QTVERSION >= '4.2.0':
                                    item.setBackground(qt.QBrush(color))
                            item.setFlags(qt.Qt.ItemIsSelectable|qt.Qt.ItemIsEnabled)                            
                        col=col+1
                    if recolor:
                        if not alreadyforced:
                            alreadyforced = 1
                            if QTVERSION < '4.0.0':
                                self.ensureCellVisible(line,0)
                            else:
                                self.scrollToItem(self.item(line, 0))
                    i += 1 

        i = 0
        for label in self.labels:
            if QTVERSION < '4.0.0':
                self.adjustColumn(i)
            else:
                self.resizeColumnToContents(i)
            i=i+1
        ndict = {}
        ndict['event'] = 'McaTableFilled'
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaTableSignal'),(ndict,))
        else:
            self.emit(qt.SIGNAL('McaTableSignal'), ndict)


    def __getfitpar(self,result):
        hypermet = 0
        if  string.find(result['fitconfig']['fittheory'],"Area") != -1:
            fitlabel='Area'
        elif string.find(result['fitconfig']['fittheory'],"Hypermet") != -1:
            fitlabel='Area'
            hypermet = 1
        else:
            fitlabel='Height'
        values = []
        sigmavalues = []
        i = 0
        for param in result['paramlist']:
            if string.find(param['name'],'ST_Area')!= -1:
                values[-1]      = value * (1.0 + param['fitresult'])
                #just an approximation
                sigmavalues[-1] = sigmavalue * (1.0 + param['fitresult'])
            elif string.find(param['name'],'LT_Area')!= -1:
                pass
            elif string.find(param['name'],fitlabel)!= -1:
                value      = param['fitresult']
                sigmavalue = param['sigma'] 
                values.append(value)
                sigmavalues.append(sigmavalue)
        return fitlabel, values, sigmavalues


    def __myslot(self,*var):
        ddict={}
        if len(var) == 0:
            #selection changed event
            #get the current selection
            ddict['event']       = 'McaTableClicked'
            row = self.currentRow()
        else:
            #Header click
            ddict['event']       = 'McaTableRowHeaderClicked'
            row = var[0]
        ccol = self.currentColumn()
        ddict['row'  ]       = row
        ddict['col']         = ccol
        ddict['labelslist']  = self.labels
        if row >= 0:
            col = 0
            for label in self.labels:
                if QTVERSION < '4.0.0':
                    text = str(self.text(row,col))
                else:
                    text = str(self.item(row, col).text())
                try:
                    ddict[label] = string.atof(text)
                except:
                    ddict[label] = text
                col +=1
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaTableSignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('McaTableSignal'), ddict)
