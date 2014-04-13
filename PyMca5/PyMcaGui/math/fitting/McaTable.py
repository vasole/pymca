#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
else:
    QString = str
    
QTVERSION = qt.qVersion()

QTable = qt.QTableWidget
    
DEBUG=0

class McaTable(QTable):
    def __init__(self, *args,**kw):
        QTable.__init__(self, *args)
        self.setRowCount(1)
        self.setColumnCount(1)            
        self.labels=['Parameter','Estimation','Fit Value','Sigma',
                     'Restrains','Min/Parame','Max/Factor/Delta/']
        self.code_options=["FREE","POSITIVE","QUOTED",
                 "FIXED","FACTOR","DELTA","SUM","IGNORE","ADD","SHOW"]

        i=0
        if 'labels' in kw:
            self.labels=[]
            for label in kw['labels']:
                self.labels.append(label)
        else:
            self.labels=['Position','Fit Area','MCA Area','Sigma','Fwhm','Chisq',
                         'Region','XBegin','XEnd']

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
                chisq=QString("%6.2f" % (result['chisq']))
            else:
                chisq=QString("Fit Error")
            if 1:
                xbegin=QString("%6g" % (result['xbegin']))
                xend=QString("%6g" % (result['xend']))
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
                    tregion=QString(str(region))
                    pos=QString("%6g" % (pos))
                    fitpar = QString("%6g" % (fitpars[i]))
                    if fitlabel == 'Area':
                        sigma = max(sigma,fitsigmas[i])
                    areastr=QString("%6g" % (area))
                    sigmastr=QString("%6.3g" % (sigma))
                    fwhm=QString("%6g" % (fwhm))
                    tregion=QString("%6g" % (region))
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
        if  result['fitconfig']['fittheory'].find("Area") != -1:
            fitlabel='Area'
        elif result['fitconfig']['fittheory'].find("Hypermet") != -1:
            fitlabel='Area'
        else:
            fitlabel='Height'
        values = []
        sigmavalues = []
        for param in result['paramlist']:
            if param['name'].find('ST_Area')!= -1:
                # value and sigmavalue known via fitlabel
                values[-1]      = value * (1.0 + param['fitresult'])
                #just an approximation
                sigmavalues[-1] = sigmavalue * (1.0 + param['fitresult'])
            elif param['name'].find('LT_Area')!= -1:
                pass
            elif param['name'].find(fitlabel)!= -1:
                value      = param['fitresult']
                sigmavalue = param['sigma'] 
                values.append(value)
                sigmavalues.append(sigmavalue)
        return fitlabel, values, sigmavalues


    def __myslot(self, *var):
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
                    ddict[label] = float(text)
                except:
                    ddict[label] = text
                col +=1
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaTableSignal'),(ddict,))
        else:
            self.emit(qt.SIGNAL('McaTableSignal'), ddict)
