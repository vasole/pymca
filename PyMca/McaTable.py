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
import sys
import qt
import qttable
import string

DEBUG=0

class McaTable(qttable.QTable):
    def __init__(self, *args,**kw):
        apply(qttable.QTable.__init__, (self, ) + args)
        self.setNumRows(1)
        self.setNumCols(1)
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
        self.setNumCols(len(self.labels))
        
        for label in self.labels:
            qt.QHeader.setLabel(self.horizontalHeader(),i,label)
            #if (i != 1) and (i!=2):
            #if (i != 2):
            if 1:
                self.adjustColumn(i)
            i=i+1
                
        self.regionlist=[]
        self.regiondict={}
        self.verticalHeader().setClickEnabled(1)
        self.connect(self.verticalHeader(),qt.SIGNAL('clicked(int)'),self.__myslot)
        self.connect(self,qt.SIGNAL("selectionChanged()"),self.__myslot)
        #self.setSelectionMode(qttable.QTable.SingleRow)

                
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
                qt.QHeader.setLabel(self.horizontalHeader(),1,"Fit "+fitlabel)
                i = 0
                for (pos,area,sigma,fwhm) in result['mca_areas']:
                    line0=line0+1
                    nlines=self.numRows()
                    if (line0 > nlines):
                        self.setNumRows(line0)
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
                        if recolor:
                            key=MyQTableItem(self,qttable.QTableItem.Never,field,color=color)
                        else:
                            key=qttable.QTableItem(self,qttable.QTableItem.Never,field)
                        self.setItem(line,col,key)
                        col=col+1
                    if recolor:
                        if not alreadyforced:
                            alreadyforced = 1
                            self.ensureCellVisible(line,0)
                    i += 1 

        i = 0
        for label in self.labels:
            self.adjustColumn(i)
            i=i+1
        ndict = {}
        ndict['event'] = 'McaTableFilled'
        self.emit(qt.PYSIGNAL('McaTableSignal'),(ndict,))


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
        dict={}
        if len(var) == 0:
            #selection changed event
            #get the current selection
            dict['event']       = 'McaTableClicked'
            row = self.currentRow()
        else:
            #Header click
            dict['event']       = 'McaTableRowHeaderClicked'
            row = var[0]
        ccol = self.currentColumn()
        dict['row'  ]       = row
        dict['col']         = ccol
        dict['labelslist']  = self.labels
        if row >= 0:
            col = 0
            for label in self.labels:
                try:
                    dict[label] = string.atof(str(self.text(row,col)))
                except:
                    dict[label] = str(self.text(row,col))
                col +=1
        self.emit(qt.PYSIGNAL('McaTableSignal'),(dict,))

class MyQTableItem(qttable.QTableItem):
         def __init__(self, table, edittype, text,color=qt.Qt.white):
                 qttable.QTableItem.__init__(self, table, edittype, text)
                 self.color = color
         def paint(self, painter, colorgroup, rect, selected):
                 cg = qt.QColorGroup(colorgroup)
                 cg.setColor(qt.QColorGroup.Base, self.color)
                 qttable.QTableItem.paint(self,painter, cg, rect, selected)
