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
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
        if qt.qVersion() < '4.0.0':
            print "WARNING: Using Qt %s version" % qt.qVersion()
    except:
        import qt
else:
    import qt
QTVERSION = qt.qVersion()
if QTVERSION < '3.0.0':
    import Myqttable as qttable
elif QTVERSION < '4.0.0':
    import qttable
import string

DEBUG=0

if QTVERSION < '4.0.0':
    class QTable(qttable.QTable):
        def __init__(self, parent=None, name=""):
            qttable.QTable.__init__(self, parent, name)
            self.rowCount    = self.numRows
            self.columnCount = self.numCols
            self.setRowCount = self.setNumRows
            self.setColumnCount = self.setNumCols
            self.resizeColumnToContents = self.adjustColumn
        
else:
    QTable = qt.QTableWidget

class McaTable(QTable):
    def __init__(self, *args,**kw):
        apply(QTable.__init__, (self, ) + args)
        if kw.has_key('labels'):
            self.labels=[]
            for label in kw['labels']:
                self.labels.append(label)
        else:
            #self.labels=['Element','Group','Energy','Ratio','Fit Area','MCA Area','Sigma','Fwhm','Chisq']
            self.labels=['Element','Group','Fit Area','Sigma','Energy','Ratio','FWHM','Chi square']
        self.setColumnCount(len(self.labels))

        if QTVERSION < '4.0.0':
            i=0        
            for label in self.labels:
                qt.QHeader.setLabel(self.horizontalHeader(),i,label)
                self.adjustColumn(i)
                i=i+1
        else:
            for i in range(len(self.labels)):
                item = self.horizontalHeaderItem(i)
                if item is None:
                    item = qt.QTableWidgetItem(self.labels[i],
                                               qt.QTableWidgetItem.Type)
                item.setText(self.labels[i])
                self.setHorizontalHeaderItem(i,item)            
                
        self.regionlist=[]
        self.regiondict={}
        if QTVERSION < '4.0.0':
            self.verticalHeader().setClickEnabled(1)
        else:
            if DEBUG:
                print "vertical header click to enable"
        self.connect(self.verticalHeader(),qt.SIGNAL('clicked(int)'),self.__myslot)
        #self.connect(self.verticalHeader(),qt.SIGNAL('clicked(int)'),self.__myslot)
        self.connect(self,qt.SIGNAL("selectionChanged()"),self.__myslot)
        #self.setSelectionMode(qttable.QTable.SingleRow)

                
    def fillfrommca(self,result,diag=1):
        line=0
        region=0
        #calculate the number of rows
        nrows = 0
        for group in result['groups']:
            nrows += 1
            for peak in result[group]['peaks']:
                nrows += 1
            for peak0 in result[group]['escapepeaks']:
                peak  = peak0+"esc"
                if result[group][peak]['ratio'] > 0.0:
                    nrows += 1
        self.setRowCount(nrows)
        alreadyforced = 0
        for group in result['groups']:
            ele,group0 = string.split(group)
            fitarea    = qt.QString("%.4e" % (result[group]['fitarea']))
            sigmaarea  = qt.QString("%.2e" % (result[group]['sigmaarea']))
            fields = [ele,group0,fitarea,sigmaarea]
            col = 0
            color = qt.QColor('white')
            nlines = self.rowCount()
            if (line+1) > nlines:
                self.setRowCount(line+1)
            for i in range(len(self.labels)):
                if i < len(fields):
                    if qt.qVersion() < '4.0.0':
                        item=ColorQTableItem(self,qttable.QTableItem.OnTyping,
                                            fields[i],color=color,bold=1)
                    else:
                        item = self.item(line, col)
                        text = fields[i]
                        if item is None:
                            item = qt.QTableWidgetItem(text,
                                                       qt.QTableWidgetItem.Type)
                        else:
                            item.setText(text)
                            item.setBackgroundColor(color)
                            item.setFlags(qt.Qt.ItemIsSelectable|
                                          qt.Qt.ItemIsEnabled)                    
                    self.setItem(line, col, item)
                else:
                    if qt.qVersion() < '4.0.0':
                        self.clearCell(line,col)
                    else:
                        item = self.item(line, col)
                        if item is not None:
                            item.setText("")
                        self.setItem(line, col, item)
                col=col+1
            line += 1
            #Lemon Chiffon = (255,250,205)
            color = qt.QColor(255,250,205)
            for peak in result[group]['peaks']:
                name  = peak
                energy = qt.QString("%.3f" % (result[group][peak]['energy']))
                ratio  = qt.QString("%.5f" % (result[group][peak]['ratio']))
                area   = qt.QString("%.4e" % (result[group][peak]['fitarea']))
                sigma  = qt.QString("%.2e" % (result[group][peak]['sigmaarea']))
                fwhm   = qt.QString("%.3f" % (result[group][peak]['fwhm']))
                chisq  = qt.QString("%.2f" % (result[group][peak]['chisq']))
                if (line+1) > nlines:
                    self.setRowCount(line+1)
                fields = [name,area,sigma,energy,ratio,fwhm,chisq]
                col = 1
                for field in fields:
                    if qt.qVersion() < '4.0.0':
                        item=ColorQTableItem(self, qttable.QTableItem.Never,
                                            field,color=color)
                    else:
                        item = self.item(line, col)
                        text = field
                        if item is None:
                            item = qt.QTableWidgetItem(text,
                                                       qt.QTableWidgetItem.Type)
                        else:
                            item.setText(text)
                        item.setBackgroundColor(color)
                        item.setFlags(qt.Qt.ItemIsSelectable|
                                      qt.Qt.ItemIsEnabled)                    
                    self.setItem(line, col, item)
                    col=col+1
                line+=1
            for peak0 in result[group]['escapepeaks']:
                peak  = peak0+"esc"
                if result[group][peak]['ratio'] > 0.0:
                    energy = qt.QString("%.3f" % (result[group][peak]['energy']))
                    ratio  = qt.QString("%.5f" % (result[group][peak]['ratio']))
                    area   = qt.QString("%.4e" % (result[group][peak]['fitarea']))
                    sigma  = qt.QString("%.2e" % (result[group][peak]['sigmaarea']))
                    fwhm   = qt.QString("%.3f" % (result[group][peak]['fwhm']))
                    chisq  = qt.QString("%.2f" % (result[group][peak]['chisq']))
                    if (line+1) > nlines:
                        self.setRowCount(line+1)
                    fields = [peak,area,sigma,energy,ratio,fwhm,chisq]
                    col = 1
                    for field in fields:
                        if qt.qVersion() < '4.0.0':
                            item=ColorQTableItem(self,
                                                 qttable.QTableItem.Never,
                                                 field,color=color)
                        else:
                            item = self.item(line, col)
                            if item is None:
                                item = qt.QTableWidgetItem(field,
                                                           qt.QTableWidgetItem.Type)
                            else:
                                item.setText(field)
                            item.setBackgroundColor(color)
                            item.setFlags(qt.Qt.ItemIsSelectable|
                                          qt.Qt.ItemIsEnabled)
                        self.setItem(line, col, item)
                        col=col+1
                    line+=1
        for i in range(self.columnCount()):
            if i>-1:
                self.resizeColumnToContents(i)

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
        if QTVERSION < '4.0.0':
            self.emit(qt.PYSIGNAL('McaTableSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('McaTableSignal'), dict)

    def gettext(self):
        lemon=string.upper("#%x%x%x" % (255,250,205))
        if QTVERSION < '4.0.0':
            if QTVERSION < '3.0.0':
                hcolor = string.upper("#%x%x%x" % (230,240,249))
            else:
                hb = self.horizontalHeader().paletteBackgroundColor()
                hcolor = string.upper("#%x%x%x" % (hb.red(),hb.green(),hb.blue()))
        else:
            if DEBUG:print "color background to implement"
            hcolor = string.upper("#%x%x%x" % (230,240,249))
        text=""
        text+=("<nobr>")
        text+=( "<table>")
        text+=( "<tr>")
        if QTVERSION < '4.0.0':
            ncols = self.numCols()
        else:
            ncols = self.columnCount()
        for l in range(ncols):
            text+=('<td align="left" bgcolor="%s"><b>' % hcolor)
            if QTVERSION < '4.0.0':
                text+=(str(self.horizontalHeader().label(l)))
            else:
                text+=(str(self.horizontalHeaderItem(l).text()))
            text+=("</b></td>")
        text+=("</tr>")
        #text+=( str(qt.QString("</br>"))
        if QTVERSION < '4.0.0':
            nrows = self.numRows()
        else:
            nrows = self.rowCount()
        for r in range(nrows):
            text+=("<tr>")
            if QTVERSION < '4.0.0':
                moretext = str(self.text(r,0))
            else:
                moretext = ""
                item = self.item(r, 0)
                if item is not None:
                    moretext = str(item.text()) 
            if len(moretext):
                color = "white"
                b="<b>"                
            else:
                b=""
                color = lemon
            for c in range(ncols):
                if QTVERSION < '4.0.0':
                    moretext = str(self.text(r,c))
                else:
                    moretext = ""
                    item = self.item(r, c)
                if item is not None:
                    moretext = str(item.text()) 
                if len(moretext):
                    finalcolor = color
                else:
                    finalcolor = "white"
                if c<2:
                    text+=('<td align="left" bgcolor="%s">%s' % (finalcolor,b))
                else:
                    text+=('<td align="right" bgcolor="%s">%s' % (finalcolor,b))
                text+= moretext
                if len(b):
                    text+=("</td>")
                else:
                    text+=("</b></td>") 
            if QTVERSION < '4.0.0':
                moretext = str(self.text(r,0))
            else:
                moretext = ""
                item = self.item(r, 0)
                if item is not None:
                    moretext = str(item.text()) 
            if len(moretext):
                text+=("</b>")
            text+=("</tr>")
            #text+=( str(qt.QString("<br>"))
            text+=("\n")
        text+=("</table>")
        text+=("</nobr>")
        return text

    def closeEvent(self, event):
        QTable.closeEvent(self, event)
        ddict={}
        ddict['event']= 'closed'
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('closed'),(ddict,))
        else:
            self.emit(qt.SIGNAL('closed'), ddict)

if QTVERSION < '4.0.0':
    class ColorQTableItem(qttable.QTableItem):
             def __init__(self, table, edittype, text,color=qt.Qt.white,bold=0):
                     qttable.QTableItem.__init__(self, table, edittype, text)
                     self.color = color
                     self.bold  = bold
             def paint(self, painter, colorgroup, rect, selected):
                painter.font().setBold(self.bold)
                cg = qt.QColorGroup(colorgroup)
                cg.setColor(qt.QColorGroup.Base, self.color)
                qttable.QTableItem.paint(self,painter, cg, rect, selected)
                painter.font().setBold(0)
