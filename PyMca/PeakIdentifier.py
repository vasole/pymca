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
__revision__= "$Revision: 1.6 $"
__author__="V.A. Sole - ESRF BLISS Group"
try:
    import PyQt4.Qt as qt
    if qt.qVersion() < '4.0.0':
        print "WARNING: Using Qt %s version" % qt.qVersion()
except:
    import qt
import Elements
from QPeriodicTable import QPeriodicTable
DEBUG = 0
class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed))

class PeakIdentifier(qt.QWidget):
    def __init__(self,parent=None,energy=None,threshold=None,useviewer=None,
                 name="Peak Identifier",fl=0):
        if qt.qVersion() < '4.0.0':
            qt.QWidget.__init__(self,parent,name,fl)
            self.setCaption(name)
        else:
            if fl == 0:
                qt.QWidget.__init__(self, parent)
            else:
                qt.QWidget.__init__(self, parent, fl)
            self.setAccessibleName(name)
            self.setWindowTitle(name)

        if energy    is None: energy    = 5.9
        if threshold is None: threshold = 0.010
        if useviewer is None: useviewer = 0
        self.__useviewer = useviewer
        
        layout = qt.QVBoxLayout(self)
        #heading
        hbox=qt.QWidget(self)
        hbox.layout = qt.QHBoxLayout(hbox)
        layout.addWidget(hbox)
        hbox.layout.addWidget(HorizontalSpacer(hbox))

        l1=qt.QLabel(hbox)
        l1.setText('<b><nobr>Energy (keV)</nobr></b>')
        hbox.layout.addWidget(l1)
        self.energy=MyQLineEdit(hbox)
        self.energy.setText("%.3f" % energy)
        if qt.qVersion() < '4.0.0':
            qt.QToolTip.add(self.energy,'Press enter to validate your energy')
        else:
            self.energy.setToolTip('Press enter to validate your energy')
        hbox.layout.addWidget(self.energy)
        hbox.layout.addWidget(HorizontalSpacer(hbox))
        self.connect(self.energy,qt.SIGNAL('returnPressed()'),self._energySlot)
        #parameters
        hbox2 = qt.QWidget(self)
        layout.addWidget(hbox2)
        hbox2.layout = qt.QHBoxLayout(hbox2)
        font=hbox2.font()
        font.setBold(1)
        hbox2.setFont(font)
        

        l2=qt.QLabel(hbox2)
        l2.setText('Energy Threshold (eV)')
        self.threshold=qt.QSpinBox(hbox2)
        if qt.qVersion() < '4.0.0':
            self.threshold.setMinValue(0)
            self.threshold.setMaxValue(1000)
        else:
            self.threshold.setMinimum(0)
            self.threshold.setMaximum(1000)
        self.threshold.setValue(int(threshold*1000))
        self.k = qt.QCheckBox(hbox2)
        self.k.setText('K')
        self.k.setChecked(1)
        self.l1 = qt.QCheckBox(hbox2)
        self.l1.setText('L1')
        self.l1.setChecked(1)
        self.l2 = qt.QCheckBox(hbox2)
        self.l2.setText('L2')
        self.l2.setChecked(1)
        self.l3 = qt.QCheckBox(hbox2)
        self.l3.setText('L3')
        self.l3.setChecked(1)
        self.m = qt.QCheckBox(hbox2)
        self.m.setText('M')
        self.m.setChecked(1)
        self.connect(self.threshold,qt.SIGNAL('valueChanged(int)'),self.myslot)
        self.connect(self.k,qt.SIGNAL('clicked()'),self.myslot)
        self.connect(self.l1,qt.SIGNAL('clicked()'),self.myslot)
        self.connect(self.l2,qt.SIGNAL('clicked()'),self.myslot)
        self.connect(self.l3,qt.SIGNAL('clicked()'),self.myslot)
        self.connect(self.m,qt.SIGNAL('clicked()'),self.myslot)

        hbox2.layout.addWidget(l2)
        hbox2.layout.addWidget(self.threshold)
        hbox2.layout.addWidget(self.k)
        hbox2.layout.addWidget(self.l1)
        hbox2.layout.addWidget(self.l2)
        hbox2.layout.addWidget(self.l3)
        hbox2.layout.addWidget(self.m)
        
        if self.__useviewer:
            if qt.qVersion() < '4.0.0':
                self.__browsertext= qt.QTextView(self)
            else:
                self.__browsertext = qt.QTextEdit(self)
        layout.addWidget(self.__browsertext)

        
    def _energySlot(self):
        qstring = self.energy.text()
        try:
            value = float(str(qstring))
            self.energyvalue = value
            self.myslot(event='coeff')
            self.energy.setPaletteBackgroundColor(qt.Qt.white)
            self.threshold.setFocus()
        except:
            msg=qt.QMessageBox(self.energy)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText(" Float")
            if qt.qVersion() < '4.0.0':
                msg.exec_loop()
            else:
                msg.exec_()
            self.energy.setFocus()
        
    def myslot(self,*var,**kw):
        energy    = float(str(self.energy.text()))
        threshold = float(str(self.threshold.text()))/1000.
        lines=[]
        if self.k.isChecked():
             lines.append('K')
        if self.l1.isChecked():
             lines.append('L1')
        if self.l2.isChecked():
             lines.append('L2')
        if self.l3.isChecked():
             lines.append('L3')
        if self.m.isChecked():
             lines.append('M')
        dict=Elements.getcandidates(energy,threshold,lines)[0]
        dict['text'] =self.getHtmlText(dict)
        dict['event']='Candidates'
        dict['lines']=lines
        if self.__useviewer:
            if qt.qVersion() < '4.0.0':
                self.__browsertext.setText(dict['text'])
            else:
                self.__browsertext.clear()
                #self.__browsertext.insertHtml("<CENTER>"+dict['text']+\
                #                              "</CENTER>")
                self.__browsertext.insertHtml(dict['text'])
        if qt.qVersion() < '4.0.0':
            self.emit(qt.PYSIGNAL('PeakIdentifierSignal'),(dict,))
        else:
            self.emit(qt.SIGNAL('PeakIdentifierSignal'),dict)
        
        
    def getHtmlText(self,dict):
        text  = ""
        text += "<br>"
        labels=['Element','Line','Energy','Rate'] 
        lemmon=("#%x%x%x" % (255,250,205))
        lemmon = lemmon.upper()
        hcolor = ("#%x%x%x" % (230,240,249))
        hcolor = hcolor.upper()
        text+="<CENTER>"
        text+=("<nobr>")
        text+=( "<table WIDTH=80%%>")
        text+=( "<tr>")
        for l in labels:
            text+=('<td align="left" bgcolor="%s"><b>' % hcolor)
            text+=l
            text+=("</b></td>")
        text+=("</tr>")
        for ele in dict['elements']:
            oldline=""
            for line in dict[ele]:
                if   line[0][0:1] == 'K':
                    group0 = 'K  rays'
                elif line[0][0:2] == 'L1':
                    group0 = 'L1 rays'
                elif line[0][0:2] == 'L2':
                    group0 = 'L2 rays'
                elif line[0][0:2] == 'L3':
                    group0 = 'L3 rays'
                elif line[0][0:1] == 'M':
                    group0 = 'M rays'
                else:
                    group0 = 'Unknown'
                if group0 != oldline:
                    text +="<tr>"
                    text += '<td align="left"><b>%s</b></td>' % ele
                    text += '<td align="left"><b>%s</b></td>' % group0
                    text += '</tr>'
                    oldline = group0
            #for peak in result[group]['peaks']:
                text += '<tr><td></td>'
                name   = line[0]
                energy = ("%.3f" % line[1])
                ratio  = ("%.5f" % line[2])
                fields = [name,energy,ratio]
                for field in fields:
                    if field == name:
                        text+=('<td align="left"  bgcolor="%s">%s</td>' % (lemmon,field))
                    else:
                        text+=('<td align="right" bgcolor="%s">%s</td>' % (lemmon,field))
                text+="</tr>"
        text+=("</table>")
        text+=("</nobr>")
        text+="</CENTER>"
        return text

class MyQLineEdit(qt.QLineEdit):
    def __init__(self,parent=None,name=None):
        qt.QLineEdit.__init__(self,parent)
        
    def setPaletteBackgroundColor(self, color):
        if qt.qVersion() < '3.0.0':
            pass
        elif qt.qVersion() < '4.0.0':
            qt.QLineEdit.setPaletteBackgroundColor(self,color)
        else:
            palette = self.palette()
            role = self.backgroundRole()
            palette.setColor(role,color)
            self.setPalette(palette)
            

    def focusInEvent(self,event):
        if qt.qVersion() < '3.0.0':
            pass        
        elif qt.qVersion() < '4.0.0 ':
            #self.backgroundcolor = self.paletteBackgroundColor()
            self.setPaletteBackgroundColor(qt.QColor('yellow'))
        else:
            self.setPaletteBackgroundColor(qt.QColor('yellow'))
    
    def focusOutEvent(self,event):
        if qt.qVersion() < '3.0.0':
            pass        
        else:
            pass
            self.setPaletteBackgroundColor(qt.QColor('white'))
        #self.emit(qt.SIGNAL("returnPressed()"),())


if __name__ == "__main__":
    import sys
    app  = qt.QApplication(sys.argv)
    winpalette = qt.QPalette(qt.QColor(230,240,249),qt.QColor(238,234,238))
    app.setPalette(winpalette)
    if len(sys.argv) > 1:
        ene = float(sys.argv[1])
    else:
        ene = 5.9
    mw = qt.QWidget()
    l  = qt.QVBoxLayout(mw)
    if 0:
       w= PeakIdentifier(mw,energy=ene)
       browsertext= qt.QTextView(mw)
       def myslot(dict):
           browsertext.setText(dict['text'])
       mw.connect(w,qt.PYSIGNAL('PeakIdentifierSignal'),myslot)
    else:
       w= PeakIdentifier(mw,energy=ene,useviewer=1)
       #######w.myslot()
    l.addWidget(w)
    if qt.qVersion() < '4.0.0':
        app.setMainWidget(mw)
        mw.show()
        app.exec_loop()
    else:
        mw.show()
        app.exec_()
        
