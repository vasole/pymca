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
from PyMca import PyMcaQt as qt
from PyMca import SpecfitGUI
from PyMca import Specfit
QTVERSION = qt.qVersion()


class ScanFit(qt.QWidget):
    def __init__(self, parent=None, name="ScanFit", specfit=None, fl=0):
                #fl=qt.Qt.WDestructiveClose):
        qt.QWidget.__init__(self, parent)
        if QTVERSION < '4.0.0':
            self.setCaption(name)
        else:
            self.setWindowTitle(name)

        if specfit is None:
            self.specfit = Specfit.Specfit()
        else:
            self.specfit = specfit
        self.info = None
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        ##############
        self.headerlabel = qt.QLabel(self)
        self.headerlabel.setAlignment(qt.Qt.AlignHCenter)
        self.setHeader('<b>Fit of XXXXXXXXXX from X XXXXX to XXXX<\b>')
        ##############
        if not len(self.specfit.theorylist):
            funsFile = "SpecfitFunctions.py"
            if not os.path.exists(funsFile):
                funsFile = os.path.join(os.path.dirname(Specfit.__file__),
                                        funsFile)
            self.specfit.importfun(funsFile)
        if 'Area Gaussians' not in self.specfit.theorylist:
            funsFile = "SpecfitFunctions.py"
            if not os.path.exists(funsFile):
                funsFile = os.path.join(os.path.dirname(Specfit.__file__),
                                        funsFile)
            self.specfit.importfun(funsFile)
        self.specfit.settheory('Area Gaussians')
        self.specfit.setbackground('Linear')
        fitconfig = {}
        fitconfig.update(self.specfit.fitconfig)
        fitconfig['WeightFlag'] = 0
        fitconfig['ForcePeakPresence'] = 1
        fitconfig['McaMode']    = 0
        self.specfit.configure(**fitconfig)
        self.specfitGUI = SpecfitGUI.SpecfitGUI(self, config=1, status=1,
                                                buttons=0,
                                                specfit=self.specfit,
                                                eh=self.specfit.eh)
        #self.specfitGUI.updateGUI(configuration=fitconfig)
        #self.setdata = self.specfit.setdata

        self.specfitGUI.guiconfig.MCACheckBox.setEnabled(1)
        palette = self.specfitGUI.guiconfig.MCACheckBox.palette()
        if QTVERSION < '4.0.0':
            palette.setDisabled(palette.active())
        ##############
        hbox = qt.QWidget(self)
        hboxlayout = qt.QHBoxLayout(hbox)
        hboxlayout.setContentsMargins(0, 0, 0, 0)
        hboxlayout.setSpacing(0)
        self.estimatebutton = qt.QPushButton(hbox)
        self.estimatebutton.setText("Estimate")
        self.fitbutton = qt.QPushButton(hbox)
        self.fitbutton.setText("Fit")
        hboxlayout.addWidget(self.estimatebutton)
        hboxlayout.addWidget(qt.HorizontalSpacer(hbox))
        hboxlayout.addWidget(self.fitbutton)

        self.dismissbutton = qt.QPushButton(hbox)
        self.dismissbutton.setText("Dismiss")
        self.connect(self.estimatebutton, qt.SIGNAL("clicked()"), self.estimate)
        self.connect(self.fitbutton, qt.SIGNAL("clicked()"), self.fit)
        self.connect(self.dismissbutton, qt.SIGNAL("clicked()"), self.dismiss)
        if QTVERSION < '4.0.0':
            self.connect(self.specfitGUI,
                         qt.PYSIGNAL('SpecfitGUISignal') ,
                         self._specfitGUISignal)
        else:
            self.connect(self.specfitGUI,
                         qt.SIGNAL('SpecfitGUISignal') ,
                         self._specfitGUISignal)
        hboxlayout.addWidget(qt.HorizontalSpacer(hbox))
        hboxlayout.addWidget(self.dismissbutton)
        layout.addWidget(self.headerlabel)
        layout.addWidget(self.specfitGUI)
        layout.addWidget(hbox)

    def setData(self, *var, **kw):
        self.info = {}
        if 'legend' in kw:
            self.info['legend'] = kw['legend']
            del kw['legend']
        else:
            self.info['legend'] = 'Unknown Origin'
        if 'xlabel' in kw:
            self.info['xlabel'] = kw['xlabel']
            del kw['xlabel']
        else:
            self.info['xlabel'] = 'X'
        self.specfit.setdata(var, **kw)
        try:
            self.info['xmin'] = "%.3f" % self.specfit.xdata[0]
        except:
            self.info['xmin'] = 'First'
        try:
            self.info['xmax'] = "%.3f" % self.specfit.xdata[-1]
        except:
            self.info['xmax'] = 'Last'
        self.setHeader(text="Fit of %s from %s %s to %s" % (self.info['legend'],
                                                            self.info['xlabel'],
                                                            self.info['xmin'],
                                                            self.info['xmax']))

    def setheader(self, *var, **kw):
        return self.setHeader(*var, **kw)

    def setHeader(self, *var, **kw):
        if len(var):
            text = var[0]
        elif 'text' in kw:
            text = kw['text']
        elif 'header' in kw:
            text = kw['header']
        else:
            text = ""
        self.headerlabel.setText("<b>%s<\b>" % text)

    def fit(self):
        if self.specfit.fitconfig['McaMode']:
            fitconfig = {}
            fitconfig.update(self.specfit.fitconfig)
            self.specfitGUI.updateGUI(configuration=fitconfig)
            #the GUI already takes care of mcafit
            self.specfitGUI.estimate()
        else:
            #exception handler to be implemented
            #self.specfitGUI.estimate()
            self.specfitGUI.startfit()

    def estimate(self):
        fitconfig = {}
        fitconfig.update(self.specfit.fitconfig)
        self.specfitGUI.updateGUI(configuration=fitconfig)
        self.specfitGUI.estimate()

    def _specfitGUISignal(self, ddict):
        if not hasattr(ddict, "keys"):
            return
        if 'event' in ddict:
            if ddict['event'].upper() == "PRINT":
                h = self.__htmlheader()
                if __name__ == "__main__":
                    self.__print(h + ddict['text'])
                else:
                    ndict = {}
                    ndict['event'] = "ScanFitPrint"
                    ndict['text' ] = h + ddict['text']
                    ndict['info' ] = {}
                    ndict['info'].update(self.info)
                    if QTVERSION < '4.0.0':
                        self.emit(qt.PYSIGNAL('ScanFitSignal'), (ndict,))
                    else:
                        self.emit(qt.SIGNAL('ScanFitSignal'), ndict)
            else:
                if self.info is None:
                    self.info = {}
                ddict['info'] = {}
                ddict['info'].update(self.info)
                if QTVERSION < '4.0.0':
                    self.emit(qt.PYSIGNAL('ScanFitSignal'), (ddict,))
                else:
                    self.emit(qt.SIGNAL('ScanFitSignal'), ddict)

    def dismiss(self):
        self.close()

    def __htmlheader(self):
        try:
            header = "Fit of %s from %s %s to %s" % (self.info['legend'],
                                                     self.info['xlabel'],
                                                     self.info['xmin'],
                                                     self.info['xmax'])
        except:
            # I cannot afford any unicode, key or whatever error, so,
            # provide a default value for the label.
            header = 'Fit of XXXXXXXXXX from Channel XXXXX to XXXX'
        if self.specfit.fitconfig['WeightFlag']:
            weight = "YES"
        else:
            weight = "NO"
        if self.specfit.fitconfig['McaMode']:
            mode = "YES"
        else:
            mode = "NO"
        theory   = self.specfit.fitconfig['fittheory']
        bkg      = self.specfit.fitconfig['fitbkg']
        fwhm     = self.specfit.fitconfig['FwhmPoints']
        scaling  = self.specfit.fitconfig['Yscaling']
        h = ""
        h += "    <CENTER>"
        h += "<B>%s</B>" % header
        h += "<BR></BR>"
        h += "<TABLE>"
        h += "<TR>"
        h += "    <TD ALIGN=LEFT><B>Function</B></TD>"
        h += "    <TD><B>:</B></TD>"
        h += "    <TD ALIGN=LEFT>%s</TD>" % theory
        h += "    <TD><SPACER TYPE=BLOCK WIDTH=50></TD>"
        h += "    <TD ALIGN=RIGHT><B>Weight</B></TD>"
        h += "    <TD><B>:</B></TD>"
        h += "    <TD ALIGN=LEFT>%s</TD>" % weight
        h += "    <TD><SPACER TYPE=BLOCK WIDTH=10></B></TD>"
        h += "    <TD ALIGN=RIGHT><B>FWHM</B></TD>"
        h += "    <TD><B>:</B></TD></TD>"
        h += "    <TD ALIGN=LEFT>%d</TD>" % fwhm
        h += "</TR>"
        h += "<TR>"
        h += "    <TD ALIGN=LEFT><B>Background</B></TH>"
        h += "    <TD><B>:</B></TD>"
        h += "    <TD ALIGN=LEFT>%s</TD>" % bkg
        h += "    <TD><SPACER TYPE=BLOCK WIDTH=50></B></TD>"
        h += "    <TD ALIGN=RIGHT><B>MCA Mode</B></TD>"
        h += "    <TD><B>:</B></TD>"
        h += "    <TD ALIGN=LEFT>%s</TD>" % mode
        h += "    <TD><SPACER TYPE=BLOCK WIDTH=10></B></TD>"
        h += "    <TD ALIGN=RIGHT><B>Scaling</B></TD>"
        h += "    <TD><B>:</B></TD>"
        h += "    <TD ALIGN=LEFT>%g</TD>" % scaling
        h += "</TR>"
        h += "</TABLE>"
        h += "</CENTER>"
        return h

    def __print(self, text):
        printer = qt.QPrinter()
        if printer.setup(self):
            painter = qt.QPainter()
            if not(painter.begin(printer)):
                return 0
            try:
                metrics = qt.QPaintDeviceMetrics(printer)
                dpiy    = metrics.logicalDpiY()
                margin  = int((2 / 2.54) * dpiy)  # 2cm margin
                body = qt.QRect(0.5 * margin, margin, metrics.width() - margin,
                                metrics.height() - 2 * margin)
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
                painter.end()
            except:
                painter.end()
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("%s" % sys.exc_info()[1])
                msg.exec_loop()

    def getText(self):
        try:
            header = "Fit of %s from %s %s to %s" % (self.info['legend'],
                                                     self.info['xlabel'],
                                                     self.info['xmin'],
                                                     self.info['xmax'])
        except:
            # I cannot afford any unicode, key or whatever error, so,
            # provide a default value for the header text.
            header = 'Fit of XXXXXXXXXX from Channel XXXXX to XXXX'
        text = header + "\n"
        if self.specfit.fitconfig['WeightFlag']:
            weight = "YES"
        else:
            weight = "NO"
        if self.specfit.fitconfig['McaMode']:
            mode = "YES"
        else:
            mode = "NO"
        theory   = self.specfit.fitconfig['fittheory']
        bkg      = self.specfit.fitconfig['fitbkg']
        fwhm     = self.specfit.fitconfig['FwhmPoints']
        scaling  = self.specfit.fitconfig['Yscaling']
        text += "Fit Function: %s\n" % theory
        text += "Background: %s\n" % bkg
        text += "Weight: %s  McaMode: %s  FWHM: %d  Yscaling: %f\n" % (weight[0],
                                                                       mode[0],
                                                                       fwhm,
                                                                       scaling)
        text += self.specfitGUI.guiparameters.getText()
        return text

    def getConfiguration(self):
        return self.specfit.configure()

    def setConfiguration(self, fitconfig):
        self.specfit.configure(**fitconfig)
        self.specfitGUI.updateGUI(configuration=fitconfig)


def main():
    app = qt.QApplication([])
    w = ScanFit()
    qt.QObject.connect(app, qt.SIGNAL("lasWindowClosed()"),
                       app, qt.SLOT("quit()"))
    w.show()
    if QTVERSION < '4.0.0':
        app.setMainWidget(w)
        app.exec_loop()
    else:
        app.exec_()

if __name__ == "__main__":
    main()
