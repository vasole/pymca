#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2007 European Synchrotron Radiation Facility
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
import Numeric
"""This module implements an info widget containing :
       - source name, scan name
       - h,k,l infos
       - peak, peak position
       - fwhm, center of fwhm
       - center of mass
"""


DEBUG=0
STATISTICS=1
class SpecArithmetic:
    """
    This class tries to mimic SPEC operations.
    Correct peak positions and fwhm information
    have to be made via a fit.
    """
    def search_peak(self, xdata,ydata):
         """
         Search a peak and its position in arrays xdata ad ydata. 
         Return three integer:
           - peak position
           - peak value
           - index of peak position in array xdata
             This result may accelerate the fwhm search.
         """
         ymax   = max(ydata)
         idx    = self.__give_index(ymax,ydata)
         return xdata[idx],ymax,idx


    def search_com(self, xdata,ydata):
        """
        Return the center of mass in arrays xdata and ydata
        """
        num    = Numeric.sum(xdata*ydata)
        denom  = Numeric.sum(ydata)
        try:
           result = num/denom
        except ZeroDivisionError:
           result = 0
        return result


    def search_fwhm(self, xdata,ydata,peak=None,index=None):
        """
        Search a fwhm and its center in arrays xdata and ydatas.
        If no fwhm is found, (0,0) is returned.
        peak and index which are coming from search_peak result, may
        accelerate calculation
        """
        if peak is None or index is None:
            x,mypeak,index_peak = self.search_peak(xdata,ydata)
        else:
            mypeak     = peak
            index_peak = index
        
        hm = mypeak/2
        idx = index_peak

        try:
            while ydata[idx] >= hm:
               idx = idx-1
            x0 = xdata[idx]
            x1 = xdata[idx+1]
            y0 = ydata[idx]
            y1 = ydata[idx+1]
        
            lhmx = (hm*(x1-x0) - (y0*x1)+(y1*x0)) / (y1-y0)
        except ZeroDivisionError:
            lhmx = 0 
        except IndexError:
            lhmx = xdata[0]

        idx = index_peak
        try:
            while ydata[idx] >= hm:
                idx = idx+1
        
            x0 = xdata[idx]
            x1 = xdata[idx+1]
            y0 = ydata[idx]
            y1 = ydata[idx+1]
        
            uhmx = (hm*(x1-x0) - (y0*x1)+(y1*x0)) / (y1-y0)
        except ZeroDivisionError:
            uhmx = 0
        except IndexError:
            uhmx = xdata[-1]

        FWHM  = uhmx - lhmx
        CFWHM = (uhmx+lhmx)/2
        return FWHM,CFWHM


    def __give_index(self, elem,array):
         """
         Return the index of elem in array
         """
         mylist = array.tolist()
         return mylist.index(elem)

class HKL(qt.QWidget):
    def __init__(self, parent = None, h= "", k= "", l=""):
        qt.QWidget.__init__(self, parent)
        layout = qt.QHBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(5)
        

        hlabel = qt.QLabel(self)
        hlabel.setText('H:')
        self.h = qt.QLineEdit(self)
        self.h.setReadOnly(True)

        klabel = qt.QLabel(self)
        klabel.setText('K:')
        self.k = qt.QLineEdit(self)
        self.k.setReadOnly(True)

        llabel = qt.QLabel(self)
        llabel.setText('L:')
        self.l = qt.QLineEdit(self)
        self.l.setReadOnly(True)
        
        self.setHKL(h, k, l)

        layout.addWidget(hlabel)
        layout.addWidget(self.h)
        layout.addWidget(klabel)
        layout.addWidget(self.k)
        layout.addWidget(llabel)
        layout.addWidget(self.l)
        
    def setHKL(self, h="", k="", l=""):
        format = "%.4f"
        if type(h) == type (""):
            self.h.setText(h)
        else:
            self.h.setText(format % h)
        if type(k) == type (""):
            self.k.setText(k)
        else:
            self.k.setText(format % k)
        if type(l) == type (""):
            self.l.setText(l)
        else:
            self.l.setText(format % l)

class GraphInfoWidget(qt.QWidget):
    def __init__(self, parent):
        qt.QWidget.__init__(self, parent)
        layout = qt.QGridLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)

        #peak
        peak   = qt.QLabel(self)
        peak.setText("Peak:  ")
        self.peak = qt.QLineEdit(self)
        self.peak.setReadOnly(True)
        hboxPeak   = qt.QWidget(self)
        hboxPeak.l = qt.QHBoxLayout(hboxPeak) 
        hboxPeak.l.setMargin(0)
        hboxPeak.l.setSpacing(0)
        peakAt     = qt.QLabel(hboxPeak)
        peakAt.setText(" at:")
        self.peakAt = qt.QLineEdit(hboxPeak)
        self.peak.setReadOnly(True)
        hboxPeak.l.addWidget(peakAt)
        hboxPeak.l.addWidget(self.peakAt)

        #fwhm
        fwhm   = qt.QLabel(self)
        fwhm.setText("Fwhm: ")
        self.fwhm = qt.QLineEdit(self)
        self.fwhm.setReadOnly(True)
        hboxFwhm   = qt.QWidget(self)
        hboxFwhm.l = qt.QHBoxLayout(hboxFwhm) 
        hboxFwhm.l.setMargin(0)
        hboxFwhm.l.setSpacing(0)
        fwhmAt     = qt.QLabel(hboxFwhm)
        fwhmAt.setText(" at:")
        self.fwhmAt = qt.QLineEdit(hboxFwhm)
        self.fwhm.setReadOnly(True)
        hboxFwhm.l.addWidget(fwhmAt)
        hboxFwhm.l.addWidget(self.fwhmAt)

        #statistics
        #COM
        com   = qt.QLabel(self)
        com.setText("COM:")
        self.com = qt.QLineEdit(self)
        self.com.setReadOnly(True)

        #mean
        mean   = qt.QLabel(self)
        mean.setText("Mean:")
        self.mean = qt.QLineEdit(self)
        self.mean.setReadOnly(True)

        #STD
        std   = qt.QLabel(self)
        std.setText("STD:")
        self.std = qt.QLineEdit(self)
        self.std.setReadOnly(True)

        #Max
        maximum   = qt.QLabel(self)
        maximum.setText("Max:")
        self.maximum= qt.QLineEdit(self)
        self.maximum.setReadOnly(True)

        #mean
        minimum   = qt.QLabel(self)
        minimum.setText("Min:")
        self.minimum= qt.QLineEdit(self)
        self.minimum.setReadOnly(True)

        #STD
        delta   = qt.QLabel(self)
        delta.setText("Delta:")
        self.delta = qt.QLineEdit(self)
        self.delta.setReadOnly(True)

        layout.addWidget(peak,      0, 0)
        layout.addWidget(self.peak, 0, 1)
        layout.addWidget(hboxPeak,  0, 2)
        layout.addWidget(com,       0, 3)
        layout.addWidget(self.com,  0, 4)
        layout.addWidget(mean,      0, 5)
        layout.addWidget(self.mean, 0, 6)
        layout.addWidget(std,       0, 7)
        layout.addWidget(self.std,  0, 8)

        layout.addWidget(fwhm,          1, 0)
        layout.addWidget(self.fwhm,     1, 1)
        layout.addWidget(hboxFwhm,      1, 2)
        layout.addWidget(maximum,       1, 3)
        layout.addWidget(self.maximum,  1, 4)
        layout.addWidget(minimum,       1, 5)
        layout.addWidget(self.minimum,  1, 6)
        layout.addWidget(delta,         1, 7)
        layout.addWidget(self.delta,    1, 8)
        self.specArithmetic = SpecArithmetic()

    def updateFromDataObject(self, dataObject):
        self.updateFromXY(dataObject.x[0],
                          dataObject.y[0])


    def updateFromXY(self, xdata, ydata):
        if len(ydata):
            peakpos,peak,myidx = self.specArithmetic.search_peak(xdata,ydata)
            com                = self.specArithmetic.search_com(xdata,ydata)
            fwhm,cfwhm         = self.specArithmetic.search_fwhm(xdata,ydata,
                                                  peak=peak,index=myidx)
            ymax  = max(ydata)
            ymin  = min(ydata)
            ymean = sum(ydata) / len(ydata)
            if len(ydata) > 1:
                ystd  = Numeric.sqrt(sum((ydata-ymean)*(ydata-ymean))/len(ydata))
            else:
                ystd = 0
            delta   = ymax - ymin
            format = "%.7g"
            peakpos = format % peakpos
            peak    = format % peak
            myidx   = "%d" % myidx 
            com     = format % com
            fwhm    = format % fwhm
            cfwhm   = format % cfwhm
            ymean   = format % ymean
            ystd    = format % ystd
            ymax    = format % ymax
            ymin    = format % ymin
            delta   = format % delta         
        else:
            peakpos = "----"
            peak    = "----"
            myidx   = "----"
            com     = "----"
            fwhm    = "----"
            cfwhm   = "----" 
            ymean   = "----" 
            ystd    = "----"
            ymax    = "----"
            ymin    = "----"
            delta   = "----"
        self.peak.setText(peak)
        self.peakAt.setText(peakpos)
        self.fwhm.setText(fwhm)
        self.fwhmAt.setText(cfwhm)
        self.com.setText(com)
        self.mean.setText(ymean)
        self.std.setText(ystd)
        self.minimum.setText(ymin)
        self.maximum.setText(ymax)
        self.delta.setText(delta)


class ScanInfoWidget(qt.QWidget):
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        layout = qt.QGridLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)

        #scan info
        sourceLabel = qt.QLabel(self)
        sourceLabel.setText('Source:')
        self.sourceLabel = qt.QLineEdit(self)
        self.sourceLabel.setReadOnly(True)        
        
        scanLabel = qt.QLabel(self)
        scanLabel.setText('Scan:')
        self.scanLabel = qt.QLineEdit(self)
        self.scanLabel.setReadOnly(True)        

        self.hkl = HKL(self)
        layout.addWidget(sourceLabel, 0, 0)
        layout.addWidget(self.sourceLabel, 0, 1, 1, 5)
        layout.addWidget(scanLabel,        1, 0)
        layout.addWidget(self.scanLabel,   1, 1)
        layout.addWidget(self.hkl,         1, 2)

    def updateFromDataObject(self, dataObject):
        info = dataObject.info
        source = info.get('SourceName', None)
        if source is None:
            self.sourceLabel.setText("")
        else:
            self.sourceLabel.setText(source[0])
        scan   = info.get('Header', None)
        if scan is None:
            self.scanLabel.setText("")
        else:
            self.scanLabel.setText(scan[0])
        hkl    = info.get('Hkl', None)
        if hkl is None:
            self.hkl.setHKL("----", "----", "----")
        else:
            self.hkl.setHKL(*hkl)

class ScanWindowInfoWidget(qt.QWidget):
    def __init__(self, parent = None):
        qt.QWidget.__init__(self, parent)
        layout = qt.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        
        self.scanInfo  = ScanInfoWidget(self)
        self.graphInfo = GraphInfoWidget(self)

        layout.addWidget(self.scanInfo)
        layout.addWidget(self.graphInfo)
        #print "hiding graph info"
        #self.graphInfo.hide()

    def updateFromDataObject(self, dataObject):
        self.scanInfo.updateFromDataObject(dataObject)
        self.graphInfo.updateFromDataObject(dataObject)

def test():
        app = qt.QApplication([])
        w   = ScanWindowInfoWidget()
        qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                        app,qt.SLOT("quit()"))
        """
        winfo.grid(sticky='wesn')
        if STATISTICS:
          winfo.configure(h=65,k=45621,l=32132,peak=6666876,
                            fwhm=0.2154,com=544,
                            ymax=10.,ymin=4,ystd=1,ymean=5)
        else:
          winfo.configure(h=65,k=45621,l=32132,peak=6666876,
                        fwhm=0.2154,com=544)
        """
        w.show()
        app.exec_()


if __name__ == '__main__':
        test()
