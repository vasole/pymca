#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
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
DEBUG = 0
if QTVERSION < '4.0.0':
    from Q3PyMcaPrintPreview import PrintPreview
else:
    from Q4PyMcaPrintPreview import PyMcaPrintPreview as PrintPreview
    
#SINGLETON
if 0:
    #It seems sip gets confused by this singleton implementation
    class PyMcaPrintPreview(PrintPreview):
        _instance = None
        def __new__(self, *var, **kw):
            if self._instance is None:
                self._instance = PrintPreview.__new__(self,*var, **kw)
            return self._instance    
else:
    #but sip is happy about this one
    class PyMcaPrintPreview(PrintPreview):
        _instance = None
        def __new__(self, *var, **kw):
            if self._instance is None:
                self._instance = PrintPreview(*var, **kw)
            return self._instance

def testPreview():
    """
    """
    import sys
    import os.path

    if len(sys.argv) < 2:
        print "give an image file as parameter please."
        sys.exit(1)

    if len(sys.argv) > 2:
        print "only one parameter please."
        sys.exit(1)

    filename = sys.argv[1]

    a = qt.QApplication(sys.argv)
 
    p = qt.QPrinter()
    p.setOutputFileName(os.path.splitext(filename)[0]+".ps")
    p.setColorMode(qt.QPrinter.Color)

    w = PyMcaPrintPreview( parent = None, printer = p, name = 'Print Prev',
                      modal = 0, fl = 0)
    w.resize(400,500)
    if QTVERSION < '4.0.0':
        w.addPixmap(qt.QPixmap(qt.QImage(filename)))
    else:
        w.addPixmap(qt.QPixmap.fromImage(qt.QImage(filename)))
    w.addImage(qt.QImage(filename))
    if 0:
        w2 = PyMcaPrintPreview( parent = None, printer = p, name = '2Print Prev',
                      modal = 0, fl = 0)
        w.exec_()
        w2.resize(100,100)
        w2.show()
        sys.exit(w2.exec_())
    if QTVERSION < '4.0.0':
        sys.exit(w.exec_loop())
    else:
        sys.exit(w.exec_())
    
if  __name__ == '__main__':
    testPreview()

 
 
