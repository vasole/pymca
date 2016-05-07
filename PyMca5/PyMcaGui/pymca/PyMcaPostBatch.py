#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os
from PyMca5 import PyMcaDirs
from PyMca5.PyMcaGui import PyMcaFileDialogs
from PyMca5.PyMcaGui import RGBCorrelator
from PyMca5.PyMcaGui import PyMcaQt as qt
if hasattr(qt, "QString"):
    QString = qt.QString
    QStringList = qt.QStringList
else:
    QString = qt.safe_str
    QStringList = list
QTVERSION = qt.qVersion()

class PyMcaPostBatch(RGBCorrelator.RGBCorrelator):
    def addBatchDatFile(self, filename, ignoresigma=None):
        #test if filename is an EDF ...
        #this is a more complete test
        #but it would rewire to import too many things
        #import QDataSource
        #sourceType = QDataSource.getSourceType(filename)
        #if sourceType.upper().startswith("EDFFILE"):
        #    return self.addFileList([filename])
        f = open(filename, 'rb')
        twoBytes = f.read(2)
        f.close()
        if sys.version < '3.0':
            twoChar = twoBytes
        else:
            try:
                twoChar = twoBytes.decode('utf-8')
            except:
                twoChar = "__dummy__"
        if twoChar in ["II", "MM", "\n{"] or\
           twoChar[0] in ["{"] or\
           filename.lower().endswith('cbf')or\
           (filename.lower().endswith('spe') and twoChar[0] not in ['$']):
            #very likely wrapped as EDF
            return self.addFileList([filename])

        text = qt.safe_str(self.windowTitle())
        text += ": " + qt.safe_str(os.path.basename(filename))

        self.setWindowTitle(text)

        if len(filename) > 4:
            if filename[-4:] == ".csv":
                csv = True
            else:
                csv = False
        self.controller.addBatchDatFile(filename, ignoresigma, csv=csv)

    def addFileList(self, filelist):
        """
        Expected to work just with EDF files
        """
        text = qt.safe_str(self.windowTitle())
        if len(filelist) == 1:
            text += ": " + qt.safe_str(os.path.basename(filelist[0]))
        else:
            text += ": from " + qt.safe_str(os.path.basename(filelist[0])) + \
                    " to " + qt.safe_str(os.path.basename(filelist[-1]))
        self.setWindowTitle(text)

        self.controller.addFileList(filelist)

    def _getStackOfFiles(self):
        wdir = PyMcaDirs.inputDir
        fileTypeList = ["Batch Result Files (*dat)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "TIFF Files (*tif *tiff *TIF *TIFF)",
                        "Image Files (* jpg *jpeg *tif *tiff *png)",
                        "All Files (*)"]
        message = "Open ONE Batch result file or SEVERAL EDF files"

        filelist = PyMcaFileDialogs.getFileList(parent=self,
                                filetypelist=fileTypeList,
                                message=message,
                                currentdir=wdir,
                                mode="OPEN",
                                single=False)
        if not len(filelist):
            return []
        PyMcaDirs.inputDir = os.path.dirname(filelist[0])
        return filelist

def test():
    sys.excepthook = qt.exceptionHandler
    app = qt.QApplication([])
    app.lastWindowClosed.connect(app.quit)

    import getopt
    options=''
    longoptions=["nativefiledialogs=","transpose=", "fileindex="]
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)
    transpose=False
    for opt,arg in opts:
        if opt in '--nativefiledialogs':
            if int(arg):
                PyMcaDirs.nativeFileDialogs=True
            else:
                PyMcaDirs.nativeFileDialogs=False
        elif opt in '--transpose':
            if int(arg):
                transpose=True
        elif opt in '--fileindex':
            if int(arg):
                transpose=True
    filelist=args
    w = PyMcaPostBatch()
    w.layout().setContentsMargins(11, 11, 11, 11)
    if not len(filelist):
        filelist = w._getStackOfFiles()
    if not len(filelist):
        print("Usage:")
        print("python PyMcaPostBatch.py PyMCA_BATCH_RESULT_DOT_DAT_FILE")
        sys.exit(app.quit())
    if len(filelist) == 1:
        if filelist[0].lower().endswith("dat"):
            try:
                w.addBatchDatFile(filelist[0])
            except ValueError:
                w.addFileList(filelist)
        else:
            w.addFileList(filelist)
    else:
        w.addFileList(filelist)
    if transpose:
        w.transposeImages()
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()

