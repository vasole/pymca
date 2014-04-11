#!/usr/bin/env python
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
__author__ = "V.A. Sole - ESRF Software Group"
import sys
import os
from PyMca5 import PyMcaDirs
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
        #if (QTVERSION < '4.3.0') and (sys.platform != 'darwin'):
        if PyMcaDirs.nativeFileDialogs:
            filetypes = ""
            for filetype in fileTypeList:
                filetypes += filetype+"\n"
            filelist = qt.QFileDialog.getOpenFileNames(self,
                        message,
                        wdir,
                        filetypes)
            if not len(filelist):return []
        else:
            fdialog = qt.QFileDialog(self)
            fdialog.setModal(True)
            fdialog.setWindowTitle(message)
            strlist = QStringList()
            for filetype in fileTypeList:
                strlist.append(filetype.replace("(","").replace(")",""))
            fdialog.setFilters(strlist)
            fdialog.setFileMode(fdialog.ExistingFiles)
            fdialog.setDirectory(wdir)
            ret = fdialog.exec_()
            if ret == qt.QDialog.Accepted:
                filelist = fdialog.selectedFiles()
                fdialog.close()
                del fdialog                        
            else:
                fdialog.close()
                del fdialog
                return []
        filelist = [qt.safe_str(x) for x in filelist]
        if not len(filelist):
            return []
        PyMcaDirs.inputDir = os.path.dirname(filelist[0])
        filelist.sort()
        return filelist

def test():
    app = qt.QApplication([])
    qt.QObject.connect(app,
                       qt.SIGNAL("lastWindowClosed()"),
                       app,
                       qt.SLOT('quit()'))

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
        
