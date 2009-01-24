#!/usr/bin/env python
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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import os
import PyMcaDirs
import RGBCorrelator
qt = RGBCorrelator.qt
import numpy.oldnumeric as Numeric
QTVERSION = qt.qVersion()

class PyMcaPostBatch(RGBCorrelator.RGBCorrelator):
    def addBatchDatFile(self, filename, ignoresigma=None):
        #test if filename is an EDF ...
        f = open(filename)
        line = f.readline()
        if not len(line.replace("\n","")):
            line = f.readline()
        f.close()
        if line[0] == "{":
            return self.addFileList([filename])
        
        text = str(self.windowTitle())
        text += ": " + str(os.path.basename(filename))

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
        text = str(self.windowTitle())
        if len(filelist) == 1:
            text += ": " + str(os.path.basename(filelist[0]))
        else:
            text += ": from " + str(os.path.basename(filelist[0])) + \
                    " to " + str(os.path.basename(filelist[-1]))
        self.setWindowTitle(text)

        self.controller.addFileList(filelist)

    def _getStackOfFiles(self):
        wdir = PyMcaDirs.inputDir
        fileTypeList = ["Batch Result Files (*dat)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
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
            strlist = qt.QStringList()
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
        filelist = map(str, filelist)
        if not len(filelist):return []
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
    w.layout().setMargin(11)
    if not len(filelist):
        filelist = w._getStackOfFiles()
    if not len(filelist):
        print "Usage:"
        print "python PyMcaPostBatch.py PyMCA_BATCH_RESULT_DOT_DAT_FILE"
        sys.exit(app.quit())        
    if len(filelist) == 1:
        try:
            w.addBatchDatFile(filelist[0])
        except ValueError:
            w.addFileList(filelist)
    else:
        w.addFileList(filelist)
    if transpose:
        w.transposeImages()
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
