#!/usr/bin/env python
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
__author__ = "V.A. Sole - ESRF BLISS Group"
import sys
import os
import EdfFileDataSource
DataReader = EdfFileDataSource.EdfFileDataSource
import RGBCorrelator
qt = RGBCorrelator.qt
import Numeric

class PyMcaPostBatch(RGBCorrelator.RGBCorrelator):
    def addBatchDatFile(self, filename, ignoresigma=None):
        text = str(self.windowTitle())
        text += ": " + str(os.path.basename(filename))
        self.setWindowTitle(text)

        f = open(filename)
        lines = f.readlines()
        f.close()
        labels = lines[0].replace("\n","").split("  ")
        i = 1
        while (not len( lines[-i].replace("\n",""))):
               i += 1
        nlabels = len(labels)
        nrows = len(lines) - i
        if ignoresigma is None:
            step  = 1
            if len(labels) > 4:
                if len(labels[2]) == (len(labels[3])-3):
                    if len(labels[3]) > 5:
                        if labels[3][2:-1] == labels[2]:
                            step = 2
        elif ignoresigma:
            step = 2
        else:
            step = 1
        totalArray = Numeric.zeros((nrows, nlabels), Numeric.Float)
        for i in range(nrows):
            totalArray[i, :] = map(float, lines[i+1].split())

        nrows = int(max(totalArray[:,0]) + 1)
        ncols = int(max(totalArray[:,1]) + 1)
        singleArray = Numeric.zeros((nrows* ncols, 1), Numeric.Float)
        for i in range(2, nlabels, step):
            singleArray[:, 0] = totalArray[:,i] * 1
            self.addImage(Numeric.resize(singleArray, (nrows, ncols)), labels[i])

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

        for fname in filelist:
            source = DataReader(fname)
            for key in source.getSourceInfo()['KeyList']:
                dataObject = source.getDataObject(key)
                self.controller.addImage(dataObject.data,
                                         os.path.basename(fname)+" "+key)

    def _getStackOfFiles(self):
        wdir = os.getcwd()
        fileTypeList = ["Batch Result Files (*dat)",
                        "EDF Files (*edf)",
                        "EDF Files (*ccd)",
                        "All Files (*)"]
        message = "Open ONE Batch result file or SEVERAL EDF files"
        if sys.platform != 'darwin':
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
    longoptions=[]
    opts, args = getopt.getopt(
                    sys.argv[1:],
                    options,
                    longoptions)      
    for opt,arg in opts:
        pass
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
    w.show()
    app.exec_()

if __name__ == "__main__":
    test()
        
