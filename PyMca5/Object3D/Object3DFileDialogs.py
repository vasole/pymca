#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import Object3DQt as qt
try:
    from PyMca5 import PyMcaDirs as Object3DDirs
except:
    from . import Object3DDirs
import os
QTVERSION = qt.qVersion()

def getFileList(parent=None, filetypelist=None, message=None, mode=None, getfilter=None):
    if filetypelist is None:
        fileTypeList = ['All Files (*)']
    else:
        fileTypeList = filetypelist
    if message is None:
        message = "Please select a file"
    if mode is None:
        mode = "OPEN"
    else:
        mode = mode.upper()
    if mode == "OPEN":
        wdir = Object3DDirs.inputDir
    else:
        wdir = Object3DDirs.outputDir
    if getfilter is None:
        getfilter = False
    if getfilter:
        if QTVERSION < '4.5.1':
            native_possible = False
        else:
            native_possible = True
    else:
        native_possible = True
    filterused = None
    if native_possible and Object3DDirs.nativeFileDialogs:
        filetypes = ""
        for filetype in fileTypeList:
            filetypes += filetype+"\n"
        if getfilter:
            if mode == "OPEN":
                filelist, filterused = qt.QFileDialog.getOpenFileNamesAndFilter(parent,
                        message,
                        wdir,
                        filetypes)
                filterused = str(filterused)
            else:
                filelist = qt.QFileDialog.getSaveFileNameAndFilter(parent,
                        message,
                        wdir,
                        filetypes)
                if len(filelist):
                    filterused = filelist[1]
                    filelist=filelist[0]
                else:
                    filelist=[]
        else:
            if mode == "OPEN":
                filelist = qt.QFileDialog.getOpenFileNames(parent,
                        message,
                        wdir,
                        filetypes)
            else:
                filelist = qt.QFileDialog.getSaveFileName(parent,
                        message,
                        wdir,
                        filetypes)
                filelist = str(filelist)
                if len(filelist):
                    filelist = [filelist]
                else:
                    filelist = []
        if not len(filelist):
            if getfilter:
                return [], filterused
            else:
                return []
        else:
            sample  = str(filelist[0])
            for filetype in fileTypeList:
                ftype = filetype.replace("(", "")
                ftype = ftype.replace(")", "")
                extensions = ftype.split()[2:]
                for extension in extensions:
                    if sample.endswith(extension[-3:]):
                        filterused = filetype
                        break
    else:
        fdialog = qt.QFileDialog(parent)
        fdialog.setModal(True)
        fdialog.setWindowTitle(message)
        strlist = qt.QStringList()
        for filetype in fileTypeList:
            strlist.append(filetype)
        fdialog.setFilters(strlist)
        if mode == "OPEN":
            fdialog.setFileMode(fdialog.ExistingFiles)
        else:
            fdialog.setAcceptMode(fdialog.AcceptSave)
            fdialog.setFileMode(fdialog.AnyFile)
            
        fdialog.setDirectory(wdir)
        if QTVERSION > '4.3.0':
            history = fdialog.history()
            if len(history) > 6:
                fdialog.setHistory(history[-6:])
        ret = fdialog.exec_()
        if ret != qt.QDialog.Accepted:
            fdialog.close()
            del fdialog
            if getfilter:
                return [], filterused
            else:
                return []
        else:            
            filelist = fdialog.selectedFiles()
            filterused = str(fdialog.selectedFilter())
            if mode != "OPEN":
                if "." in filterused:
                    extension = filterused.replace(")", "")
                    if "(" in extension:   
                        extension = extension.split("(")[-1]
                    extensionList = extension.split()
                    txt = str(filelist[0])
                    for extension in extensionList:
                        extension = extension.split(".")[-1]
                        if extension != "*":
                            txt = str(filelist[0])
                            if txt.endswith(extension):
                                break
                            else:
                                txt = txt+"."+extension
                    filelist[0] = txt
            fdialog.close()
            del fdialog
    filelist = map(str, filelist)
    if not(len(filelist)): return []
    if mode == "OPEN":
        Object3DDirs.inputDir = os.path.dirname(filelist[0])
        if Object3DDirs.outputDir is None:
            Object3DDirs.outputDir = os.path.dirname(filelist[0])
    else:
        Object3DDirs.outputDir = os.path.dirname(filelist[0])
        if Object3DDirs.inputDir is None:
            Object3DDirs.inputDir = os.path.dirname(filelist[0])        
    filelist.sort()
    if getfilter:
        return filelist, filterused
    else:
        return filelist

if __name__ == "__main__":
    app = qt.QApplication([])
    fileTypeList = ['PNG Files (*.png *.jpg)']
    print(getFileList(None, fileTypeList,"Please select a file", "SAVE", True))
    #app.exec_()
