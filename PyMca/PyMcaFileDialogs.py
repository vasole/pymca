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
import os
from PyMca import PyMcaQt as qt
from PyMca import PyMcaDirs
QTVERSION = qt.qVersion()

def getExistingDirectory(parent=None, message=None, mode=None):
    if message is None:
        message = "Please select a directory"
    if mode is None:
        mode = "OPEN"
    else:
        mode = mode.upper()
    if mode == "OPEN":
        wdir = PyMcaDirs.inputDir
    else:
        wdir = PyMcaDirs.outputDir
    if PyMcaDirs.nativeFileDialogs:
        outdir=str(qt.QFileDialog.getExistingDirectory(parent,
                        message,
                        wdir))
    else:
        outfile = qt.QFileDialog(parent)
        outfile.setWindowTitle("Output Directory Selection")
        outfile.setModal(1)
        outfile.setDirectory(wdir)
        outfile.setFileMode(outfile.DirectoryOnly)
        ret = outfile.exec_()
        if ret:
            outdir=str(outfile.selectedFiles()[0])
        else:
            outdir = ""
            outfile.close()
            del outfile
    if len(outdir):
        if mode == "OPEN":
            PyMcaDirs.inputDir = os.path.dirname(outdir)
            if PyMcaDirs.outputDir is None:
                PyMcaDirs.outputDir = os.path.dirname(outdir)
        else:
            PyMcaDirs.outputDir = os.path.dirname(outdir)
            if PyMcaDirs.inputDir is None:
                PyMcaDirs.inputDir = os.path.dirname(outdir)
    return outdir

def getFileList(parent=None, filetypelist=None, message=None,
                mode=None, getfilter=None, single=False, currentfilter=None):
    if filetypelist is None:
        fileTypeList = ['All Files (*)']
    else:
        fileTypeList = filetypelist
    if currentfilter is None:
        currentfilter = filetypelist[0]
    if message is None:
        if single:
            message = "Please select one file"
        else:
            message = "Please select one or more files"
    if mode is None:
        mode = "OPEN"
    else:
        mode = mode.upper()
    if mode == "OPEN":
        wdir = PyMcaDirs.inputDir
    else:
        wdir = PyMcaDirs.outputDir
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
    if native_possible and PyMcaDirs.nativeFileDialogs:
        filetypes = currentfilter+"\n"
        for filetype in fileTypeList:
            if filetype != currentfilter:
                filetypes += filetype+"\n"
        if getfilter:
            if mode == "OPEN":
                if single:
                    filelist, filterused = qt.QFileDialog.getOpenFileNameAndFilter(parent,
                        message,
                        wdir,
                        filetypes)
                    filelist =[filelist]
                else:
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
                if len(filelist[0]):
                    filterused = str(filelist[1])
                    filelist=[filelist[0]]
                else:
                    filelist = []
        else:
            if mode == "OPEN":
                if single:
                    filelist = [qt.QFileDialog.getOpenFileName(parent,
                            message,
                            wdir,
                            filetypes)]
                else:
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
        if hasattr(qt, "QStringList"):
            strlist = qt.QStringList()
        else:
            strlist = []
        strlist.append(currentfilter)
        for filetype in fileTypeList:
            if filetype != currentfilter:
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
            if single:
                filelist = [filelist[0]]
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
    filelist = [str(x) for x in  filelist]
    if not(len(filelist)):
        return []
    if mode == "OPEN":
        PyMcaDirs.inputDir = os.path.dirname(filelist[0])
        if PyMcaDirs.outputDir is None:
            PyMcaDirs.outputDir = os.path.dirname(filelist[0])
    else:
        PyMcaDirs.outputDir = os.path.dirname(filelist[0])
        if PyMcaDirs.inputDir is None:
            PyMcaDirs.inputDir = os.path.dirname(filelist[0])
    #do not sort file list
    #filelist.sort()
    if getfilter:
        return filelist, filterused
    else:
        return filelist

if __name__ == "__main__":
    app = qt.QApplication([])
    fileTypeList = ['PNG Files (*.png *.jpg)']
    print(getExistingDirectory())
    PyMcaDirs.nativeFileDialogs = False
    print(getExistingDirectory())
    PyMcaDirs.nativeFileDialogs = True
    print(getFileList(None, fileTypeList,"Please select a file", "SAVE", True, single=True))
    PyMcaDirs.nativeFileDialogs = False
    print(getFileList(None, fileTypeList,"Please select files", "LOAD", getfilter=False, single=False))
    #app.exec_()
