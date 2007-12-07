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
# is a problem for you.
#############################################################################*/
import os
import EdfFile

def save2DArrayListAsASCII(datalist, filename, labels = None, csv=False):
    if type(datalist) != type([]):
        datalist = [datalist]
    r, c = datalist[0].shape
    ndata = len(datalist)
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except:
            pass
    if labels is None:
        labels = []
        for i in range(len(datalist)):
            if csv:
                labels.append("Array_%d" % i)
            else:
                labels.append('"Array_%d"' % i)
    if len(labels) != len(datalist):
        raise ValueError, "Incorrect number of labels"
    if csv:
        header = '"row","column"'
        for label in labels:
            header +=';"%s"' % label
    else:
        header = "row  column"
        for label in labels:
            header +="  %s" % label
    filehandle=open(filename,'w+')
    filehandle.write('%s\n' % header)
    fileline=""
    if csv:
        for row in range(r):
            for col in range(c):
                fileline += "%d" % row
                fileline += ";%d" % col
                for i in range(ndata):
                    fileline +=";%g" % datalist[i][row, col]
                fileline += "\n"
                filehandle.write("%s" % fileline)
                fileline =""
    else:
        for row in range(r):
            for col in range(c):
                fileline += "%d" % row
                fileline += "  %d" % col
                for i in range(ndata):
                    fileline +="  %g" % datalist[i][row, col]
                fileline += "\n"
                filehandle.write("%s" % fileline)
                fileline =""
    filehandle.write("\n") 
    filehandle.close()

def save2DArrayListAsEDF(datalist, filename, labels = None):
    if type(datalist) != type([]):
        datalist = [datalist]
    ndata = len(datalist)
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except:
            pass
    if labels is None:
        labels = []
        for i in range(ndata):
            labels.append("Array_%d" % i) 
    if len(labels) != ndata:
        raise "ValueError", "Incorrect number of labels"
    edfout   = EdfFile.EdfFile(filename)
    for i in range(ndata):
        edfout.WriteImage ({'Title':labels[i]} , datalist[i], Append=1)
    del edfout #force file close
