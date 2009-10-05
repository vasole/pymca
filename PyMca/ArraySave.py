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


HDF5 = True
try:
    import h5py
    PHYNX=True
    try:
        from PyMca import phynx
    except ImportError:
        try:
            import phynx
        except ImportError:
            PHYNX = False
except ImportError:
    HDF5 = False

"""
try:
    from PyMca import phynx
except ImportError:
    try:
        import phynx
    except ImportError:
        HDF5 = False
"""

def save2DArrayListAsASCII(datalist, filename, labels = None, csv=False, csvseparator=";"):
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
            labels.append("Array_%d" % i)
    if len(labels) != len(datalist):
        raise ValueError, "Incorrect number of labels"
    if csv:
        header = '"row"%s"column"' % csvseparator
        for label in labels:
            header +='%s"%s"' % (csvseparator,label)
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
                fileline += "%s%d" % (csvseparator,col)
                for i in range(ndata):
                    fileline +="%s%g" % (csvseparator, datalist[i][row, col])
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

def save2DArrayListAsEDF(datalist, filename, labels = None, dtype=None):
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
        raise ValueError, "Incorrect number of labels"
    edfout   = EdfFile.EdfFile(filename)
    for i in range(ndata):
        if dtype is None:
            edfout.WriteImage ({'Title':labels[i]} , datalist[i], Append=1)
        else:
            edfout.WriteImage ({'Title':labels[i]} ,
                               datalist[i].astype(dtype),
                               Append=1)
    del edfout #force file close


def save3DArrayAsHDF5(data, filename, labels = None, dtype=None, mode='nexus'):
    if not HDF5:
        raise IOError, 'h5py does not seem to be installed in your system'
    shape = data.shape
    if dtype is None:
        dtype =data.dtype
    if mode.lower() == 'nexus':
        raise IOError, 'NeXus data saving not implemented yet'
        hdf = phynx.File(filename, 'a')
        entryName = "%4d.%d %s" % (1, 1, 'title')
        nxEntry = hdf.require_group(entryName, type='Entry')
        #nxEntry.require_dataset('title', data = XXX)
        #nxEntry['start_time'] = get_date()
        #nxData = hdf.require_group(entryName, type='Data')
        #nxData.require_dataset('data', dtype=dtype, data, chunksize=10)
    elif mode.lower() == 'simplest':
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                raise IOError, "Cannot overwrite existing file!"
        hdf = h5py.File(filename, 'a')
        hdf.require_dataset('data',
                           shape=shape,
                           dtype=dtype,
                           data=data,
                           chunks=(1, shape[1], shape[2]))
        hdf.flush()
    else:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                raise IOError, "Cannot overwrite existing file!"
        shape = data.shape
        dtype = data.dtype
        hdf = h5py.File(filename, 'a')
        dataGroup = hdf.require_group('data')
        dataGroup.require_dataset('data',
                           shape=shape,
                           dtype=dtype,
                           data=data,
                           chunks=(1, shape[1], shape[2]))
        hdf.flush()
    hdf.close()


if __name__ == "__main__":
    import numpy
    a=numpy.arange(1000000.)
    a.shape = 20, 50, 1000
    save3DArrayAsHDF5(a, '/test.h5', mode='simplest')
