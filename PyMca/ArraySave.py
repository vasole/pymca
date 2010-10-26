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
import numpy
import time

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

def getHDF5FileInstanceAndBuffer(filename, shape,
                                 buffername="data",
                                 dtype=numpy.float32):
    if not HDF5:
        raise IOError, 'h5py does not seem to be installed in your system'

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except:
            raise IOError, "Cannot overwrite existing file!"
    hdf = phynx.File(filename, 'a')
    entryName = "data"
    nxEntry = hdf.require_group(entryName, type='Entry')
    nxEntry.require_dataset('title', data = "PyMca saved 3D Array")
    nxEntry['start_time'] = getDate()
    nxData = nxEntry.require_group('NXdata', type='Data')
    data = nxData.require_dataset(buffername,
                           shape=shape,
                           dtype=dtype,
                           chunks=(1, shape[1], shape[2]))
    data.attrs['signal'] = 1
    for i in range(len(shape)):
        dim = numpy.arange(shape[i]).astype(numpy.float32)
        dset = nxData.require_dataset('dim_%d' % i,
                               dim.shape,
                               dim.dtype,
                               dim,
                               chunks=dim.shape)
        dset.attrs['axis'] = i + 1
    nxEntry['end_time'] = getDate()
    return hdf, data

def save3DArrayAsHDF5(data, filename, labels = None, dtype=None, mode='nexus',
                      mcaindex=-1, interpretation=None):
    if not HDF5:
        raise IOError, 'h5py does not seem to be installed in your system'
    if (mcaindex == 0) and (interpretation in ["spectrum", None]):
        #stack of images to be saved as stack of spectra
        modify = True
        shape  = [data.shape[1], data.shape[2], data.shape[0]]
    elif (mcaindex != 0 ) and (interpretation in ["image"]):
        #stack of spectra to be saved as stack of images
        modify = True
        shape = [data.shape[2], data.shape[0], data.shape[1]]
    else:
        modify = False
        shape = data.shape
    if dtype is None:
        dtype =data.dtype
    if mode.lower() in ['nexus', 'nexus+']:
        #raise IOError, 'NeXus data saving not implemented yet'
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                raise IOError, "Cannot overwrite existing file!"
        hdf = phynx.File(filename, 'a')
        entryName = "data"
        nxEntry = hdf.require_group(entryName, type='Entry')
        nxEntry.require_dataset('title', data = "PyMca saved 3D Array")
        nxEntry['start_time'] = getDate()
        nxData = nxEntry.require_group('NXdata', type='Data')
        if modify:
            if interpretation in ["image"]:
                dset = nxData.require_dataset('data',
                                   shape=shape,
                                   dtype=dtype,
                                   chunks=(1, shape[1], shape[2]))
                for i in range(data.shape[-1]):
                    tmp = data[:,:, i:i+1]
                    tmp.shape = 1, shape[1], shape[2]
                    dset[i, :, :] = tmp
            elif 0:
                #if I do not match the input and output shapes it takes ages
                #to save the images as spectra. However, it is much faster
                #when performing spectra operations.
                dset = nxData.require_dataset('data',
                               shape=shape,
                               dtype=dtype,
                               chunks=(1, shape[1], shape[2]))
                for i in range(data.shape[1]): #shape[0]
                    chunk = numpy.zeros((1, data.shape[2], data.shape[0]), dtype)
                    for k in range(data.shape[0]): #shape[2]
                        if 0:
                            tmpData = data[k:k+1]
                            for j in range(data.shape[2]): #shape[1]
                                tmpData.shape = data.shape[1], data.shape[2]
                                chunk[0, j, k] = tmpData[i, j]
                        else:
                            tmpData = data[k:k+1, i, :]
                            tmpData.shape = -1
                            chunk[0, :, k] = tmpData
                    print "Saving item ", i, "of ", data.shape[1]
                    dset[i, :, :] = chunk
            else:
                #if I do not match the input and output shapes it takes ages
                #to save the images as spectra. This is a very fast saving, but
                #the perfromance is awful when reading.
                dset = nxData.require_dataset('data',
                               shape=shape,
                               dtype=dtype,
                               chunks=(shape[0], shape[1], 1))
                for i in range(data.shape[0]):
                    tmp = data[i:(i+1),:,:]
                    tmp.shape = shape[0], shape[1], 1
                    dset[:, :, i:(i+1)] = tmp
        else:
            dset = nxData.require_dataset('data',
                               shape=shape,
                               dtype=dtype,
                               data=data,
                               chunks=(1, shape[1], shape[2]))
        dset.attrs['signal'] = "1"
        if interpretation is not None:
            dset.attrs['interpretation'] = interpretation
        for i in range(len(shape)):
            dim = numpy.arange(shape[i]).astype(numpy.float32)
            dset = nxData.require_dataset('dim_%d' % i,
                                   dim.shape,
                                   dim.dtype,
                                   dim,
                                   chunks=dim.shape)
            dset.attrs['axis'] = i + 1
        nxEntry['end_time'] = getDate()
        if mode.lower() == 'nexus+':
            #create link
            g = h5py.h5g.open(hdf.fid, '/')
            g.link('/data/NXdata/data', '/data/data', h5py.h5g.LINK_HARD)
        
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

def getDate():
    localtime = time.localtime()
    #year, month, day, hour, minute, second,\
    #      week_day, year_day, delta = time.localtime()
    year   = localtime[0]
    month  = localtime[1]
    day    = localtime[2]
    hour   = localtime[3]
    minute = localtime[4]
    second = localtime[5]
    return "%4d-%02d-%02d %02d:%02d:%02d" % (year, month, day, hour, minute, second)


if __name__ == "__main__":
    a=numpy.arange(1000000.)
    a.shape = 20, 50, 1000
    save3DArrayAsHDF5(a, '/test.h5', mode='nexus')
