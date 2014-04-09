#/*##########################################################################
# Copyright (C) 2004-2014 European Synchrotron Radiation Facility
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
"""
Generic access to data sources.

"""
import sys
import os
from PyMca import PyMcaQt as qt
QTVERSION = qt.qVersion()

from PyMca import SpecFileDataSource
from PyMca import EdfFileDataSource
from PyMca import QEdfFileWidget
from PyMca import QSpecFileWidget

if sys.platform == "win32":
    source_types = { SpecFileDataSource.SOURCE_TYPE: SpecFileDataSource.SpecFileDataSource,
                     EdfFileDataSource.SOURCE_TYPE:  EdfFileDataSource.EdfFileDataSource}

    source_widgets = { SpecFileDataSource.SOURCE_TYPE: QSpecFileWidget.QSpecFileWidget,
                       EdfFileDataSource.SOURCE_TYPE: QEdfFileWidget.QEdfFileWidget}
    sps = None 
else:
    from PyMca import QSpsDataSource
    sps = QSpsDataSource.SpsDataSource.sps
    from PyMca import QSpsWidget
    source_types = { SpecFileDataSource.SOURCE_TYPE: SpecFileDataSource.SpecFileDataSource,
                     EdfFileDataSource.SOURCE_TYPE:  EdfFileDataSource.EdfFileDataSource,
                     QSpsDataSource.SOURCE_TYPE: QSpsDataSource.QSpsDataSource}

    source_widgets = { SpecFileDataSource.SOURCE_TYPE: QSpecFileWidget.QSpecFileWidget,
                       EdfFileDataSource.SOURCE_TYPE: QEdfFileWidget.QEdfFileWidget,
                       QSpsDataSource.SOURCE_TYPE: QSpsWidget.QSpsWidget}

NEXUS = True
try:
    from PyMca import NexusDataSource
    from PyMca import PyMcaNexusWidget
    import h5py
except:
    # HDF5 file format support is not mandatory
    NEXUS = False


if NEXUS:
    source_types[NexusDataSource.SOURCE_TYPE] = NexusDataSource.NexusDataSource
    source_widgets[NexusDataSource.SOURCE_TYPE] = PyMcaNexusWidget.PyMcaNexusWidget

def getSourceType(sourceName0):
    if type(sourceName0) == type([]):
        sourceName = sourceName0[0]
    else:
        sourceName = sourceName0
    if sps is not None:
        if sourceName in sps.getspeclist():
            return QSpsDataSource.SOURCE_TYPE
    if not os.path.exists(sourceName):
        if ('%' in sourceName):
            try:
                f = h5py.File(sourceName, 'r', driver='family')
                f.close()
                f = None
                return NexusDataSource.SOURCE_TYPE
            except:
                pass        
    if os.path.exists(sourceName):
        f = open(sourceName, 'rb')
        if sys.version < '3.0':
            line = f.readline()
            if not len(line.replace("\n","")):
                line = f.readline()
        else:
            tiff = False
            try:
                twoChars = f.read(2)
                if twoChars in [eval("b'II'"), eval("b'MM'")]:
                    f.close()
                    return EdfFileDataSource.SOURCE_TYPE
            except:
                pass
            f.seek(0)                
            try:
                line = str(f.readline().decode())
                if not len(line.replace("\n","")):
                    line = str(f.readline().decode())
            except UnicodeDecodeError:
                line = str(f.readline().decode("latin-1"))
                if not len(line.replace("\n","")):
                    line = str(f.readline().decode("latin-1"))
                            
        f.close()
        if sourceName.lower().endswith('.cbf'):
            #pilatus CBF
            mccd = True
        elif len(line) < 2:
            mccd = False
        elif line[0:2] in ["II","MM"]:
            #this also accounts for TIFF
            mccd = True
        elif sourceName.lower().endswith('.spe') and\
             (line[0:1] not in ['$', '#']):
            #Roper images
            mccd = True
        else:
            mccd = False
        if (line.startswith("{")) or mccd:
            return EdfFileDataSource.SOURCE_TYPE
        elif sourceName.lower().endswith('edf.gz') or\
             sourceName.lower().endswith('ccd.gz') or\
             sourceName.lower().endswith('raw.gz') or\
             sourceName.lower().endswith('edf.bz2') or\
             sourceName.lower().endswith('ccd.bz2') or\
             sourceName.lower().endswith('raw.bz2'):
            return EdfFileDataSource.SOURCE_TYPE
        else:
            if NEXUS:
                ishdf5 = False
                try:
                    ishdf5 = h5py.is_hdf5(sourceName)
                except TypeError:
                    if sys.version > '2.9':
                        if sourceName.endswith('.h5') or\
                           sourceName.endswith('.hdf') or\
                           sourceName.endswith('.nxs'):
                            ishdf5 = True
                if ishdf5:
                    return NexusDataSource.SOURCE_TYPE
                try:
                    f = h5py.File(sourceName, 'r')
                    f.close()
                    f = None
                    return NexusDataSource.SOURCE_TYPE
                except:
                    pass                
            return SpecFileDataSource.SOURCE_TYPE
    else:
        return QSpsDataSource.SOURCE_TYPE

def QDataSource(name=None, source_type=None):
    if name is None:
        raise ValueError("Invalid Source Name")
    if source_type is None:
        source_type = getSourceType(name)    
    try:
        sourceClass = source_types[source_type]
    except KeyError:
        #ERROR invalid source type
        raise TypeError("Invalid Source Type, source type should be one of %s" % source_types.keys())
    return sourceClass(name)
  
  
if __name__ == "__main__":
    try:
        sourcename=sys.argv[1]
        key       =sys.argv[2]        
    except:
        print("Usage: QDataSource <sourcename> <key>")
        sys.exit()
    #one can use this:
    #obj = EdfFileDataSource(sourcename)
    #or this:
    obj = QDataSource(sourcename)
    #data = obj.getData(key,selection={'pos':(10,10),'size':(40,40)})
    #data = obj.getData(key,selection={'pos':None,'size':None})
    data = obj.getDataObject(key)
    print("info = ",data.info)
    print("data shape = ",data.data.shape)


