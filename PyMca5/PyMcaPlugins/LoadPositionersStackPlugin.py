# /*#########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""
This plugin is used to load motor positions from a CSV or HDF5 file.

The number of values associated with a given positioner should be equal
to the number of pixels in the stack image. A single scalar value can also be
provided for a motor, if it didn't move during the experiment.

A CSV file should have unique motor names in the header line, and can have
an arbitrary number of motors/columns.

Motor positions in a HDF5 files are 1-dimensional datasets whose names are
the motor names. The user is allowed to select the HDF5 group containing all
motor datasets.

Data loaded with this plugin can then be used by other tools, such as the
"Stack motor positions" plugin.
"""
__authors__ = ["P. Knobel"]
__license__ = "MIT"

import logging

from PyMca5 import StackPluginBase
from PyMca5.PyMcaGui.io.hdf5.HDF5Widget import getGroupNameDialog
from PyMca5.PyMcaGui.io import PyMcaFileDialogs
from PyMca5.PyMcaIO import specfilewrapper

try:
    import h5py
except ImportError:
    h5py = None

try:
    # if silx is available, we can open SPEC
    from silx.io import open as silx_open
    h5open = silx_open
    from silx.io import is_dataset
except ImportError:
    silx_open = None
    if h5py is not None:
        # at least we can open hdf5 files
        def h5open(filename):
            return h5py.File(filename, "r")

        def is_dataset(item):
            return isinstance(item, h5py.Dataset)
    else:
        # no luck, only CSV files available
        h5open = None
        is_dataset = None


# suppress errors and warnings if fabio is missing
if silx_open is not None:
    logging.getLogger("silx.io.fabioh5").setLevel(logging.CRITICAL)

_logger = logging.getLogger(__name__)


class LoadPositionersStackPlugin(StackPluginBase.StackPluginBase):
    def __init__(self, stackWindow):
        if _logger.getEffectiveLevel() == logging.DEBUG:
            StackPluginBase.pluginBaseLogger.setLevel(logging.DEBUG)
        StackPluginBase.StackPluginBase.__init__(self, stackWindow)
        self.methodDict = {'Load positioners': [self._loadFromFile,
                                                "Load positioners from file"]}
        self.__methodKeys = ['Load positioners']

    #Methods implemented by the plugin
    def getMethods(self):
        return self.__methodKeys

    def getMethodToolTip(self, name):
        return self.methodDict[name][1]

    def getMethodPixmap(self, name):
        return self.methodDict[name][2]

    def applyMethod(self, name):
        return self.methodDict[name][0]()

    def _loadFromFile(self):
        stack = self.getStackDataObject()
        if stack is None:
            return
        mcaIndex = stack.info.get('McaIndex')
        if not (mcaIndex in [0, -1, 2]):
            raise IndexError("1D index must be 0, 2, or -1")

        # test io dependencies
        if h5py is None:
            filefilter = []
        else:
            filefilter = ['HDF5 (*.h5 *.nxs *.hdf *.hdf5)']
        filefilter.append('CSV (*.csv *.txt)')
        if silx_open is not None:
            filefilter.append('Any (*)')

        filename, ffilter = PyMcaFileDialogs.\
                    getFileList(parent=None,
                        filetypelist=filefilter,
                        message='Load',
                        mode='OPEN',
                        single=True,
                        getfilter=True,
                        currentfilter=filefilter[0])
        if len(filename):
            _logger.debug("file name = %s file filter = %s", filename, ffilter)
        else:
            _logger.debug("nothing selected")
            return
        filename = filename[0]

        positioners = {}
        if not ffilter.startswith('CSV'):
            h5GroupName = getGroupNameDialog(filename)
            if h5GroupName is None:
                return
            with h5open(filename) as h5f:
                h5Group = h5f[h5GroupName]
                positioners = {}
                for dsname in h5Group:
                    # links and subgroups just ignored for the time being
                    if not is_dataset(h5Group[dsname]):
                        continue
                    positioners[dsname] = h5Group[dsname][()]
        else:
            sf = specfilewrapper.Specfile(filename)
            scan = sf[0]
            labels = scan.alllabels()
            data = scan.data()
            scan = None
            sf = None
            for i, label in enumerate(labels):
                positioners[label] = data[i, :]

        self._stackWindow.setPositioners(positioners)


MENU_TEXT = "Load positioners from file"


def getStackPluginInstance(stackWindow, **kw):
    ob = LoadPositionersStackPlugin(stackWindow)
    return ob
