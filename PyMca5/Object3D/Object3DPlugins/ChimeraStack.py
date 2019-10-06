#/*##########################################################################
# Copyright (C) 2004-2019 V.A. Sole, European Synchrotron Radiation Facility
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
import os
import h5py
import Object3DStack

from PyMca5.PyMcaGui import PyMcaFileDialogs
from PyMca5.PyMcaGui import PyMcaQt as qt

class ChimeraStack(Object3DStack.Object3DStack):
    pass

MENU_TEXT = '4D Chimera'
def getObject3DInstance(config=None):
    #for the time being a former configuration
    #for serializing purposes is not implemented

    #I do the import here for the case PyMca is not installed
    #because the modules could be instanstiated without using
    #this method
    try:
        from PyMca5.PyMcaIO import EDFStack
    except ImportError:
        import EDFStack

    fileTypeList = ['Chimera Stack (*cmp)',
                    'Chimera Stack (*)']
    old = PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs * 1
    #PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = False
    fileList, filterUsed = PyMcaFileDialogs.getFileList(
        parent=None,
        filetypelist=fileTypeList,
        message="Please select the object file(s)",
        mode="OPEN",
        getfilter=True)
    PyMcaFileDialogs.PyMcaDirs.nativeFileDialogs = old
    if not len(fileList):
        return None
    if filterUsed == fileTypeList[0]:
        fileindex = 2
    else:
        fileindex = 1
    #file index is irrelevant in case of an actual 3D stack.
    filename = fileList[0]
    legend = os.path.basename(filename)
    with h5py.File(filename, mode='r') as f:
        stack = f['Image']['data'][()]
    if stack is None:
        raise IOError("Problem reading stack.")
    object3D = ChimeraStack(name=legend)
    object3D.setStack(stack)
    return object3D

if __name__ == "__main__":
    import sys
    from Object3D import SceneGLWindow
    import os
    try:
        from PyMca5.PyMcaIO import EDFStack
        from PyMca5.PyMcaIO import EdfFile
    except ImportError:
        import EDFStack
        import EdfFile
    import getopt
    options = ''
    longoptions = ["fileindex=","begin=", "end="]
    try:
        opts, args = getopt.getopt(
                     sys.argv[1:],
                     options,
                     longoptions)
    except:
        print(sys.exc_info()[0])
        sys.exit(1)
    fileindex = 2
    begin = None
    end = None
    for opt, arg in opts:
        if opt in '--begin':
            begin = int(arg)
        elif opt in '--end':
            end = int(arg)
        elif opt in '--fileindex':
            fileindex = int(arg)
    app = qt.QApplication(sys.argv)
    window = SceneGLWindow.SceneGLWindow()
    window.show()
    if len(sys.argv) == 1:
        object3D = getObject3DInstance()
        if object3D is not None:
            window.addObject(object3D)
    else:
        filename = sys.argv[1]
        if not os.path.exists(filename):
            print("File does not exists")
            sys.exit(1)
        with h5py.File(filename, mode='r') as f:
            stack = f['Image']['data'][()]
        if stack is None:
            raise IOError("Problem reading stack.")
        object3D = ChimeraStack(name=os.path.basename(filename))
        object3D.setStack(stack)
        f = None
        window.addObject(object3D, os.path.basename(filename))
        window.setSelectedObject(os.path.basename(filename))

    window.glWidget.setZoomFactor(1.0)
    window.show()
    app.exec_()
