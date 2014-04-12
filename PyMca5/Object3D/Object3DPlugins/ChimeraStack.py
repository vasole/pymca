import os
import h5py
import Object3DStack
try:
    from PyMca5.Object3D import Object3DFileDialogs
except ImportError:
    from Object3D import Object3DFileDialogs
qt = Object3DFileDialogs.qt

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
    old = Object3DFileDialogs.Object3DDirs.nativeFileDialogs * 1
    #Object3DFileDialogs.Object3DDirs.nativeFileDialogs = False
    fileList, filterUsed = Object3DFileDialogs.getFileList(None, fileTypeList,
                                               "Please select the object file(s)",
                                               "OPEN",
                                               True)
    Object3DFileDialogs.Object3DDirs.nativeFileDialogs = old
    if not len(fileList):
        return None
    if filterUsed == fileTypeList[0]:
        fileindex = 2
    else:
        fileindex = 1
    #file index is irrelevant in case of an actual 3D stack.
    filename = fileList[0]
    legend = os.path.basename(filename)
    f = h5py.File(filename)
    stack = f['Image']['data'].value
    f = None
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
    except getopt.error,msg:
        print msg
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
            print "File does not exists"
            sys.exit(1)
        f = h5py.File(filename)
        stack = f['Image']['data'].value
        if stack is None:
            raise IOError, "Problem reading stack."
        object3D = ChimeraStack(name=os.path.basename(filename))
        object3D.setStack(stack)
        f = None
        window.addObject(object3D, os.path.basename(filename))
        window.setSelectedObject(os.path.basename(filename))
            
    window.glWidget.setZoomFactor(1.0)
    window.show()
    app.exec_()
