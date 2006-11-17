import sys
if 'qt' not in sys.modules:
    try:
        import PyQt4.Qt as qt
    except:
        import qt
else:
    import qt
    
QTVERSION = qt.qVersion()
DEBUG = 0
if QTVERSION < '4.0.0':
    import Q3PyMcaPrintPreview as PrintPreview
else:
    from Q4PyMcaPrintPreview import PyMcaPrintPreview as PrintPreview
    
#SINGLETON
class PyMcaPrintPreview(PrintPreview):
    _instance = None
    def __new__(self, *var, **kw):
        if self._instance is None:
            self._instance = PrintPreview.__new__(self,*var, **kw)
        return self._instance    

def testPreview():
    """
    """
    import sys
    import os.path

    if len(sys.argv) < 2:
        print "give an image file as parameter please."
        sys.exit(1)

    if len(sys.argv) > 2:
        print "only one parameter please."
        sys.exit(1)

    filename = sys.argv[1]

    a = qt.QApplication(sys.argv)
 
    p = qt.QPrinter()
    p.setOutputFileName(os.path.splitext(filename)[0]+".ps")
    p.setColorMode(qt.QPrinter.Color)

    w = PyMcaPrintPreview( parent = None, printer = p, name = 'Print Prev',
                      modal = 0, fl = 0)
    w.resize(400,500)
    w.addPixmap(qt.QPixmap.fromImage(qt.QImage(filename)))
    w.addImage(qt.QImage(filename))
    if 0:
        w2 = PyMcaPrintPreview( parent = None, printer = p, name = '2Print Prev',
                      modal = 0, fl = 0)
        w.exec_()
        w2.resize(100,100)
        w2.show()
        sys.exit(w2.exec_())
    else:
        sys.exit(w.exec_())
    
if  __name__ == '__main__':
    testPreview()

 
 
