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

class QSelectorWidget(qt.QWidget):
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.mainLayout = qt.QVBoxLayout(self)
        self._build()
        self._buildActions()

    def _build(self):
        """
        Method to be overwritten to build the main widget
        """
        if DEBUG:print "_build():Method to be overwritten"
        pass

    def _buildActions(self):
        self.buttonBox = qt.QWidget(self)
        buttonBox = self.buttonBox
        self.buttonBoxLayout = qt.QHBoxLayout(buttonBox)
        
        self.addButton = qt.QPushButton(buttonBox)
        self.addButton.setText("ADD")
        self.removeButton = qt.QPushButton(buttonBox)
        self.removeButton.setText("REMOVE")
        self.replaceButton = qt.QPushButton(buttonBox)
        self.replaceButton.setText("REPLACE")
        
        self.buttonBoxLayout.addWidget(self.addButton)
        self.buttonBoxLayout.addWidget(self.removeButton)
        self.buttonBoxLayout.addWidget(self.replaceButton)
        
        self.mainLayout.addWidget(buttonBox)
        
        self.connect(self.addButton, qt.SIGNAL("clicked()"), 
                    self._addClicked)

        self.connect(self.removeButton, qt.SIGNAL("clicked()"), 
                    self._removeClicked)

        self.connect(self.replaceButton, qt.SIGNAL("clicked()"), 
                    self._replaceClicked)

    def _addClicked(self):
        if DEBUG: print "_addClicked()"
    
    def _removeClicked(self):
        if DEBUG: print "_removeClicked()"     
        
    def _replaceClicked(self):
        if DEBUG: print "_replaceClicked()"

            
def test():
    app = qt.QApplication([])
    w = QSelectorWidget()
    w.show()
    qt.QObject.connect(app, qt.SIGNAL("lastWindowClosed()"),
                       app, qt.SLOT("quit()"))
    if QTVERSION < '4.0.0':
        app.exec_loop()
    else:
        app.exec_()
        
if __name__ == "__main__":
    test()
