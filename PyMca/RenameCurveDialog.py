import PyQt4.Qt as qt

class RenameCurveDialog(qt.QDialog):
    def __init__(self, parent = None, current="", curves = []):
	qt.QDialog.__init__(self, parent)
	self.setWindowTitle("Rename Curve %s" % current)
	self.curves = curves
	layout = qt.QVBoxLayout(self)
	self.lineEdit = qt.QLineEdit(self)
	self.lineEdit.setText(current)
	self.hbox = qt.QWidget(self)
	self.hboxLayout = qt.QHBoxLayout(self.hbox)
	self.hboxLayout.addWidget(HorizontalSpacer(self.hbox))
	self.okButton    = qt.QPushButton(self.hbox)
	self.okButton.setText('OK')
	self.hboxLayout.addWidget(self.okButton)
	self.cancelButton = qt.QPushButton(self.hbox)
	self.cancelButton.setText('Dismiss')
	self.hboxLayout.addWidget(self.cancelButton)
	self.hboxLayout.addWidget(HorizontalSpacer(self.hbox))
	layout.addWidget(self.lineEdit)
	layout.addWidget(self.hbox)
	self.connect(self.okButton, qt.SIGNAL('clicked()'), self.preAccept)
	self.connect(self.cancelButton, qt.SIGNAL('clicked()'), self.reject)

    def preAccept(self):
	text = str(self.lineEdit.text())
	addedText = "" 
	if len(text):
	    if text not in self.curves:
		self.accept()
		return
	    else:
		addedText = "Curve already exists."
	text = "Invalid Curve Name"
	msg = qt.QMessageBox(self)
	msg.setIcon(qt.QMessageBox.Critical)
	msg.setWindowTitle(text)
	text += "\n%s" % addedText
	msg.setText(text)
	msg.exec_()

    def getText(self):
	return str(self.lineEdit.text())
	
class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
      
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                           qt.QSizePolicy.Fixed))


if __name__ == "__main__":
   app = qt.QApplication([])
   w=RenameCurveDialog(None, 'curve1', ['curve1', 'curve2', 'curve3'])
   ret = w.exec_()
   if ret == qt.QDialog.Accepted:
       print "newcurve = ", str(w.lineEdit.text())
   else:
       print "keeping old curve"
