import sys
import time
try:
    from PyMca import PyMcaQt as qt
except ImportError:
    import PyMcaQt as qt

class CalculationThread(qt.QThread):
    def __init__(self, parent=None, calculation_method=None):
        qt.QThread.__init__(self, parent)
        self.calculation_method = calculation_method

    def run(self):
        try:
            self.result = self.calculation_method()
        except:
            self.result = ("Exception",) + sys.exc_info()


def waitingMessageDialog(thread, message=None, parent=None):
    try:
        if message is None:
            message = "Please wait. Calculation going on." 
        msg = qt.QDialog(parent)#, qt.Qt.FramelessWindowHint)
        msg.setModal(1)
        msg.setWindowTitle("Please Wait")
        layout = qt.QHBoxLayout(msg)
        layout.setMargin(0)
        layout.setSpacing(0)
        l1 = qt.QLabel(msg)
        l1.setFixedWidth(l1.fontMetrics().width('##'))
        l2 = qt.QLabel(msg)
        l2.setText("%s" % message)
        l3 = qt.QLabel(msg)
        l3.setFixedWidth(l3.fontMetrics().width('##'))
        layout.addWidget(l1)
        layout.addWidget(l2)
        layout.addWidget(l3)
        msg.show()
        qt.qApp.processEvents()
        t0 = time.time()
        i = 0
        ticks = ['-','\\', "|", "/","-","\\",'|','/']
        while (thread.isRunning()):
            i = (i+1) % 8
            l1.setText(ticks[i])
            l3.setText(" "+ticks[i])
            qt.qApp.processEvents()
            time.sleep(2)
    finally:
        msg.close()
