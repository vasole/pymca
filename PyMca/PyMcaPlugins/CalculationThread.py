import sys
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


