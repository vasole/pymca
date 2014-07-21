#/*##########################################################################
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
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import time
try:
    from PyMca5.PyMcaGui import PyMcaQt as qt
except ImportError:
    import PyMcaQt as qt

class CalculationThread(qt.QThread):
    def __init__(self, parent=None, calculation_method=None,
                 calculation_vars=None, calculation_kw=None):
        qt.QThread.__init__(self, parent)
        self.calculation_method = calculation_method
        self.calculation_vars = calculation_vars
        self.calculation_kw = calculation_kw
        self.result = None

    def run(self):
        try:
            if self.calculation_vars is None and self.calculation_kw is None:
                self.result = self.calculation_method()
            elif self.calculation_vars is None:
                self.result = self.calculation_method(**self.calculation_kw)
            elif self.calculation_kw is None:
                self.result = self.calculation_method(*self.calculation_vars)
            else:
                self.result = self.calculation_method(*self.calculation_vars,
                                                      **self.calculation_kw)
        except:
            self.result = ("Exception",) + sys.exc_info()
        finally:
            self.calculation_vars = None
            self.calculation_kw = None

def waitingMessageDialog(thread, message=None, parent=None, modal=True, update_callback=None):
    """
    thread  - The thread to be polled
    message - The initial message to be diplayed
    parent  - The parent QWidget. It is used just to provide a convenient localtion
    modal   - Default is True. The dialog will prevent user from using other widgets
    update_callback - The function to be called to provide progress feedback. It is expected
             to return a dictionnary. The recognized key words are:
             message: The updated message to be displayed.
             title: The title of the window title.
             progress: A number between 0 and 100 indicating the progress of the task.
             status: Status of the calculation thread.
    """
    try:
        if message is None:
            message = "Please wait. Calculation going on."
        windowTitle = "Please Wait"
        msg = qt.QDialog(parent)#, qt.Qt.FramelessWindowHint)
        #if modal:
        #    msg.setWindowFlags(qt.Qt.Window | qt.Qt.CustomizeWindowHint | qt.Qt.WindowTitleHint)
        msg.setModal(modal)
        msg.setWindowTitle(windowTitle)
        layout = qt.QHBoxLayout(msg)
        layout.setContentsMargins(0, 0, 0, 0)
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
        qApp = qt.QApplication.instance()
        qApp.processEvents()
        t0 = time.time()
        i = 0
        ticks = ['-','\\', "|", "/","-","\\",'|','/']
        if update_callback is None:
            while (thread.isRunning()):
                i = (i+1) % 8
                l1.setText(ticks[i])
                l3.setText(" "+ticks[i])
                qApp = qt.QApplication.instance()
                qApp.processEvents()
                time.sleep(2)
        else:
            while (thread.isRunning()):
                updateInfo = update_callback()
                message = updateInfo.get('message', message)
                windowTitle = updateInfo.get('title', windowTitle)
                msg.setWindowTitle(windowTitle)
                i = (i+1) % 8
                l1.setText(ticks[i])
                l2.setText(message)
                l3.setText(" "+ticks[i])
                qApp = qt.QApplication.instance()
                qApp.processEvents()
                time.sleep(2)
    finally:
        msg.close()
