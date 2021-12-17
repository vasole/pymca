#/*##########################################################################
# Copyright (C) 2004-2019 European Synchrotron Radiation Facility
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

USE_MOVE_TO_THREAD = False
QTHREAD = True
if QTHREAD:
    QThread = qt.QThread
else:
    import threading
    QThread =  threading.Thread
    # Python threads do not support moveToThread (untested)
    USE_MOVE_TO_THREAD = False

class CalculationObject(qt.QObject):
    finished = qt.pyqtSignal()
    started = qt.pyqtSignal()
    def __init__(self, parent=None, calculation_method=None,
                 calculation_vars=None, calculation_kw=None,
                 expand_vars=True, expand_kw=True):
        qt.QObject.__init__(self, parent)
        self.calculation_method = calculation_method
        self.calculation_vars = calculation_vars
        self.calculation_kw = calculation_kw
        self.expand_vars = expand_vars
        self.expand_kw = expand_kw
        if self.expand_vars:
            if not self.expand_kw:
                raise ValueError("Cannot expand vars without expanding kw")
        self.__result = None

    def run(self):
        try:
            self.__result = None
            if self.calculation_vars is None and self.calculation_kw is None:
                self.__result = self.calculation_method()
            elif self.calculation_vars is None:
                if self.expand_kw:
                    self.__result = self.calculation_method(**self.calculation_kw)
                else:
                    self.__result = self.calculation_method(self.calculation_kw)
            elif self.calculation_kw is None:
                if self.expand_vars:
                    self.__result = self.calculation_method(*self.calculation_vars)
                else:
                    self.__result = self.calculation_method(self.calculation_vars)
            elif self.expand_vars and self.expand_kw:
                self.__result = self.calculation_method(*self.calculation_vars,
                                                      **self.calculation_kw)
            elif self.expand_kw:
                self.__result = self.calculation_method(self.calculation_vars,
                                                      **self.calculation_kw)
            else:
                print("Impossible combination of vars and kw")
                self._threadRunning = False
                raise ValueError("Impossible combination of vars and kw")
        except:
            self.__result = ("Exception",) + sys.exc_info()
        finally:
            # comment lines to allow to other call ????
            self.calculation_vars = None
            self.calculation_kw = None
            self.finished.emit()

    def getResult(self):
        return self.__result
        
class NewCalculationThread(qt.QObject):
    finished = qt.pyqtSignal()
    started = qt.pyqtSignal()
    def __init__(self, parent=None, calculation_method=None,
                 calculation_vars=None, calculation_kw=None,
                 expand_vars=True, expand_kw=True):
        qt.QObject.__init__(self, parent)
        self._calculationObject = CalculationObject(parent,
                                    calculation_method=calculation_method,
                                    calculation_vars=calculation_vars,
                                    calculation_kw=calculation_kw,
                                    expand_vars=expand_vars,
                                    expand_kw=expand_kw)
        self._calculationThread = qt.QThread()
        self._calculationObject.moveToThread(self._calculationThread)
        self._calculationObject.finished.connect(self._calculationThread.quit,
                                                qt.Qt.DirectConnection)
        self._calculationThread.started.connect(self._calculationObject.run,
                                                qt.Qt.DirectConnection)
        self._calculationThread.finished.connect(self._threadFinishedSlot,
                                                qt.Qt.DirectConnection)
        self._threadRunning = False

    def isRunning(self):
        #return self.calculationThread.isRunning()
        return self._threadRunning

    def _threadFinishedSlot(self):
        self._threadRunning = False
        self.finished.emit()

    def start(self):
        self._threadRunning = True
        self._calculationThread.start()
        self.started.emit()

    def getResult(self):
        return self._calculationObject.getResult()

    result = property(getResult)

class OldCalculationThread(QThread):
    def __init__(self, parent=None, calculation_method=None,
                 calculation_vars=None, calculation_kw=None,
                 expand_vars=True, expand_kw=True):
        if QTHREAD:
            QThread.__init__(self, parent)
        else:
            QThread.__init__(self)
            self._threadRunning = False
        self.calculation_method = calculation_method
        self.calculation_vars = calculation_vars
        self.calculation_kw = calculation_kw
        self.expand_vars = expand_vars
        self.expand_kw = expand_kw
        if self.expand_vars:
            if not self.expand_kw:
                raise ValueError("Cannot expand vars without expanding kw")

    if not QTHREAD:
        def isRunning(self):
            return self._threadRunning

    def run(self):
        try:
            self._threadRunning = True
            self.result = None
            if self.calculation_vars is None and self.calculation_kw is None:
                self.result = self.calculation_method()
            elif self.calculation_vars is None:
                if self.expand_kw:
                    self.result = self.calculation_method(**self.calculation_kw)
                else:
                    self.result = self.calculation_method(self.calculation_kw)
            elif self.calculation_kw is None:
                if self.expand_vars:
                    self.result = self.calculation_method(*self.calculation_vars)
                else:
                    self.result = self.calculation_method(self.calculation_vars)
            elif self.expand_vars and self.expand_kw:
                self.result = self.calculation_method(*self.calculation_vars,
                                                      **self.calculation_kw)
            elif self.expand_kw:
                self.result = self.calculation_method(self.calculation_vars,
                                                      **self.calculation_kw)
            else:
                print("Impossible combination of vars and kw")
                self._threadRunning = False
                raise ValueError("Impossible combination of vars and kw")
        except:
            self._threadRunning = False
            self.result = ("Exception",) + sys.exc_info()
        finally:
            self.calculation_vars = None
            self.calculation_kw = None
            self._threadRunning = False

    def getResult(self):
        if hasattr(self, "result"):
            return self.result
        else:
            return None

if USE_MOVE_TO_THREAD:
    CalculationThread=NewCalculationThread
else:
    CalculationThread=OldCalculationThread

def waitingMessageDialog(thread, message=None, parent=None,
                         modal=True, update_callback=None,
                         frameless=False):
    """
    thread  - The thread to be polled
    message - The initial message to be diplayed
    parent  - The parent QWidget. It is used just to provide a convenient localtion
    modal   - Default is True. The dialog will prevent user from using other widgets
    update_callback - The function to be called to provide progress feedback. It is expected
             to return a dictionary. The recognized key words are:
             message: The updated message to be displayed.
             title: The title of the window title.
             progress: A number between 0 and 100 indicating the progress of the task.
             status: Status of the calculation thread.
    """
    if message is None:
        message = "Please wait. Calculation going on."
    windowTitle = "Please Wait"
    if frameless:
        msg = qt.QDialog(None, qt.Qt.FramelessWindowHint)
    else:
        msg = qt.QDialog(None)

    #if modal:
    #    msg.setWindowFlags(qt.Qt.Window | qt.Qt.CustomizeWindowHint | qt.Qt.WindowTitleHint)
    msg.setModal(modal)
    msg.setWindowTitle(windowTitle)
    layout = qt.QHBoxLayout(msg)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    l1 = qt.QLabel(msg)
    l1.setFixedWidth(l1.fontMetrics().maxWidth()*len('##'))
    l2 = qt.QLabel(msg)
    l2.setText("%s" % message)
    l3 = qt.QLabel(msg)
    l3.setFixedWidth(l3.fontMetrics().maxWidth()*len('##'))
    layout.addWidget(l1)
    layout.addWidget(l2)
    layout.addWidget(l3)
    msg.show()
    if parent is not None:
        # place the dialog at appropriate point
        parentGeometry = parent.geometry()
        x = parentGeometry.x() + 0.5 * parentGeometry.width()
        y = parentGeometry.y() + 0.5 * parentGeometry.height()
        msg.move(int(x - 0.5 * msg.width()), int(y))
    t0 = time.time()
    i = 0
    ticks = ['-','\\', "|", "/","-","\\",'|','/']
    qApp = qt.QApplication.instance()
    if update_callback is None:
        while (thread.isRunning()):
            i = (i+1) % 8
            l1.setText(ticks[i])
            l3.setText(" "+ticks[i])
            qApp.processEvents()
            if QTHREAD:
                qt.QThread.msleep(1000)
            else:
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
            qApp.processEvents()
            if QTHREAD:
                qt.QThread.msleep(1000)
            else:
                time.sleep(2)
    msg.close()
