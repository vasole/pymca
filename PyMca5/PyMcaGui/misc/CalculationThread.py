#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
        qt.qApp.processEvents()
        t0 = time.time()
        i = 0
        ticks = ['-','\\', "|", "/","-","\\",'|','/']
        if update_callback is None:
            while (thread.isRunning()):
                i = (i+1) % 8
                l1.setText(ticks[i])
                l3.setText(" "+ticks[i])
                qt.qApp.processEvents()
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
                qt.qApp.processEvents()
                time.sleep(2)            
    finally:
        msg.close()
