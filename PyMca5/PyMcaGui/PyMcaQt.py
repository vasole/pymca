#/*##########################################################################
# Copyright (C) 2004-2022 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
__author__ = "V.A. Sole - ESRF"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import os
import sys
import traceback
import logging

"""
This module simplifies writing code that has to deal with with PySideX and PyQtX.

"""

_logger = logging.getLogger(__name__)

BINDING = None
"""The name of the Qt binding in use: PyQt5, PyQt4, PySide2, PySide6"""

HAS_SVG = False
"""True if Qt provides support for Scalable Vector Graphics (QtSVG)."""

HAS_OPENGL = False
"""True if Qt provides support for OpenGL (QtOpenGL)."""

# force cx_freeze to consider sip among the modules to add
# to the binary packages
if 'PySide2.QtCore' in sys.modules:
    BINDING = 'PySide2'

elif 'PySide6.QtCore' in sys.modules:
    BINDING = 'PySide6'

elif 'PyQt5.QtCore' in sys.modules:
    BINDING = 'PyQt5'

elif 'PyQt4.QtCore' in sys.modules:
    BINDING = 'PyQt4'
    _logger = logging.critical("PyQt4 already imported and not supported")

elif hasattr(sys, 'argv') and ('--binding=PySide2' in sys.argv):
    BINDING = 'PySide2'

elif hasattr(sys, 'argv') and ('--binding=PySide6' in sys.argv):
    # argv might not be defined for embedded python (e.g., in Qt designer)
    BINDING = 'PySide6'

if BINDING is None: # Try the different bindings
    try:
        import PyQt5.QtCore
        BINDING = "PyQt5"
    except ImportError:
        if "PyQt5" in sys.modules:
            del sys.modules["PyQt5"]
        try:
            import PySide2.QtCore
            BINDING = "PySide2"
        except ImportError:
            if "PySide2" in sys.modules:
                del sys.modules["PySide2"]
            try:
                import PySide6.QtCore
                BINDING = "PySide6"
            except ImportError:
                if 'PySide6' in sys.modules:
                    del sys.modules["PySide6"]
                raise ImportError(
                    'No Qt wrapper found. Install PyQt5, PySide2 or PySide6.')

_logger.info("BINDING set to %s" % BINDING)

if BINDING == "PyQt5":
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtPrintSupport import *

    try:
        from PyQt5.QtOpenGL import *
        HAS_OPENGL = True
    except:
        _logger.info("PyQt5.QtOpenGL not available")

    try:
        from PyQt5.QtSvg import *
        HAS_SVG = True
    except:
        _logger.info("PyQt5.QtSVG not available")

    Signal = pyqtSignal
    Slot = pyqtSlot

elif BINDING == "PySide2":
    # try PySide2 (experimental)
    from PySide2.QtCore import *
    from PySide2.QtGui import *
    from PySide2.QtWidgets import *
    from PySide2.QtPrintSupport import *

    try:
        from PySide2.QtOpenGL import *
        HAS_OPENGL = True
    except:
        _logger.info("PySide2.QtOpenGL not available")


    try:
        from PySide2.QtSvg import *
        HAS_SVG = True
    except:
        _logger.info("PySide2.QtSVG not available")
    pyqtSignal = Signal
    pyqtSlot = Slot

    # Qt6 compatibility:
    # with PySide2 `exec` method has a special behavior
    class _ExecMixIn:
        """Mix-in class providind `exec` compatibility"""
        def exec(self, *args, **kwargs):
            return super().exec_(*args, **kwargs)

    # QtWidgets
    QApplication.exec = QApplication.exec_
    class QColorDialog(_ExecMixIn, QColorDialog): pass
    class QDialog(_ExecMixIn, QDialog): pass
    class QErrorMessage(_ExecMixIn, QErrorMessage): pass
    class QFileDialog(_ExecMixIn, QFileDialog): pass
    class QFontDialog(_ExecMixIn, QFontDialog): pass
    class QInputDialog(_ExecMixIn, QInputDialog): pass
    class QMenu(_ExecMixIn, QMenu): pass
    class QMessageBox(_ExecMixIn, QMessageBox): pass
    class QProgressDialog(_ExecMixIn, QProgressDialog): pass
    #QtCore
    class QCoreApplication(_ExecMixIn, QCoreApplication): pass
    class QEventLoop(_ExecMixIn, QEventLoop): pass
    if hasattr(QTextStreamManipulator, "exec_"):
        # exec_ only wrapped in PySide2 and NOT in PyQt5
        class QTextStreamManipulator(_ExecMixIn, QTextStreamManipulator): pass
    class QThread(_ExecMixIn, QThread): pass

    # workaround not finding the Qt platform plugin "windows" in "" error
    # when creating a QApplication
    if sys.platform.startswith("win")and QApplication.instance() is None:
        _platform_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH",
                                                 None)
        if _platform_plugin_path:
            if not os.path.exists(_platform_plugin_path):
                _logger.info("QT_QPA_PLATFORM_PLUGIN_PATH <%s> ignored" % \
                             _platform_plugin_path)
                _platform_plugin_path = None
        if not _platform_plugin_path:
            import PySide2
            _platform_plugin_path = os.path.join( \
                            os.path.dirname(PySide2.__file__),
                            "plugins", "platforms")
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = \
                                                _platform_plugin_path
            _logger.info("QT_QPA_PLATFORM_PLUGIN_PATH set to <%s>" % \
                             _platform_plugin_path)
elif BINDING == 'PySide6':
    _logger.debug('Using PySide6 bindings')

    import PySide6 as QtBinding  # noqa

    from PySide6.QtCore import *  # noqa
    from PySide6.QtGui import *  # noqa
    from PySide6.QtWidgets import *  # noqa
    from PySide6.QtPrintSupport import *  # noqa

    try:
        from PySide6.QtOpenGL import *  # noqa
        from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa
    except ImportError:
        _logger.info("PySide6.QtOpenGL not available")
        HAS_OPENGL = False
    else:
        HAS_OPENGL = True

    try:
        from PySide6.QtSvg import *  # noqa
    except ImportError:
        _logger.info("PySide6.QtSvg not available")
        HAS_SVG = False
    else:
        HAS_SVG = True

    pyqtSignal = Signal

    # use a (bad) replacement for QDesktopWidget
    class QDesktopWidget:
        def height(self):
            _logger.info("Using obsolete classes")
            screen = QApplication.instance().primaryScreen() 
            return screen.availableGeometry().height()


        def width(self):
            _logger.info("Using obsolete classes")
            screen = QApplication.instance().primaryScreen() 
            return screen.availableGeometry().width()

else:
    raise ImportError('No Qt wrapper found. Install one of PyQt5, PySide2 or PySide6')

# provide a exception handler but not implement it by default
def exceptionHandler(type_, value, trace):
    _logger.error("%s %s %s", type_, value, ''.join(traceback.format_tb(trace)))
    if QApplication.instance():
        msg = QMessageBox()
        msg.setWindowTitle("Unhandled exception")
        msg.setIcon(QMessageBox.Critical)
        msg.setInformativeText("%s %s\nPlease report details" % (type_, value))
        msg.setDetailedText(("%s " % value) + \
                            ''.join(traceback.format_tb(trace)))
        msg.raise_()
        msg.exec()

# Overwrite the QFileDialog to make sure that by default it
# returns non-native dialogs as it was the traditional behavior of Qt
_QFileDialog = QFileDialog
class QFileDialog(_QFileDialog):
    def __init__(self, *args, **kwargs):
        try:
            _QFileDialog.__init__(self, *args, **kwargs)
        except:
            # not all versions support kwargs
            _QFileDialog.__init__(self, *args)
        try:
            self.setOptions(_QFileDialog.DontUseNativeDialog)
        except:
            print("WARNING: Cannot force default QFileDialog behavior")

class HorizontalSpacer(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Fixed))

class VerticalSpacer(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,
                                          QSizePolicy.Expanding))
_QToolButton = QToolButton
class QToolButton(_QToolButton):
    def __init__(self, *var, **kw):
        _QToolButton.__init__(self, *var, **kw)
        if "silx" in sys.modules:
            try:
                # this should be set via a user accessible parameter
                tb = QToolBar()
                size = tb.iconSize()
                if (size.width() > 15) and (size.height() > 15):
                    self.setIconSize(size)
            except:
                print("unable")
                pass

if sys.version_info < (3,):
    import types
    # perhaps a better name would be safe unicode?
    # should this method be a more generic tool to
    # be found outside PyMcaQt?
    def safe_str(potentialQString):
        if type(potentialQString) == types.StringType or\
           type(potentialQString) == types.UnicodeType:
            return potentialQString
        try:
            # default, just str
            x = str(potentialQString)
        except UnicodeEncodeError:

            # try user OS file system encoding
            # expected to be 'mbcs' under windows
            # and 'utf-8' under MacOS X
            try:
                x = unicode(potentialQString, sys.getfilesystemencoding())
                return x
            except:
                # on any error just keep going
                pass
            # reasonable tries are 'utf-8' and 'latin-1'
            # should I really go beyond those?
            # In fact, 'utf-8' is the default file encoding for python 3
            encodingOptions = ['utf-8', 'latin-1', 'utf-16', 'utf-32']
            for encodingOption in encodingOptions:
                try:
                    x = unicode(potentialQString, encodingOption)
                    break
                except UnicodeDecodeError:
                    if encodingOption == encodingOptions[-1]:
                        raise
        return x
else:
    safe_str = str


class CLocaleQDoubleValidator(QDoubleValidator):
    """
    A QDoubleValidator using C locale
    """
    def __init__(self, *var):
        QDoubleValidator.__init__(self, *var)
        self._localeHolder = QLocale("C")
        self.setLocale(self._localeHolder)
