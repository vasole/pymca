#/*##########################################################################
# Copyright (C) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMca; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# PyMca follows the dual licensing model of Riverbank's PyQt and cannot be
# used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#############################################################################*/
import sys
"""
This module simplifies writing code that has to deal with with PyQt and PyQt4.

In the future may also be used to choose between PyQt4 and PySide depending
on the one that has been previously chosen by the end user.

"""
# force cx_freeze to consider sip among the modules to add
# to the binary packages
import sip
if 'qt' not in sys.modules:
    try:
        from PyQt4.QtCore import *
        from PyQt4.QtGui import *
        try:
            # In case PyQwt is compiled with QtSvg this forces
            # cx_freeze to add PyQt4.QtSvg to the list of modules
            from PyQt4.QtSvg import *
        except:
            pass
    except:
        from qt import *
else:
    from qt import *

# Overwrite the QFileDialog to make sure that by default it 
# returns non-native dialogs as it was the traditional behavior of Qt
_QFileDialog = QFileDialog
class QFileDialog(_QFileDialog):
    def __init__(self, *args, **kwargs):
        _QFileDialog.__init__(self, *args, **kwargs)
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

if sys.version < '3.0':
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
