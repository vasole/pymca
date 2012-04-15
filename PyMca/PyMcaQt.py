#/*##########################################################################
# Copyright (C) 2004-2012 European Synchrotron Radiation Facility
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
