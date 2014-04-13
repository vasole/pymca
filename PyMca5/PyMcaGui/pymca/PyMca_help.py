#/*##########################################################################
# Copyright (C) 2004-2014 E. Papillon, V.A. Sole, European Synchrotron Radiation Facility
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
__author__ = "E. Papillon & V.A. Sole - ESRF Software Group"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
FileOpen= \
""" Click this button to open a <em>new file</em>.<br><br>
You can also select the <b>Open</b> command from the <b>File</b> menu."""

FileSave= \
"""Click this button to save the file you are editing.<br><br>
You will be prompted for a filename.<br><br>
You can also select the <b>Save</b> command from the <b>File</b> menu."""

SpecOpen= \
"""<img source="spec">
Click this button to open a <em>new spec shared array</em>.<br><br>
You can also select then <b>Open Spec</b> command from the <b>File</b> menu."""

FilePrint= \
"""Click this button to print the file you are editing.<br><br>
You can also select the <b>Print</b> command from the <b>File</b> menu."""

FullScreen= \
"""<b>Maximize</b> current active window.<br>
The window will occupy all application window.
"""

NoFullScreen= \
"""Redisplay all windows using current window geometry.<br>
Window geometry could be:<br>
<b>Cascade</b>, <b>tile</b>, <b>tile horizontally</b> or <b>vertically</b>
"""

HelpDict= {
	"fileopen":	FileOpen,
	"filesave":	FileSave,
    "specopen":     SpecOpen,
	"fileprint":	FilePrint,
	"fullscreen":	FullScreen,
	"nofullscreen": NoFullScreen,
	
}
