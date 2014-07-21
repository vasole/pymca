#/*##########################################################################
# Copyright (C) 2004-2014 E. Papillon, V.A. Sole, European Synchrotron Radiation Facility
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
__author__ = "E. Papillon & V.A. Sole - ESRF Software Group"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
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
