#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
"""
A 2D plugin is a module that can be added to the PyMca 2D window in order to
perform user defined operations of the plotted 2D data.

Plugins can be automatically installed provided they are in the appropriate place:

    - In the user home directory (POSIX systems): *${HOME}/.pymca/plugins*
      or *${HOME}/PyMca/plugins* (older PyMca installation)
    - In *"My Documents\\\\PyMca\\\\plugins"* (Windows)

A plugin inherits the :class:`Plugin2DBase` class and implements the methods:

    - :meth:`Plugin2DBase.getMethods`
    - :meth:`Plugin2DBase.getMethodToolTip` (optional but convenient)
    - :meth:`Plugin2DBase.getMethodPixmap` (optional)
    - :meth:`Plugin2DBase.applyMethod`

and modifies the static module variable :const:`MENU_TEXT` and the static module function
:func:`getPlugin2DInstance` according to the defined plugin.

It may also optionally implement :meth:`Plugin2DBase.activeImageChanged`.

"""
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import weakref


class Plugin2DBase(object):
    def __init__(self, plotWindow, **kw):
        """
        plotWindow is the plot on which the plugin operates.


        """
        self._plotWindow = weakref.proxy(plotWindow)

    # Methods to be implemented by the plugin
    def getMethods(self, plottype=None):
        """

        :return:  A list with the NAMES  associated to the callable methods
         that are applicable to the specified type plot. The list can be empty.
        :rtype: list[string]
        """
        print("getMethods not implemented")
        return []

    def getMethodToolTip(self, name):
        """
        Returns the help associated to the particular method name or None.

        :param name: The method for which a tooltip is asked
        :rtype: string
        """
        return None

    def getMethodPixmap(self, name):
        """
        :param name: The method for which a pixmap is asked
        :rtype: QPixmap or None
        """
        return None

    def applyMethod(self, name):
        """
        The plugin is asked to apply the method associated to name.
        """
        print("applyMethod not implemented")
        return

    def activeImageChanged(self, prev, new):
        """A plugin may implement this method which is called
        when the active image changes in the plot.

        :param prev: Legend of the previous active image,
            or None if no image was active.
        :param new: Legend of the new active curve,
            or None if no image is currently active.
        """
        pass


MENU_TEXT = "Plugin2D Base"
"""This is the name of the plugin, as it appears in the plugins menu."""


def getPlugin2DInstance(plotWindow, **kw):
    """
    This function will be called by the plot window instantiating and calling
    the plugins. It passes itself as first argument, but the default implementation
    of the base class only keeps a weak reference to prevent circular references.
    """
    ob = Plugin2DBase(plotWindow)
    return ob
