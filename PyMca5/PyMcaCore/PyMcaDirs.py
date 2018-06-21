#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
import os
import logging

_logger = logging.getLogger(__name__)

inputDir  = None
outputDir = None
nativeFileDialogs = False

class __ModuleWrapper:
  def __init__(self, wrapped):
    self.__dict__["_ModuleWrapper__wrapped"] = wrapped

  def __getattr__(self, name):
    _logger.debug("getting %s", name)
    if name == "inputDir":
        if self.__wrapped.__dict__[name] is None:
            if self.__wrapped.__dict__['outputDir'] is not None:
                value = self.__wrapped.__dict__['outputDir']
            else:
                value = os.getcwd()
            if not os.path.isdir(value):
                value = os.getcwd()
            self.__setattr__('inputDir', value)
    elif name == "outputDir":
        if self.__wrapped.__dict__[name] is None:
            if self.__wrapped.__dict__['inputDir'] is not None:
                value = self.__wrapped.__dict__['inputDir']
            else:
                value = os.getcwd()
            if not os.path.isdir(value):
                value = os.getcwd()
            self.__setattr__('outputDir', value)
    _logger.debug("got %s %s", name, getattr(self.__wrapped, name))
    return getattr(self.__wrapped, name)

  def __setattr__(self, name, value):
    _logger.debug("setting %s %s", name, value)
    if name == "inputDir":
        if os.path.isdir(value):
            self.__wrapped.__dict__[name]=value
        else:
            if not len("%s" % value):
                self.__wrapped.__dict__[name] = os.getcwd()
            else:
                raise ValueError("Non-existing directory <%s>" % value)
    elif name == "outputDir":
        if os.path.isdir(value):
            self.__wrapped.__dict__[name]=value
        else:
            if not len("%s" % value):
                self.__wrapped.__dict__[name] = os.getcwd()
            else:
                raise ValueError("Non-existing directory <%s>" % value)
    elif name == "nativeFileDialogs":
        self.__wrapped.__dict__[name]=value
    elif name.startswith("__"):
        self.__dict__[name]=value
    else:
        raise AttributeError("Invalid attribute %s" % name)
        #self.__wrapped.__dict__[name]=value

sys.modules[__name__]=__ModuleWrapper(sys.modules[__name__])


