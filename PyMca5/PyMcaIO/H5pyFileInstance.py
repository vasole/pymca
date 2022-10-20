#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
Context manager to handle transparently either a h5py.File instances or a file
name.

This provides a level of abstraction for functions to accept both a file path
or an already opened file object.

Instead of writing::

    def func(file_):
        if isinstance(file_, h5py.File):
            h5f = file_
            must_be_closed = False
        else:
            h5f = h5py.File(file_, "w")
            must_be_closed = False

        # do some work with h5f...

        if must_be_closed:
            h5f.close()

you can write::

    def func(file_):
        with H5pyFileInstance(file_, "w") as h5f:
            # do some work with h5f...

"""
import h5py

__author__ = "P.Knobel - ESRF Data Analysis"
__contact__ = "pierre.knobel@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"


class H5pyFileInstance(object):
    """This class is a context manager returning a h5py.File instance.

    The constructor accepts either an already opened file object, or a
    filename to be opened on entry and closed on exit.

    When providing a file name to the constructor, it is guaranteed that
    the file we be closed on exiting the ``with`` block.
    """
    def __init__(self, file_, mode="r"):
        """

        :param file_: Either a filename or a h5py.File instance
        :param str mode: Mode in which to open file; one of ("w", "r", "r+",
                         "a", "w-"). Ignored if :param:`file_` is a h5py.File
                         instance.
        """
        if not isinstance(file_, h5py.File):
            # assume file_ is a valid path and let h5py.File raise errors if
            # it isn't
            self.file_obj = h5py.File(file_, mode)
            self._must_be_closed = True
        else:
            self.file_obj = file_
            self._must_be_closed = False

    def __enter__(self):
        return self.file_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._must_be_closed:
            self.file_obj.close()
