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
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
import sys
import os
import ctypes
import traceback
import logging


_logger = logging.getLogger(__name__)


def loadCLibrary(name="libc.so"):
    try:
        libc = ctypes.CDLL(name)
    except OSError:
        text = traceback.format_exc()
        if "invalid ELF header" in text:
            library = text.split(": ")
            if len(library) > 1:
                libraryFile = library[1]
                if os.path.exists(libraryFile):
                    # to read some line
                    f = open(libraryFile, 'r')
                    for i in range(10):
                        line = f.readline()
                        if name in line:
                            f.close()
                            lineSplit = line.split(name)
                            libraryFile = lineSplit[0].split()[-1] +\
                                          name+\
                                          lineSplit[1].split()[0]
                            break
                    libraryFile = line[line.index(libraryFile):].split()[0]
                    libc = ctypes.CDLL(libraryFile)
        else:
            raise
    return libc

if sys.platform == 'win32':
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

        def __init__(self):
            # have to initialize this to the size of MEMORYSTATUSEX
            self.dwLength = ctypes.sizeof(self)
            super(MEMORYSTATUSEX, self).__init__()

    def getPhysicalMemory():
        #print("MemoryLoad: %d%%" % (stat.dwMemoryLoad))
        #print("Physical memory = %d" % stat.ullTotalPhys)
        #print(stat.ullAvailPhys)
        stat = MEMORYSTATUSEX()
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys

    def getAvailablePhysicalMemory():
        stat = MEMORYSTATUSEX()
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        value = stat.ullAvailPhys
        return value

    def getAvailablePhysicalMemoryOrNone():
        try:
            value = getAvailablePhysicalMemory()
            if value < 0:
                # Value makes no sense.
                # return None as requested in case of failure
                print("WARNING: Returned physical memory does not make sense %d" % \
                          value)
                return None
            else:
                return value
        except:
            return None

elif sys.platform.startswith('linux'):
    def getPhysicalMemory():
        return os.sysconf('SC_PAGESIZE') * os.sysconf('SC_PHYS_PAGES')

elif sys.platform == 'darwin':
    def getPhysicalMemory():
        libc = loadCLibrary("libc.dylib")
        memsize = ctypes.c_uint64(0)
        length = ctypes.c_size_t(ctypes.sizeof(memsize))
        name = "hw.memsize"
        if hasattr(name, "encode"):
            # Passing a string was returning 0 memory size under Python 3.5
            name = name.encode()
        libc.sysctlbyname(name, ctypes.byref(memsize), ctypes.byref(length), None, 0)
        return memsize.value
else:
    def getPhysicalMemory():
        return None

def getPhysicalMemoryOrNone():
    try:
        value = getPhysicalMemory()
        if value <= 0:
            # Value makes no sense.
            # return None as requested in case of failure
            _logger.warning("WARNING: Returned physical memory does not make sense %d",
                            value)
            return None
        else:
            return value
    except:
        return None

if __name__ == "__main__":
    print("Physical memory = %d" % getPhysicalMemory())
