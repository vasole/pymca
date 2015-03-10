#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
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
__doc__="Translation from EADL index to actual shell (Table VI)"
SHELL_LIST = ['K (1s1/2)',
             'L (2)',
             'L1 (2s1/2)',
             'L23 (2p)',
             'L2 (2p1/2)',
             'L3 (2p3/2)',
             'M (3)',
             'M1 (3s1/2)',
             'M23 (3p)',
             'M2 (3p1/2)',
             'M3 (3p3/2)',
             'M45 (3d)',
             'M4 (3d3/2)',
             'M5 (3d5/2)',
             'N (4)',
             'N1 (4s1/2)',
             'N23 (4p)',
             'N2 (4p1/2)',
             'N3 (4p3/2)',
             'N45 (4d)',
             'N4 (4d3/2)',
             'N5 (4d5/2)',
             'N67 (4f)',
             'N6 (4f5/2)',
             'N7 (4f7/2)',
             'O (5)',
             'O1 (5s1/2)',
             'O23 (5p)',
             'O2 (5p1/2)',
             'O3 (5p3/2)',
             'O45 (5d)',
             'O4 (5d3/2)',
             'O5 (5d5/2)',
             'O67 (5f)',
             'O6 (5f5/2)',
             'O7 (5f7/2)',
             'O89 (5g)',
             'O8 (5g7/2)',
             'O9 (5g9/2)',
             'P (6)',
             'P1 (6s1/2)',
             'P23 (6p)',
             'P2 (6p1/2)',
             'P3 (6p3/2)',
             'P45 (6d)',
             'P4 (6d3/2)',
             'P5 (6d5/2)',
             'P67 (6f)',
             'P6 (6f5/2)',
             'P7 (6f7/2)',
             'P89 (6g)',
             'P8 (6g7/2)',
             'P9 (6g9/2)',
             'P1011 (6h)',
             'P10 (6h9/2)',
             'P11 (6h11/2)',
             'Q (7)',
             'Q1 (7s1/2)',
             'Q23 (7p)',
             'Q2 (7p1/2)',
             'Q3 (7p3/2)']

def getSubshellFromValue(value):
    idx = int(value) - 1
    if idx < 0:
        raise IndexError("Invalid EADL Atomic Subshell Designator")
    return SHELL_LIST[idx]

def getValueFromSubshell(subshell):
    """
    Returns the float value associated to the respective shell or subshell
    """
    if subshell.startswith('K'):
        return 1.0

    #cleanup subshell
    wshell = subshell.replace(" ","")
    wshell = wshell.split("(")[0]
    wshell = wshell.upper()

    #test
    i = 0
    for shell in SHELL_LIST:
        i += 1
        if wshell == shell.split(" ")[0]:
            return float(i)
    raise ValueError("Invalid shell name %s" % subshell)
