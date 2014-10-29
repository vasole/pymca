# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2014 European Synchrotron Radiation Facility
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
# ###########################################################################*/
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module implements graph labels layout using nice numbers by Paul Heckbert
From "Graphics Gems", Academic Press, 1990
http://tog.acm.org/resources/GraphicsGems/gems/Label.c
"""


# import ######################################################################

import math


# Nice Numbers ################################################################


def _niceNum(value, round_=False):
    expValue = math.floor(math.log10(value))
    frac = value/pow(10., expValue)
    if round_:
        if frac < 1.5:
            niceFrac = 1.
        elif frac < 3.:
            niceFrac = 2.
        elif frac < 7.:
            niceFrac = 5.
        else:
            niceFrac = 10.
    else:
        if frac <= 1.:
            niceFrac = 1.
        elif frac <= 2.:
            niceFrac = 2.
        elif frac <= 5.:
            niceFrac = 5.
        else:
            niceFrac = 10.
    return niceFrac * pow(10., expValue)


def niceNumbers(min_, max_, nTick):
    """Return tick positions
    :param float min_: The min value on the axis
    :param float max_: The max value on the axis
    :param int nTick: The number of ticks to position
    :returns: min, max, increment value of tick positions and
    number of fractional digit to show
    :rtype: tuple
    """
    range_ = _niceNum(max_ - min_, False)
    tickSpacing = _niceNum(range_/nTick, True)
    graphMin = math.floor(min_/tickSpacing) * tickSpacing
    graphMax = math.ceil(max_/tickSpacing) * tickSpacing
    nFrac = max(int(-math.floor(math.log10(tickSpacing))), 0)
    return (graphMin, graphMax, tickSpacing, nFrac)


# main ########################################################################

if __name__ == "__main__":
    nTick = 5
    for min_, max_ in [(0.5, 10.5), (10000., 10000.5), (0.001, 0.005)]:
        print(min_, max_, niceNumbers(min_, max_, nTick))
