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
This module implements labels layout on graph axes.
"""


# import ######################################################################

import math


# utils #######################################################################

def numberOfDigits(tickSpacing):
    """Returns the number of digits to display for text label.

    :param float tickSpacing: Step between ticks in data space.
    :return: Number of digits to show for labels.
    :rtype: int
    """
    nFrac = int(-math.floor(math.log10(tickSpacing)))
    if nFrac < 0:
        nFrac = 0
    return nFrac


# Nice Numbers ################################################################

def _niceNum(value, isRound=False):
    expValue = math.floor(math.log10(value))
    frac = value/pow(10., expValue)
    if isRound:
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


def niceNumbers(vMin, vMax, nTicks=5):
    """Returns tick positions.

    This function implements graph labels layout using nice numbers
    by Paul Heckbert from "Graphics Gems", Academic Press, 1990.
    See `C code <http://tog.acm.org/resources/GraphicsGems/gems/Label.c>`_.

    :param float vMin: The min value on the axis
    :param float vMax: The max value on the axis
    :param int nTicks: The number of ticks to position
    :returns: min, max, increment value of tick positions and
              number of fractional digit to show
    :rtype: tuple
    """
    vRange = _niceNum(vMax - vMin, False)
    tickSpacing = _niceNum(vRange / nTicks, True)
    graphMin = math.floor(vMin / tickSpacing) * tickSpacing
    graphMax = math.ceil(vMax / tickSpacing) * tickSpacing
    nFrac = numberOfDigits(tickSpacing)
    return graphMin, graphMax, tickSpacing, nFrac


def niceNumbersAdaptative(vMin, vMax, axisLength, tickDensity):
    """Returns tick positions using :func:`niceNumbers` and a
    density of ticks.

    axisLength and tickDensity are based on the same unit (e.g., pixel).

    :param float vMin: The min value on the axis
    :param float vMax: The max value on the axis
    :param float axisLength: The length of the axis.
    :param float tickDensity: The density of ticks along the axis.
    :returns: min, max, increment value of tick positions and
              number of fractional digit to show
    :rtype: tuple
    """
    # At least 2 ticks
    nTicks = max(2, int(round(tickDensity * axisLength)))
    tickMin, tickMax, step, nbFrac = niceNumbers(vMin, vMax, nTicks)

    return tickMin, tickMax, step, nbFrac


# Nice Numbers for log scale ##################################################

def niceNumbersForLog10(minLog, maxLog, nTicks=5):
    """Return tick positions for logarithmic scale

    :param float minLog: log10 of the min value on the axis
    :param float maxLog: log10 of the max value on the axis
    :param int nTicks: The number of ticks to position
    :returns: log10 of min, max and increment value of tick positions
    :rtype: tuple of int
    """
    graphMinLog = math.floor(minLog)
    graphMaxLog = math.ceil(maxLog)
    rangeLog = graphMaxLog - graphMinLog

    if rangeLog <= nTicks:
        tickSpacing = 1.
    else:
        tickSpacing = math.floor(rangeLog / nTicks)

        graphMinLog = math.floor(graphMinLog / tickSpacing) * tickSpacing
        graphMaxLog = math.ceil(graphMaxLog / tickSpacing) * tickSpacing

    return int(graphMinLog), int(graphMaxLog), int(tickSpacing)


# main ########################################################################

if __name__ == "__main__":
    niceNumbersTests = [
        (0.5, 10.5),
        (10000., 10000.5),
        (0.001, 0.005)
    ]

    for vMin, vMax in niceNumbersTests:
        print("niceNumbers({0}, {1}): {2}".format(
            vMin, vMax, niceNumbers(vMin, vMax)))

    niceNumbersForLog10Tests = [  # This are log10 min, max
        (0., 3.),
        (-3., 3),
        (-32., 0.)
    ]

    for vMin, vMax in niceNumbersForLog10Tests:
        print("niceNumbersForLog10({0}, {1}): {2}".format(
            vMin, vMax, niceNumbersForLog10(vMin, vMax)))
