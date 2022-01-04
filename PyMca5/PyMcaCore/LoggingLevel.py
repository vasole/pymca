
#!/usr/bin/env python
#/*##########################################################################
# Copyright (C) 2018-2022 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF.
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
"""Module for parsing command line options related to the logging level."""
__author__ = "P. Knobel"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import logging


DEFAULT_LOGGING_LEVEL = logging.WARNING


def getLoggingLevel(opts):
    """Find logging level from the output of `getopt.getopt()`.
    This level can be specified via one of two long options:
    --debug or --logging. If both are specified, --logging overrules
    --debug.

    When specifying the level with --logging, the level can be
    specified explicitly as a string (debug, info, warning, error, critical),
    or as an integer in the range 0--4, in increasing order of verbosity
    (0 is "critical", 4 is "debug").

    The option --debug only allows to chose between the default logging level
    (--debug=0) or debugging mode with maximum verbosity (--debug=1).

    :param opts: Command line options as a list of 2-tuples of strings
        (e.g. ``[('--logging', 'debug'), ('--cfg', 'config.ini')]``).
    :returns: logging level
    :rtype: int"""
    logging_level = None
    for opt, arg in opts:
        if opt == '--logging':
            levels_dict = {
                # Explicit args
                'debug': logging.DEBUG,
                'info': logging.INFO,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'critical': logging.CRITICAL,
                # int args sorted by increasing verbosity
                '0': logging.CRITICAL,
                '1': logging.ERROR,
                '2': logging.WARNING,
                '3': logging.INFO,
                '4': logging.DEBUG}

            logging_level = levels_dict.get(arg.lower())
            if logging_level is None:
                raise ValueError("Unknown logging level <%s>" % arg)
            # if --logging is specified, ignore --debug
            return logging_level
        if opt == '--debug':
            # simpler option to choose between the default logging or DEBUG
            if arg.lower() in ["0", 0, "false"]:
                logging_level = DEFAULT_LOGGING_LEVEL
            elif arg.lower() in ["1", 1, "true"]:
                logging_level = logging.DEBUG
            else:
                raise ValueError("Incorrect debug parameter <%s> (should be 0 or 1)" % arg)
    if logging_level is None:
        return DEFAULT_LOGGING_LEVEL
    return logging_level
