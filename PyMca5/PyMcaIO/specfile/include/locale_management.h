#/*##########################################################################
# Copyright (C) 2012-2017 European Synchrotron Radiation Facility
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
#ifndef PyMca_LOCALE_MANAGEMENT_H
#define PyMca_LOCALE_MANAGEMENT_H

#include <locale.h>
#ifdef _GNU_SOURCE
#  ifdef __GLIBC__
#    include <features.h>
#    if !((__GLIBC__ > 2) || ((__GLIBC__ == 2) && (__GLIBC_MINOR__ > 25)))
#      /* strtod_l has been moved to stdlib.h since glibc 2.26 */
#      include <xlocale.h>
#    endif
#  else
#    include <xlocale.h>
#  endif
#else
#  ifdef SPECFILE_POSIX
#    ifndef LOCALE_NAME_MAX_LENGTH
#           define LOCALE_NAME_MAX_LENGTH 85
#    endif
#  endif
#endif

double PyMcaAtof(const char*);

#endif
