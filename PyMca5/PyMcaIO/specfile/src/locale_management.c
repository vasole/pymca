# /*##########################################################################
# Copyright (C) 2012-2022 European Synchrotron Radiation Facility
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
# ############################################################################*/

#define _GNU_SOURCE
#include <stdlib.h>
#include <locale_management.h>
#include <string.h>

double PyMcaAtof(const char * inputString)
{
#if defined(_MSC_VER) && defined(_MSC_FULL_VER)
	double result;
	_locale_t newLocale = _create_locale(LC_NUMERIC, "C");
	result = _atof_l(inputString, newLocale);
	_free_locale(newLocale);
	return result;
#elif defined __GLIBC__ || defined__GLIBCXX__
	double result;
	locale_t newLocale = newlocale(LC_NUMERIC_MASK, "C", NULL);
	result = strtod_l(inputString, NULL, newLocale);
	freelocale(newLocale);
	return result;
#elif defined SPECFILE_POSIX
#   ifndef LOCALE_NAME_MAX_LENGTH
#           define LOCALE_NAME_MAX_LENGTH 85
#   endif
	char *currentLocaleBuffer;
	char *restoredLocaleBuffer;
	char localeBuffer[LOCALE_NAME_MAX_LENGTH + 1] = {'\0'};
	double result;
	currentLocaleBuffer = setlocale(LC_NUMERIC, NULL);
	strcpy(localeBuffer, currentLocaleBuffer);
	setlocale(LC_NUMERIC, "C\0");
	result = atof(inputString);
	restoredLocaleBuffer = setlocale(LC_NUMERIC, localeBuffer);
	return(result);
#else
	return atof(inputString);
#endif
}
