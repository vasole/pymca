#/*##########################################################################
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
#############################################################################*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void smooth1d(double *data, int size);
void smooth2d(double *data, int size0, int size1);
void smooth3d(double *data, int size0, int size1, int size2);

void smooth1d(double *data, int size)
{
	long i;
	double oldy;
	double newy;

	if (size < 3)
	{
		return;
	}
	oldy = data[0];
	for (i=0; i<(size-1); i++)
	{
		newy = 0.25 * (oldy + 2 * data[i] + data[i+1]);
		oldy = data[i];
		data[i] = newy;
	}
	data[size-1] = 0.25 * oldy + 0.75 * data[size-1];
	return;
}

void smooth2d(double *data, int size0, int size1)
{
	long i, j;
	double *p;

	/* smooth the first dimension */
	for (i=0; i < size0; i++)
	{
		smooth1d(&data[i*size1], size1);
	}

	/* smooth the 2nd dimension */
	p = (double *) malloc(size0 * sizeof(double));
	for (i=0; i < size1; i++)
	{
		for (j=0; j<size0; j++)
		{
			p[j] = data[j*size1+i];
		}
		smooth1d(p, size0);
	}
	free(p);
}

void smooth3d(double *data, int size0, int size1, int size2)
{
	long i, j, k, ihelp, jhelp;
	double *p;
	int size;


	size = size1*size2;

	/* smooth the first dimension */
	for (i=0; i < size0; i++)
	{
		smooth2d(&data[i*size], size1, size2);
	}

	/* smooth the 2nd dimension */
	size = size0 * size2;
	p = (double *) malloc(size * sizeof(double));

	for (i=0; i < size1; i++)
	{
		ihelp = i * size2;
		for (j=0; j<size0; j++)
		{
			jhelp = j * size1 * size2 + ihelp;
			for(k=0; k<size2; k++)
			{
				p[j*size2+k] = data[jhelp+k];
			}
		}
		smooth2d(p, size0, size2);
	}
	free(p);

	/* smooth the 3rd dimension */
	size = size0 * size1;
	p = (double *) malloc(size * sizeof(double));

	for (i=0; i < size2; i++)
	{
		for (j=0; j<size0; j++)
		{
			jhelp = j * size1 * size2 + i;
			for(k=0; k<size1; k++)
			{
				p[j*size1+k] = data[jhelp+k*size2];
			}
		}
		smooth2d(p, size0, size1);
	}
	free(p);
}
