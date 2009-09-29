#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program.
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
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