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

void lls(double *data, int size);
void lls_inv(double *data, int size);
void snip1d(double *data, int size, int niter);

void lls(double *data, int size)
{
	int i;
	for (i=0; i< size; i++)
	{
		data[i] = log(log(sqrt(data[i]+1.0)+1.0)+1);
	}
}

void lls_inv(double *data, int size)
{
	int i;
	double tmp;
	for (i=0; i< size; i++)
	{
		tmp = exp(exp(data[i]-1.0)-1.0);
		data[i] = tmp * tmp - 1.0;
	}
}

void snip1d(double *data, int size, int niter)
{
	int i;
	int p;
	double *w;


	w = (double *) malloc(size * sizeof(double));

	for (p=niter; p > 0; p--)
	{
		for (i=p; i<(size-p); i++)
		{
			w[i] = MIN(data[i], 0.5*(data[i-p]+data[i+p]));
		}
		for (i=p; i<(size-p); i++)
		{
			data[i] = w[i];
		}
	}
	free(w);
}
