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

void lls(double *input, int size, double *output);
void lls_inv(double *input, int size, double *output);

void snip2d(double *input, int nrows, int ncolumns, int niter, double *output)
{
	int i, j;
	int p;
	int size;
	double *w;
	double P1, P2, P3, P4;
	double S1, S2, S3, S4;
	double dhelp;
	int	iminuspxncolumns; /* (i-p) * ncolumns */
	int	ixncolumns; /*  i * ncolumns */
	int	ipluspxncolumns; /* (i+p) * ncolumns */

	size = nrows * ncolumns;
	w = (double *) malloc(size * sizeof(double));
    memcpy(output, input, size * sizeof(double));

	for (p=niter; p > 0; p--)
	{
		for (i=p; i<(nrows-p); i++)
		{
			iminuspxncolumns = (i-p) * ncolumns;
			ixncolumns = i * ncolumns;
			ipluspxncolumns = (i+p) * ncolumns; 
			for (j=p; j<(ncolumns-p); j++)
			{
				P4 = w[ iminuspxncolumns + (j-p)]; /* P4 = w[i-p][j-p] */
				S4 = w[ iminuspxncolumns + j];     /* S4 = w[i-p][j]   */
				P2 = w[ iminuspxncolumns + (j+p)]; /* P2 = w[i-p][j+p] */
				S3 = w[ ixncolumns + (j-p)];       /* S3 = w[i][j-p]   */
				S2 = w[ ixncolumns + (j+p)];       /* S2 = w[i][j+p]   */
				P3 = w[ ipluspxncolumns + (j-p)];  /* P3 = w[i+p][j-p] */
				S1 = w[ ipluspxncolumns + j];      /* S1 = w[i+p][j]   */
				P1 = w[ ipluspxncolumns + (j+p)];  /* P1 = w[i+p][j+p] */
				dhelp = 0.5*(P1+P3);
				S1 = MAX(S1, dhelp) - dhelp;
				dhelp = 0.5*(P1+P2);
				S2 = MAX(S2, dhelp) - dhelp;
				dhelp = 0.5*(P3+P4);
				S3 = MAX(S3, dhelp) - dhelp;
				dhelp = 0.5*(P2+P4);
				S4 = MAX(S4, dhelp) - dhelp;
				w[ixncolumns + j] = MIN(output[ixncolumns + j], 0.5 * (S1+S2+S3+S4) + 0.25 * (P1+P2+P3+P4));
			}
		}
		for (i=p; i<(nrows-p); i++)
		{
			for (j=p; j<(ncolumns-p); j++)
			{
				output[ixncolumns + j] = w[ixncolumns + j];
			}
		}
	}
	free(w);
}
