#/*##########################################################################
# Copyright (C) 2004-2013 European Synchrotron Radiation Facility
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
/* 
   Implementation of the algorithm SNIP in 1D described in
   Miroslav Morhac et al. Nucl. Instruments and Methods in Physics Research A401 (1997) 113-132.

   The original idea for 1D and the low-statistics-digital-filter (lsdf) come from
   C.G. Ryan et al. Nucl. Instruments and Methods in Physics Research B34 (1988) 396-402.
*/
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void lls(double *data, int size);
void lls_inv(double *data, int size);
void snip1d(double *data, int n_channels, int snip_width);
void snip1d_multiple(double *data, int n_channels, int snip_width, int n_spectra);
void lsdf(double *data, int size, int fwhm, double f, double A, double M, double ratio);

void lls(double *data, int size)
{
	int i;
	for (i=0; i< size; i++)
	{
		data[i] = log(log(sqrt(data[i]+1.0)+1.0)+1.0);
	}
}

void lls_inv(double *data, int size)
{
	int i;
	double tmp;
	for (i=0; i< size; i++)
	{
		/* slightly different than the published formula because
		   with the original formula:
		   
		   tmp = exp(exp(data[i]-1.0)-1.0);
		   data[i] = tmp * tmp - 1.0;
		   
		   one does not recover the original data */

		tmp = exp(exp(data[i])-1.0)-1.0;
		data[i] = tmp * tmp - 1.0;
	}
}

void lsdf(double *data, int size, int fwhm, double f, double A, double M, double ratio)
{
	int channel, i, j;
	double L, R, S;
	int width;
	double dhelp;

	width = (int) (f * fwhm);
	for (channel=width; channel<(size-width); channel++)
	{
		i = width;
		while(i>0)
		{
			L=0;
			R=0;
			for(j=channel-i; j<channel; j++)
			{
				L += data[j];
			}
			for(j=channel+1; j<channel+i; j++)
			{
				R += data[j];
			}
			S = data[channel] + L + R;
			if (S<M)
			{
				data[channel] = S /(2*i+1);
				break;
			}
			dhelp = (R+1)/(L+1); 
			if ((dhelp < ratio) && (dhelp > (1/ratio)))
			{
				if (S<(A*sqrt(data[channel])))
				{
					data[channel] = S /(2*i+1);
					break;
				}				
			}
			i=i-1;
		}
	}
}


void snip1d(double *data, int n_channels, int snip_width)
{
	snip1d_multiple(data, n_channels, snip_width, 1);
}

void snip1d_multiple(double *data, int n_channels, int snip_width, int n_spectra)
{
	int i;
	int j;
	int p;
	int offset;
	double *w;

	i = (int) (0.5 * snip_width);
	/* lsdf(data, size, i, 1.5, 75., 10., 1.3); */
	
	w = (double *) malloc(n_channels * sizeof(double));

	for (j=0; j < n_spectra; j++)
	{
		offset = j * n_channels;
		for (p = snip_width; p > 0; p--)
		{
			for (i=p; i<(n_channels - p); i++)
			{
				w[i] = MIN(data[i + offset], 0.5*(data[i + offset - p] + data[ i + offset + p]));
			}
			for (i=p; i<(n_channels - p); i++)
			{
				data[i+offset] = w[i];
			}
		}
	}
	free(w);
}
