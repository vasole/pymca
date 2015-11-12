#/*##########################################################################
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
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
#include <stdio.h>
#include <math.h>
#include "polspl.h"
#define POLABS(x) (x>0) ? x : -x

void polspl(double *xx, double *yy, double *w, int npts, \
            double *xl, double *xh, int *nc, int nr, \
            double *c, int csize) /* c must have enough memory to host Sum(nc[i] * nr) */
  {

   int i,j,ibl,k,nk,ik,m,n,n1,ns,ne,ncol,i1,ni,ni1,nm1,ns1;
   double df[26], a[36][37];
   double t;
   double xk[10];
   int p, nbs[11];

   for (i=0; i < 26; i++)
   {
       df[i] = 0.0;
   }
   for (i=0; i < 11; i++)
   {
       nbs[i] = 0;
   }
   for (i=0; i < 10; i++)
   {
       xk[i] = 0.0;
   }
   for (i=0; i < 36; i++)
   {
       for (j=0; j < 37; j++)
       {
           a[i][j] = 0.0;
       }
   }

   n = 0;
   nbs[1] = 1;

   for(i=1; i < nr+1 ;i++){
     n = n + nc[i];
     nbs[i+1] = n + 1 ;
     if(xl[i] >= xh[i]){
        t = xl[i];
        xl[i] = xh[i];
        xh[i] = t;
     }
   }
   n = n + 2 * (nr - 1);
   n1 = n + 1;
   xl[nr + 1] = 0.0;
   xh[nr + 1] = 0.0;

   for(ibl=1; ibl < nr + 1 ;ibl++){
      xk[ibl] = 0.5 * (xh[ibl] + xl[ibl+1]);
      if(xl[ibl] > xl[ibl+1])
      {
        xk[ibl] = 0.5 * (xl[ibl] + xh[ibl+1]);
      }
      ns = nbs[ibl];
      ne = nbs[ibl+1] - 1;
      for(i=1; i < npts+1 ; i++){
        if((xx[i] >= xl[ibl]) && (xx[i] <= xh[ibl])){
            df[ns] = 1.0;
            ns1 = ns + 1;
            for(j=ns1; j<ne+1 ;j++)
            {
                df[j] = df[j-1] * xx[i];
            }
            for(j=ns; j<ne + 1 ;j++)
            {
                for(k=j; k< ne + 1;k++)
                {
                    a[j][k] = a[j][k] + df[j] * df[k] * w[i];
                }
               a[j][n1] = a[j][n1] + df[j] * yy[i] * w[i];
            }
          }
       }
    }
    ncol = nbs[nr+1] - 1;
    nk = nr - 1;

    if(nk != 0 )
    {
       for(ik=1; ik<nk+1 ;ik++)
       {
            ncol++;
            ns = nbs[ik];
            ne = nbs[ik+1] -1 ;
            a[ns][ncol] = -1.0;
            ns++;
            for(i=ns; i<ne+1 ; i++)
            {
                a[i][ncol] = a[i-1][ncol] * xk[ik];
            }
            ncol++;
            a[ns][ncol] = -1.0;
            ns++;
            if(ns <= ne)
            {
                for(i=ns; i<ne+1 ;i++)
                {
                    p = i - ns + 1;
                    a[i][ncol] = (ns - i -2) * pow(xk[ik], p);
                }
            }
            ncol--;
            ns = nbs[ik+1];
            ne = nbs[ik+2]-1 ;
            a[ns][ncol] = 1.0;
            ns++;
            for(i=ns; i < ne +1; i++)
            {
                a[i][ncol] = a[i-1][ncol]*xk[ik];
            }
            ncol++;
            a[ns][ncol] = 1.0;
            ns++;
            if (ns <= ne)
            {
                for(i=ns; i<ne+1 ;i++)
                {
                    p = i -ns + 1;
                    a[i][ncol] = (i - ns + 2) * pow(xk[ik], p);
                }
            }
       }
    }

    for(i=1; i<n+1 ;i++)
    {
       i1 = i-1 ;
       for(j=1; j<i1+1 ;j++)
       {
            a[i][j] = a[j][i];
       }
    }

    nm1 = n -1;

    for(i=1; i<nm1+1 ;i++)
    {
        i1 = i + 1;
        m = i;
        t = POLABS(a[i][i]);
        for(j=i1; j<n+1 ;j++)
        {
            if(t < POLABS(a[j][i]))
            {
                m = j;
                t = POLABS(a[j][i]);
            }
        }
        if(m != i)
        {
            for(j=1; j<n1+1 ;j++)
            {
                t = a[i][j];
                a[i][j] = a[m][j];
                a[m][j] = t;
            }
        }
        for(j=i1; j<n1+1 ;j++)
        {
            t = a[j][i] / a[i][i];
            for(k=i1; k<n1+1 ;k++)
            {
                a[j][k] = a[j][k] - t * a[i][k];
            }
        }
    }

    c[n] = a[n][n1] / a[n][n];

    for(i=1; i<nm1+1 ;i++)
    {
        ni = n - i;
        t = a[ni][n1];
        ni1 = ni + 1;
        for(j=ni1; j<n+1 ;j++)
        {
            t = t - c[j] * a[ni][j];
        }
        if (ni == 0)
            printf("t = %f, a = %f\n", t, a[0][0]);
        c[ni] = t / a[ni][ni];
    }
}
