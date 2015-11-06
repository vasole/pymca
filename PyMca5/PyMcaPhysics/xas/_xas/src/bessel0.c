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
#include <math.h>
#include "bessel0.h"
void j0Multiple(double *x, int n)
{
    int i;
    for (i=0; i < n; i++)
    {
        x[i] = j0Single(x[i]);
    }
}

double j0Single(double x)
{
    double f0, theta0;
    double tmpDouble;
    double tmpDouble3;
    if (x < 0)
        x = -x;
    if (x > 3)
    {
        /* Abramowitz and Stegun 9.4.3 */
        /* Absolute error < 1.6E-08 */
        tmpDouble = 3. / x;
        tmpDouble3 = pow(tmpDouble, 3);
        f0 = 0.79788456 - tmpDouble * (0.00000077 + 0.00552740 *  tmpDouble) + \
              tmpDouble3 * ( 0.00137237 * tmpDouble - 0.00009513) + \
              tmpDouble3 * (0.00014476 * tmpDouble3 - 0.00072805 * tmpDouble * tmpDouble);
        theta0 = x - 0.78539816 - 0.04166397 * tmpDouble - \
               0.00003954 * tmpDouble * tmpDouble + \
               tmpDouble3 * (0.00262573 - 0.00054125 * tmpDouble) + \
               tmpDouble3 * (0.00013558 * tmpDouble3 - 0.00029333 * tmpDouble * tmpDouble);
        return pow(x, -0.5) * f0 * cos(theta0);
    }
    else
    {       
        /* Abramowitz and Stegun 9.4.1 */
        /* Absolute error < 5.0E-08 */
        tmpDouble = pow(x/3., 2);
        tmpDouble = 1.0 - 2.2499997 * tmpDouble + \
                    1.2656208 * tmpDouble * tmpDouble - \
                    0.3163866 * tmpDouble * tmpDouble * tmpDouble + \
                    0.0444479 * tmpDouble * tmpDouble * tmpDouble * tmpDouble - \
                    0.0039444 * tmpDouble * tmpDouble * tmpDouble * tmpDouble * tmpDouble + \
                    0.0002100 * tmpDouble * tmpDouble * tmpDouble * tmpDouble * tmpDouble * tmpDouble;
        return tmpDouble;
    }
}
