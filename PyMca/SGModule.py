#
# The code to calculate the Savitzky-Golay filter coefficients
# is a shameless copy of the sg_filter.py module from Uwe Schmitt
# available from http://public.procoders.net/sg_filter
#
# Therefore PyMca author(s) do not claim any ownership of that code
# and are very grateful to Uwe for making his code available to the
# community.
#
#

import numpy
from numpy.linalg import solve

def calc_coeff(num_points, pol_degree, diff_order=0):

    """ calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

        num_points   means that 2*num_points+1 values contribute to the
                     smoother.

        pol_degree   is degree of fitting polynomial

        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first 
                                                 derivative of function.
                     and so on ...

    """

    # setup interpolation matrix
    # ... you might use other interpolation points
    # and maybe other functions than monomials ....

    x = numpy.arange(-num_points, num_points+1, dtype=numpy.int)
    monom = lambda x, deg : pow(x, deg)

    A = numpy.zeros((2*num_points+1, pol_degree+1), numpy.float)
    for i in range(2*num_points+1):
        for j in range(pol_degree+1):
            A[i,j] = monom(x[i], j)
        
    # calculate diff_order-th row of inv(A^T A)
    ATA = numpy.dot(A.transpose(), A)
    rhs = numpy.zeros((pol_degree+1,), numpy.float)
    rhs[diff_order] = 1
    wvec = solve(ATA, rhs)

    # calculate filter-coefficients
    coeff = numpy.dot(A, wvec)

    return coeff

def smooth(signal, coeff):
    
    """ applies coefficients calculated by calc_coeff()
        to signal """
    N = numpy.size(coeff-1)/2
    res = numpy.convolve(signal, coeff)
    return res[N:-N]

def getSavitzkyGolay(spectrum, npoints=3, degree=1, order=0):
    coeff = calc_coeff(npoints, degree, order)
    N = numpy.size(coeff-1)/2
    if order < 1:
        result = 1.0 * spectrum
    else:
        result = 0.0 * spectrum
    result[N:-N] = numpy.convolve(spectrum, coeff, mode='valid')
    return result

def replaceStackWithSavitzkyGolay(stack, npoints=3, degree=1, order=0):
    #Warning: Not checked if the last dimension is the one containing the
    #spectra as it is assuming!!!!
    coeff = calc_coeff(npoints, degree, order)
    N = numpy.size(coeff-1)/2
    convolve = numpy.convolve
    if hasattr(stack, "info") and hasattr(stack, "data"):
        data = stack.data
    else:
        data = stack
    oldShape = data.shape
    data.shape = -1, oldShape[-1]        
    for i in range(data.shape[0]):
        data[i,N:-N] = convolve(data[i,:],coeff, mode='valid')
        if order > 0:
            data[i, :N] = 0
            data[i, -N:] = 0
    data.shape = oldShape
    return
    

