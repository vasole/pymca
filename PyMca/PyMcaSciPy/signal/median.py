import mediantools
from numpy import asarray

def medfilt2d(input_data, kernel_size=3):
    """Median filter two 2-dimensional arrays.

  Description:

    Apply a median filter to the input array using a local window-size
    given by kernel_size (must be odd).

  Inputs:

    in -- An 2 dimensional input array.
    kernel_size -- A scalar or an length-2 list giving the size of the
                   median filter window in each dimension.  Elements of
                   kernel_size should be odd.  If kernel_size is a scalar,
                   then this scalar is used as the size in each dimension.

  Outputs: (out,)

    out -- An array the same size as input containing the median filtered
           result.

    """
    image = asarray(input_data)
    if kernel_size is None:
        kernel_size = [3] * 2
    kernel_size = asarray(kernel_size)
    if len(kernel_size.shape) == 0:
        kernel_size = [kernel_size.item()] * 2
    kernel_size = asarray(kernel_size)

    for size in kernel_size:
        if (size % 2) != 1:
            raise ValueError, "Each element of kernel_size should be odd."

    return mediantools._medfilt2d(image, kernel_size)

