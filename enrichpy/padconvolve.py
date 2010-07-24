"""Routines to pad convolution to control the centering of the kernel.

"""
import numpy
def pad(x, klength, origin=0, value=0.):
    """Pad an array that is going to be convolved.

    Set origin to zero to have only positive delays. Set origin to
    klength-1 to have entirely negative delays. Set origin to the
    center of the kernel for centering on zero delay.

    Parameters
    ----------
    klength: scalar
        Length of the kernel.
    origin: scalar
        The index of the kernel value you want at the origin (default 0).
    value: scalar 
        Value to pad the array with (default 0). 

    """
    if (origin > klength-1):
        raise ValueError("Origin can't be > klength-1.")
    elif (origin < 0):
        raise ValueError("Origin can't be < 0.")
    return numpy.hstack((numpy.zeros(klength-1-origin) + value,
                         x,
                         numpy.zeros(origin) + value))

def padded_convolve(x, kernel, origin=0, value=0.0):
    """Convolve an array with padding.
    
    See docstring for pad for more info.
    """
    return numpy.convolve(pad(x, len(kernel), origin, value), 
                          kernel, 
                          mode='valid')

