#!/usr/bin/env python2

class ShapeError(Exception):
    """Raised when invalid in/output dimensionality causes an error

    Typically, this means that the network tried to compile before
    input or output dimensionality has been set, or with invalid shape.
    Can also mean that input or output shape has been set, but doesn't
    match the shape of the data.
    Attributes:
        Xshape (tuple of int): the shape of the input data (X)
            A value of None means that input shape has not been set.
        yshape (tuple of int): the shape of the output data (y)
            A value of None means that output shape has not been set.
    """
    def __init__(self,Xshape,yshape):
        self.Xshape = Xshape
        self.yshape = yshape
