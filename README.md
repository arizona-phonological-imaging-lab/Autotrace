Autotrace
=========

A collection of tools to analyze tongue surface contours in ultrasound images.

##[Matlab-version](https://github.com/jjberry/Autotrace/tree/master/matlab-version)
The older, fully-functioning AutoTrace system requires Matlab and several python dependencies related to GTK+.  Because of GTK+, this version is best run on Linux, although it can be installed on Mac OS X using MacPorts (http://www.macports.org/).



##[Matlab-free version](https://github.com/bamartin-ua/Autotrace) (under-development)
A Matlab-free version is currently under development. This project draws inspiration from the Matlab version (above), and from an older attempt at a Matlab-free version (which can be fount at https://github.com/jjberry/Autotrace/tree/master/under-development), making this the third in the Autotrace line, and thus the codename Autotres.

The code for training deep networks uses [Lasagne](https://github.com/Lasagne/Lasagne) and [Theano](http://deeplearning.net/software/theano/)
These allow the network to be trained on a CUDA-capable GPU if present (with limited support for open-cl).
If no GPU is present, theano will use the CPU. For best results, a BLAS library with multithreading support is suggested, such as [OpenBLAS](http://www.openblas.net).

Currently, the project lacks a graphical interface, and has only been tested on Ubuntu 14.04. With luck, future versions will rectify these shortcomings.
