# ANNFMP

Software repository for the "Approximate Nearest-Neighbour Fields via Massively-Parallel Propagation-Assisted K-D Trees" paper, presented at IEEE BigData 2020, MLBD special session.
The annfmp package provides a highly-efficient parallel implementation for computing nearest neighbor fields.

## Documentation

See the [documentation](http://annfmp.readthedocs.org) for details and examples.

## Dependencies

The annfmp package has been tested under Python 3.6 to 3.9. The required Python dependencies are:

- numpy==1.16.3
- pyopencl==2018.2.5
- sklearn

Furthermore, [OpenCL](https://www.khronos.org/opencl) needs to be available.
When installed from source, [SWIG](http://www.swig.org/) is required.

## Quickstart

The package can easily be installed via pip via::

  `pip install annfmp`

To install the package from the sources, first get the current stable release via::

  `git clone https://github.com/diku-dk/annfmp.git`

Afterwards, on Linux systems, you can install the package locally for the current user via::

  `python setup.py install --user`

## Disclaimer

The source code is published under the GNU General Public License (GPLv3). The authors are not responsible for any implications that stem from the use of this software.
