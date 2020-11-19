import os
import sys
import shutil
from distutils.command.clean import clean as Clean

DISTNAME = 'annfmp'
DESCRIPTION = 'A Python library for the massively-parallel computation of nearest neighbor fields.'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Fabian Gieseke'
MAINTAINER_EMAIL = 'fabian.gieseke@uni-muenster.de'
URL = 'http://www.annfmp.pydata.org'
LICENSE = 'GNU GENERAL PUBLIC LICENSE'
DOWNLOAD_URL = 'https://github.com/gieseke/annfmp'

import annfmp
VERSION = annfmp.__version__

# use setuptools for certain commands
if len(set(('develop', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist_wininst', 'install_egg_info', 'build_sphinx',
           'egg_info', 'easy_install', 'upload', 'bdist_wheel',
           '--single-version-externally-managed',
            )).intersection(sys.argv)) > 0:
    import setuptools
    extra_setuptools_args = dict(
        zip_safe=False,
        include_package_data=True,
    )
else:
    import setuptools
    extra_setuptools_args = dict()


# define new clean command
class CleanCommand(Clean):

    description = "Removes build directories and compiled files in the source tree."

    def run(self):

        Clean.run(self)

        if os.path.exists('build'):
            shutil.rmtree('build')

        for dirpath, dirnames, filenames in os.walk('annfmp'):
            for filename in filenames:
                if (filename.endswith('.so') or \
                    filename.endswith('.pyd') or \
                    filename.endswith('.dll') or \
                    filename.endswith('.pyc') or \
                    filename.endswith('_wrap.c') or \
                    filename.startswith('wrapper_') or \
                    filename.endswith('~')):
                        os.unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == '__pycache__' or dirname == 'build':
                    shutil.rmtree(os.path.join(dirpath, dirname))


def configuration(parent_package='', top_path=None):

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('annfmp')

    return config


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    download_url=DOWNLOAD_URL,
                    long_description=LONG_DESCRIPTION,
                    packages=setuptools.find_packages(),
                    install_requires=[
                        'numpy>=1.11.0',
                        #'pyopencl==2018.2.5',
                    ],
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.3',
                                 'Programming Language :: Python :: 3.4',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 ],
                    cmdclass={'clean': CleanCommand},
                    setup_requires=["numpy>=1.11.0"],
                    **extra_setuptools_args)

    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean'))):

        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION

    else:

        from numpy.distutils.core import setup
        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
