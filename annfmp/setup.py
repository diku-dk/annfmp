def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
        
    config = Configuration('annfmp', parent_package, top_path)

    config.add_subpackage('openmp', subpackage_path='openmp')
    config.add_subpackage('openmp/kdtree', subpackage_path='openmp/kdtree')

    config.add_subpackage('openclopt', subpackage_path='openclopt')
    
    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
