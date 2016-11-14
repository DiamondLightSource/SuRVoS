#!/usr/bin/env python


import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

from survos.build import custom_build_ext

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
	config = Configuration('survos', parent_package, top_path,
						   cmdclass={'build_ext': custom_build_ext})
	config.add_subpackage('lib')
	return config


if __name__ == '__main__':
	config = configuration(top_path='').todict()
	setup(**config)
