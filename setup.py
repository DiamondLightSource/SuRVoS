#!/usr/bin/env python


import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
	config = Configuration('', parent_package, top_path)
	config.add_subpackage('survos')
	return config


if __name__ == '__main__':
	config = configuration(top_path='').todict()
	setup(**config)
