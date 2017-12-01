#!/usr/bin/env python


import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

import urllib
try:
    import urllib.request
except:
    pass

import shutil, glob
import zipfile

from survos.build import custom_build_ext, locate_cuda

base_path = os.path.abspath(os.path.dirname(__file__))
source_path = os.path.join(base_path, 'qpbo_src')

def get_qpbo():
    if os.path.isdir(source_path):
        return
    else:
        os.mkdir(source_path)

    if hasattr(urllib, "urlretrieve"):
        urlretrieve = urllib.urlretrieve
    else:
        urlretrieve = urllib.request.urlretrieve

    qpbo_version = 'QPBO-v1.4.src'
    qpbo_file = '{}.zip'.format(qpbo_version)

    urlretrieve('http://pub.ist.ac.at/~vnk/software/{}'.format(qpbo_file), qpbo_file)
    with zipfile.ZipFile(qpbo_file) as zf:
        zf.extractall(source_path)

    for f in glob.glob(os.path.join(source_path, qpbo_version, '*')):
        shutil.move(f, os.path.dirname(os.path.dirname(f)))

    os.rmdir(os.path.join(source_path, qpbo_version))
    os.remove(qpbo_file)

def cuda_extension(config, name, sources, CUDA, gcc_args=[], gpp_args=[]):
    config.add_extension(name, sources=sources,
                         library_dirs=[CUDA['lib64']],
                         libraries=['cudart', 'stdc++'], language='c++',
                         runtime_library_dirs=[CUDA['lib64']],
                         extra_compile_args={
                            'gcc': gcc_args,
                            'g++': gpp_args,
                            'nvcc': ['-arch=sm_30',
                                     '-gencode=arch=compute_20,code=sm_20',
                                     '-gencode arch=compute_30,code=sm_30',
                                     '-gencode arch=compute_35,code=sm_35',
                                     '-gencode arch=compute_37,code=sm_37',
                                     '-gencode arch=compute_50,code=sm_50',
                                     '-gencode arch=compute_52,code=sm_52',
                                     '-gencode arch=compute_52,code=compute_52',
                                     '-gencode=arch=compute_60,code=sm_60',
                                     '-gencode=arch=compute_61,code=sm_61',
                                     '--ptxas-options=-v', '-c',
                                     '--compiler-options', "'-fPIC'"]
                         },
                         include_dirs = [get_numpy_include_dirs()[0],
                                         CUDA['include'], 'src'])


def configuration(parent_package='', top_path=None):
    config = Configuration('lib', parent_package, top_path,
                           cmdclass={'build_ext': custom_build_ext})

    CUDA = locate_cuda()

    files = ['_supersegments.pyx']
    config.add_extension('_supersegments', sources=files,
                         language='c++', libraries=["stdc++"],
                         include_dirs=[get_numpy_include_dirs()])

    sources = ['src/cuda.cu', 'src/tv.cu', 'src/diffusion.cu',
               'src/chambolle2005.cu', 'src/chambolle2011.cu',
               'src/bregman.cu',  '_preprocess.pyx']
    cuda_extension(config, '_preprocess', sources, CUDA)

    sources = ['src/slic.cu', 'src/cuda.cu', '_superpixels.pyx']
    cuda_extension(config, '_superpixels', sources, CUDA)

    sources = ['src/convolutions_raw.cu',
               'src/convolutions_separable.cu',
               'src/convolutions_separable_shared.cu',
               'src/cuda.cu', '_convolutions.pyx']
    cuda_extension(config, '_convolutions', sources, CUDA)

    sources = ['src/symmetric_eigvals3S.cu', 'src/cuda.cu', '_channels.pyx']
    cuda_extension(config, '_channels', sources, CUDA)

    config.add_extension('_spencoding', sources=['_spencoding.pyx'],
                         include_dirs=get_numpy_include_dirs())
    config.add_extension('_features', sources=['_features.pyx'],
                         include_dirs=get_numpy_include_dirs())
    config.add_extension('_rag', sources=['_rag.pyx'],
                         include_dirs=get_numpy_include_dirs())

    config.add_extension('_dist', sources=['_dist.pyx'],
                         include_dirs=get_numpy_include_dirs())

    get_qpbo()
    qpbo_directory = source_path
    files = ["QPBO.cpp", "QPBO_extra.cpp", "QPBO_maxflow.cpp",
             "QPBO_postprocessing.cpp"]
    files = [os.path.join(qpbo_directory, f) for f in files]
    files = ['_qpbo.pyx'] + files
    config.add_extension('_qpbo', sources=files, language='c++',
                         libraries=["stdc++"],
                         include_dirs=[qpbo_directory, get_numpy_include_dirs()],
                         library_dirs=[qpbo_directory],
                         extra_compile_args=[],
                         extra_link_args=[])

    return config


if __name__ == '__main__':
    config = configuration(top_path='').todict()
    setup(**config)
