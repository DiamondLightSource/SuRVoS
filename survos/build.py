
import os
import numpy as np
from Cython.Distutils import build_ext
from os.path import join as pjoin

#adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/

def find_in_path(name, path):
    "Find a file in a search path"
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig

CUDA = locate_cuda()

def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')

    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        ext = os.path.splitext(src)[1]
        if ext == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        elif type(extra_postargs) == dict:
            if ext == '.cpp':
                postargs = extra_postargs['g++']
            else:
                postargs = extra_postargs['gcc']
        else:
            postargs = extra_postargs

        super(obj, src, ext, cc_args, postargs, pp_opts)

        self.compiler_so = default_compiler_so

    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)
