
import os
import numpy as np
from Cython.Distutils import build_ext
import os

#adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/

def find_in_path(name, path):
    "Find a file in a search path"
    for cdir in path.split(os.pathsep):
        binpath = os.path.join(cdir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)

    return None

def locate_cuda():
    nvcc = None
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc_path = os.path.join(home, 'bin', 'nvcc')
        if os.path.isfile(nvcc_path):
            nvcc = nvcc_path

    if nvcc is None:
        nvcc_path = find_in_path('nvcc', os.environ['PATH'])
        if nvcc_path is None or not os.path.isfile(nvcc_path):
            raise EnvironmentError('The nvcc binary could not be located in your'
                                   ' $PATH. Either add it to your path, or set'
                                   ' appropiate $CUDAHOME.')
        nvcc = nvcc_path
        home = os.path.dirname(os.path.dirname(nvcc_path))

    cudaconfig = {
        'home': home,
        'nvcc': nvcc,
        'include': os.path.join(home, 'include'),
        'lib64': os.path.join(home, 'lib64')
    }

    errmsg = 'The CUDA {} path could not be located in {}'

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(errmsg.format(k, v))

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
