
import os
import numpy as np
from Cython.Distutils import build_ext
PATH = os.environ.get('PATH')
from distutils.spawn import spawn, find_executable
import os
from sys import platform
import sys
import distutils
from distutils.debug import DEBUG
from distutils.errors import DistutilsPlatformError, DistutilsExecError


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
        if platform == "win32" :
            nvcc_path = os.path.join(home, 'bin', 'nvcc.exe')
        else:
            nvcc_path = os.path.join(home, 'bin', 'nvcc')
        if os.path.isfile(nvcc_path):
            nvcc = nvcc_path

    if nvcc is None:
        nvcc_path = find_in_path('nvcc.exe', os.environ['PATH'])
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
        'lib64': os.path.join(home, 'lib\\x64')
    }

    errmsg = 'The CUDA {} path could not be located in {}'

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(errmsg.format(k, v))

    return cudaconfig

CUDA = locate_cuda()

def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    self.initialize()
    default_compiler_so = self.cc    
#    super = self._compile
    self.cc = CUDA['nvcc']
    print("CUDA Compiler:"+CUDA['nvcc'])
    self._cpp_extensions.append('.cu')
    self.src_extensions.append('.cu')
   
#    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
#        print("CUDA Compiler"+CUDA['nvcc'])
#        ext = os.path.splitext(src)[1]
#        if ext == '.cu':
#            self.set_executable('cc', CUDA['nvcc'])
#            postargs = extra_postargs['nvcc']
#        elif type(extra_postargs) == dict:
#            if ext == '.cpp':
#                postargs = extra_postargs['g++']
#            else:
#                postargs = extra_postargs['gcc']
#        else:
#            postargs = extra_postargs
#
#        super(obj, src, ext, cc_args, postargs, pp_opts)

        #self.cc = default_compiler_so
 #   self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):

    def build_extensions(self):
        print("Setting custom extension")
        customize_compiler_for_nvcc(self.compiler)
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)



    def spawn(self, cmd, search_path=1, verbose=1, dry_run=0):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        print("Invoking custom spawn")
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        elif self.compiler.compiler_type == 'msvc':
            # There are several things we need to do to change the commands
            # issued by MSVCCompiler into one that works with nvcc. In the end,
            # it might have been easier to write our own CCompiler class for
            # nvcc, as we're only interested in creating a shared library to
            # load with ctypes, not in creating an importable Python extension.
            # - First, we replace the cl.exe or link.exe call with an nvcc
            #   call. In case we're running Anaconda, we search cl.exe in the
            #   original search path we captured further above -- Anaconda
            #   inserts a MSVC version into PATH that is too old for nvcc.
            cmd[:1] = ['nvcc', '--compiler-bindir',
                       '"'+os.path.dirname(find_executable("cl.exe", PATH))+'"'
                       or cmd[0]]
            print(cmd)
            # - Secondly, we fix a bunch of command line arguments.
            for idx, c in enumerate(cmd):
                # create .dll instead of .pyd files
                if '.pyd' in c: cmd[idx] = c = c.replace('.pyd', '.dll')
                # replace /c by -c
                if c == '/c': cmd[idx] = '-c'
                # replace /DLL by --shared
                elif c == '/DLL': cmd[idx] = '--shared'
                # remove --compiler-options=-fPIC
                elif '-fPIC' in c: del cmd[idx]
                # replace /Tc... by ...
                elif c.startswith('/Tc'): cmd[idx] = c[3:]
                elif c.startswith('/Tp'): cmd[idx] = c[3:]
                elif c.startswith('/MD'): cmd[idx] = '/MT'
                elif c.startswith('/MANIFEST:EMBED,ID=2'): cmd[idx] = ' '
                # replace /Fo... by -o ...
                elif c.startswith('/Fo'): cmd[idx:idx+1] = ['-o ', c[3:]]
                # replace /LIBPATH:... by -L...
                elif c.startswith('/LIBPATH:'): cmd[idx] = '-L' + c[9:]
                # replace /OUT:... by -o ...
                elif c.startswith('/OUT:'): cmd[idx:idx+1] = ['-o', c[5:]]
                # remove /EXPORT:initlibcudamat or /EXPORT:initlibcudalearn
                elif c.startswith('/EXPORT:'): del cmd[idx]
                # replace cublas.lib by -lcublas
                elif c == 'cublas.lib': cmd[idx] = '-lcublas'
            
            # - Finally, we pass on all arguments starting with a '/' to the
            #   compiler or linker, and have nvcc handle all other arguments
            if '--shared' in cmd:
                pass_on = '--cudart=shared --linker-options='
                # we only need MSVCRT for a .dll, remove CMT if it sneaks in:
                #cmd.append('/NODEFAULTLIB:libcmt.lib')
            else:
                pass_on = '--compiler-options='
            cmd = ([c for c in cmd if c[0] != '/'] +
                   [pass_on + ','.join(c for c in cmd if c[0] == '/')])
            # For the future: Apart from the wrongly set PATH by Anaconda, it
            # would suffice to run the following for compilation on Windows:
            # nvcc -c -O -o <file>.obj <file>.cu
            # And the following for linking:
            # nvcc --shared -o <file>.dll <file1>.obj <file2>.obj -lcublas
            # This could be done by a NVCCCompiler class for all platforms.
        self.spawn_nt(cmd, search_path, verbose, dry_run)
        
    
    def spawn_nt(self, cmd, search_path=1, verbose=0, dry_run=0):
        executable = cmd[0]
        if search_path:
            # either we find one or it stays the same
            executable = find_executable(executable) or executable
        print(' '.join([executable] + cmd[1:]))
        if not dry_run:
            # spawn for NT requires a full path to the .exe
            try:
                rc = os.spawnv(os.P_WAIT, executable, cmd)
            except OSError as exc:
                # this seems to happen when the command isn't found
                if not DEBUG:
                    cmd = executable
                raise DistutilsExecError(
                      "command %r failed: %s" % (cmd, exc.args[-1]))
            if rc != 0:
                # and this reflects the command running but failing
                if not DEBUG:
                    cmd = executable
                raise DistutilsExecError(
    "command %r failed with exit status %d" % (cmd, rc))
