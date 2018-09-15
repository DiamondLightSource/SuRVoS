#!/usr/bin/env python
import setuptools
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os
import numpy
import platform	
import sys
import urllib
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import zipfile
import glob
import shutil

extra_include_dirs = [numpy.get_include(), '../functions/']
extra_library_dirs = []
extra_compile_args = []
extra_link_args    = []
extra_libraries    = ['survos_cuda']

if platform.system() == 'Windows':
    extra_compile_args += ['/DWIN32']
elif platform.system() == 'Darwin':
    extra_compile_args += [ '-O2', '-Wall']
    extra_library_dirs += ['.']
    extra_libraries += ['m']    
else:
    extra_compile_args += [ '-O2', '-Wall', '-std=c99']
    extra_library_dirs += ['.']
    extra_libraries += ['m']  
    
base_path = os.path.abspath(os.path.dirname(__file__))
source_path = os.path.join(base_path, 'survos', 'lib', 'qpbo_src')

def apply_QPBO_fix(qpbo_dir):
    content = []
    with open(os.path.join(qpbo_dir,'instances.inc'),"r") as f:
        content = f.readlines()
    content.insert(8, "template <>") 
    content.insert(9, "inline void QPBO<int>::get_type_information(const char*& type_name, const char*& type_format);")
    content.insert(10, "template <>")
    content.insert(11, "inline void QPBO<float>::get_type_information(const char*& type_name, const char*& type_format);")
    content.insert(12, "template <>") 
    content.insert(13, "inline void QPBO<double>::get_type_information(const char*& type_name, const char*& type_format);")
    with open(os.path.join(qpbo_dir,'instances.inc'),"w") as f:
       f .writelines(content)

def get_qpbo():
    if os.path.isdir(source_path):
        return
    else:
        os.mkdir(source_path)


    qpbo_version = 'QPBO-v1.4.src'
    qpbo_file = '{}.zip'.format(qpbo_version)

    urlretrieve('http://pub.ist.ac.at/~vnk/software/{}'.format(qpbo_file), qpbo_file)
    with zipfile.ZipFile(qpbo_file) as zf:
        zf.extractall(source_path)

    for f in glob.glob(os.path.join(source_path, qpbo_version, '*')):
        shutil.move(f, os.path.dirname(os.path.dirname(f)))

    os.rmdir(os.path.join(source_path, qpbo_version))
    os.remove(qpbo_file)
    #Apply patch for Mac
    if platform.system() == 'Darwin':
        apply_QPBO_fix(source_path)

get_qpbo()
qpbo_directory = source_path
files = ["QPBO.cpp", "QPBO_extra.cpp", "QPBO_maxflow.cpp",
             "QPBO_postprocessing.cpp"]
files = [os.path.join(qpbo_directory, f) for f in files]   
extra_include_dirs +=  [qpbo_directory]
print(extra_include_dirs)
extensions = [
    Extension("survos.lib._channels", 
              sources = ["survos/lib/_channels.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),
    Extension("survos.lib._preprocess",
              sources = ["survos/lib/_preprocess.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),
    Extension("survos.lib._convolutions", 
              sources = ["survos/lib/_convolutions.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),
    Extension("survos.lib._dist", 
              sources = ["survos/lib/_dist.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),
    Extension("survos.lib._features", 
              sources = ["survos/lib/_features.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),
    Extension("survos.lib._rag", 
              sources = ["survos/lib/_rag.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),
    Extension("survos.lib._spencoding", 
              sources = ["survos/lib/_spencoding.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),
    Extension("survos.lib._superpixels", 
              sources = ["survos/lib/_superpixels.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args), 
    Extension("survos.lib._supersegments", 
              sources = ["survos/lib/_supersegments.pyx",
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),  
    Extension("survos.lib._qpbo", 
              sources = ["survos/lib/_qpbo.pyx",
                         "survos/lib/qpbo_src/QPBO.cpp",
                         "survos/lib/qpbo_src/QPBO_extra.cpp",
                         "survos/lib/qpbo_src/QPBO_maxflow.cpp",
                         "survos/lib/qpbo_src/QPBO_postprocessing.cpp",              
                        ],
              include_dirs = extra_include_dirs,
              library_dirs = extra_library_dirs,
              extra_compile_args = extra_compile_args,
              libraries = extra_libraries,
              extra_link_args = extra_link_args),                
    ]    
setup(
     name = 'survos',
	 version = "1.1.3",
	 description = '(Su)per (R)egion (Vo)lume (S)egmentaton workbench',
	 license='GPL',
     packages = find_packages(),
     include_package_data=True,
     install_requires=['numpy','scipy','matplotlib'],
	 zip_safe = False,
     ext_modules = cythonize(extensions, language='c++'),
     package_data={'survos.images':['survos/images/*.png',],
                   'survos.images.PNG':['survos/images/PNG/*.png',],
                   'survos.images.SVG':['survos/images/SVG/*.svg',],
                   'survos.launcher':['survos/launcher/*.qcs',],                                      
                  },
	 entry_points={
					'console_scripts': [
										'SuRVos=survos.launcher.survos_launcher:main'
									   ],
				  },
     
)
