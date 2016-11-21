
all: ; python setup.py build_ext --inplace

clean: clean_so clean_c clean_cpp clean_pyc clean_build


clean_so: ; find . -name "*.so" -exec rm {} \;

clean_c: ; find . -name "_*.c" -exec rm {} \;

clean_cpp: ; find . -name "_*.cpp" -exec rm {} \;

clean_pyc: ; find . -name "*.pyc" -exec rm {} \;

clean_build: ; rm -rf build/
