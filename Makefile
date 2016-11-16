
all:
    python setup.py build_ext --inplace

clean:
    find . -name "*.so" -exec rm {} \;
    find . -name "_*.cpp" -exec rm {} \;
    find . -name "_*.c" -exec rm {} \;
    rm -rf survos/build/;
