ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%"
cd "%SRC_DIR%"
cmake -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%PREFIX%\lib" -DINSTALL_BIN_DIR="%PREFIX%\bin" -DINSTALL_LIB_DIR="%PREFIX%\lib" %SRC_DIR%\survos\lib\src
if errorlevel 1 exit 1

make build
make install

python setup.py build_ext
python setup.py install