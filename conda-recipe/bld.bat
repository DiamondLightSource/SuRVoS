ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%"
cd "%SRC_DIR%"
cmake -G "NMake Makefiles" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%PREFIX%\Library" -DINSTALL_BIN_DIR="%PREFIX%\Library\bin" -DINSTALL_LIB_DIR="%PREFIX%\Library\lib" %SRC_DIR%\survos\lib\src
if errorlevel 1 exit 1

nmake
if errorlevel 1 exit 1

nmake install
if errorlevel 1 exit 1

python setup.py build_ext
python setup.py install --single-version-externally-managed --record=record.txt
