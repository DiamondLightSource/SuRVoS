
cp -rv ${RECIPE_DIR}/../ ${SRC_DIR}
cd ${SRC_DIR}
python setup.py build
python setup.py install
