#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}
TENSORRT_TAR="/private/ningqingqun/tensorrt/TensorRT-5.1.5.0.Ubuntu-16.04.5.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz"
PYCUDA_TAR="/private/ningqingqun/tensorrt/pycuda-2019.1.2.tar.gz"
INSTALL_PREFIX="/usr/local/"
CWD=`pwd` && cd ${INSTALL_PREFIX}

sudo tar xzvf ${TENSORRT_TAR} -C ${INSTALL_PREFIX}
sudo ln -s TensorRT-5.1.5.0 tensorrt

# install python lib
cd tensorrt/python
sudo ${PYTHON} -m pip install tensorrt-5.1.5.0-cp36-none-linux_x86_64.whl

# install graphsurgeon
cd ../graphsurgeon
sudo ${PYTHON} -m pip install graphsurgeon-0.4.1-py2.py3-none-any.whl

# install pycuda
sudo tar xzvf ${PYCUDA_TAR} -C ${INSTALL_PREFIX}
cd ${INSTALL_PREFIX}/pycuda-2019.1.2
sudo ${PYTHON} configure.py --cuda-root=/usr/local/cuda
sudo make install

# test pycuda
sudo ${PYTHON} -m pip install pytest
cd test
${PYTHON} test_driver.py

cd ${CWD}
echo "install done!"
