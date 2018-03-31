---
title: Installation
layout: documentation
toc-section: 1
---

## Requirements

- Python Distribution (Python 2.7 or 3.6)/ Anaconda (recommended)

- NVIDIA GPU with at least 1024 threads

- CUDA SDK 9.0+ already installed and configured.

  If CUDA is not yet installed in the system, follow instructions in:

  [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

::: warning
NOTE: SuRVoS is not supported on Windows Python 2.7
:::
   
 
## 1. Installing a Python distribution

If there is no Python distribution installed on your system, we recommend installing [Anaconda](https://docs.continuum.io/anaconda/). Otherwise, skip to **step 2**.

### 1.1 Download Anaconda:

[https://www.continuum.io/downloads](https://www.continuum.io/downloads)

### 1.2 Install Anaconda :

**(Windows)** Just double click on the installer and follow instructions.

**(Linux)** Open a terminal and type the following commands:

```bash
$> cd /path/to/anaconda/
$> chmod a+x Anaconda2-4.0.0-Linux-x86_64.sh
$> ./Anaconda2-4.0.0-Linux-x86_64.sh
    1. press ENTER
    2. press Q
    3. enter "yes"
    4. pres ENTER
    5. enter "yes"
$> source ~/.bashrc
```

**NOTE:** Replace **2-4.0.0** with your version of Anaconda.

## 2. Install SuRVoS from conda channel

```bash
$> conda install -c conda-forge -c numba -c ccpi survos
```


## 3. Install SuRVoS from source

### 3.1 Download SuRVoS

Navigate in a terminal (using `cd`) to a folder where you want to save SuRVoS and type the following commands

```bash
$> git clone https://github.com/DiamondLightSource/SuRVoS.git
$> cd SuRVoS
```

### 3.2 Using Anaconda

#### 3.2.1 Windows

```bash
$> conda create -n ccpi python=3.6
$> activate ccpi
$[ccpi]> conda build conda-recipe -c conda-forge -c numba --python=3.6
$[ccpi]> conda install --use-local survos -c conda-forge -c numba --python=3.6
```

#### 3.2.2 Linux
Replace the &lt;CUDA_HOME&gt; with the path to the CUDA install directory. Generally CUDA SDK 9.0 is installed in /usr/local/cuda-9.0.

```bash
$> conda create -n ccpi python=3.6
$> activate ccpi
$[ccpi]> export PATH=<CUDA_HOME>/bin:$PATH
$[ccpi]> export LD_LIBRARY_PATH=<CUDA_HOME>/lib64:$LD_LIBRARY_PATH
$[ccpi]> conda build conda-recipe -c conda-forge -c numba --python=3.6
$[ccpi]> conda install --use-local survos -c conda-forge -c numba --python=3.6
```
**NOTE:** Replace **3.6** with 2.7 for building using python 2.7.

### 3.3 Using standard python

#### 3.3.1 Windows
**NOTE** Not yet supported.

#### 3.3.2 Linux
This step requires CUDA already installed and NVCC compiler in the path (type `which nvcc` to verify it).

```bash
$> export PATH=<CUDA_HOME>/bin:$PATH
$> export LD_LIBRARY_PATH=<CUDA_HOME>/lib64:$LD_LIBRARY_PATH
$> python -m venv ccpi
$> . ccpi/bin/activate
$(ccpi)> pip install cmake cython numpy scipy matplotlib h5py pyqt5==5.8.2 tifffile networkx scikit-image scikit-learn seaborn 
$(ccpi)> cmake -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${VIRTUAL_ENV}" -DINSTALL_BIN_DIR="${VIRTUAL_ENV}/bin" -DINSTALL_LIB_DIR="${VIRTUAL_ENV}/lib64" survos/lib/src
$(ccpi)> make
$(ccpi)> make install
$(ccpi)> python setup.py build
$(ccpi)> python setup.py install
$(ccpi)> export LD_LIBRARY_PATH=${VIRTUAL_ENV}/lib64:$LD_LIBRARY_PATH
$(ccpi)> SuRVoS
```
## 4. Run SuRVoS

From the SuRVoS folder:

```bash
$> SuRVoS
```
