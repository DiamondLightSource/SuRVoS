---
title: Installation
layout: documentation
toc-section: 1
---

## Requirements

- Python Distribution

- NVIDIA GPU with at least 1024 threads

- CUDA SDK already installed and configured.

  If CUDA is not yet installed in the system, follow instructions in:

  [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

## 1. Installing a Python distribution

If there is no Python distribution installed on your system, we recommend installing [Anaconda](https://docs.continuum.io/anaconda/). Otherwise, skip to **step 2**.

### 1.1 Download Anaconda:

[https://www.continuum.io/downloads](https://www.continuum.io/downloads)

### 1.2 Install Anaconda:

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

## 2. Installing Dependencies

With Anaconda:

```bash
$> conda update conda
$> conda install numpy scipy matplotlib scikit-learn scikit-image cython seaborn networkx
$> pip install scikit-tensor
```

With another python distribution:

```bash
$> pip install --upgrade pip
$> pip install --upgrade numpy scipy matplotlib scikit-learn scikit-image cython seaborn networkx scikit-tensor
```

## 3. Install SuRVoS

### 3.1 Download SuRVoS

Navigate in a terminal (using `cd`) to a folder where you want to save SuRVoS and type the following commands

```bash
$> git clone https://github.com/DiamondLightSource/SuRVoS.git
$> cd SuRVoS
```

### 3.2 Compile SuRVoS features

This step requires CUDA already installed and NVCC compiler in the path (type `which nvcc` to verify it).

```bash
$> env PYTHONPATH=.. python setup.py build_ext -i
```

## 4. Run SuRVoS

From the SuRVoS folder:

```bash
$> env PYTHONPATH=.. ./SuRVoS
```
