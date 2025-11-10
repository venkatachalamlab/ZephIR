![logo](https://user-images.githubusercontent.com/39420322/179288567-257d5aa4-c19f-42b3-be58-cd77bd18d561.png)

![release](https://img.shields.io/github/v/release/venkatachalamlab/zephir)
[![PyPI](https://img.shields.io/pypi/v/zephir)](https://pypi.org/project/zephir/)
[![Downloads](https://pepy.tech/badge/zephir)](https://pepy.tech/project/zephir)
[![GitHub](https://img.shields.io/github/license/venkatachalamlab/ZephIR)](https://github.com/venkatachalamlab/ZephIR/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/venkatachalamlab/ZephIR.svg?style=social&label=Star)](https://github.com/venkatachalamlab/ZephIR)
[![Youtube](https://img.shields.io/badge/YouTube-Demo-red)](https://youtu.be/4O9aIftvoqM)


ZephIR is a multiple object tracking algorithm based on image registration and built on PyTorch. Check out the [publication](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012075) and [tutorial video](https://youtu.be/4O9aIftvoqM)!

### Latest Updates in v1.0.6

* bug fixes for deprecated dependencies
* see [release notes](https://github.com/venkatachalamlab/ZephIR/releases) for full list of changes

## About

ZephIR tracks keypoints in a 2D or 3D movie by registering image descriptors sampled around each keypoint.
Image registration loss is combined with three additional regularization terms:
- spring connections between neighboring objects allow a flexible spatial model of loosely correlated motion
- feature detection optimizes results towards centers of detected features
- temporal smoothing of pixel intensity a small patch of frames limit fluctuations in activity

Overview of tracking loss:

![loss](https://user-images.githubusercontent.com/39420322/179583408-79b86ebc-7d44-4fd0-ab80-a53eee300c16.png)


ZephIR is fast, efficient, and designed to run on laptops instead of powerful desktop workstations. 
It requires no prior training of any model weights, and it is capable of generalizing to a wide diversity of datasets with small tweaks to parameters. 
This makes ZephIR ideal for analyzing datasets that lack a large corpus of training data, and for tracking fluorescent sources in moving and deforming tissue, both of which create a particularly challenging environment for modern deep learning techniques.
ZephIR can also serve as a data augmentation tool in some cases.
We provide some support for exporting ZephIR results to [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut).


## Installation

### Quick start

```bash
pip install zephir
```

### Dependencies

Make sure that **Python (>=3.11)** and the following packages are installed (prefer uv over pip):
  - dataclasses (>=0.8)
  - docopt-ng (>=0.9.0)
  - flask (>=3.1.2)
  - gevent (>=25.9.0)
  - h5py (>=3.15.1)
  - matplotlib (>=3.10.7)
  - numpy (>=2.0)
  - opencv-python (>=4.6.0)
  - pandas (>=2.3.3)
  - scikit-image (>=0.19.2)
  - scikit-learn (>=1.7.2)
  - scipy (>=1.16.0)
  - setuptools (>=80.9.0)
  - torch (>=2.9.0) (see [PyTorch.org](https://pytorch.org/get-started/locally/) for instructions on installing with CUDA)
  - tqdm (>=4.64.0)

> **NOTE**: To enable GPU acceleration, you need to compile PyTorch with CUDA by following the instructions on [PyTorch.org](https://pytorch.org/get-started/locally/). 
Using the quick installation command will *not* install CUDA-enabled PyTorch automatically. 
This installation may also produce an `OMP: Error #15`. 
If you run into this issue, check your `numpy` and `numpy-base` versions and if they disagree, update one or the other to match.  

> **NOTE**: If you run into an issue when launching the annotator, you may need to downgrade `Flask` (2.1.2) and `Pandas` (1.4.2).


### Build from source

1. Clone git repository: 
  ```bash
  git clone https://github.com/venkatachalamlab/ZephIR.git
  ```  

2. Navigate to the cloned directory on your local machine.

3. Checkout the current release:
```bash
git checkout v1.0.6
```
Use the following command to see what's new in the most recent release:
```bash
git show v1.0.6
```

4. Install:
  ```bash
  python3 -m pip install .
  ```
  or install in development mode:
  ```bash
  python3 -m pip install -e .
  ```

## Getting Started

Run from command line:
  ```bash
  zephir --dataset=. [options]
  ```

We provide a detailed guide for running ZephIR as well as some example workflows for using ZephIR [here](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-ZephIR.md).

## Parameters

For a list of all CLI options and user-tunable parameters, see [here](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-parameters.md).

To help figure out what options may be right for you, check out the list of examples with explanations for the chosen parameters [here](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/examples.md).

## Interfacing with Annotator

ZephIR includes an annotator GUI with custom Python macros for interacting with the data from the GUI. 

Run from command line:
```bash
annotator --dataset=. [--port=5005]
```

Learn more about the annotator and its features [here](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/annotatorGUI.md).

We also provide a more detailed user guide for using the GUI as a part of a ZephIR workflow [here](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md).
