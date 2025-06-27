Installation
============

First, download and install the latest `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.

Create a new environment with Python 3.10 (we used Anaconda).

Ensure Git is installed (we used Anaconda):
::

   conda install anaconda::git

Update dependencies:
::

   pip install --upgrade pip setuptools wheel

Install `PyTorch <https://pytorch.org/>`_ (Make sure the CUDA version matches if using GPU):
::

   pip3 install torch torchvision torchaudio

Install `Detectron2 <https://github.com/facebookresearch/detectron2>`_:
::

   pip install git+https://github.com/facebookresearch/detectron2

Install additional dependencies:
::

   pip install numpy==1.23 opencv-python filterpy super-gradients

Install BubbleID:
::

   pip install bubbleid
