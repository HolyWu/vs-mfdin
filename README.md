# MFDIN
Multiframe Joint Enhancement for Early Interlaced Videos, based on https://github.com/anymyb/MFDIN.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.5.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.3.0 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.5.0 or later

To install the latest nightly build of PyTorch and Torch-TensorRT, run:
```
pip install -U packaging setuptools wheel
pip install --pre -U torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
pip install --no-deps --pre -U torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu124
pip install -U tensorrt-cu12 tensorrt-cu12_bindings tensorrt-cu12_libs --extra-index-url https://pypi.nvidia.com
```


## Installation
```
pip install -U vsmfdin
```


## Usage
```python
from vsmfdin import mfdin

ret = mfdin(clip)
```

See `__init__.py` for the description of the parameters.
