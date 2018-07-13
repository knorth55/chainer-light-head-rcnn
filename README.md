chainer-light-head-rcnn - Light Head RCNN
=========================================

![Build Status](https://travis-ci.com/knorth55/chainer-light-head-rcnn.svg?branch=master)

![Example](static/coco_example.png)

This is [Chainer](https://github.com/chainer/chainer) implementation of [Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://arxiv.org/abs/1711.07264).

Original TensorFlow repository is [zengarden/light_head_rcnn](https://github.com/zengarden/light_head_rcnn).

Requirement
-----------

- [CuPy](https://github.com/cupy/cupy)
- [Chainer](https://github.com/chainer/chainer)
- [ChainerCV](https://github.com/chainer/chainercv)
- OpenCV2

Additional Requirement
----------------------
- For COCO Dataset class
  - [Cython](http://cython.org/)
  - [pycocotools](https://github.com/cocodataset/cocoapi)

TODO
----
- COCO
  - [ ] Reproduce original repo training accuracy
  - [ ] Refine evaluation code

Installation
------------

We recommend to use [Anacoda](https://anaconda.org/).

```bash
# Requirement installation
conda create -n fcis python=3.6
conda install -c menpo opencv
source activate light-head-rcnn
pip install cupy

# Installation
git clone https://github.com/knorth55/chainer-light-head-rcnn.git
pip install -e .
```

Inference
---------
```bash
cd examples/
python demo.py <image.jpg> --gpu <gpu>
```

Training
--------

**In Progress**

LICENSE
-------
[MIT LICENSE](LICENSE)
