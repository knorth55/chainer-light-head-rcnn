# COCO Object Detection

![Example](../static/coco_example.png)

## Note 

Both inference and training takes very long time with CPU.

I recommend to use GPUs.

## Inference

Converted model can be dowloaded [here](https://github.com/knorth55/chainer-light-head-rcnn/releases/download/v0.0/light_head_rcnn_resnet101_converted_2018_07_12.npz).

This model is converted from one trained by original repo.

```bash
python demo.py --gpu <gpu>
```
## Training

**In progress**

### Requirements
- [Cython](http://cython.org/)
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [OpenMPI](https://www.open-mpi.org/)
- [nccl](https://developer.nvidia.com/nccl)
- [ChainerMN](https://github.com/chainer/chainermn)

```bash
mpiexec -n <gpu_num> python train_multi.py
```

## Evaluation

**In progress**


```bash
python eval_coco.py --gpu <gpu>
```

## Model conversion

Convert model to chainer model.

Original tensorflow parameters can be downloaded [here](https://github.com/knorth55/chainer-light-head-rcnn/releases/download/v0.0/tf_light_head_rcnn_extracted_2018_07_12.npz).

```bash
python tfnpz2npz.py <tfmodelpath> --out <chainermodelpath>
```
