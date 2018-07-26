# COCO Object Detection

![Example](../static/coco_example.png)

## Note 

Both inference and training takes very long time with CPU.

I recommend to use GPUs.

## Inference

### COCO Trained Model
Trained model can be dowloaded [here](https://github.com/knorth55/chainer-light-head-rcnn/releases/download/v0.0/light_head_rcnn_resnet101_trained_2018_07_19.npz).

This model is trained with this repository.

### Command

```bash
python demo.py --gpu <gpu>
```
## Training

### Requirements

- [Cython](http://cython.org/)
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [OpenMPI](https://www.open-mpi.org/)
- [nccl](https://developer.nvidia.com/nccl)
- [ChainerMN](https://github.com/chainer/chainermn)

### Command

```bash
mpiexec -n <gpu_num> python train_multi.py
```

## Evaluation

### Evaluation Score

| Implementation | mAP@0.5:0.95/all | mAP@0.5/all | mAP@0.75/all | mAP:0.5:0.95/small | mAP:0.5:0.95/medium | mAP:0.5:0.95/large |
|:--------------:|:----------------:|:-----------:|:------------:|:------------------:|:-------------------:|:------------------:|
| [Original](https://github.comzengarden/light_head_rcnn) | 0.400 | 0.621 | 0.429 | 0.225 | 0.446 | 0.540 |
| Ours | 0.391 | 0.607 | 0.419 | 0.212 | 0.428 | 0.541 |

### Command

```bash
python eval_coco.py --gpu <gpu>
```

## Model conversion

Convert model to chainer model.

Original tensorflow parameters can be downloaded [here](https://github.com/knorth55/chainer-light-head-rcnn/releases/download/v0.0/tf_light_head_rcnn_resnet101_extracted_2018_07_12.npz).

```bash
python tfnpz2npz.py <tfmodelpath> --out <chainermodelpath>
```
