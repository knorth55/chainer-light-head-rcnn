from __future__ import division

import argparse
import numpy as np

import chainer
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
from chainercv.chainer_experimental.datasets.sliceable \
    import ConcatenatedDataset
from chainercv.chainer_experimental.datasets.sliceable \
    import TransformDataset
from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.extensions import DetectionCOCOEvaluator
from chainercv import transforms
import chainermn

from light_head_rcnn.extensions import ManualScheduler
from light_head_rcnn.links import LightHeadRCNNResNet101
from light_head_rcnn.links import LightHeadRCNNTrainChain


class Transform(object):

    def __init__(self, light_head_rcnn):
        self.light_head_rcnn = light_head_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.light_head_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: LightHeadRCNN')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    # chainermn
    comm = chainermn.create_communicator()
    device = comm.intra_rank

    np.random.seed(args.seed)

    label_names = coco_bbox_label_names
    train_dataset = ConcatenatedDataset(
        COCOBboxDataset(split='train'),
        COCOBboxDataset(split='valminusminival'))
    test_dataset = COCOBboxDataset(
        split='minival', use_crowded=True,
        return_crowded=True, return_area=True)

    light_head_rcnn = LightHeadRCNNResNet101(
        n_fg_class=len(label_names), pretrained_model='imagenet')
    light_head_rcnn.use_preset('evaluate')
    model = LightHeadRCNNTrainChain(light_head_rcnn)

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    train_dataset = TransformDataset(
        train_dataset, ('img', 'bbox', 'label', 'scale'),
        Transform(model.light_head_rcnn))

    if comm.rank == 0:
        indices = np.arange(len(train_dataset))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train_dataset = train_dataset.slice[indices]
    train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size=1)

    if comm.rank == 0:
        test_iter = chainer.iterators.SerialIterator(
            test_dataset, batch_size=1, repeat=False, shuffle=False)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(momentum=0.9), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

    for param in model.params():
        if param.name in ['beta', 'gamma']:
            param.update_rule.enabled = False
    model.light_head_rcnn.extractor.conv1.disable_update()
    model.light_head_rcnn.extractor.res2.disable_update()

    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=device)

    trainer = chainer.training.Trainer(
        updater, (30, 'epoch'), out=args.out)

    def lr_schedule(updater):
        base_lr = 0.0005 * 1.25 * comm.size
        warm_up_duration = 500
        warm_up_rate = 1 / 3

        iteration = updater.iteration
        epoch = updater.epoch
        if iteration < warm_up_duration:
            rate = warm_up_rate \
                + (1 - warm_up_rate) * iteration / warm_up_duration
        elif epoch < 20:
            rate = 1
        elif epoch < 26:
            rate = 0.1
        else:
            rate = 0.01
        return base_lr * rate

    trainer.extend(ManualScheduler('lr', lr_schedule))

    if comm.rank == 0:
        # interval
        log_interval = 100, 'iteration'
        plot_interval = 3000, 'iteration'
        print_interval = 20, 'iteration'

        # training extensions
        model_name = model.light_head_rcnn.__class__.__name__
        trainer.extend(
            chainer.training.extensions.snapshot_object(
                model.light_head_rcnn,
                savefun=chainer.serializers.save_npz,
                filename='%s_model_iter_{.updater.iteration}.npz'
                         % model_name),
            trigger=(1, 'epoch'))
        trainer.extend(
            extensions.observe_lr(),
            trigger=log_interval)
        trainer.extend(
            extensions.LogReport(log_name='log.json', trigger=log_interval))
        report_items = [
            'iteration', 'epoch', 'elapsed_time', 'lr',
            'main/loss',
            'main/rpn_loc_loss',
            'main/rpn_cls_loss',
            'main/roi_loc_loss',
            'main/roi_cls_loss',
            'validation/main/map/iou=0.50:0.95/area=all/max_dets=100',
        ]

        trainer.extend(
            extensions.PrintReport(report_items), trigger=print_interval)
        trainer.extend(
            extensions.ProgressBar(update_interval=10))

        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(
                    ['main/loss'],
                    file_name='loss.png', trigger=plot_interval),
                trigger=plot_interval)

        trainer.extend(
            DetectionCOCOEvaluator(
                test_iter, model.light_head_rcnn, label_names=label_names),
            trigger=ManualScheduleTrigger([20, 26], 'epoch'))
        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()