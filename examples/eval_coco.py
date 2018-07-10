import argparse

import chainer
from chainer import iterators
from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.evaluations import eval_detection_coco
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from light_head_rcnn.links import LightHeadRCNNResNet101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    model = LightHeadRCNNResNet101(
        n_fg_class=len(coco_bbox_label_names),
        pretrained_model=args.pretrained_model)
    model.use_preset('evaluate')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = COCOBboxDataset(
        split='minival', use_crowded=True,
        return_crowded=True, return_area=True)
    iterator = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    pred_bboxes, pred_labels, pred_scores = out_values
    gt_bboxes, gt_labels, gt_areas, gt_crowdeds = rest_values

    result = eval_detection_coco(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_areas, gt_crowdeds)

    keys = [
        'map/iou=0.50:0.95/area=all/max_dets=100',
        'map/iou=0.50/area=all/max_dets=100',
        'map/iou=0.75/area=all/max_dets=100',
        'map/iou=0.50:0.95/area=small/max_dets=100',
        'map/iou=0.50:0.95/area=medium/max_dets=100',
        'map/iou=0.50:0.95/area=large/max_dets=100',
        'mar/iou=0.50:0.95/area=all/max_dets=1',
        'mar/iou=0.50:0.95/area=all/max_dets=10',
        'mar/iou=0.50:0.95/area=all/max_dets=100',
        'mar/iou=0.50:0.95/area=small/max_dets=100',
        'mar/iou=0.50:0.95/area=medium/max_dets=100',
        'mar/iou=0.50:0.95/area=large/max_dets=100',
    ]

    print('')
    for key in keys:
        print('{:s}: {:f}'.format(key, result[key]))


if __name__ == '__main__':
    main()
