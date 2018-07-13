import numpy as np
import warnings

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator

from light_head_rcnn.links.model.utils.proposal_target_creator import\
    ProposalTargetCreator


class LightHeadRCNNTrainChain(chainer.Chain):

    """Calculate losses for Light Head R-CNN and report them.

    This is used to train Light Head R-CNN in the joint training scheme
    [#LHRCNN]_.

    .. [#LHRCNN] Zeming Li, Chao Peng, Gang Yu, Xiangyu Zhang, Yangdong Deng, \
    Jian Sun. Light-Head R-CNN: In Defense of Two-Stage Object Detector. \
    arXiv preprint arXiv:1711.07264.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.

    Args:
        light_head_rcnn (~light_head_rcnn.links.light_head_rcnn.LightHeadRCNN):
            A Light Head R-CNN model that is going to be trained.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#LHRCNN]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#LHRCNN]_.
        anchor_target_creator: An instantiation of
            :class:`~chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator: An instantiation of
            :class:`~light_head_rcnn.links.model.utils.ProposalTargetCreator`.

    """

    def __init__(
            self, light_head_rcnn,
            rpn_sigma=3., roi_sigma=1., n_ohem_sample=256,
            anchor_target_creator=AnchorTargetCreator(),
            proposal_target_creator=ProposalTargetCreator()
    ):
        super(LightHeadRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.light_head_rcnn = light_head_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.n_ohem_sample = n_ohem_sample

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

        self.loc_normalize_mean = light_head_rcnn.loc_normalize_mean
        self.loc_normalize_std = light_head_rcnn.loc_normalize_std

    def __call__(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            bboxes (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.array
        if isinstance(labels, chainer.Variable):
            labels = labels.array
        if isinstance(scale, chainer.Variable):
            scale = scale.array
        scale = np.asscalar(cuda.to_cpu(scale))
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        rpn_features, roi_features = self.light_head_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.light_head_rcnn.rpn(rpn_features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi, bbox, label, self.loc_normalize_mean, self.loc_normalize_std)
        sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)
        roi_cls_loc, roi_score = self.light_head_rcnn.head(
            roi_features, sample_roi, sample_roi_index)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head.
        roi_loc_loss, roi_cls_loss = _ohem_loss(
            roi_score, roi_cls_loc, gt_roi_label, gt_roi_loc,
            self.n_ohem_sample, self.roi_sigma)
        roi_loc_loss = 2 * roi_loc_loss

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
        chainer.reporter.report({'rpn_loc_loss': rpn_loc_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'roi_loc_loss': roi_loc_loss,
                                 'roi_cls_loss': roi_cls_loss,
                                 'loss': loss},
                                self)
        return loss


def _ohem_loss(
        roi_score, roi_cls_loc, gt_roi_label, gt_roi_loc,
        n_ohem_sample, roi_sigma=1.0
):
    xp = cuda.get_array_module(roi_cls_loc)
    n_sample = roi_cls_loc.shape[0]
    roi_cls_loc = roi_cls_loc.reshape((n_sample, -1, 4))
    roi_loc = roi_cls_loc[xp.arange(n_sample), gt_roi_label]
    roi_loc_loss = _fast_rcnn_loc_loss(
        roi_loc, gt_roi_loc, gt_roi_label, roi_sigma, reduce='no')
    roi_cls_loss = F.softmax_cross_entropy(
        roi_score, gt_roi_label, reduce='no')
    assert roi_loc_loss.shape == roi_cls_loss.shape

    roi_cls_loc_loss = roi_loc_loss.array + roi_cls_loss.array
    n_ohem_sample = min(n_ohem_sample, n_sample)
    indices = roi_cls_loc_loss.argsort(axis=0)[::-1][:n_ohem_sample]
    roi_loc_loss = F.sum(roi_loc_loss[indices]) / n_ohem_sample
    roi_cls_loss = F.sum(roi_cls_loss[indices]) / n_ohem_sample

    return roi_loc_loss, roi_cls_loss


def _smooth_l1_loss_base(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.array < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return F.sum(y, axis=1)


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma, reduce='mean'):
    xp = cuda.get_array_module(pred_loc)

    in_weight = xp.zeros_like(gt_loc)
    # Localization loss is calculated only for positive rois.
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss_base(pred_loc, gt_loc, in_weight, sigma)
    # Normalize by total number of negtive and positive rois.
    if reduce == 'mean':
        loc_loss = F.sum(loc_loss) / xp.sum(gt_label >= 0)
    elif reduce != 'no':
        warnings.warn('no reduce option: {}'.format(reduce))
    return loc_loss
