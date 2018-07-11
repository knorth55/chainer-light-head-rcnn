import argparse
import numpy as np
import re

import chainer

from light_head_rcnn.links import LightHeadRCNNResNet101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out', '-o', type=str, default='chainer_light_head_rcnn.npz')
    parser.add_argument('tfmodel')
    args = parser.parse_args()
    tf_data = np.load(args.tfmodel)
    model = LightHeadRCNNResNet101(n_fg_class=80, pretrained_model='imagenet')

    for key in tf_data.keys():
        value = tf_data[key]
        key_list = re.split('/|:', key)
        if key_list[1] == 'conv1':  # conv1
            if key_list[2] == 'weights':  # conv
                value = value.transpose((3, 2, 0, 1))
                assert model.extractor.conv1.conv.W.shape == value.shape
                np.testing.assert_array_equal(
                    model.extractor.conv1.conv.W.array, value)
                model.extractor.conv1.conv.W.array[:] = value
            elif key_list[2] == 'BatchNorm':  # bn
                if key_list[3] == 'gamma':  # gamma
                    assert model.extractor.conv1.bn.gamma.shape == value.shape
                    np.testing.assert_array_equal(
                        model.extractor.conv1.bn.gamma.array, value)
                    model.extractor.conv1.bn.gamma.array[:] = value
                elif key_list[3] == 'beta':  # beta
                    assert model.extractor.conv1.bn.beta.shape == value.shape
                    np.testing.assert_array_equal(
                        model.extractor.conv1.bn.beta.array, value)
                    model.extractor.conv1.bn.beta.array[:] = value
                elif key_list[3] == 'moving_mean':  # avg_mean
                    np.testing.assert_array_equal(
                        model.extractor.conv1.bn.avg_mean, value)
                    assert model.extractor.conv1.bn.avg_mean.shape \
                        == value.shape
                    model.extractor.conv1.bn.avg_mean[:] = value
                elif key_list[3] == 'moving_variance':  # avg_var
                    np.testing.assert_array_equal(
                        model.extractor.conv1.bn.avg_var, value)
                    assert model.extractor.conv1.bn.avg_var.shape \
                        == value.shape
                    model.extractor.conv1.bn.avg_var[:] = value
                else:
                    print('param: {} is not converted'.format(key))
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1].startswith('block'):  # res2-4
            resblock_num = int(key_list[1][-1]) + 1
            resblock_name = 'res{}'.format(resblock_num)
            resblock = getattr(model.extractor, resblock_name)
            unit_num = int(key_list[2].split('_')[-1])
            if unit_num == 1:
                unit_name = 'a'
            else:
                unit_name = 'b{}'.format(unit_num - 1)
            unit = getattr(resblock, unit_name)
            if key_list[4].startswith('conv'):
                convbn_name = key_list[4]
            elif key_list[4] == 'shortcut':
                convbn_name = 'residual_conv'
            else:
                print('unknow convbn name: {}'.format(key_list[4]))
            convbn = getattr(unit, convbn_name)
            if key_list[5] == 'weights':  # conv
                value = value.transpose((3, 2, 0, 1))
                assert convbn.conv.W.shape == value.shape
                convbn.conv.W.array[:] = value
            elif key_list[5] == 'BatchNorm':  # bn
                if key_list[6] == 'gamma':  # gamma
                    assert convbn.bn.gamma.shape == value.shape
                    convbn.bn.gamma.array[:] = value
                elif key_list[6] == 'beta':  # beta
                    assert convbn.bn.beta.shape == value.shape
                    convbn.bn.beta.array[:] = value
                elif key_list[6] == 'moving_mean':  # avg_mean
                    assert convbn.bn.avg_mean.shape == value.shape
                    convbn.bn.avg_mean[:] = value
                elif key_list[6] == 'moving_variance':  # avg_var
                    assert convbn.bn.avg_var.shape == value.shape
                    convbn.bn.avg_var[:] = value
                else:
                    print('param: {} is not converted'.format(key))
            else:
                print('param: {} is not converted'.format(key))

        elif key_list[1] == 'rpn_conv':  # rpn conv
            if key_list[3] == 'weights':
                value = value.transpose((3, 2, 0, 1))
                assert model.rpn.conv1.W.shape == value.shape
                model.rpn.conv1.W.array[:] = value
            elif key_list[3] == 'biases':
                assert model.rpn.conv1.b.shape == value.shape
                model.rpn.conv1.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'rpn_cls_score':  # rpn score
            if key_list[2] == 'weights':
                value = value.transpose((3, 2, 0, 1))
                assert model.rpn.score.W.shape == value.shape
                model.rpn.score.W.array[:] = value
            elif key_list[2] == 'biases':
                assert model.rpn.score.b.shape == value.shape
                model.rpn.score.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'rpn_bbox_pred':  # rpn loc
            if key_list[2] == 'weights':
                value = value.reshape((1, 1, 512, 15, 4))
                value = value.transpose((3, 4, 2, 0, 1))
                value = value[:, [1, 0, 3, 2]]
                value = value.reshape((60, 512, 1, 1))
                assert model.rpn.loc.W.shape == value.shape
                model.rpn.loc.W.array[:] = value
            elif key_list[2] == 'biases':
                value = value.reshape((15, 4))
                value = value[:, [1, 0, 3, 2]]
                value = value.reshape((60))
                assert model.rpn.loc.b.shape == value.shape
                model.rpn.loc.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'conv_new_1_conv15_w_pre':  # col_max
            if key_list[2] == 'weights':
                value = value.transpose((3, 2, 0, 1))
                assert model.head.global_context_module.col_max.W.shape \
                    == value.shape
                model.head.global_context_module.col_max.W.array[:] = value
            elif key_list[2] == 'biases':
                assert model.head.global_context_module.col_max.b.shape \
                    == value.shape
                model.head.global_context_module.col_max.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'conv_new_1_conv15_w':  # col
            if key_list[2] == 'weights':
                value = value.transpose((3, 2, 0, 1))
                assert model.head.global_context_module.col.W.shape \
                    == value.shape
                model.head.global_context_module.col.W.array[:] = value
            elif key_list[2] == 'biases':
                assert model.head.global_context_module.col.b.shape \
                    == value.shape
                model.head.global_context_module.col.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'conv_new_1_conv15_h_pre':  # global_row_max
            if key_list[2] == 'weights':
                value = value.transpose((3, 2, 0, 1))
                assert model.head.global_context_module.row_max.W.shape \
                    == value.shape
                model.head.global_context_module.row_max.W.array[:] = value
            elif key_list[2] == 'biases':
                assert model.head.global_context_module.row_max.b.shape \
                    == value.shape
                model.head.global_context_module.row_max.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'conv_new_1_conv15_h':  # global_row
            if key_list[2] == 'weights':
                value = value.transpose((3, 2, 0, 1))
                assert model.head.global_context_module.row.W.shape \
                    == value.shape
                model.head.global_context_module.row.W.array[:] = value
            elif key_list[2] == 'biases':
                assert model.head.global_context_module.row.b.shape \
                    == value.shape
                model.head.global_context_module.row.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'ps_fc_1':  # fc1
            if key_list[2] == 'weights':
                value = value.reshape((49, 10, 2048))
                value = value.transpose((2, 1, 0))
                value = value.reshape((2048, 490))
                assert model.head.fc1.W.shape == value.shape
                model.head.fc1.W.array[:] = value
            elif key_list[2] == 'biases':
                assert model.head.fc1.b.shape == value.shape
                model.head.fc1.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'cls_fc':  # score
            if key_list[2] == 'weights':
                value = value.transpose((1, 0))
                assert model.head.score.W.shape == value.shape
                model.head.score.W.array[:] = value
            elif key_list[2] == 'biases':
                assert model.head.score.b.shape == value.shape
                model.head.score.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        elif key_list[1] == 'bbox_fc':  # loc
            if key_list[2] == 'weights':
                value = value.reshape((2048, 81, 4))
                value = value.transpose((1, 2, 0))
                value = value[:, [1, 0, 3, 2]]
                value = value.reshape((81 * 4, 2048))
                assert model.head.cls_loc.W.shape == value.shape
                model.head.cls_loc.W.array[:] = value
            elif key_list[2] == 'biases':
                value = value.reshape((81, 4))
                value = value[:, [1, 0, 3, 2]]
                value = value.reshape((4 * 81, ))
                assert model.head.cls_loc.b.shape == value.shape
                model.head.cls_loc.b.array[:] = value
            else:
                print('param: {} is not converted'.format(key))
        else:
            print('param: {} is not converted'.format(key))

    chainer.serializers.save_npz(args.out, model)


if __name__ == '__main__':
    main()
