# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_final_preds
from utils.utils import create_logger
from utils.transforms import get_affine_transform
import numpy as np
import cv2
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--img-file',
                        help='input your test img',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file

def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # Loading an image
    image_file = args.img_file
    data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    # object detection box
    # need to be given [left_top, w, h]
    # box = [391, 99, 667-391, 524-99]
    box = [743, 52, 955-743, 500-52]
    # box = [93, 262, 429-93, 595-262]
    c, s = _box2cs(box, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])
    print(c)
    r = 0

    trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
    print(trans.shape)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input = transform(input).unsqueeze(0)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = model(input)
        preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))

        image = data_numpy.copy()
        for mat in preds[0]:
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

        # vis result
        # cv2.imwrite("test_lp50.jpg", image)
        cv2.imshow('demo', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()

