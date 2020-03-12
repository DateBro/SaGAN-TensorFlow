from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data
import models

import os


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', help='experiment_name')
parser.add_argument('--gpu', type=str, default='all', help='gpu')
parser.add_argument('--dataroot', type=str, default='/data/Datasets/CelebA/Img')
# if assigned, only given images will be tested.
parser.add_argument('--img', type=int, nargs='+', default=None, help='e.g., --img 182638 202599')
# for multiple attributes
parser.add_argument('--test_att', type=str, default='Bangs')
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)

# model
atts = args['target_att']
n_att = len(atts)
img_size = args['img_size']
label = args['label']

dataroot = args_.dataroot
img = args_.img
print('Using selected images:', img)

gpu = args_.gpu
if gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

#### testing
# single attribute
test_int = args_.test_int

thres_int = args['thres_int']
# others
use_cropped_img = args['use_cropped_img']
experiment_name = args_.experiment_name


# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = tl.session()
te_data = data.Celeba(dataroot, atts, img_size, 1, part='test', sess=sess, crop=not use_cropped_img, im_no=img)
# models
G = partial(models.G, dim=32)
D = partial(models.D, dim=32, n_layers=6)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[1, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[1, ])
raw_b_sample = tf.placeholder(tf.float32, shape=[1, ])

# sample
test_label = _b_sample
x_sample = G(xa_sample,  test_label, is_training=False)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
tl.load_checkpoint(ckpt_dir, sess)

# test
try:
    for idx, batch in enumerate(te_data):
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]
        # 这个操作之后b是比a多一维的
        b_sample_ipt_list = [a_sample_ipt.copy() for _ in range(1)]
        # test_single_attributes
        for i in range(len(atts)):
            tmp = np.array(a_sample_ipt, copy=True)
            tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
            tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
            b_sample_ipt_list.append(tmp)

        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
        raw_a_sample_ipt = a_sample_ipt.copy()
        raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
        for i, b_sample_ipt in enumerate(b_sample_ipt_list):
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
            if not test_slide:
                if i > 0:   # i == 0 is for reconstruction
                    _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int
            x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                   _b_sample: _b_sample_ipt,
                                                                   raw_b_sample: raw_a_sample_ipt}))
        sample = np.concatenate(x_sample_opt_list, 2)

        save_folder = 'sample_testing'
        save_dir = './output/%s/%s' % (experiment_name, save_folder)
        pylib.mkdir(save_dir)
        im.imwrite(sample.squeeze(0), '%s/%06d%s.png' % (save_dir,
                                                         idx + 182638 if img is None else img[idx], 
                                                         '_%s' % ''))

        print('%06d.png done!' % (idx + 182638 if img is None else img[idx]))
except:
    traceback.print_exc()
finally:
    sess.close()