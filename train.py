from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
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
def boolean(s):
    return s.lower() in ('true', 't', 'yes', 'y', '1')

parser = argparse.ArgumentParser()
# settings
dataroot_default = '/data/Datasets/CelebA/Img'
parser.add_argument('--dataroot', type=str, default=dataroot_default)
parser.add_argument('--gpu', type=str, default='all',
                    help='Specify which gpu to use by `CUDA_VISIBLE_DEVICES=num python train.py **kwargs`\
                          or `python train.py --gpu num` if you\'re running on a multi-gpu enviroment.\
                          You need to do nothing if your\'re running on a single-gpu environment or\
                          the gpu is assigned by a resource manager program.')
parser.add_argument('--threads', type=int, default=-1,
                    help='Control parallel computation threads,\
                          please leave it as is if no heavy cpu burden is observed.')
# model
target_att_default = ['Bangs']
att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
               'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
parser.add_argument('--atts', default=target_att_default, choices=data.Celeba.att_dict.keys(),
                    help='Target attributes to modify by the model')
parser.add_argument('--img_size', type=int, default=128, help='input image size')
# training
parser.add_argument('--epoch', type=int, default=200, help='# of epochs')
parser.add_argument('--init_epoch', type=int, default=100, help='# of epochs with init lr.')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--n_d', type=int, default=3, help='# of d updates per g update')
parser.add_argument('--n_sample', type=int, default=64, help='# of sample images')
parser.add_argument('--save_freq', type=int, default=0,
                    help='save model every save_freq iters, 0 means to save evary epoch.')
parser.add_argument('--sample_freq', type=int, default=0,
                    help='eval on validation set every sample_freq iters, 0 means to save every epoch.')
# others
parser.add_argument('--use_cropped_img', action='store_true')
parser.add_argument('--experiment_name', default=datetime.datetime.now().strftime("%Y.%m.%d-%H%M%S"))
parser.add_argument('--num_ckpt', type=int, default=1)
parser.add_argument('--clear', default=False, action='store_true')

args = parser.parse_args()
# settings
dataroot = args.dataroot
if args.gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
threads = args.threads
# model
atts = args.atts
n_att = len(atts)
img_size = args.img_size
# training
epoch = args.epoch
init_epoch = args.init_epoch
batch_size = args.batch_size
lr_base = args.lr
n_d = args.n_d
n_sample = args.n_sample
save_freq = args.save_freq
sample_freq = args.sample_freq
# others
use_cropped_img = args.use_cropped_img
experiment_name = args.experiment_name
num_ckpt = args.num_ckpt
clear = args.clear

pylib.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
if threads >= 0:
    cpu_config = tf.ConfigProto(intra_op_parallelism_threads = threads//2,
                                inter_op_parallelism_threads = threads//2,
                                device_count = {'CPU': threads})
    sess = tf.Session(config=cpu_config)
else:
    sess = tl.session()
crop_ = not use_cropped_img
tr_data = data.Celeba(dataroot, atts, img_size, batch_size, part='train', sess=sess, crop=crop_)
val_data = data.Celeba(dataroot, atts, img_size, n_sample, part='val', shuffle=False, sess=sess, crop=crop_)

# models
G = partial(models.G)
D = partial(models.D)

# inputs
lr = tf.placeholder(dtype=tf.float32, shape=[])

xa = tr_data.batch_op[0]
a = tr_data.batch_op[1]
b = 1 - a
_a = tf.to_float(a)
_b = tf.to_float(b)

xa_sample = tf.placeholder(tf.float32, shape=[n_sample, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[n_sample, ])

# generate
xb_, mask_b_ = G(xa, _b)

with tf.control_dependencies([xb_]):
    xa_, mask_xa_ = G(xa, _a)
    xb__a, mask_xb__a = G(xb_, _a)

# discriminate
xa_logit_gan, xa_logit_att = D(xa)
xb__logit_gan, xb__logit_att = D(xb_)

# discriminator losses
wd = tf.reduce_mean(xa_logit_gan) - tf.reduce_mean(xb__logit_gan)
d_loss_gan = -wd
gp = models.gradient_penalty(D, xa, xb_)

# attribute manipulation loss
xa_loss_att = tf.losses.sigmoid_cross_entropy(a, xa_logit_att)

d_loss = d_loss_gan + gp * 10.0 + xa_loss_att

# generator losses
xb__loss_gan = -tf.reduce_mean(xb__logit_gan)

xb__loss_att = tf.losses.sigmoid_cross_entropy(b, xb__logit_att)
xa__loss_rec = 20 * tf.losses.absolute_difference(xa, xb__a) + 100 * tf.losses.absolute_difference(xa, xa_)

g_loss = xb__loss_gan + xb__loss_att + xa__loss_rec

# optim
d_var = tl.trainable_variables('D')
d_step = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_var)

g_var = tl.trainable_variables('G')
g_step = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_var)

# summary
show_weights = False

d_summary = tl.summary({
    d_loss_gan: 'd_loss_gan',
    gp: 'gp',
    xa_loss_att: 'xa_loss_att',
}, scope='D')

lr_summary = tl.summary({lr: 'lr'}, scope='Learning_Rate')


g_summary = tl.summary({
    xb__loss_gan: 'xb__loss_gan',
    xb__loss_att: 'xb__loss_att',
    xa__loss_rec: 'xa__loss_rec',
}, scope='G')
if show_weights:
    d_histogram = tf.summary.merge([tf.summary.histogram(
        name=i.name,
        values=i
    ) for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'D')])

    gamn_histogram = tf.summary.merge([tf.summary.histogram(
        name=i.name,
        values=i
    ) for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'Gamn')])

    gsan_histogram = tf.summary.merge([tf.summary.histogram(
        name=i.name,
        values=i
    ) for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, 'Gsan')])

    d_summary = tf.summary.merge([d_summary, lr_summary, d_histogram])
    g_summary = tf.summary.merge([g_summary, gamn_histogram, gsan_histogram])
else:
    d_summary = tf.summary.merge([d_summary, lr_summary])

# sample
test_label = _b_sample
x_sample, mask_sample = models.G(xa_sample, test_label, is_training=False)

# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# iteration counter
it_cnt, update_cnt = tl.counter()

# saver
saver = tf.train.Saver(max_to_keep=num_ckpt)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)

try:
    assert clear == False
    tl.load_checkpoint(ckpt_dir, sess)
except:
    print('NOTE: Initializing all parameters...')
    sess.run(tf.global_variables_initializer())

# train
try:
    # data for sampling
    xa_sample_ipt, a_sample_ipt = val_data.get_next()
    b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
    tmp = np.array(a_sample_ipt, copy=True)
    tmp[:] = 1 - tmp[:]   # inverse attribute
    b_sample_ipt_list.append(tmp)

    it_per_epoch = len(tr_data) // (batch_size * (n_d + 1))
    max_it = epoch * it_per_epoch
    for it in range(sess.run(it_cnt), max_it):
        with pylib.Timer(is_output=False) as t:
            sess.run(update_cnt)

            # which epoch
            epoch = it // it_per_epoch
            it_in_epoch = it % it_per_epoch + 1

            # learning rate
            lr_ipt = lr_base / (10 ** (epoch // init_epoch))

            # train D
            for i in range(n_d):
                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(d_summary_opt, it)

            # train G
            g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(g_summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d) Time: %s!" % (epoch, it_in_epoch, it_per_epoch, t))

            # save
            if (it + 1) % (save_freq if save_freq else it_per_epoch) == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt'%(ckpt_dir, epoch, it_in_epoch, it_per_epoch))
                print('Model is saved at %s!' % save_path)

            # sample
            if (it + 1) % (sample_freq if sample_freq else it_per_epoch) == 0:
                x_sample_opt_list = [xa_sample_ipt, np.full((n_sample, img_size, img_size // 10, 3), -1.0)]
                mask_sample_opt_list = [np.full((n_sample, img_size, img_size // 10, 1), -1.0)]
                for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                    x_sample_result, mask_sample_result = sess.run([x_sample, mask_sample], feed_dict={xa_sample: xa_sample_ipt,
                                                                                                       _b_sample: b_sample_ipt})
                    x_sample_opt_list.append(x_sample_result)
                    mask_sample_opt_list.append(mask_sample_result)
                    last_images = x_sample_opt_list[-1]
                    print('last_images shape: ', last_images.shape)
                    if i > 0:   # add a mark (+/-) in the upper-left corner to identify add/remove an attribute
                        for nnn in range(last_images.shape[0]):
                            last_images[nnn, 2:5, 0:7, :] = 1.
                            if b_sample_ipt[nnn] > 0:
                                last_images[nnn, 0:7, 2:5, :] = 1.
                                last_images[nnn, 1:6, 3:4, :] = -1.
                            last_images[nnn, 3:4, 1:6, :] = -1.
                sample = np.concatenate(x_sample_opt_list, 2)
                masks = np.concatenate(mask_sample_opt_list, 2)

                save_dir = './output/%s/sample_training' % experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(im.immerge(sample, n_sample, 1), '%s/Epoch_(%d)_(%dof%d).jpg' % \
                           (save_dir, epoch, it_in_epoch, it_per_epoch))
                im.imwrite(im.immerge(masks, n_sample, 1), '%s/Mask_Epoch_(%d)_(%dof%d).jpg' % \
                           (save_dir, epoch, it_in_epoch, it_per_epoch))
except:
    traceback.print_exc()
finally:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
    print('Model is saved at %s!' % save_path)
    sess.close()
