from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim

import tflib as tl


conv = partial(slim.conv2d, activation_fn=None, padding='SAME', weights_initializer=tf.random_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
valid_conv = partial(slim.conv2d, activation_fn=None, padding='VALID', weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = slim.instance_norm
conv_in_relu = partial(conv, normalizer_fn=instance_norm, activation_fn=relu)
dconv_in_relu = partial(dconv, normalizer_fn=instance_norm, activation_fn=relu)

MAX_DIM = 64 * 16

def Gamn(x, a, is_training=True):
    dim = 32
    enc_layers = 4
    residual_blocks = 4
    dec_layers = 3
    residual_channels = 256
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu)
    conv_bn = partial(conv, normalizer_fn=bn)

    def _concat(x, a):
        images = x
        atts = a
        feats = [images]
        atts = tf.reshape(atts, [tl.shape(atts)[-1], 1, 1, -1])
        atts = tf.tile(atts, [1, tl.shape(images)[1], tl.shape(images)[2], 1])
        feats.append(atts)

        return tf.concat(feats, axis=3)

    x_a_cat = _concat(x, a)

    with tf.variable_scope('Gamn', reuse=tf.AUTO_REUSE):
        z = x_a_cat
        # encoder
        for i in range(enc_layers):
            d = min(dim * 2**i, MAX_DIM)
            if i == 0:
                z = conv_in_relu(z, d, 7, 1)
            else:
                z = conv_in_relu(z, d, 4, 2)

        # residual blocks
        for i in range(residual_blocks):
            bottle_neck = conv_bn_relu(z, residual_channels, 3, 1)
            bottle_neck = conv_bn(bottle_neck, residual_channels, 3, 1)
            z = relu(z + bottle_neck)

        # decoder
        for i in range(dec_layers):
            d = min(dim * 2**(dec_layers - i - 1), MAX_DIM)
            z = dconv_in_relu(z, d, 4, 2)

        z = tanh(conv(z, 3, 7, 1))

        return z

def Gsan(x, is_training=True):
    dim = 32
    enc_layers = 4
    residual_blocks = 4
    dec_layers = 3
    residual_channels = 256
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu)
    conv_bn = partial(conv, normalizer_fn=bn)

    with tf.variable_scope('Gsan', reuse=tf.AUTO_REUSE):
        z = x
        # encoder
        for i in range(enc_layers):
            d = min(dim * 2**i, MAX_DIM)
            if i == 0:
                z = conv_in_relu(z, d, 7, 1)
            else:
                z = conv_in_relu(z, d, 4, 2)

        # residual blocks
        for i in range(residual_blocks):
            bottle_neck = conv_bn_relu(z, residual_channels, 3, 1)
            bottle_neck = conv_bn(bottle_neck, residual_channels, 3, 1)
            z = z + bottle_neck
            # z = relu(z + bottle_neck)

        # decoder
        for i in range(dec_layers):
            d = min(dim * 2**(dec_layers - i - 1), MAX_DIM)
            z = dconv_in_relu(z, d, 4, 2)

        z = sigmoid(conv(z, 1, 7, 1))

        return z

def G(x, a, is_training=True):
    y = Gamn(x, a, is_training=is_training)
    mask = Gsan(x, is_training=is_training)
    z = y * mask + x * (1 - mask)
    return z, mask

def D(x, dim=32, n_layers=6):
    conv_lrelu = partial(conv, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            y = conv_lrelu(y, d, 4, 2)

        # distinguish for real/fake
        logit_gan = conv(y, 1, 3, 1)

        # distinguish for attributes
        logit_att = valid_conv(y, 1, 2, 1)
        logit_att = tf.squeeze(logit_att)

        return logit_gan, logit_att

def gradient_penalty(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                _, variance = tf.nn.moments(a, range(a.shape.ndims))
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
