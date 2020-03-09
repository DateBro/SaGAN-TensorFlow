from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl


# 简化写法，所以说pytorch确实比TensorFlow简洁
conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = slim.instance_norm

MAX_DIM = 64 * 16


# encoder部分就是用多层卷积得到每一层的feature
# 输入的x的shape:[batch, height, width, channel]
# 输出的zs的shape:[n_layer, batch, height, width, output_channel]
def Genc(x, dim=64, n_layers=5, multi_inputs=1, is_training=True):
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    with tf.variable_scope('Genc', reuse=tf.AUTO_REUSE):
        h, w = x.shape[1:3]
        z = x
        zs = []
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            if multi_inputs > i > 0:
                z = tf.concat([z, tf.image.resize_bicubic(x, (h//(2**i), w//(2**i)))], 3)
            # d是卷积核个数，4是kernel_size，2是stride，为什么要用偶数的卷积核啊？
            # 因为默认的padding都是SAME？
            z = conv_bn_lrelu(z, d, 4, 2)
            zs.append(z)
        return zs

def ConvGRUCell(in_data, state, out_channel, is_training=True, kernel_size=3, norm='none', pass_state='lstate'):
    if norm == 'bn':
        norm_fn = partial(batch_norm, is_training=is_training)
    elif norm == 'in':
        norm_fn = instance_norm
    else:
        norm_fn = None
    gate = partial(conv, normalizer_fn=norm_fn, activation_fn=sigmoid)
    info = partial(conv, normalizer_fn=norm_fn, activation_fn=tanh)
    with tf.name_scope('ConvGRUCell'):
        state_ = dconv(state, out_channel, 4, 2)  # upsample and make `channel` identical to `out_channel`
        # 上面将state upsample成对应decoder layer的shape，和这里in_data的shape相同，所以可以直接concat
        reset_gate = gate(tf.concat([in_data, state_], axis=3), out_channel, kernel_size)
        update_gate = gate(tf.concat([in_data, state_], axis=3), out_channel, kernel_size)
        new_state = reset_gate * state_
        new_info = info(tf.concat([in_data, new_state], axis=3), out_channel, kernel_size)
        output = (1-update_gate)*state_ + update_gate*new_info
        if pass_state == 'gru':
            return output, output
        elif pass_state == 'direct':
            return output, state_
        else: # 'stu'
            return output, new_state

# 输出是encoder前四层的transformed feature和最后一层的feature
def Gstu(zs, _a, dim=64, n_layers=1, inject_layers=0, is_training=True, kernel_size=3, norm='none', pass_state='stu'):
    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            # _a的shape：[batch, 1, 1, attributes]
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            # _a的shape：[batch, height, weight, attributes]
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
            feats.append(_a)
            # 此时feats里面就只有_a和z两个元素，所以后面进行concatenation
        return tf.concat(feats, axis=3)
    
    with tf.variable_scope('Gstu', reuse=tf.AUTO_REUSE):
        # 得到encoder最后一层的feature作为hidden state
        zs_ = [zs[-1]]
        # 先将difference attribute vector stretch到和上一层的state同样的spatial size
        state = _concat(zs[-1], None, _a)
        for i in range(n_layers): # n_layers <= 4
            # 果然是从encoder后面的layer开始，这里计算的 dim 对应的是decoder相应layer的output channel
            d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
            # 这里的output不应该就是ConvGRUCell里的output吗？难道还是output和new_state的列表？
            output = ConvGRUCell(zs[n_layers - 1 - i], state, d, is_training=is_training,
                                 kernel_size=kernel_size, norm=norm, pass_state=pass_state)
            # 将最新的output放到list最前面
            zs_.insert(0, output[0])
            if inject_layers > i:
                state = _concat(output[1], None, _a)
            else:
                state = output[1]
        return zs_

# _a是attribute vector
def Gdec(zs, _a, dim=64, n_layers=5, shortcut_layers=1, inject_layers=0, is_training=True, one_more_conv=0):
    bn = partial(batch_norm, is_training=is_training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

    shortcut_layers = min(shortcut_layers, n_layers - 1)
    inject_layers = min(inject_layers, n_layers - 1)

    def _concat(z, z_, _a):
        feats = [z]
        if z_ is not None:
            feats.append(z_)
        if _a is not None:
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
            feats.append(_a)
        return tf.concat(feats, axis=3)

    with tf.variable_scope('Gdec', reuse=tf.AUTO_REUSE):
        z = _concat(zs[-1], None, _a)
        for i in range(n_layers):
            if i < n_layers - 1:
                d = min(dim * 2**(n_layers - 1 - i), MAX_DIM)
                # z 是decoder每个layer的feature
                z = dconv_bn_relu(z, d, 4, 2)
                if shortcut_layers > i:
                    # 这里应该就是合并decoder的feature和对应的STU的transformed feature
                    z = _concat(z, zs[n_layers - 2 - i], None)
                if inject_layers > i:
                    # 这里不知道为啥叫inject_layer，是因为插入了attribute vector？如果是默认的0的话岂不是这里没用？
                    z = _concat(z, None, _a)
            else:
                if one_more_conv: # add one more conv after the decoder
                    z = dconv_bn_relu(z, dim//4, 4, 2)
                    x = tf.nn.tanh(dconv(z, 3, one_more_conv))
                else:
                    x = z = tf.nn.tanh(dconv(z, 3, 4, 2))
        return x


def D(x, n_att, dim=64, fc_dim=MAX_DIM, n_layers=5):
    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            y = conv_in_lrelu(y, d, 4, 2)

        # distinguish for real/fake
        logit_gan = lrelu(fc(y, fc_dim))
        logit_gan = fc(logit_gan, 1)

        # distinguish for attributes
        logit_att = lrelu(fc(y, fc_dim))
        logit_att = fc(logit_att, n_att)

        return logit_gan, logit_att


# WGAN-GP 中提出的一个惩罚项，加在原来的loss上
# 最近看看WGAN-GP，好好学习一下公式推导
def gradient_penalty(f, real, fake=None):
    # 抓住生成样本、真实样本以及夹在它们中间的区域
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
