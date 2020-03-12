import tensorflow as tf
import tflib as tl
import numpy as np
import torch

def concat(images, atts):
    feats = [images]
    if atts is not None:
        atts = tf.reshape(atts, [-1, 1, 1, tl.shape(atts)[-1]])
        # atts = tf.reshape(atts, [tl.shape(atts)[0], 1, 1, -1])
        print('atts shape: ', atts.shape)
        atts = tf.tile(atts, [1, tl.shape(images)[1], tl.shape(images)[2], 1])
        print('atts shape: ', atts.shape)
        feats.append(atts)
    return tf.concat(feats, axis=3)

def torch_concat(images, atts):
    print('atts shape: ', atts.shape)
    atts = atts.unsqueeze(2).unsqueeze(3).repeat(1, 1, images.size(2), images.size(3))
    print('atts shape: ', atts.shape)
    x = torch.cat((images, atts), dim=1)
    return x

def test_torch():
    images = torch.zeros((16, 128, 128, 3))
    atts = torch.ones((16))
    atts = atts[:, None]
    concatenation = torch_concat(images, atts)
    print('torch_concat shape: ', concatenation.shape)

def test_tf():
    images = np.zeros((16, 128, 128, 3))
    images = tf.convert_to_tensor(images)
    print('images shape: ', images.shape)

    atts = np.ones((16, ))
    atts = tf.convert_to_tensor(atts)
    print('atts shape: ', atts.shape)

    concatenation = concat(images, atts)
    print('concatenation shape: ', concatenation.shape)

if __name__ == '__main__':
    # test_tf()

    # test_torch()

    a = np.full((64, 128, 128 // 10, 3), -1.0)
    print(a.shape)
