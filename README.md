# Spatial Attention Generative Adversarial Network

This repository contains the TensorFlow implementation of the ECCV 2018 paper "Generative Adversarial Network with Spatial Attention for Face Attribute Editing" ([pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf)).

My results with images and attention masks on CelebA 128 _(original, eyeglasses, mouth_slightly_open, no_beard, smiling)_

![Results](https://github.com/elvisyjlin/SpatialAttentionGAN/blob/master/pics/4_attr_results.jpg)


## Requirements

* Python 3.6
* TensorFlow 1.15.0

The training procedure described in paper takes 5.5GB memory on a single GPU.

* Datasets
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    * Put _Align&Cropped Images_ in `./data/celeba/*.jpg`
    * Put _Attributes Annotations_ in `./data/list_attr_celeba.txt`

* Pretrained models (download from http://bit.ly/sagan-results and decompress the zips to `./results`)
  ```
    results
    ├── celeba_128_eyeglasses
    ├── celeba_128_mouth_slightly_open
    ├── celeba_128_no_beard
    └── celeba_128_smiling
  ```

## Usage

Train a model with a target attribute

```bash
 python train.py --experiment_name 128_Bangs --atts Bangs --dataroot ./data/Datasets/CelebA/Img
```

Generate images from trained models

```bash
python3 test.py --experiment-name celeba_128_eyeglasses --gpu
```