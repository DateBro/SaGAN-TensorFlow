# Spatial Attention Generative Adversarial Network

**Tensorflow** implementation of [**Generative Adversarial Network with Spatial Attention for Face Attribute Editing**](http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf)


## Preparation

- **Prerequisites**
    - Tensorflow (r1.4 - r1.12 should work fine)
    - Python 3.x with matplotlib, numpy and scipy

- **Dataset**
    - [CelebA](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf) dataset (Find more details from the [project page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))
        - [Images](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM) should be placed in ***DATAROOT/img_align_celeba/\*.jpg***
        - [Attribute labels](https://drive.google.com/open?id=0B7EVK8r0v71pblRyaVFSWGxPY0U) should be placed in ***DATAROOT/list_attr_celeba.txt***
        - If google drive is unreachable, you can get the data from [Baidu Cloud](http://pan.baidu.com/s/1eSNpdRG)
    - We follow the settings of AttGAN, kindly refer to [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow) for more dataset preparation details

## Usage

Train a model with a target attribute

```bash
 python train.py --experiment_name 128_Bangs --atts Bangs --dataroot ./data/Datasets/CelebA/Img
```

Generate images from trained models

```bash
python test.py --experiment-name 128_Bangs --gpu
```

### NOTE:

- You should give the path of the data by adding `--dataroot DATAROOT`;
- You can specify which GPU to use by adding `--gpu GPU`, e.g., `--gpu 0`;
- You can specify which image(s) to test by adding `--img num` (e.g., `--img 182638`, `--img 200000 200001 200002`), where the number should be no larger than 202599 and is suggested to be no smaller than 182638 as our test set starts at 182638.png.

## Acknowledgement
The code is built upon [STGAN](https://github.com/csmliu/STGAN) and [AttGAN](https://github.com/LynnHo/AttGAN-Tensorflow), thanks for their excellent work!