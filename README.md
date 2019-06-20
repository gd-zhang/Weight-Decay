# Three Mechanisms of Weight Decay Regularization
This repo contains the official implementations of [Three Mechanisms of Weight Decay Regularization](https://openreview.net/forum?id=B1lz-3Rct7). 

1. The config file for the experiments are under the directory of `configs/`.
2. The modified optimization algorithms are in `libs/`. 

# Citation
To cite this work, please use
```
@inproceedings{
  zhang2018three,
  title={Three Mechanisms of Weight Decay Regularization},
  author={Guodong Zhang and Chaoqi Wang and Bowen Xu and Roger Grosse},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=B1lz-3Rct7},
}
```

# Requirements
This project uses Python 3.5.2. Before running the code, you have to install
* [Tensorflow 1.4](https://www.tensorflow.org/)
* [PyTorch](http://pytorch.org/)
* [Numpy](http://www.numpy.org/)
* [tqdm](https://pypi.python.org/pypi/tqdm)

# How to run?

```
# example
$ python main.py --config configs/cifar100/resnet32/kfac/mb128_lr5e2_damp1e3_cov95_bn_aug.json
```


# Contact
If you have any questions or suggestions about the code or paper, please do not hesitate to contact with Guodong Zhang(`gdzhang.cs@gmail.com` or `gdzhang@cs.toronto.edu`) and Chaoqi Wang(`alecwangcq@gmail.com` or `cqwang@cs.toronto.edu`).
