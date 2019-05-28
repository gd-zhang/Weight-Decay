from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from misc.utils import get_logger, get_args, makedirs
from misc.config import process_config
from core.model import Model
from core.train import Trainer
from data_loader import load_pytorch


_INPUT_DIM = {
    'fmnist': [784],
    'mnist': [784],
    'cifar10': [32, 32, 3],
    'cifar100': [32, 32, 3]
}

_OUTPUT_DIM = {
    'fmnist': 10,
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100
}


def main():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    try:
        args = get_args()
        config = process_config(args.config)
        config.input_dim = _INPUT_DIM[config.dataset]
        config.output_dim = _OUTPUT_DIM[config.dataset]
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'core/model.py')
    path_train = os.path.join(path, 'core/train.py')
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=path_model, package_files=[path_train])
    logger.info(dict(config))

    # load data
    train_loader, test_loader = load_pytorch(config)

    # define computational graph
    sess = tf.Session()

    model = Model(config, sess)
    trainer = Trainer(sess, model, train_loader, test_loader, config, logger)

    trainer.train()


if __name__ == "__main__":
    main()
