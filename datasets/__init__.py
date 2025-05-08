# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.domain_net import SequentialDomainNet
from datasets.imagenet_c import SequentialImageNet_C
from datasets.imagenet_r import SequentialImageNet_R
from datasets.imagenet_cr import SequentialImageNet_CR

from datasets.cddb import SequentialCDDB

from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

# string : instance
NAMES = {
    SequentialDomainNet.NAME: SequentialDomainNet,
    SequentialImageNet_C.NAME: SequentialImageNet_C,
    SequentialImageNet_R.NAME: SequentialImageNet_R,
    SequentialImageNet_CR.NAME: SequentialImageNet_CR,
    SequentialCDDB.NAME: SequentialCDDB,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
