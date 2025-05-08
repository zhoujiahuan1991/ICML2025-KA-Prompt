# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch
import torch.nn as nn
import os
import random
import numpy as np
from timm.models import create_model
import torchvision.transforms as T


class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, domain,data_path):   
        self.data_path=data_path
        self.domain=domain    
        self.download_data()  

    def download_data(self):
        name=self.domain
        
        train_dataset = []
        root_ = os.path.join(self.data_path, name, 'train')
        # sub_classes = os.listdir(root_) if self.multiclass[id] else ['']
        sub_classes=['']
        for cls in sub_classes:
            for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 ))
            for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1))
                
       
        
        test_dataset = []
        root_ = os.path.join(self.data_path, name, 'val')
        # sub_classes = os.listdir(root_) if self.multiclass[id] else ['']
        sub_classes=['']
        for cls in sub_classes:
            for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 ))
            for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 ))
                                
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        
        
class CDDB(Dataset):
    """
    Overrides the dataset to change the getitem function.
    """
    def __init__(self, domain_id=0, mode='train', transform=None,
                 target_transform=None) -> None:
        super().__init__()
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        self.target_transform = target_transform
        self.data_root='/home/xukunlun/DATA/CIL/CDDB'

        self.mode = mode           
        self.task_name = ["gaugan", "biggan", "wild", "whichfaceisreal", "san"]
        self.domain = self.task_name[domain_id]                  
        
        self.data=iGanFake(self.domain,self.data_root)# 读取数据
        
        self.path = self.get_path()
        self.length = len(self.path)
        
        
        

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.path[index]
        # img_path = os.path.join(self.data_root, img_path)
        img = Image.open(img_path)
        img=img.convert('RGB')
        
        # print("img.shape:",img.shape)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.mode == 'train':
            return img, target, 1
        elif self.mode == 'test':
            return img, target
    
    def get_path(self):
        if 'train' == self.mode:
            images=self.data.test_dataset
            random.shuffle(images)  
        else:
            images=self.data.train_dataset
                
        return images



class SequentialCDDB(ContinualDataset):

    NAME = 'cddb'
    SETTING = 'domain-il'
    N_DOMAINS_PER_TASK = 1
    N_TASKS = 5
    N_CLASSES = 345
    
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])
    TRANSFORM_DOMAIN_NET = transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    def get_trans_train(args):
        return transforms.Compose(
            [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63/255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
    )
        
    #     return transforms.Compose(
    #         [transforms.Resize(args.resize_train),
    #         transforms.RandomCrop(224),
    #         transforms.RandomHorizontalFlip(0.5),
    #         transforms.ToTensor()]
    # )
    
        
    def get_trans_test(args):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        )
        # return transforms.Compose(
        #     [transforms.Resize(args.resize_test),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor()]            
    # )

    def get_trans_basic(args):
        return T.Compose([
        T.Resize((224, 224), interpolation=3),
        T.ToTensor(),
    ])

    def get_finetune_all_dataloader(self, mode="train"):
        dataset = DomainNetAll(mode=mode)
        if mode == "train":
            shu = True
        else:
            shu = False
        return DataLoader(dataset,
                        batch_size=self.args.batch_size, 
                        shuffle=shu, 
                        num_workers=4, 
                        pin_memory=True)

    def get_test_dataloaders(self, task_id):
        test_loader_all = []
        for i in range(task_id + 1):
            test_dataset = CDDB(domain_id=i, mode='test', 
                                        transform=SequentialCDDB.get_trans_test(self.args))
            test_loader = DataLoader(test_dataset,
                                    batch_size=self.args.batch_size*4, shuffle=False, num_workers=4, pin_memory=True)
                                    # batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            test_loader_all.append(test_loader)
        return test_loader_all

    def get_dataset(self, task_id, mode='train'):
        return CDDB(domain_id=task_id, mode=mode, 
                                    transform=SequentialCDDB.TRANSFORM_DOMAIN_NET)

    def get_data_loaders(self, task_id,eval=False):
        if eval:
            train_dataset = CDDB(domain_id=task_id, mode='train', 
                                        transform=SequentialCDDB.get_trans_test(self.args))
            train_loader = DataLoader(train_dataset,
                                    batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        else:

            # transform = self.TRANSFORM

            train_dataset = CDDB(domain_id=task_id, mode='train', 
                                        transform=SequentialCDDB.get_trans_train(self.args))
            train_loader = DataLoader(train_dataset,
                                    batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        return train_loader
    
    def get_original_data_loaders(self, task_id):
        train_dataset = CDDB(domain_id=task_id, mode='train', 
                                    transform=SequentialCDDB.get_trans_train(self.args))

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader


    # def not_aug_dataloader(self, batch_size):
    #     transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

    #     train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
    #                               download=True, transform=transform)
    #     train_loader = get_previous_train_loader(train_dataset, batch_size, self)

    #     return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCDDB.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        model = create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=False,
            num_classes=21843,
            drop_block_rate=None,
        )
        weight_path = "models/vit_base_p16_224_in22k.pth"
        model.load_state_dict(torch.load(weight_path, weights_only=True), False)
        model.head = torch.nn.Linear(768,SequentialCDDB.N_CLASSES)
        return model

    @staticmethod
    def get_loss():
        return nn.CrossEntropyLoss()

    @staticmethod
    #! https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar100.py
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    #! https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar100.py
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform


if __name__ == '__main__':
    dataset = CDDB()
    dataset[0]

