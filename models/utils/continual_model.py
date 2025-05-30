import torch.nn as nn
from torch.optim import SGD, Adam
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from transformers import AdamW


# todo: require_proto
class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.require_task_id = False
        
        if args.area == "NLP":
            self.opt = AdamW(self.net.parameters(), lr=self.args.lr)
        else:
            self.opt = Adam(self.net.parameters(), 
                            lr=self.args.lr, 
                            betas=(0.9, 0.999))
            # self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        
        if args.cuda != -1:
            self.device = torch.device("cuda:{}".format(self.args.cuda) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        #self.device = get_device()
    
    def reset_opt(self):
        if self.args.area == "NLP":
            self.opt = AdamW(self.net.parameters(), lr=self.args.lr)
        else:
            self.opt = Adam(self.net.parameters(), 
                lr=self.args.lr, 
                betas=(0.9, 0.999))
            # self.opt = SGD(self.net.parameters(), lr=self.args.lr)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)
    
    def forward_nlp(self, x: torch.Tensor, x_mask: torch.Tensor, task_id=None):
        return self.net(x, x_mask, task_id=task_id)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, task_id=None) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs:
        :param labels:
        :param not_aug_inputs:
        :param task_id:
        :return:
        """
        pass
