import importlib
import torch
import numpy as np
from datasets import get_dataset

from models import get_all_models
from models import get_model

from argparse import ArgumentParser
from utils.args import add_management_args

from trainer import train_cprompt,test_pretrained_cprompt
from utils.conf import set_random_seed

from utils.loggers import save_args_to_file

from plot_tendency import plot_res
import os
def main():

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = ArgumentParser(description='KA-Prompt', allow_abbrev=False)
    parser.add_argument('--model', type=str, 
                        default='kaprompt_vit',
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)

    
    args = parser.parse_known_args()[0]
    
    mod = importlib.import_module('models.' + args.model)
    
    
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()
    
   
    if args.seed is not None:
        set_random_seed(args.seed)
    
    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform()) 
    
    # print(model.device)

    # if torch.cuda.is_available():
    #     print("put the model to gpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.device = device

    save_args_to_file(args)
    if args.evaluate:
        test_pretrained_cprompt(model, dataset, args)
    else:
        train_cprompt(model, dataset, args)
        plot_res(os.path.join(args.output_path, args.info,'results'))

if __name__ == '__main__':
    main()
