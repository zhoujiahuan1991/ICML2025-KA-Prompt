# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
# from backbone import get_supported_ptm
from utils.lr_scheduler import get_all_scheduler


# modularized arguments management
def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, 
                        default="domain-net",
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')    
    parser.add_argument('--model', type=str,
                        default="kaprompt_vit",
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--lr', type=float, 
                        default=0.005,
                        # default=0.005,
                        help='Learning rate.')
    parser.add_argument('--lr_scheduler', type=str, choices=get_all_scheduler(), required=False, default='uniform',
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, 
                        default=128,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, 
                        # default=2,
                        default=5,
                        help='The number of epochs for each task.')
    parser.add_argument('--topN', type=int, 
                        default=1,
                        help='The number of prompts to add.')
    parser.add_argument('--prompt_comp', action='store_true', 
                        default=False,
                        help='change the prompt to compositional prompt.')    
    parser.add_argument('--use_prompt_penalty_3', action='store_true', 
                        default=False,
                        help='use prompt penalty 3 to split the prompt pool.')
    
    parser.add_argument('--pool_size', type=int, 
                        default=-1,
                        help='The size of prompt pool.')
    parser.add_argument('--prompt_per_task', type=int, 
                        default=1,
                        help='the prompt num of per task.')
    parser.add_argument('--prompt_num', type=int, 
                        default=20,
                        help='The length of E-prompt.')
    # use gpu
    parser.add_argument('--use_gpu', type=bool, required=False, default=True, help='Define devices')
    parser.add_argument('--pltf', type=str, required=False, default="m", help='Define devices')
    parser.add_argument('--cuda', type=int, required=False, default=0, help='Define CUDA num')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, 
                        default=3407,
                        # default=100,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument('--area', type=str, 
                        help="CV or NLP", default="CV")
    
    parser.add_argument('--prob_type', required=False, default="", type=str, choices=["proto", "final"],
                        help="conduct prototype based probing.")
    parser.add_argument("--prob_all_tasks", required=False, action="store_true", help="conduct probing for all tasks")
    parser.add_argument('--info', type=str, 
                        default='debug',
                        help='introduction.')
    parser.add_argument('--output_path', type=str, 
                        default='./output')
    parser.add_argument('--use_ema', action='store_true',
                        default=False,
                        help='whether to use ema')
    parser.add_argument('--ema_beta', type=float,
                        default=0.9999,
                        help='the beta value of ema')
    parser.add_argument('--use_ema_c', action='store_true',
                        default=False,
                        help='whether to use ema for classifier')
    
    
    
    parser.add_argument('--ema_beta_c', type=float,
                        default=0.9999,
                        help='the beta value of ema for classifier')
    parser.add_argument('--update_ema_c', type=str,
                        default='batch',
                        help='the update frequency of ema_c')
    parser.add_argument('--update_ema_g', type=str,
                        default='batch',
                        help='the update frequency of ema_g')
    parser.add_argument('--adapt_ema_c', action='store_true',
                        default=False,
                        help='whether to use adaptive ema for classifier')
    parser.add_argument('--adapt_h', type=float,
                        default=10,
                        help='whether to use adaptive ema for classifier')
    parser.add_argument('--test_fuse_c', action='store_true',
                        default=False,
                        help='fuse the train and ema classifier in testing')

    parser.add_argument('--train_fuse_c', action='store_true',
                        default=False,
                        help='fuse the train and ema classifier in training')
    parser.add_argument('--fuse_prompt', action='store_true',
                        default=False,
                        help='fuse the selected N E-Prompt')
    
    parser.add_argument('--freeze_g_prompt', action='store_true',
                        default=False,
                        help='freeze the G-Prompt after task 1')
    parser.add_argument('--save_prompt', action='store_true',
                        default=True,
                        help='save the prompt (general_prompt and prompt pool)')
    parser.add_argument('--save_model', action='store_true',
                        default=True,
                        help='save the model')
    parser.add_argument('--save_ema', action='store_true',
                        default=True,
                        help='save the ema')

 
    
    parser.add_argument('--csv_log', action='store_true',
                        default=True,
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        default=False,
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
 

    
    # # use gpu
    # parser.add_argument('--use_gpu', type=bool, required=False, default=True, help='Define devices')
    parser.add_argument('--resize_train', type=int, default=-1, help="")
    parser.add_argument('--resize_test', type=int, default=-1, help="")
                   
    
    parser.add_argument('--greedy_init', action='store_true', help="use greedy rearch to initialize prompts")
        
    parser.add_argument('--ema_after_domain_2', action='store_true', help="ema_after_domain_2")
    
    
    parser.add_argument('--evaluate', action='store_true', help="test the pretrained model")
    
    
    parser.add_argument('--aux_weight', type=float, default=2.0,help="weight of the mix-prompt guided loss")
    parser.add_argument('--tau', type=float, default=0.01,help="temperature parameter of mix prompt weight")
    
    
    parser.add_argument('--query_pos', type=int, default=1,help="temperature parameter of mix prompt weight")
    
    parser.add_argument('--e_s', type=int, default=2,help="temperature parameter of mix prompt weight")
    parser.add_argument('--e_e', type=int, default=11,help="temperature parameter of mix prompt weight")
    
    
    