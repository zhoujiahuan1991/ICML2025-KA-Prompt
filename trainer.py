import torch
import time
import numpy as np
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import gc
from utils.map import MAP
import matplotlib.pyplot as plt
import cv2
import time

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:] = -float('inf')



def evaluate_cprompt(model: ContinualModel, dataset: ContinualDataset, 
                        args: Namespace, t, last=False) -> Tuple[list, list]:
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    # using ema
    if args.use_ema or args.use_ema_c:
        if args.ema_after_domain_2 and 0==t:
            pass
        else:
            model.ema_before_eval(t)

    if args.dataset == "domain-net":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-c":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-r":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-cr":
        test_loaders = dataset.get_test_dataloaders(t)
    else:
        test_loaders = dataset.get_test_dataloaders(t)

    with torch.no_grad():
        keys_all=[]
        domains_all=[]
        for k, test_loader in enumerate(test_loaders):
            if last and k == len(test_loaders) - 1:                
                continue
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0            
            
            print(f"*******domian {k} *******")
            tmp = 0
            for data in test_loader:
                tmp += 1
                if tmp%50 == 0:
                    print(tmp," / ", len(test_loader))
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs, keys = model.forward_model(inputs)             
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()     
                
                total += labels.shape[0]
                # print("correct:", correct)
                # print("total:", total)`
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
                
                # break
                if last:
                    keys_all.append(keys)
                    domains_all.append(torch.ones_like(keys.view(-1))*k)
                    
            
            
            accs.append(correct / total * 100)
            accs_mask_classes.append(correct_mask_classes / total * 100)
    consistency=False
    if last and consistency:
        keys_all=torch.cat(keys_all)  # 统计提示的域一致性
        keys_all=keys_all//args.prompt_per_task
        first_id=keys_all[:,:1]
        first_id=first_id+keys_all*0
        res=first_id!=keys_all
        res=res.sum(-1).float()
        print("seed:",args.seed)
        print("cross domain prompt sampling rate",res.sum().item()/len(res))
    
    if last:
        keys_all=[x.view(-1) for x in keys_all]
        keys_all=torch.cat(keys_all)//args.prompt_per_task
        domains_all=torch.cat(domains_all)
        res=(keys_all==domains_all).float()
        
        print("domain match correct rate",res.sum().item()/len(res))
    
    
    model.net.train(status)

    # using ema
    if args.use_ema or args.use_ema_c:
        if args.ema_after_domain_2 and 0==t:
            pass
        else:
            model.ema_after_eval()

    l = len(accs)
    for i in range(l):
        accs[i] = round(accs[i], 2)
        accs_mask_classes[i] = round(accs_mask_classes[i], 2)

    return accs, accs_mask_classes


def test_pretrained_cprompt(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # visual_features(model,dataset, args)
    # visual_prompts(model,dataset, args)
    # exit(0)
    
    model.net.to(model.device)
    print(model.device)
    
    results, results_mask_classes = [], []
    
    model_stash = create_stash(model, args, dataset)
    
          
    model.load_prompt_from_file(dataset.N_TASKS-1)
    model.load_model_from_file(dataset.N_TASKS-1)
    model.load_ema_from_file(dataset.N_TASKS-1)
    t=dataset.N_TASKS-1
            
    torch.cuda.empty_cache()
    
    # ''' the code for heatmap visualize'''
    # from utils.visualize_attention import visyalize_compare
    # import copy
    # compared_model=copy.deepcopy(model)
    # # path='/home/xukunlun/CODE/DIL/CPrompt-Aligner/Cprompt_models/ImageNetR'
    # compared_model.args.output_path='/home/xukunlun/CODE/DIL/CPrompt-Aligner/Cprompt_models'
    # compared_model.args.info='DomainNet' #'ImageNetR'
    # compared_model.load_prompt_from_file(dataset.N_TASKS-1)
    # compared_model.load_model_from_file(dataset.N_TASKS-1)
    # compared_model.load_ema_from_file(dataset.N_TASKS-1)
    # accs = visyalize_compare(model,compared_model, dataset, args, t, last=True)
    # exit(0)
    
    # model.shuffle_prompt(t)

    accs = evaluate_cprompt(model, dataset, args, t, last=True)
    results.append(accs[0])
    results_mask_classes.append(accs[1])
    
    mean_acc = np.mean(accs, axis=1)
    print(accs)
    print_mean_accuracy(args, mean_acc, t + 1, dataset.SETTING)
    # save_mean_accuracy_to_file(mean_acc, t + 1, dataset.SETTING, args, t, accs, loss)
    
    # # freeze G Prompt
    # if t == 0 and args.freeze_g_prompt:
    #     model.set_g_prompt_not_update()

    # model_stash['mean_accs'].append(mean_acc)
      

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def train_cprompt(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # visual_features(model,dataset, args)
    # visual_prompts(model,dataset, args)
    # exit(0)
    model.net.to(model.device)
    print(model.device)
    
    results, results_mask_classes = [], []
    
    model_stash = create_stash(model, args, dataset)
    
    print(file=sys.stderr)
    start_time = time.time()
    for t in range(dataset.N_TASKS):
        gc.collect()
        model.net.train()
        if args.dataset == "domain-net":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-c":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-r":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-cr":
            train_loader = dataset.get_data_loaders(task_id=t)
     
            
        model.init_opt(args, t, train_loader)
      
        
        for epoch in range(args.n_epochs):
            epoch_start = time.time()
            batch_avg=AverageMeter()
            start = time.time()
            for i, data in enumerate(train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                
                
                loss = model.observe(inputs, labels,dataset,t)
                
                end = time.time()
                elapsed_time = (end - start) 
                batch_avg.update(elapsed_time)

                if args.adapt_ema_c:
                    model.cal_g_change()

                # update ema_c
                if args.use_ema_c and args.update_ema_c == 'batch':
                    model.ema_c.update()
                if args.use_ema and args.update_ema_g == 'batch':
                    model.update_ema_g()

                progress_bar(i, len(train_loader), epoch, t, loss, args)
                
                model_stash['batch_idx'] = i + 1
                
                start = time.time()
                
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0


            if args.use_ema_c and args.update_ema_c == 'epoch':
                model.ema_c.update()
            if args.use_ema and args.update_ema_g == 'epoch':
                model.update_ema_g()

            print("batch time:",batch_avg.avg)

        if args.use_ema_c and args.update_ema_c == 'task':
            model.ema_c.update()
        if args.use_ema and args.update_ema_g == 'task':
            model.update_ema_g()


        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0
        
        if args.ema_after_domain_2 and 0==t:
            print("*"*40)
            print("escape ema at the first domain")
            model.re_init_ema()
            # exit(0)

        # save the prompt and model
        if args.save_prompt and args.model != 'finetune_vit' and args.model != "classifier_vit":
            model.save_prompt_to_file(t)
        if args.save_model:
            model.save_model_to_file(t)
        if args.use_ema_c and args.save_ema:
            model.save_ema_to_file(t)
            
        torch.cuda.empty_cache()

        accs = evaluate_cprompt(model, dataset, args, t)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(args, mean_acc, t + 1, dataset.SETTING)
        save_mean_accuracy_to_file(mean_acc, t + 1, dataset.SETTING, args, t, accs, loss)
        
        # freeze G Prompt
        if t == 0 and args.freeze_g_prompt:
            model.set_g_prompt_not_update()

        model_stash['mean_accs'].append(mean_acc)
        tmp = time.time() - start_time

    running_time = time.time() - start_time

def visual_features(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:

    model.net.to(model.device)
    model.eval()
    print(model.device)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime']
    markers=['v','s','o','^','p','D']
    results, results_mask_classes = [], []
    
    model_stash = create_stash(model, args, dataset)
    
    print(file=sys.stderr)
    start_time = time.time()
    
    all_feats={}
    for t in range(dataset.N_TASKS):
        gc.collect()
        model.net.train()
        if args.dataset == "domain-net":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-c":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-r":
            train_loader = dataset.get_data_loaders(task_id=t)
        elif args.dataset == "imagenet-cr":
            train_loader = dataset.get_data_loaders(task_id=t)
        feats=[]
        Labels=[]
        
        epoch_start = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            representations = model.net.forward_features(inputs)  # [128, 197, 768]
            query = representations[:, 1, :]
            feats.append(query.cpu())
            Labels.append(labels.cpu())
        # print(Labels)
        all_feats[t]= {
                "feats":torch.cat(feats),
                "labels":torch.cat(Labels)
            }
                
    running_time = time.time() - start_time
    
    print("feature extraction time...", running_time)
    
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from matplotlib import rcParams
    import random
    # plt.scatter(6, 50, s=200, color="orange", marker="*")
    config = {
    #"font.family":'serif',
    "font.size": 18,
    "mathtext.fontset":'stix',
    #"font.serif": ['SimSun'],
    }
    rcParams.update(config)

    plt.rcParams["font.family"] = "Times New Roman"
    selected_feats=[]
    selected_labels=[]
    domain_ids=[]
    vis_num=200
    class_id=[1,2,3,4,5]
    for t in range(dataset.N_TASKS):
        feats=all_feats[t]['feats']
        labels=all_feats[t]['labels']
        # print(labels)
        # print(set(labels.tolist()))
        if class_id is not None:
            Keep=labels<-1
            # print(Keep)
            for c_id in class_id:
                Keep=Keep+(labels==c_id)
            # print(Keep)    
            labels=labels[Keep]
            feats=feats[Keep]
            # print(Keep)
        if len(feats)>0:
            selected_feats.append(feats[:vis_num])
            selected_labels.append(labels[:vis_num])
            domain_ids.append(torch.ones(len(feats[:vis_num]))*t)
    selected_feats=torch.cat(selected_feats).numpy()
    selected_labels=torch.cat(selected_labels).numpy().astype('int64')
    domain_ids=torch.cat(domain_ids).numpy().astype('int64')
    
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    feature_2D = tsne_2D.fit_transform(selected_feats)
    x_min, x_max = np.min(feature_2D, 0), np.max(feature_2D, 0)
    feature_2D = (feature_2D - x_min) / (x_max - x_min)
    plt.cla()
    
    if class_id is not None:
        unique_list = list(range(len(feature_2D)))
        random.shuffle(unique_list)    
        for i in unique_list:   
            plt.scatter(
                    feature_2D[i, 0],
                    feature_2D[i, 1],
                    c=[colors[domain_ids[i]]],
                    edgecolors='black',
                    marker=markers[selected_labels[i]],
                    linewidths=1.0,
                    s=20  
            )
    else:
        plt.scatter(
                feature_2D[:, 0],
                feature_2D[:, 1],
                c=[colors[d_id] for d_id in domain_ids],
                edgecolors='black',
                # marker=[markers[l_id] for l_id in selected_labels],
                linewidths=1.0,
                s=20  
        )

    


    # print(i)
    plt.gca().spines['right'].set_color('darkgrey')
    plt.gca().spines['left'].set_color('darkgrey')
    plt.gca().spines['bottom'].set_color('darkgrey')
    plt.gca().spines['top'].set_color('darkgrey')
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.tight_layout()
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')
    
    if class_id is not None:
        save_name = os.path.join(args.output_path,args.info, 'vis_class.png')
    else:
        save_name = os.path.join(args.output_path,args.info, 'vis.png')
    # save_name=method+'_'+dataset_name+'_'+str(j)+'.png'
    # save_name='visual_results/'+method+'/'+method+'_'+dataset_name+'_'+str(j)+'.png'
    
    plt.savefig(save_name)


def visual_prompt_keys(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:

    model.net.to(model.device)
    model.eval()
    print(model.device)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime']
    markers=['v','s','o','^','p','D']
    results, results_mask_classes = [], []
    
    
    print(file=sys.stderr)
    start_time = time.time()
    
    # model.load_model_from_file(dataset.N_TASKS-1)
    # model.load_ema_from_file(dataset.N_TASKS-1)
    model.load_prompt_from_file(dataset.N_TASKS-1)   
      
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from matplotlib import rcParams
    import random
    # plt.scatter(6, 50, s=200, color="orange", marker="*")
    config = {
    #"font.family":'serif',
    "font.size": 18,
    "mathtext.fontset":'stix',
    #"font.serif": ['SimSun'],
    }
    rcParams.update(config)

    plt.rcParams["font.family"] = "Times New Roman"
    
    # vis_num=200
    # class_id=[1,2,3,4,5]
    layers=[0]
    for l_id in layers:
        selected_feats=[]
        selected_labels=[]
        domain_ids=[]
        for t in range(dataset.N_TASKS):
            feats=model.pool.key_list[l_id][t*args.prompt_per_task:(t+1)*args.prompt_per_task]
            # print(feats)
        
            selected_feats.append(torch.stack(feats))
            domain_ids.append(torch.ones(len(feats))*t)
        selected_feats=torch.cat(selected_feats).detach().cpu().numpy()
        domain_ids=torch.cat(domain_ids).numpy().astype('int64')
        
        tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
        feature_2D = tsne_2D.fit_transform(selected_feats)
        x_min, x_max = np.min(feature_2D, 0), np.max(feature_2D, 0)
        feature_2D = (feature_2D - x_min) / (x_max - x_min)
        plt.cla()
        
    
        plt.scatter(
                feature_2D[:, 0],
                feature_2D[:, 1],
                c=[colors[d_id] for d_id in domain_ids],
                edgecolors='black',
                # marker=[markers[l_id] for l_id in selected_labels],
                linewidths=1.0,
                s=20  
        )    


        # print(i)
        plt.gca().spines['right'].set_color('darkgrey')
        plt.gca().spines['left'].set_color('darkgrey')
        plt.gca().spines['bottom'].set_color('darkgrey')
        plt.gca().spines['top'].set_color('darkgrey')
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.tight_layout()
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')
        
        save_name = os.path.join(args.output_path,args.info, f'vis_prompt_keys_layer-{l_id}.png')
        plt.savefig(save_name)


def visual_prompts(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:

    model.net.to(model.device)
    model.eval()
    print(model.device)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime']
    markers=['s','o','^','p','D', 'x', '*','v','P', '>','<', '2','H','+','X']
    results, results_mask_classes = [], []
    
    
    print(file=sys.stderr)
    start_time = time.time()
    
    # model.load_model_from_file(dataset.N_TASKS-1)
    # model.load_ema_from_file(dataset.N_TASKS-1)
    model.load_prompt_from_file(dataset.N_TASKS-1)   
      
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from matplotlib import rcParams
    import random
    # plt.scatter(6, 50, s=200, color="orange", marker="*")
    config = {
    #"font.family":'serif',
    "font.size": 18,
    "mathtext.fontset":'stix',
    #"font.serif": ['SimSun'],
    }
    rcParams.update(config)

    plt.rcParams["font.family"] = "Times New Roman"
    
    # vis_num=200
    # class_id=[1,2,3,4,5]
    layers=[2,3,4]
    for l_id in layers:
        selected_feats_key=[]
        selected_feats_value=[]
        selected_labels=[]
        domain_ids=[]
        
        group_ids=[]
        for t in range(dataset.N_TASKS):
            feats=model.pool.prompt_list[l_id][t*args.prompt_per_task:(t+1)*args.prompt_per_task]
            # print(feats)
            feats_key=[x[:args.prompt_num//2] for x  in feats]
            feats_value=[x[args.prompt_num//2:] for x  in feats]
        
            selected_feats_key.append(torch.cat(feats_key))
            selected_feats_value.append(torch.cat(feats_value))
            
            domain_ids.append(torch.ones(len(feats_key)*args.prompt_num//2)*t)
            group_ids+=[torch.ones(args.prompt_num//2)*t for t in range(len(feats_key))]
            # group_ids.append((torch.range(0, len(feats_key)-1).repeat(1,args.prompt_num//2)).view(len(feats_key)*args.prompt_num//2))
            # print(group_ids)
        domain_ids=torch.cat(domain_ids).numpy().astype('int64')
        group_ids=torch.cat(group_ids).numpy().astype('int64')
        def plot_prompt(selected_feats, domain_ids, group_ids, save_path):
            # indices = torch.randperm(len(domain_ids))
            # selected_feats=(torch.cat(selected_feats).detach().cpu()[indices]).numpy()
            # domain_ids=domain_ids[indices].numpy().astype('int64')
            # group_ids=group_ids[indices].numpy().astype('int64')
            selected_feats=torch.cat(selected_feats).detach().cpu().numpy()
            
            
            tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
            feature_2D = tsne_2D.fit_transform(selected_feats)
            x_min, x_max = np.min(feature_2D, 0), np.max(feature_2D, 0)
            feature_2D = (feature_2D - x_min) / (x_max - x_min)
            plt.cla()
            
            
            unique_list = list(range(len(selected_feats)))
            random.shuffle(unique_list)    
            for i in unique_list:   
                if group_ids[i]>= len(markers):
                    continue
                plt.scatter(
                        feature_2D[i, 0],
                        feature_2D[i, 1],
                        c=[colors[domain_ids[i]]],
                        edgecolors='black',
                        marker=markers[group_ids[i]],
                        linewidths=1.0,
                        s=20 
                )
        
      


            # print(i)
            plt.gca().spines['right'].set_color('darkgrey')
            plt.gca().spines['left'].set_color('darkgrey')
            plt.gca().spines['bottom'].set_color('darkgrey')
            plt.gca().spines['top'].set_color('darkgrey')
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['right'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)
            plt.gca().spines['top'].set_linewidth(1.5)
            plt.tight_layout()
            plt.xticks([])  # 去掉x轴
            plt.yticks([])  # 去掉y轴
            plt.axis('off')
            
            # save_name = os.path.join(args.output_path,args.info, f'vis_prompt_keys_layer-{l_id}.png')
            plt.savefig(save_path)
        save_path_key=os.path.join(args.output_path,args.info, f'vis_prompt_keys_layer-{l_id}.png')
        plot_prompt(selected_feats_key, domain_ids,group_ids, save_path_key)
        save_path_value=os.path.join(args.output_path,args.info, f'vis_prompt_valuess_layer-{l_id}.png')
        plot_prompt(selected_feats_value, domain_ids, group_ids, save_path_value)

