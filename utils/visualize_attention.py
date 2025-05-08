import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import copy
import os
import cv2
# 定义一个函数来将热图叠加在原始图像上
def overlay_heatmap(image, heatmap):
    # heatmap_resized = resize_heatmap(heatmap, (image.shape[0], image.shape[1]))
    heatmap = np.uint8(255 * heatmap)  # 将热图数据转换为0-255范围的整数
    # print(image.min(),image.max())
    image = np.uint8(255*image) 
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 
    # print(image.shape, heatmap.shape)
    overlaid_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)  # 将热图叠加到原始图像上
    # print(overlaid_image.shape)
    return overlaid_image
def merge_img(image_list, transpose=False):
    # 假设每幅图像的大小相同
    image_height, image_width, _ = image_list[0][0].shape
    
    image_list = list(map(list, zip(*image_list)))
    
    if transpose:
        image_list=[[row[i] for row in image_list] for i in range(len(image_list[0]))]

    # 创建一幅新的图像
    new_image = np.ones((len(image_list) * image_height + 10*len(image_list), len(image_list[0]) * image_width + len(image_list[0])*10, 3), dtype=np.uint8) * 255
    # print(new_image.shape)
    # 在新图像上绘制每幅图像
    for i in range(len(image_list)):
        for j in range(len(image_list[0])):
            x_offset = j * (image_width + 10)
            y_offset = i * (image_height + 10)
            new_image[y_offset:y_offset+image_height, x_offset:x_offset+image_width, :] = image_list[i][j]
    return new_image
# 缩放热图以匹配图像尺寸
# def resize_heatmap(heatmap, target_size):
#     heatmap_resized = torch.nn.functional.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
#     return heatmap_resized.squeeze(0).squeeze(0)

class AttentionVisualization(nn.Module):
    def __init__(self, model):
        super(AttentionVisualization, self).__init__()
        # model=copy.deepcopy(model)
        # model.train()
        self.model = model
        self.attentions = []
        self.hooks = []
        for name, module in self.model.named_modules():
            # if 'attn.q_norm' in name:
            if 'my_atten' in name:
                # 注册钩子来捕获注意力分数
                hook = module.register_forward_hook(self.get_attention_hook())
                self.hooks.append(hook)
        #         print(name)
        #         print(module)
        # exit(0)

    def get_attention_hook(self):
        def hook(module, input, output):
          
            self.attentions.append(output)
        return hook

    def forward(self, x):
        self.attentions = []
        _ = self.model(x)
        print(len(self.attentions))
        print("finish")
        exit(0)
        return self.attentions


# def show_attention_maps(attention_maps, imgs, layer=2,betch_id=0,method='ours', args=None):
#     print("layer number:",len(attention_maps))
#     print("atyention size:", attention_maps[0].shape) #[37, 12, 197, 199]
#     print("image size:",imgs.shape)
#     attention_maps=attention_maps[layer]
#     num_heads = attention_maps.shape[1]
#     batch_size = len(attention_maps)
#     num_prompt=attention_maps.shape[-1]-attention_maps.shape[-2]

#     for i, img_attn in enumerate(attention_maps):
#         img=imgs[i]
#         img=img.detach().cpu().permute(1,2,0).numpy()
#         fig, axes = plt.subplots(num_prompt, num_heads, figsize=(5*num_heads, 5 * num_prompt))       
        
        
#         for j, head_attn in enumerate(img_attn):head_attn
#             head_attn=head_attn.permute(1, 0)
#             # prompt_images=[]
#             for k, prompt_atten in enumerate(head_attn[1:num_prompt+1]):              
#                 ax = axes[k][j]
#                 attn_map = prompt_atten[1:].reshape(14, 14).cpu().numpy()
#                 # attn_map = head_attn.detach().cpu().numpy()
#                 # 调整大小以匹配图像尺寸
#                 attn_map = F.interpolate(torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0), scale_factor=16, mode='bilinear')[0][0]
                
#                 # prompt_images.append()
#                 ax.imshow(img, alpha=0.5)
#                 ax.imshow(attn_map, cmap='hot', alpha=0.5)
#                 ax.axis('off')
#                 ax.set_title(f'Layer {i+1}, Head {j+1}')
#         plt.tight_layout()
#         # plt.show()
        
#         save_name = os.path.join(args.output_path,args.info, str(betch_id).zfill(4)+'-'+str(i).zfill(4)+'-'+method+'.png')
    
#         plt.savefig(save_name)
#         print(save_name)
def show_attention_maps_list_head(attention_maps_list, imgs, layer=2,betch_id=0,method='ours', args=None):
    attention_maps_list=[torch.stack(x[2:]) for x in attention_maps_list] # 前2层是global prompt
    attention_maps_list=torch.stack(attention_maps_list)#[N_p+1,n_layer,bs, n_head, 197,199]
    # print(attention_maps_list.shape)
    attention_maps_list=attention_maps_list.permute(1,2,0,3,4,5) #[n_layer,bs,N_p+1, n_head, 197,199]
    # print(attention_maps_list.shape)
    # exit(0)
    # print("layer number:",len(attention_maps))
    # print("atyention size:", attention_maps[0].shape) #[37, 12, 197, 199]
    # print("image size:",imgs.shape)
    
    attention_maps_list=attention_maps_list[layer-2]#[bs,N_p+1, n_head, 197,199]
    # print(attention_maps_list.shape)
    # exit(0)
    
    attention_maps=attention_maps_list #[bs,N_p+1, n_head, 197,199]
    # print(attention_maps.shape)
    # exit(0)
    num_heads = attention_maps.shape[2]
    batch_size = len(attention_maps)
    num_prompt=attention_maps.shape[-1]-attention_maps.shape[-2]
    
    # print(attention_maps.shape)
    # exit(0)

    for i, img_attn in enumerate(attention_maps):
        img=imgs[i]
        img=img.detach().cpu().permute(1,2,0).numpy()
        # fig, axes = plt.subplots(num_prompt, num_heads, figsize=(5*num_heads, 5 * num_prompt))  
        heat_imgs=[]       
        for j, prompt_attn in enumerate(img_attn):
            # print(prompt_attn.shape)
            # prompt_attn=prompt_attn.permute(0,2,1)
            prompt_images=[]
            for k, head_atten in enumerate(prompt_attn):              
                # ax = axes[k][j]
                attn_map = head_atten[1:,0].reshape(14, 14).cpu().numpy()
                attn_map=attn_map/(attn_map.max()+1e-5)
                # attn_map = head_attn.detach().cpu().numpy()
                # 调整大小以匹配图像尺寸
                # attn_map = F.interpolate(torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0), scale_factor=4)[0][0]
                attn_map = F.interpolate(torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0), scale_factor=16, mode='bilinear')[0][0]
                
                prompt_images.append(overlay_heatmap(img[:,:,::-1], attn_map.numpy()))
                # ax.imshow(img, alpha=0.5)
                # ax.imshow(attn_map, cmap='hot', alpha=0.5)
                # ax.axis('off')
                # ax.set_title(f'Layer {i+1}, Head {j+1}')
            heat_imgs.append(prompt_images)
        new_image=merge_img(heat_imgs)
        # plt.show()
        
        save_all=True
        if save_all:                   
            save_name = os.path.join(args.output_path,args.info,f'vis_layer-{layer}', str(betch_id).zfill(4)+'-'+str(i).zfill(4)+'-'+method+'.png')
            os.makedirs(os.path.dirname(save_name),exist_ok=True)
            cv2.imwrite(save_name, new_image)
            print(save_name)
        else:
            save_name = os.path.join(args.output_path,args.info,f'vis_layer-{layer}', str(betch_id).zfill(4)+'-'+str(i).zfill(4)+'-'+method+'.png')
            # plt.tight_layout`()
            # plt.savefig(save_name)
            os.makedirs(os.path.dirname(save_name),exist_ok=True)
            cv2.imwrite(save_name, new_image)
            print(save_name)


def show_attention_maps_list(attention_maps_list, imgs, layer=2,betch_id=0,method='ours', args=None,targets=None, class_names=None):
    targets=targets.cpu().tolist()
    attention_maps_list=[torch.stack(x[2:]) for x in attention_maps_list] # 前2层是global prompt
    attention_maps_list=torch.stack(attention_maps_list)#[N_p+1,n_layer,bs, n_head, 197,201]
    # print(attention_maps_list.shape)
    attention_maps_list=attention_maps_list.permute(1,2,0,3,4,5) #[n_layer,bs,N_p+1, n_head, 197,201]
    # print(attention_maps_list.shape)
    # exit(0)
    # print("layer number:",len(attention_maps))
    # print("atyention size:", attention_maps[0].shape) #[37, 12, 197, 201]
    # print("image size:",imgs.shape)
    
    attention_maps_list=attention_maps_list[layer-2]#[bs,N_p+1, n_head, 197,201]
    # print(attention_maps_list.shape)
    # exit(0)
    
    attention_maps=attention_maps_list.mean(2)#[bs,N_p+1, 197,201]
    # print(attention_maps.shape)
    # exit(0)
    # num_heads = attention_maps.shape[1]
    batch_size = len(attention_maps)
    num_prompt=attention_maps.shape[-1]-attention_maps.shape[-2]
    
    # print(attention_maps.shape)
    # exit(0)

    for i, img_attn in enumerate(attention_maps):
        img=imgs[i]
        img=img.detach().cpu().permute(1,2,0).numpy()
        # fig, axes = plt.subplots(num_prompt, num_heads, figsize=(5*num_heads, 5 * num_prompt))  
        heat_imgs=[]       
        for j, head_attn in enumerate(img_attn):
            # print(head_attn.shape)
            head_attn=head_attn.permute(1, 0)
            prompt_images=[]
            for k, prompt_atten in enumerate(head_attn[1:num_prompt+1]):    # from head to ptompt          
                # ax = axes[k][j]
                if 0==k:
                    # attn_map = head_attn[num_prompt+1:,0].reshape(14, 14).cpu().numpy()
                    attn_map = prompt_atten[1:].reshape(14, 14).cpu().numpy()
                else:
                    attn_map = prompt_atten[1:].reshape(14, 14).cpu().numpy()
                attn_map=attn_map/(attn_map.max()+1e-5)
                # attn_map = head_attn.detach().cpu().numpy()
                # 调整大小以匹配图像尺寸
                # attn_map = F.interpolate(torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0), scale_factor=4)[0][0]
                attn_map = F.interpolate(torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0), scale_factor=16, mode='bilinear')[0][0]
                
                prompt_images.append(overlay_heatmap(img[:,:,::-1], attn_map.numpy()))
                # ax.imshow(img, alpha=0.5)
                # ax.imshow(attn_map, cmap='hot', alpha=0.5)
                # ax.axis('off')
                # ax.set_title(f'Layer {i+1}, Head {j+1}')
            heat_imgs.append(prompt_images)
        new_image=merge_img(heat_imgs, transpose=True)
        
        save_single=True
        if save_single:
            # heat_imgs=[[row[i] for row in heat_imgs] for i in range(len(heat_imgs[0]))] # 转置
            for mm in range(len(heat_imgs)):
                for nn in range(len(heat_imgs[0])):
                    vis_img=heat_imgs[mm][nn]
                    
                    pos='/'+str(mm)+'-'+str(nn)
                    
                    if class_names is not None:
                        save_name = os.path.join(args.output_path,args.info,'heatmaps_single',class_names[targets[i]],f'vis_layer-{layer}', str(betch_id).zfill(4)+'-'+str(i).zfill(4)+pos+'_'+method+'.png')
                    else:
                        save_name = os.path.join(args.output_path,args.info,f'vis_layer-{layer}', str(betch_id).zfill(4)+'-'+str(i).zfill(4)+pos+'-'+method+'.png')
                    # plt.tight_layout()
                    # plt.savefig(save_name)
                    os.makedirs(os.path.dirname(save_name),exist_ok=True)
                    cv2.imwrite(save_name, vis_img)
                    print(save_name)
        else:
            
            # plt.show()
            if class_names is not None:
                save_name = os.path.join(args.output_path,args.info,'heatmaps',class_names[targets[i]],f'vis_layer-{layer}', str(betch_id).zfill(4)+'-'+str(i).zfill(4)+'-'+method+'.png')
            else:
                save_name = os.path.join(args.output_path,args.info,f'vis_layer-{layer}', str(betch_id).zfill(4)+'-'+str(i).zfill(4)+'-'+method+'.png')
            # plt.tight_layout()
            # plt.savefig(save_name)
            os.makedirs(os.path.dirname(save_name),exist_ok=True)
            cv2.imwrite(save_name, new_image)
            print(save_name)


def show_attention_maps_list_cls(attention_maps_list, imgs, layer=2,betch_id=0,method='ours', args=None):
    attention_maps_list=[torch.stack(x[2:]) for x in attention_maps_list] # 前2层是global prompt
    attention_maps_list=torch.stack(attention_maps_list)#[N_p+1,n_layer,bs, n_head, 197,199]
    # print(attention_maps_list.shape)
    attention_maps_list=attention_maps_list.permute(1,2,0,3,4,5) #[n_layer,bs,N_p+1, n_head, 197,199]
    # print(attention_maps_list.shape)
    # exit(0)
    # print("layer number:",len(attention_maps))
    # print("atyention size:", attention_maps[0].shape) #[37, 12, 197, 199]
    # print("image size:",imgs.shape)
    
    attention_maps_list=attention_maps_list[layer-2]#[bs,N_p+1, n_head, 197,199]
    # print(attention_maps_list.shape)
    # exit(0)
    
    attention_maps=attention_maps_list.mean(2)#[bs,N_p+1, 197,199]
    # print(attention_maps.shape)
    # exit(0)
    num_heads = attention_maps.shape[1]
    batch_size = len(attention_maps)
    num_prompt=attention_maps.shape[-1]-attention_maps.shape[-2]
    
    # print(attention_maps.shape)
    # exit(0)

    for i, img_attn in enumerate(attention_maps):
        img=imgs[i]
        img=img.detach().cpu().permute(1,2,0).numpy()
        fig, axes = plt.subplots(num_prompt, num_heads, figsize=(5*num_heads, 5))  
        heat_imgs=[]       
        for j, head_attn in enumerate(img_attn):
            # print(head_attn.shape)
            # head_attn=head_attn.permute(1, 0)
            prompt_images=[]
            # for k, prompt_atten in enumerate(head_attn[1:num_prompt+1]):     
            prompt_atten=head_attn[0]         
            ax = axes[j]
            attn_map = prompt_atten[1+num_prompt:].reshape(14, 14).cpu().numpy()
            attn_map=attn_map/(attn_map.max()+1e-5)
            # attn_map = head_attn.detach().cpu().numpy()
            # 调整大小以匹配图像尺寸
            # attn_map = F.interpolate(torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0), scale_factor=4)[0][0]
            attn_map = F.interpolate(torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0), scale_factor=16, mode='bilinear')[0][0]
            
            prompt_images.append(overlay_heatmap(img[:,:,::-1], attn_map.numpy()))
          
            heat_imgs.append(prompt_images)
        new_image=merge_img(heat_imgs)
        # plt.show()
        
        save_name = os.path.join(args.output_path,args.info,f'vis_layer-{layer}', str(betch_id).zfill(4)+'-'+str(i).zfill(4)+'-'+method+'.png')
        # plt.tight_layout()
        # plt.savefig(save_name)
        os.makedirs(os.path.dirname(save_name),exist_ok=True)
        cv2.imwrite(save_name, new_image)
        print(save_name)




def visyalize_compare(model_1,model_2, dataset, 
                        args, t, last=False):
    
    model_1.net.eval()
    model_2.net.eval()
    accs, accs_mask_classes = [], []

    # using ema
    if args.use_ema or args.use_ema_c:
        model_1.ema_before_eval(t)
        model_2.ema_before_eval(t)
    if args.dataset == "domain-net":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-c":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-r":
        test_loaders = dataset.get_test_dataloaders(t)
    elif args.dataset == "imagenet-cr":
        test_loaders = dataset.get_test_dataloaders(t)

    
    attn_visualizer_1 = AttentionVisualization(model_1)
    attn_visualizer_2 = AttentionVisualization(model_2)
    
    target_class=[0,1,2,3,12,25,29,31,42,51,86,184,247,275,325]
    target_class=torch.tensor(target_class).view(1,-1).to(model_1.device)
    
    class_names={
        0: "aircraft_carrier",
        1: "airplane",
    2: "alarm_clock",
    3: "ambulance",
    12: "backpack",
    25: "bed",
    29: "bicycle",
    31: "bird",
    42: "bridge",
    51: "cake",
    86: "cruise_ship",
    184: "microphone",
    247: "sailboat",
    275: "speedboat",
    325: "truck",
    
    }
    
    with torch.no_grad():
        keys_all=[]
        for k, test_loader in enumerate(test_loaders):
            if last and k < len(test_loaders) - 1:                
                continue
            correct, correct_mask_classes, total = 0.0, 0.0, 0.0
            
            
            print(f"*******domian {k} *******")
            tmp = 0
            for data in test_loader:
                tmp += 1
                if tmp%50 == 0:
                    print(tmp," / ", len(test_loader))
                inputs, labels = data
                inputs, labels = inputs.to(model_1.device), labels.to(model_1.device)
                print("batchsize:",len(inputs))
                outputs, keys = model_1.forward_model(inputs)    
                outputs=F.softmax(outputs,-1)
                outputs_2, keys_2 = model_2.forward_model(inputs)    
                outputs_2=F.softmax(outputs_2,-1)         
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()    
                
                total += labels.shape[0]
                # print("correct:", correct)
                # print("total:", total)`
                S1, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
                
                S2, pred_2 = torch.max(outputs_2.data, 1)    # 得到模型2的预测
                
                res=(pred==labels).float()-(pred_2==labels).float() # 找到 good case
                Keep =(res>0)*(S1-S2>0.5)
                
                Keep_cls=((labels.view(-1,1)-target_class)==0).sum(-1)
                # print(Keep_cls)
                Keep_cls=Keep_cls>0
                Keep=Keep_cls
                # Keep=res>=0
                
                torch.cuda.empty_cache()
                
                if torch.any(Keep):                    
                    input_tensor=inputs[Keep]
                    keys=keys[Keep]
                    keys_2=keys_2[Keep]
                    targets=labels[Keep]
                    
                    # 得到注意力分数
                    attn_visualizer_1.attentions=[]
                    model_1.forward_model(input_tensor)                      
                    attentions_1 = attn_visualizer_1.attentions
                     
                    # print(attentions_1)
                    attn_visualizer_2.attentions=[]
                    model_2.forward_model(input_tensor)                      
                    attentions_2 = attn_visualizer_2.attentions
                    
                    # show_attention_maps(attentions_1, input_tensor, betch_id=tmp,method='ours', args=args)
                    # show_attention_maps(attentions_2, input_tensor, betch_id=tmp,method='c-prompt', args=args)
                    
                    
                    # 得到注意力分数
                    
                    keys=keys.permute(1,0)
                    attentions_1_single=[attentions_1]
                    for key_one in keys:
                        attn_visualizer_1.attentions=[]
                        model_1.forward_specific_prompt(input_tensor, key_one)                      
                        attentions_1_single.append(attn_visualizer_1.attentions)
                     
                    keys_2=keys_2.permute(1,0)
                    attentions_2_single=[attentions_2]
                    for key_one in keys_2:
                        attn_visualizer_2.attentions=[]
                        model_2.forward_specific_prompt(input_tensor, key_one)                      
                        attentions_2_single.append(attn_visualizer_2.attentions)
                    
      
                    
                    # 可视化 attention map
                    # show_attention_maps_list(attentions_1_single, input_tensor,layer=3, betch_id=tmp,method='ours', args=args, targets=targets, class_names=class_names)
                    # show_attention_maps_list(attentions_2_single, input_tensor,layer=3, betch_id=tmp,method='c-prompt', args=args, targets=targets, class_names=class_names)
                    
                    # show_attention_maps_list(attentions_1_single, input_tensor,layer=4, betch_id=tmp,method='ours', args=args, targets=targets, class_names=class_names)
                    # show_attention_maps_list(attentions_2_single, input_tensor,layer=4, betch_id=tmp,method='c-prompt', args=args, targets=targets, class_names=class_names)
                    
                    show_attention_maps_list(attentions_1_single, input_tensor,layer=5, betch_id=tmp,method='ours', args=args, targets=targets, class_names=class_names)
                    # show_attention_maps_list(attentions_2_single, input_tensor,layer=5, betch_id=tmp,method='c-prompt', args=args, targets=targets, class_names=class_names)
                    show_attention_maps_list(attentions_1_single, input_tensor,layer=7, betch_id=tmp,method='ours', args=args, targets=targets, class_names=class_names)
                    # show_attention_maps_list(attentions_1_single, input_tensor,layer=9, betch_id=tmp,method='ours', args=args, targets=targets, class_names=class_names)
                    # show_attention_maps_list(attentions_1_single, input_tensor,layer=11, betch_id=tmp,method='ours', args=args, targets=targets, class_names=class_names)
             
                else:
                    attn_visualizer_1.attentions=[]
                    attn_visualizer_2.attentions=[]
                    

            
            
            accs.append(correct / total * 100)
            accs_mask_classes.append(correct_mask_classes / total * 100)
        exit()
    # using ema
    if args.use_ema or args.use_ema_c:
        model_1.ema_after_eval()
        model_2.ema_after_eval()

    return accs, accs_mask_classes