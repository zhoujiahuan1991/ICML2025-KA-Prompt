import os
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import sys
import json

# 保留有效数字
def keep_significant(num, digits=2):
    if num == 0:
        return 0
    # if num>=10:
    real_digits=digits - int(f"{abs(num):e}".split('e')[1]) - 1
    real_digits=max(real_digits, digits)
    return round(num, real_digits)
  
  
def cal_statis(model_dir):
  folder_names=os.listdir(model_dir)
  
  Avg_Acc=[]
  Avg_F=[]
  CAA=[]
  Last_res=[]
  Stage_Accs=[]
  for name in folder_names:
    res_path=os.path.join(model_dir, name, 'results')
    if os.path.isdir(res_path):
      avg_acc, avg_f, caa, last_res, stage_accs=plot_res(res_path)
      Avg_Acc.append(avg_acc)
      Avg_F.append(avg_f)
      CAA.append(caa)
      Last_res.append(last_res)
      Stage_Accs.append(stage_accs)

  
  def mean_std(results):
    if not isinstance(results, np.ndarray):      
      data_array = np.array(results)
    else:
      data_array=results
    mean_value = np.mean(data_array)
    std_deviation = np.std(data_array)
    # return keep_significant(mean_value,2), keep_significant(std_deviation,2)
    return round(mean_value,2), round(std_deviation,2)
  
  Last_res=np.array(Last_res).T
  subset_acc=[mean_std(x) for x in Last_res]
  
  Stage_Accs=np.array(Stage_Accs).T
  Stage_Accs=[mean_std(x) for x in Stage_Accs]
  
  Stage_Accs_mean=np.array(Stage_Accs)[:,0].tolist()
  Stage_Accs_var=np.array(Stage_Accs)[:,1].tolist()
  # for s_acc in Stage_Accs:
  print("domain-wise avg-acc_mean:", Stage_Accs_mean)
  print("domain-wise avg-acc_var:", Stage_Accs_var)
    
  
  return {
    'avg_acc':mean_std(Avg_Acc),
    "avg_f":mean_std(Avg_F),
    'caa':mean_std(CAA),
    'subset_acc':subset_acc,
    # ''
  }
  
  
  
def cal_forget(all_res):
  res_dict=defaultdict(list)
  for res_line in all_res:
    for c_id, acc_res in enumerate(res_line):
      res_dict[c_id].append(acc_res)
  # hist_ids=list(range(len(all_res)-1))
  max_res=[max(res_dict[x][:-1]) for x in range(len(all_res)-1)]
  final_res=[res_dict[x][-1] for x in range(len(all_res)-1)]
  avg_f=0
  for m_s,f_s in zip(max_res, final_res):
    avg_f+=m_s-f_s
  avg_f=round(avg_f/len(max_res),2)
  return avg_f
  
def plot_res(res_dir):
  num_files=len(os.listdir(res_dir))
  all_res=[]
  
  avg_accs=[]
  for idx in range(num_files):
    with open(res_dir+f'/{idx}.txt', 'r') as file:
      for idx, line in enumerate(file):
        if 1==idx: 
          if '([' in line:
                line=line.replace('[', '').replace("(",'').replace("\n",'')
                line=line.split(']')[0]
          else:
                line=line.replace('[', '').replace("]",'').replace("\n",'')       
          # line=line.replace('[', '').replace("]",'').replace("\n",'')
          res=[float(x) for x in line.split(',')]
          all_res.append(res)
        elif 3 == idx:
          final_line=line.split(': ')[-1]
          final_line=final_line.split(' ')[0]
          avg_accs.append(float(final_line))
  
  # calculate the forgetting rate       
  avg_f=cal_forget(all_res)
  # calculate the performance at each stage
  stage_accs=[round(sum(x)/len(x),2) for x in all_res]
  
  
         
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime']
  colors=colors+colors
  
  
  
  for idx in range(num_files):
    res=[]
    for nn in range(idx, num_files):
      res.append(all_res[nn][idx])
    x=range(idx, num_files)
    plt.plot(x,res , color=colors[idx], label=f'domain {idx+1}')
  plt.title(f"average forgetting rate {avg_f}")
  plt.legend()
  plt.savefig(res_dir)
  # print("saving visualization results to...", res_dir)
  last_res=all_res[-1]
  
  return avg_accs[-1], avg_f, sum(avg_accs)/len(avg_accs), last_res, stage_accs
  
if __name__=='__main__':    
  if len(sys.argv)>1:    
    res_dir=sys.argv[1]  
  else:  
    res_dir='/home/xukunlun/CODE/DIL/CPrompt-Aligner/Ablation/DomainNet/aux'
  res=cal_statis(res_dir)
  save_path=os.path.join(res_dir, 'statistics.txt')
  # print('\n')
  print('\n',res)
  with open(save_path, 'w') as file:
        json.dump(res, file)
