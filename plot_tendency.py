import os
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
 
# rc('text', usetex=True)
# rc('text.latex', preamble=r'\usepackage{custom_commands}')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.linewidth': 0.8,
    'grid.color': '#EBEBEB',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8
})


def cal_forget(all_res):
  res_dict=defaultdict(list)
  for res_line in all_res:
    for c_id, acc_res in enumerate(res_line):
      res_dict[c_id].append(acc_res)
  # hist_ids=list(range(len(all_res)-1))
  max_res=[max(res_dict[x][:-1]) for x in range(len(all_res)-1)]
  final_res=[res_dict[x][-1] for x in range(len(all_res)-1)]
  # print(max_res)
  # print(final_res)
  avg_f=0
  for m_s,f_s in zip(max_res, final_res):
    avg_f+=m_s-f_s
  avg_f=round(avg_f/len(max_res),2)
  print(f"*************\n for previous T-1 domains \n max accs:{max_res}\n final accs:{final_res}\n The average forgetting rate is {avg_f}")
  return avg_f
  
def plot_res(res_dir):
  num_files=len(os.listdir(res_dir))
  all_res=[]
  for idx in range(num_files):
    with open(res_dir+f'/{idx}.txt', 'r') as file:
      for idx, line in enumerate(file):
        if 1==idx:        
          line=line.replace('[', '').replace("]",'').replace("\n",'')
          res=[float(x) for x in line.split(',')]
          all_res.append(res)
  
  # calculate the forgetting rate       
  avg_f=cal_forget(all_res)
  
         
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime']
  colors=colors+colors
  
  plt.clf()
  
  
  # print(all_res)
  plt.figure(figsize=(10, 5))
  for idx in range(num_files):
    res=[]
    for nn in range(idx, num_files):
      # print(idx,nn)
      res.append(all_res[nn][idx])
    x=range(idx+1, num_files+1)
    plt.plot(x,res , color=colors[idx], label=f'domain {idx+1}')
   
  plt.xlabel('t',fontsize=15)
  plt.ylabel('Avg-ACC', fontsize=15)
  
  if num_files<=15:
    plt.legend(ncol=5, bbox_to_anchor=(0.45,-0.42), loc='lower center', edgecolor='w', fontsize = 15,
            handletextpad=0.5,
            columnspacing=1.3
            )
#     plt.legend(
#     loc='lower center',           
#     bbox_to_anchor=(0.5, -0.1),   
#     ncol=5,                       
#     fontsize=15, 
#     title_fontsize=15,
#     columnspacing=0.8,          
#     handletextpad=0.5             
# )
  else:
    plt.legend()
    plt.title(f"average forgetting rate {avg_f}")
  plt.savefig(res_dir, bbox_inches='tight')
  print("saving visualization results to...", res_dir)
  
  
if __name__=='__main__':        
  res_dir='/home/xukunlun/CODE/DIL/KA-Prompt/Repeat_models/KA-Prompt/ImageNet-R/seed-3407/results'
  plot_res(res_dir)