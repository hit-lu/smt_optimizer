import numpy as np
import copy
import pandas as pd

# 读取PCB数据
step_col = ["ref", "x", "y", "z", "r", "part", "fdr", "nz", "hd", "cs", "cy", "sk", "ar", "fid", "pl", "lv"]
pcb_data = pd.DataFrame(pd.read_csv('data/pcb.txt', '\t', header = None)).dropna(axis = 1)
pcb_data.columns = step_col
point_num = len(pcb_data)

# 注册元件检查
part_col = ["part", "fdr", "nz1", "nz2"]
component_data = pd.DataFrame(pd.read_csv('data/component.txt', '\t', header = None))
component_data.columns = part_col
for i in range(len(pcb_data)):
    if not pcb_data.loc[i].part in component_data['part'].values:
        raise Exception("unregisted component:  " + pcb_data.loc[i].part)