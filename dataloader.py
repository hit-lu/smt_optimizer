import numpy as np
import copy
import pandas as pd
from common_function import *

def load_data(filename: str, load_cp_data = True):
    # 读取PCB数据
    filename = 'data/' + filename
    pcb_data = pd.DataFrame(pd.read_csv(filename, '\t', header = None)).dropna(axis = 1)
    n_columns = len(pcb_data.columns)

    if n_columns == 18:
        step_col = ["ref", "x", "y", "z", "r", "part", "desc", "fdr", "nz", "hd", "cs", "cy", "sk", "bk", "ar", "fid", "pl", "lv"]
    elif n_columns == 16:
        step_col = ["ref", "x", "y", "z", "r", "part", "fdr", "nz", "hd", "cs", "cy", "sk", "ar", "fid", "pl", "lv"]
    else:
        step_col = ["ref", "x", "y", "z", "r", "part", "fdr", "nz", "hd", "cs", "cy", "sk", "bk", "ar", "fid", "pl", "lv"]

    pcb_data.columns = step_col

    # 注册元件检查
    component_data = None
    if load_cp_data:
        part_col = ["part", "fdr", "nz1", "nz2"]
        component_data = pd.DataFrame(pd.read_csv('component.txt', '\t', header = None))
        component_data.columns = part_col
        for i in range(len(pcb_data)):
            if not pcb_data.loc[i].part in component_data['part'].values:
                raise Exception("unregistered component:  " + pcb_data.loc[i].part)

    # 读取供料器基座数据
    feeder_data = None
    if load_cp_data:
        feeder_col = ['slot', 'part', 'desc', 'type', 'push', 'x', 'y', 'z', 'r', 'part_r', 'skip', 'dump', 'pt']
        feeder_data = pd.DataFrame(pd.read_csv('feeder.txt', '\t', header = None)).dropna(axis = 1)
        feeder_data.columns = feeder_col

    return pcb_data, component_data, feeder_data