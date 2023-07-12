import random
import time
import os
import pandas as pd
import numpy as np
from dataloader import load_data


def generate_pcb_file(component_data, n_points=100, x_min=0, x_max=200, y_min=0, y_max=200):

    lineinfo = ''
    for index in range(n_points):
        component_index = random.randint(0, len(component_data) - 1)
        data = component_data.iloc[component_index]
        part, nozzle = data.part, data.nz
        lineinfo += 'R' + str(index + 1) + '\t'
        pos_x, pos_y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)

        lineinfo += '{:.3f}'.format(pos_x) + '\t' + '{:.3f}'.format(
            pos_x) + '\t0.000\t0.000\t' + part + '\t\tA\t1-0 ' + nozzle + '\t1\t1\t1\t1\t1\t1\t1\tN\tL0\n'
    filepath = 'rd' + time.strftime('%d%H%M', time.localtime()) + '.txt'
    with open('data/' + filepath, 'w') as f:
        f.write(lineinfo)
    f.close()
    return filepath


def convert_pcb_file(file_path, x_min=0, x_max=200, y_min=0, y_max=200):
    pcb_data, _, _ = load_data(file_path, component_register=True)
    for idx in range(len(pcb_data)):
        pos_x, pos_y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)

        pcb_data.loc[idx, 'x'] = '{:.3f}'.format(pos_x)
        pcb_data.loc[idx, 'y'] = '{:.3f}'.format(pos_y)
        pcb_data.loc[idx, 'z'] = '0.000'
        pcb_data.loc[idx, 'r'] = '0.000'

    pcb_data.to_csv('data/convert/cvt - ' + file_path, sep='\t', index=False, header=None)


if __name__ == '__main__':

    # === 生成随机数据 ===
    # component_data = pd.DataFrame(pd.read_csv(filepath_or_buffer='component.txt', sep='\t', header=None))
    # component_data.columns = ["part", "desc", "fdr", "nz", 'camera', 'group', 'feeder-limit', 'points']
    # filepath = generate_pcb_file(component_data)
    #
    # pcb_data, component_data = load_data(filepath, feeder_limit=1, auto_register=1)

    # === 坐标随机转换 ===
    # for _, file in enumerate(os.listdir('data/')):
    #     if len(file) < 4 or file[-4:] != '.txt':
    #         continue
    #     convert_pcb_file(file)
    convert_pcb_file('PCB.txt')

