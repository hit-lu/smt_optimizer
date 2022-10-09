from optimizer_common import *

import numpy as np
import pandas as pd


# 生成坐标范围为min_v~max_v的pcb数据
def generate_pcbfile(filename, n_points=100, n_components=10, x_low=0, x_high=200, y_low=0, y_high=200):

    angle_set = [0, 30, 60, 90, 180, 270]
    component_data = pd.DataFrame(pd.read_csv('data/component.txt', '\t', header=None))
    component_data.columns = ["part", "desc", "fdr", "nz", 'camera', 'group', 'feeder-limit']
    component_data = component_data.sample(min(n_components, len(component_data)))

    with open('data/' + filename, 'w') as f:
        for index in range(n_points):
            part = str(component_data.sample(1)['part'].values)
            part = part[2: len(part) - 2]
            lineinfo = ref = 'R' + str(index + 1) + '\t'
            pos_x, pos_y = np.random.uniform(x_low, x_high), np.random.uniform(y_low, y_high)
            pos_r = np.random.choice(angle_set)
            lineinfo += '{:.3f}'.format(pos_x) + '\t' + '{:.3f}'.format(pos_x) + '\t0.000\t' + '{:.3f}'.format(pos_r) + '\t'

            lineinfo += part +'\tA\tA\t1\t1\t1\t1\t1\t1\t1\tN\tL0'
            f.write(lineinfo + '\n')
