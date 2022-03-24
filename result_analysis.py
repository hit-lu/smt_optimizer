import matplotlib.pyplot as plt
import numpy as np
from dataloader import *
import random

# 绘制各周期从供料器周期拾取的元件位置
def pickup_cycle_schematic(feederslot_result, cycle_result):
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # data
    bar_width = .7
    # for cycle in range(len(feederslot_result)):
    feederpart = np.zeros((int)(max_slot_index / 2), dtype = np.int)
    for cycle in range(len(feederslot_result)):
        label_str = '周期' + str(cycle + 1)
        cur_feederpart = np.zeros((int)(max_slot_index / 2), dtype = np.int)
        for slot in feederslot_result[cycle]:
            if slot > 0:
                cur_feederpart[slot] += cycle_result[cycle]

        plt.bar(np.arange(max_slot_index/2), cur_feederpart, bar_width, \
                edgecolor = 'black', bottom = feederpart, label = label_str)

        for slot in feederslot_result[cycle]:
            if slot > 0:
                feederpart[slot] += cycle_result[cycle]

    plt.legend()
    plt.show()


# 绘制指定周期的拾贴路径图
def placement_route_schematic(component_result, cycle_result, feederslot_result, placement_result, cycle = 0):
    pos_x, pos_y = [], []
    for i in range(len(pcb_data)):
        pos_x.append(pcb_data.loc[i]['x'])
        pos_y.append(pcb_data.loc[i]['y'])
        # plt.text(pcb_data.loc[i]['x'], pcb_data.loc[i]['y'] + 0.1, '%d' % i, ha='center', va = 'bottom', size = 8)

    mount_pos = []
    for head in range(max_head_index):
        index = placement_result[cycle][head]
        plt.text(pcb_data.loc[index]['x'], pcb_data.loc[index]['y'] + 0.1, 'HD%d' % head, ha = 'center', va = 'bottom', size = 10)
        mount_pos.append([pcb_data.loc[index]['x'] - head * slot_interval * interval_ratio, pcb_data.loc[index]['y']])
        plt.plot(mount_pos[-1][0], mount_pos[-1][1], marker = '^', color = 'red', markerfacecolor = 'white')

    mount_pos = np.sort(mount_pos, axis = 1)
    for i in range(len(mount_pos) - 1):
        plt.plot([mount_pos[i][0], mount_pos[i + 1][0]], [mount_pos[i][1], mount_pos[i + 1][1]], color = 'blue', linewidth = 1)

    plt.scatter(pos_x, pos_y, s = 8)
    # TODO: 绘制供料器位置布局
    plt.show()

# TODO: 贴装时间预估函数
def placement_time_estimate(component_result, cycle_result, feederslot_result, placement_result) -> float:
    t_pick, t_place = 1., 0.    # 贴装/拾取用时
    velocity = 1000             # 移动速度
    total_distance = .0         # 总移动距离
    cur_position = [0, 0]       # 贴装头当前位置

    return .0