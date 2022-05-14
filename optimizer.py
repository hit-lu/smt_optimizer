import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np

from result_analysis import *
from dataloader import *

from optimizer_celldivision import *
from optimizer_feederpriority import *
from optimizer_hierarchy import *
from optimizer_vrpmodel import *

from random_generator import *

parser = argparse.ArgumentParser(description = 'smt optimizer implementation')
parser.add_argument('--filename', default = 'PCB.txt', type = str, help = 'load pcb data')
parser.add_argument('--mode', default = 1, type = int, help = 'mode: 0 -directly load pcb data without optimization for data analysis, 1 -optimize pcb data')
parser.add_argument('--optimize_method', default = 'feeder_priority', type = str, help = 'optimizer algorithm')
parser.add_argument('--figure', default = 1, type = int, help = 'draw mount process figure or not')
parser.add_argument('--feeder_limit', default = 1, type = int, help = 'the upper bound of feeder assigned to the slot')
params = parser.parse_args()

pcb_data, component_data, feeder_data = load_data(params.filename)     # 加载PCB数据
component_result, cycle_result, feederslot_result, placement_result, head_sequence = [], [], [], [], []

if params.mode == 0:
    component_result, cycle_result, feederslot_result, placement_result, head_sequence = convert_pcbdata_to_result(pcb_data, component_data)
else:
    if params.optimize_method == 'cell_division':
        component_result, cycle_result, feederslot_result = optimizer_celldivision(pcb_data, component_data)
        placement_result, head_sequence = greedy_placement_route_generation(pcb_data, component_data, component_result,
                                                                     cycle_result)
    elif params.optimize_method == 'hierarchy':
        # TODO: 吸杆任务分配
        placement_result, head_sequence = greedy_placement_route_generation(pcb_data, component_data, component_result,
                                                                     cycle_result)
    elif params.optimize_method == 'feeder_priority':
        # 第1步：分配供料器位置
        feeder_allocate(component_data, pcb_data, feeder_data)
        # 第2步：扫描供料器基座，确定元件拾取的先后顺序
        component_result, cycle_result, feederslot_result = feederbase_scan(component_data, pcb_data, feeder_data)
        # 第3步：贴装路径规划
        placement_result, head_sequence = greedy_placement_route_generation(pcb_data, component_data, component_result,
                                                                     cycle_result)
    else:
        # TODO: LED 类型贴片机路径规划算法（未完成）
        component_result, cycle_result, feederslot_result, placement_result, head_sequence = optimizer_route_generation(pcb_data, component_data, render = True)


if params.figure:
    # 绘制各周期从供料器拾取的贴装点示意图
    # pickup_cycle_schematic(feederslot_result, cycle_result)

    # 绘制贴装路径图
    placement_route_schematic(pcb_data, component_result, cycle_result, feederslot_result, placement_result, head_sequence,1)

# 估算贴装用时
placement_time_estimate(pcb_data, component_result, cycle_result, feederslot_result, placement_result, head_sequence)

# 统计各类型元件的贴装点数
# component_points = [0] * len(component_data)
# for i in range(len(pcb_data)):
#     part = pcb_data.loc[i]['part']
#     component_index = component_data[component_data['part'] == part].index.tolist()[0]
#
#     component_points[component_index] += 1
#
# sort_index = np.argsort(component_points)[::-1]
#
# component_groups = []
# index = 0
# while True:
#     if index + max_head_index >= len(component_data):
#         component_groups.append(sort_index[index: len(component_data)])
#         break
#     else:
#         component_groups.append(sort_index[index: index + max_head_index])
#
#     index += max_head_index

# feeder_base[:max_slot_index // 2].plot(kind = 'bar')                # 41种有效元件
# plt.show()

