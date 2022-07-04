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

parser = argparse.ArgumentParser(description='smt optimizer implementation')
parser.add_argument('--filename', default='IPC9850.txt', type=str, help='load pcb data')
parser.add_argument('--mode', default=1, type=int, help='mode: 0 -directly load pcb data without optimization '
                                                        'for data analysis, 1 -optimize pcb data')
parser.add_argument('--optimize_method', default='cell_division', type=str, help='optimizer algorithm')
parser.add_argument('--figure', default=0, type=int, help='draw mount process figure or not')
parser.add_argument('--feeder_limit', default=1, type=int, help='the upper bound of feeder assigned to the slot')
parser.add_argument('--save', default=0, type=int, help='save the optimization result and figure')
params = parser.parse_args()

pcb_data, component_data, feeder_data = load_data(params.filename, load_feeder_data=False)  # 加载PCB数据
component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = [], [], [], [], []

if params.mode == 0:
    component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = convert_pcbdata_to_result(
        pcb_data, component_data)
else:
    if params.optimize_method == 'cell_division':
        component_result, cycle_result, feeder_slot_result = optimizer_celldivision(pcb_data, component_data)
        placement_result, head_sequence = greedy_placement_route_generation(pcb_data, component_data, component_result,
                                                                            cycle_result)
    elif params.optimize_method == 'hierarchy':
        # TODO: 吸杆任务分配
        placement_result, head_sequence = greedy_placement_route_generation(pcb_data, component_data, component_result,
                                                                            cycle_result)
    elif params.optimize_method == 'feeder_priority':
        # 第1步：吸嘴分配          TODO: 此函数不可用
        nozzle_result, nozzle_cycle = nozzle_assignment(component_data, pcb_data)
        # 第2步：分配供料器位置       TODO: 尚未考虑一类元件对应多个供料器的情形
        feeder_allocate(component_data, pcb_data, feeder_data, nozzle_result, nozzle_cycle, params.figure)
        # 第3步：扫描供料器基座，确定元件拾取的先后顺序
        component_result, cycle_result, feeder_slot_result = feederbase_scan(component_data, pcb_data, feeder_data,
                                                                            nozzle_result, nozzle_cycle)
        # 第4步：贴装路径规划
        placement_result, head_sequence = greedy_placement_route_generation(pcb_data, component_data, component_result,
                                                                            cycle_result)
    elif params.optimize_method == 'route_schedule':
        component_result, cycle_result, feeder_slot_result, _, _ = convert_pcbdata_to_result(
            pcb_data, component_data)

        # placement_result, head_sequence = cluster_based_route_generation(component_data, pcb_data, component_result,
        #                                                                  cycle_result, feeder_slot_result)
        placement_result, head_sequence = greedy_placement_route_generation(pcb_data, component_data, component_result,
                                                                            cycle_result)

if params.figure:
    # 绘制各周期从供料器拾取的贴装点示意图
    # pickup_cycle_schematic(feederslot_result, cycle_result)

    # 绘制贴装路径图
    for cycle in range(0, len(placement_result)):
        placement_route_schematic(pcb_data, component_result, cycle_result, feeder_slot_result, placement_result,
                                  head_sequence, cycle)

if params.save:
    save_placement_route_figure(params.filename, pcb_data, component_result, cycle_result, feeder_slot_result,
                                placement_result, head_sequence)

# 估算贴装用时
placement_time_estimate(component_data, pcb_data, component_result, cycle_result, feeder_slot_result, placement_result,
                        head_sequence)

