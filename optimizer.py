import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np

from result_analysis import *
from dataloader import *

from optimizer_celldivision import *
from optimizer_feederpriority import *
from optimizer_hierarchy import *
from optimizer_hybridgenetic import *
from optimizer_aggregation import *
from optimizer_hybridevolutionary import *

from random_generator import *

parser = argparse.ArgumentParser(description='smt optimizer implementation')
parser.add_argument('--filename', default='PCB.txt', type=str, help='load pcb data')
parser.add_argument('--mode', default=1, type=int, help='mode: 0 -directly load pcb data without optimization '
                                                        'for data analysis, 1 -optimize pcb data')
parser.add_argument('--optimize_method', default='feeder_priority', type=str, help='optimizer algorithm')
parser.add_argument('--figure', default=0, type=int, help='plot mount process figure or not')
parser.add_argument('--feeder_limit', default=1, type=int, help='the upper bound of feeder assigned to the slot')
parser.add_argument('--save', default=0, type=int, help='save the optimization result and figure')
params = parser.parse_args()

pcb_data, component_data, feeder_data = load_data(params.filename, load_feeder_data=True)  # 加载PCB数据
component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = [], [], [], [], []

# TODO 1: 一类元件对应多个供料器的情形
# TODO 2: 处理不同宽度喂料器 ×
# TODO 3: 如何处理供料器位置分配的边界条件（占位数量≈可用槽位数） ×
# TODO 4: 如何平衡供料器前基座和后基座之间的分配元件数量（目前仅考虑前基座优化） ×
# TODO 5: 解的质量的提升
# TODO 6: 扩大测试范围，保存中间测试数据
# TODO 7: 算法效率提升，python本身效率慢导致的求解时间长的问题
# TODO 8: 估计时间时考虑吸嘴更换等因素，降低估计时间和实际时间的差距 -
# TODO 9: 实际应用的限制：吸嘴数、供料器数、机械限位等

if params.mode == 0:
    component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = convert_pcbdata_to_result(
        pcb_data, component_data)
else:
    if params.optimize_method == 'cell_division':           # 基于元胞分裂的遗传算法
        component_result, cycle_result, feeder_slot_result = optimizer_celldivision(pcb_data, component_data)
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result)

    elif params.optimize_method == 'hierarchy':             # 分层启发式算法
        # TODO: 吸杆任务分配
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result)

    elif params.optimize_method == 'feeder_priority':       # 基于基座扫描的供料器优先算法
        # 第1步：分配供料器位置
        feeder_allocate(component_data, pcb_data, feeder_data, False)
        # 第2步：扫描供料器基座，确定元件拾取的先后顺序
        component_result, cycle_result, feeder_slot_result = feeder_base_scan(component_data, pcb_data, feeder_data)
        # 第3步：贴装路径规划
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result)

    elif params.optimize_method == 'route_schedule':        # 路径规划测试
        component_result, cycle_result, feeder_slot_result, _, _ = convert_pcbdata_to_result(
            pcb_data, component_data)

        # placement_result, head_sequence = cluster_based_route_generation(component_data, pcb_data, component_result,
        #                                                                  cycle_result, feeder_slot_result)
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result)

    elif params.optimize_method == 'hybrid_genetic':        # 基于拾取组的混合遗传算法
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_hybrid_genetic(
            pcb_data, component_data)

    elif params.optimize_method == 'aggregation':           # 基于batch-level的整数规划 + 启发式算法
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_aggregation(
            component_data, pcb_data)

    elif params.optimize_method == 'hybrid_evolutionary':   # 混合进化算法
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_hybrid_evolutionary(
            component_data, pcb_data)

if params.figure:
    # 绘制各周期从供料器拾取的贴装点示意图
    # pickup_cycle_schematic(feeder_slot_result, cycle_result)

    # 绘制贴装路径图
    for cycle in range(40, len(placement_result)):
        placement_route_schematic(pcb_data, component_result, cycle_result, feeder_slot_result, placement_result,
                                  head_sequence, cycle)

if params.save:
    save_placement_route_figure(params.filename, pcb_data, component_result, cycle_result, feeder_slot_result,
                                placement_result, head_sequence)

# 估算贴装用时
placement_time_estimate(component_data, pcb_data, component_result, cycle_result, feeder_slot_result, placement_result,
                        head_sequence)

