import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from dataloader import *


def nozzle_assignment(component_data, pcb_data):
    pass


def greedy_placement_route_generation(pcb_data, component_data, component_result, cycle_result):
    placement_result, head_sequence_result = [], []

    mount_point_index = [[] for _ in range(len(component_data))]
    mount_point_pos = [[] for _ in range(len(component_data))]

    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index[component_index].append(i)
        mount_point_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

    search_dir = 1  # 0：自左向右搜索  1：自右向左搜索
    for cycle_set in range(len(component_result)):
        floor_cycle, ceil_cycle = sum(cycle_result[:cycle_set]), sum(cycle_result[:(cycle_set + 1)])
        for cycle in range(floor_cycle, ceil_cycle):
            max_pos = [max(mount_point_pos[component_index], key=lambda x: x[0]) for component_index in
                       range(len(mount_point_pos)) if len(mount_point_pos[component_index]) > 0][0][0]
            min_pos = [min(mount_point_pos[component_index], key=lambda x: x[0]) for component_index in
                       range(len(mount_point_pos)) if len(mount_point_pos[component_index]) > 0][0][0]
            point2head_range = min(math.floor((max_pos - min_pos) / head_interval) + 1, max_head_index)

            assigned_placement = [-1 for _ in range(max_head_index)]
            # 最近邻确定
            way_point = None
            head_range = range(max_head_index - 1, -1, -1) if search_dir else range(max_head_index)

            for head_counter, head in enumerate(head_range):
                if component_result[cycle_set][head] == -1:
                    continue

                component_index = component_result[cycle_set][head]
                if way_point is None or head_counter % point2head_range == 0:
                    if way_point is None:
                        index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                    else:
                        num_points = len(mount_point_pos[component_index])
                        index = np.argmin([abs(mount_point_pos[component_index][i][0] - way_point[0]) * .1 + abs(
                            mount_point_pos[component_index][i][1] - way_point[1]) for i in
                                           range(num_points)])
                    # index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                    assigned_placement[head] = mount_point_index[component_index][index]

                    # 记录路标点
                    way_point = mount_point_pos[component_index][index]
                    way_point[0] += (max_head_index - head - 1) * head_interval if dir else -head * head_interval

                    mount_point_index[component_index].pop(index)
                    mount_point_pos[component_index].pop(index)
                else:
                    head_index, point_index = -1, -1
                    min_cheby_distance, min_euler_distance = np.inf, np.inf
                    for next_head in range(max_head_index):
                        if assigned_placement[next_head] != -1:
                            continue

                        component_index = component_result[cycle_set][next_head]
                        for counter in range(len(mount_point_pos[component_index])):
                            if dir:
                                delta_x = abs(mount_point_pos[component_index][counter][0] - way_point[0]
                                              + (max_head_index - next_head - 1) * head_interval)
                            else:
                                delta_x = abs(mount_point_pos[component_index][counter][0] - way_point[0]
                                              - next_head * head_interval)

                            delta_y = abs(mount_point_pos[component_index][counter][1] - way_point[1])
                            cheby_distance = max(delta_x, delta_y)
                            euler_distance = pow(delta_x, 2) + pow(delta_y, 2)

                            if cheby_distance < min_cheby_distance or \
                                    (abs(cheby_distance - min_cheby_distance) < 1e-9 and euler_distance < min_euler_distance):
                                min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                head_index, point_index = next_head, counter

                    component_index = component_result[cycle_set][head_index]
                    assigned_placement[head_index] = mount_point_index[component_index][point_index]

                    way_point = mount_point_pos[component_index][point_index]
                    way_point[0] += (max_head_index - head_index - 1) * head_interval if dir else -head_index * head_interval

                    mount_point_index[component_index].pop(point_index)
                    mount_point_pos[component_index].pop(point_index)

            placement_result.append(assigned_placement)  # 各个头上贴装的元件类型
            head_sequence_result.append(dynamic_programming_cycle_path(pcb_data, assigned_placement))

    return placement_result, head_sequence_result
