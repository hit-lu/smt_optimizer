import copy
import time
import math
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps

# 机器参数
max_slot_index = 120
max_head_index = 6
interval_ratio = 2
slot_interval = 15
head_interval = slot_interval * interval_ratio
head_nozzle = ['' for _ in range(max_head_index)]    # 头上已经分配吸嘴

# 位置信息
slotf1_pos, slotr1_pos = [-31.267, 44.], [807., 810.545]   # F1(前基座最左侧)、R1(后基座最右侧)位置
fix_camera_pos = [269.531, 694.823]             # 固定相机位置
anc_marker_pos = [336.457, 626.230]             # ANC基准点位置
stopper_pos = [620., 200.]        # 止档块位置

# 算法权重参数
factor_nozzle_change, factor_simultaneous_pick = 0.8, 1. / max_head_index


def find_commonpart(head_group, feeder_group):
    feeder_group_len = len(feeder_group)

    max_length, max_common_part = -1, []
    for offset in range(-max_head_index + 1, feeder_group_len - 1):
        # offset: head_group相对于feeder_group的偏移量
        length, common_part = 0, []
        for hd_index in range(max_head_index):
            fd_index = hd_index + offset
            if fd_index < 0 or fd_index >= feeder_group_len:
                common_part.append(-1)
                continue

            if head_group[hd_index] == feeder_group[fd_index] and head_group[hd_index] != -1:
                length += 1
                common_part.append(head_group[hd_index])
            else:
                common_part.append(-1)
        if length > max_length:
            max_length = length
            max_common_part = common_part

    return max_common_part


def timer_wrapper(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)

        print("function {} running time :  {} s".format(func.__name__, time.time() - start_time))
        return result
    return measure_time


def feeder_assignment(component_data, pcb_data, component_result, cycle_result, feeder_limit):
    # Section: 供料器分配结果
    feeder_slot_result, feeder_group_result = [], []
    for component_group in component_result:
        new_feeder_group = []
        for component in component_group:
            if component == -1 or feeder_limit[component] == 0 or component in new_feeder_group:
                new_feeder_group.append(-1)
            else:
                new_feeder_group.append(component)

        if len(new_feeder_group) == 0:
            continue

        while sum(i >= 0 for i in new_feeder_group) != 0:
            max_common_part, index = [], -1
            max_common_length = -1
            for feeder_index in range(len(feeder_group_result)):
                common_part = find_commonpart(new_feeder_group, feeder_group_result[feeder_index])
                if sum(i > 0 for i in max_common_part) > max_common_length:
                    max_common_length = sum(i > 0 for i in max_common_part)
                    max_common_part, index = common_part, feeder_index

            new_feeder_length = 0
            for feeder in new_feeder_group:
                if feeder != -1 and feeder_limit[feeder] > 0:
                    new_feeder_length += 1

            feeder_group_result.append([])
            if new_feeder_length > max_common_length:
                # 新分配供料器
                for feeder_index in range(len(new_feeder_group)):
                    feeder = new_feeder_group[feeder_index]
                    if feeder != -1 and feeder_limit[feeder] > 0:
                        feeder_group_result[-1].append(feeder)
                        new_feeder_group[feeder_index] = -1
                        feeder_limit[feeder] -= 1
                    else:
                        feeder_group_result[-1].append(-1)
            else:
                # 使用旧供料器
                for feeder_index in range(len(max_common_part)):
                    feeder = max_common_part[feeder_index]
                    if feeder != -1:
                        feeder_group_result[-1].append(feeder)
                        new_feeder_group[feeder_index] = -1
                        feeder_limit[feeder] -= 1
                    else:
                        feeder_group_result[-1].append(-1)

    # 去除多余的元素
    for feeder_group in feeder_group_result:
        while len(feeder_group) > 0 and feeder_group[0] == -1:
            feeder_group.pop(0)

        while len(feeder_group) > 0 and feeder_group[-1] == -1:
            feeder_group.pop(-1)

    # 确定供料器组的安装位置
    point_num = len(pcb_data)
    component_pos = [[] for _ in range(len(component_data))]
    for point_cnt in range(point_num):
        part = pcb_data.loc[point_cnt, 'part']
        index = np.where(component_data['part'].values == part)[0]
        component_pos[index[0]].append(pcb_data.loc[point_cnt, 'x'] + stopper_pos[0])

    # 供料器组分配的优先顺序
    feeder_assign_sequence = []
    for i in range(len(feeder_group_result)):
        for j in range(len(feeder_group_result)):
            if j in feeder_assign_sequence:
                continue

            if len(feeder_assign_sequence) == i:
                feeder_assign_sequence.append(j)
            else:
                seq = feeder_assign_sequence[-1]
                if cycle_result[seq] * len([k for k in feeder_group_result[seq] if k >= 0]) < cycle_result[j] * len(
                        [k for k in feeder_group_result[seq] if k >= 0]):
                    feeder_assign_sequence.pop(-1)
                    feeder_assign_sequence.append(j)

    # TODO: 暂未考虑机械限位
    feeder_group_slot = [-1] * len(feeder_group_result)
    feeder_lane_state = [0] * max_slot_index        # 0表示空，1表示已占有
    for index in feeder_assign_sequence:
        feeder_group = feeder_group_result[index]
        best_slot = []
        for cp_index, component in enumerate(feeder_group):
            if component == -1:
                continue
            best_slot.append(round((sum(component_pos[component]) / len(component_pos[component]) - slotf1_pos[
                0]) / slot_interval) + 1 - cp_index * interval_ratio)
        best_slot = round(sum(best_slot) / len(best_slot))

        dir, step = 0, 0        # dir: 1-向右, 0-向左
        prev_assign_available = True
        while True:
            assign_slot = best_slot + step if dir else best_slot - step
            if assign_slot + (len(feeder_group) - 1) * interval_ratio >= max_slot_index / 2 or assign_slot < 0:
                if not prev_assign_available:
                    raise Exception('feeder assign error!')
                prev_assign_available = False
                dir = 1 - dir
                if dir == 0:
                    step += 1
                continue

            prev_assign_available = True
            assign_available = True

            # 分配对应槽位
            slot = 0
            for slot in range(assign_slot, assign_slot + interval_ratio * len(feeder_group), interval_ratio):
                feeder_index = int((slot - assign_slot) / interval_ratio)
                if feeder_lane_state[slot] == 1 and feeder_group[feeder_index]:
                    assign_available = False
                    break

            if assign_available:
                for idx, part in enumerate(feeder_group):
                    if part != 1:
                        feeder_lane_state[slot + idx * interval_ratio] = 1
                feeder_group_slot[index] = slot
                break

            dir = 1 - dir
            if dir == 0:
                step += 1

    # 按照最大匹配原则，确定各元件周期拾取槽位
    for component_group in component_result:
        feeder_slot_result.append([-1] * max_head_index)
        head_index = [head for head, component in enumerate(component_group) if component >= 0]
        while head_index:
            max_overlap_counter = 0
            overlap_feeder_group_index, overlap_feeder_group_offset = -1, -1
            for feeder_group_idx, feeder_group in enumerate(feeder_group_result):
                # offset 头1 相对于 供料器组第一个元件的偏移量
                for offset in range(-max_head_index + 1, max_head_index + len(feeder_group)):
                    overlap_counter = 0
                    for head in head_index:
                        if 0 <= head + offset < len(feeder_group) and component_group[head] == \
                                feeder_group[head + offset]:
                            overlap_counter += 1

                    if overlap_counter > max_overlap_counter:
                        max_overlap_counter = overlap_counter
                        overlap_feeder_group_index, overlap_feeder_group_offset = feeder_group_idx, offset

            feeder_group = feeder_group_result[overlap_feeder_group_index]
            head_index_cpy = copy.deepcopy(head_index)
            # TODO: 关于供料器槽位位置分配的方法不正确
            for head in head_index_cpy:
                if 0 <= head + overlap_feeder_group_offset < len(feeder_group) and component_group[head] == \
                        feeder_group[head + overlap_feeder_group_offset]:
                    feeder_slot_result[-1][head] = feeder_group_slot[overlap_feeder_group_index] + interval_ratio * head
                    head_index.remove(head)

    return feeder_slot_result


def dynamic_programming_cycle_path(pcb_data, cycle_placement):
    head_sequence = []
    num_pos = sum([placement != -1 for placement in cycle_placement]) + 1

    pos, head_set = [], []
    average_pos_x, counter = 0, 1
    for head, placement in enumerate(cycle_placement):
        if placement == -1:
            continue
        head_set.append(head)
        pos.append([pcb_data.loc[placement]['x'] - head * head_interval + stopper_pos[0],
                    pcb_data.loc[placement]['y'] + stopper_pos[1]])
        average_pos_x = average_pos_x + (pos[-1][0] - average_pos_x) / counter

        counter += 1

    pos.insert(0, [average_pos_x, slotf1_pos[1]])

    def get_distance(pos_1, pos_2):
        return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

    # 各节点之间的距离
    dist = [[get_distance(pos_1, pos_2) for pos_2 in pos] for pos_1 in pos]

    min_dist = [[np.inf for _ in range(num_pos)] for s in range(1 << num_pos)]
    min_path = [[[] for _ in range(num_pos)] for s in range(1 << num_pos)]

    # 状压dp搜索
    for s in range(1, 1 << num_pos, 2):
        # 考虑节点集合s必须包括节点0
        if not (s & 1):
            continue
        for j in range(1, num_pos):
            # 终点i需在当前考虑节点集合s内
            if not (s & (1 << j)):
                continue
            if s == int((1 << j) | 1):
                # 若考虑节点集合s仅含节点0和节点j，dp边界，赋予初值
                # print('j:', j)
                min_path[s][j] = [j]
                min_dist[s][j] = dist[0][j]

            # 枚举下一个节点i，更新
            for i in range(1, num_pos):
                # 下一个节点i需在考虑节点集合s外
                if s & (1 << i):
                    continue
                if min_dist[s][j] + dist[j][i] < min_dist[s | (1 << i)][i]:
                    min_path[s | (1 << i)][i] = min_path[s][j] + [i]
                    min_dist[s | (1 << i)][i] = min_dist[s][j] + dist[j][i]

    ans_dist = np.inf
    ans_path = []
    # 求最终最短哈密顿回路
    for i in range(1, num_pos):
        if min_dist[(1 << num_pos) - 1][i] + dist[i][0] < ans_dist:
            # 更新，回路化
            ans_path = min_path[s][i]
            ans_dist = min_dist[(1 << num_pos) - 1][i] + dist[i][0]

    for element in ans_path:
        head_sequence.append(head_set[element - 1])

    return head_sequence


def greedy_placement_route_generation(component_data, pcb_data, component_result, cycle_result):
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
                    index = 0
                    if way_point is None:
                        index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                    else:
                        for next_head in head_range:
                            component_index = component_result[cycle_set][next_head]
                            if assigned_placement[next_head] == -1 and component_index != -1:
                                num_points = len(mount_point_pos[component_index])
                                index = np.argmin([abs(mount_point_pos[component_index][i][0] - way_point[0]) * .1 + abs(
                                    mount_point_pos[component_index][i][1] - way_point[1]) for i in
                                                   range(num_points)])
                                head = next_head
                                break
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
                        if assigned_placement[next_head] != -1 or component_result[cycle_set][next_head] == -1:
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
