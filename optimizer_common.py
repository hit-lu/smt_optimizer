import copy
import time
import math
from tqdm import tqdm

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from collections import defaultdict

# 机器参数
max_slot_index = 120
max_head_index = 6
interval_ratio = 2
slot_interval = 15
head_interval = slot_interval * interval_ratio
head_nozzle = ['' for _ in range(max_head_index)]  # 头上已经分配吸嘴

# 位置信息
slotf1_pos, slotr1_pos = [-31.267, 44.], [807., 810.545]  # F1(前基座最左侧)、R1(后基座最右侧)位置
fix_camera_pos = [269.531, 694.823]  # 固定相机位置
anc_marker_pos = [336.457, 626.230]  # ANC基准点位置
stopper_pos = [620., 200.]  # 止档块位置

# 算法权重参数
e_nz_change, e_gang_pick = 3, 0.6

# 电机参数
head_rotary_velocity = 8e-5  # 贴装头R轴旋转时间
x_max_velocity, y_max_velocity = 1.4, 1.2
x_max_acceleration, y_max_acceleration = x_max_velocity / 0.079, y_max_velocity / 0.079


def axis_moving_time(distance, axis=0):
    distance = abs(distance) * 1e-3
    Lamax = x_max_velocity ** 2 / x_max_acceleration if axis == 0 else y_max_velocity ** 2 / y_max_acceleration
    Tmax = x_max_velocity / x_max_acceleration if axis == 0 else y_max_velocity / y_max_acceleration
    if axis == 0:
        return 2 * math.sqrt(distance / x_max_acceleration) if distance < Lamax else 2 * Tmax + (
                    distance - Lamax) / x_max_velocity
    else:
        return 2 * math.sqrt(distance / y_max_acceleration) if distance < Lamax else 2 * Tmax + (
                    distance - Lamax) / y_max_velocity


def head_rotary_time(angle):
    while -180 > angle > 180:
        if angle > 180:
            angle -= 360
        else:
            angle += 360
    return abs(angle) * head_rotary_velocity


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


def feeder_assignment(component_data, pcb_data, component_result, cycle_result):
    # Section: 供料器分配结果
    feeder_slot_result, feeder_group_result = [], []
    feeder_limit = defaultdict(int)
    for component in range(len(component_data)):
        feeder_limit[component] = component_data.loc[component]['feeder-limit']

    for component_cycle in component_result:
        new_feeder_group = []
        for component in component_cycle:
            if component == -1 or feeder_limit[component] == 0:
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
                if sum(i > 0 for i in common_part) > max_common_length:
                    max_common_length = sum(i > 0 for i in common_part)
                    max_common_part, index = common_part, feeder_index

            new_feeder_length = 0
            for feeder in new_feeder_group:
                if feeder != -1 and feeder_limit[feeder] > 0:
                    new_feeder_length += 1

            if new_feeder_length > max_common_length:
                # 新分配供料器
                feeder_group_result.append([])
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
                for feeder_index, feeder_part in enumerate(max_common_part):
                    if feeder_part != -1:
                        new_feeder_group[feeder_index] = -1

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
    feeder_lane_state = [0] * max_slot_index  # 0表示空，1表示已占有
    for index in feeder_assign_sequence:
        feeder_group = feeder_group_result[index]
        best_slot = []
        for cp_index, component in enumerate(feeder_group):
            if component == -1:
                continue
            best_slot.append(round((sum(component_pos[component]) / len(component_pos[component]) - slotf1_pos[
                0]) / slot_interval) + 1 - cp_index * interval_ratio)
        best_slot = round(sum(best_slot) / len(best_slot))

        search_dir, step = 0, 0  # dir: 1-向右, 0-向左
        left_out_range, right_out_range = False, False
        while True:
            assign_slot = best_slot + step if search_dir else best_slot - step
            # 出现越界，反向搜索
            if assign_slot + (len(feeder_group) - 1) * interval_ratio >= max_slot_index / 2:
                right_out_range = True
                search_dir = 0
                step += 1
            elif assign_slot < 0:
                left_out_range = True
                search_dir = 1
                step += 1
            else:
                if left_out_range or right_out_range:
                    step += 1       # 单向搜索
                else:
                    search_dir = 1 - search_dir     # 双向搜索
                    if search_dir == 0:
                        step += 1

            assign_available = True

            # === 分配对应槽位 ===
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

        if feeder_group_slot[index] == -1:
            raise Exception('feeder assign error!')


    # 按照最大匹配原则，确定各元件周期拾取槽位
    for component_cycle in component_result:
        feeder_slot_result.append([-1] * max_head_index)
        head_index = [head for head, component in enumerate(component_cycle) if component >= 0]
        while head_index:
            max_overlap_counter = 0
            overlap_feeder_group_index, overlap_feeder_group_offset = -1, -1
            for feeder_group_idx, feeder_group in enumerate(feeder_group_result):
                # offset 头1 相对于 供料器组第一个元件的偏移量
                for offset in range(-max_head_index + 1, max_head_index + len(feeder_group)):
                    overlap_counter = 0
                    for head in head_index:
                        if 0 <= head + offset < len(feeder_group) and component_cycle[head] == \
                                feeder_group[head + offset]:
                            overlap_counter += 1

                    if overlap_counter > max_overlap_counter:
                        max_overlap_counter = overlap_counter
                        overlap_feeder_group_index, overlap_feeder_group_offset = feeder_group_idx, offset

            feeder_group = feeder_group_result[overlap_feeder_group_index]
            head_index_cpy = copy.deepcopy(head_index)

            for head in head_index_cpy:
                if 0 <= head + overlap_feeder_group_offset < len(feeder_group) and component_cycle[head] == \
                        feeder_group[head + overlap_feeder_group_offset]:
                    feeder_slot_result[-1][head] = feeder_group_slot[overlap_feeder_group_index] + interval_ratio * head
                    head_index.remove(head)

    return feeder_slot_result


def dynamic_programming_cycle_path(pcb_data, cycle_placement, assigned_feeder):
    head_sequence = []
    num_pos = sum([placement != -1 for placement in cycle_placement]) + 1

    pos, head_set = [], []
    feeder_set = set()
    for head, feeder in enumerate(assigned_feeder):
        if feeder == -1:
            continue

        head_set.append(head)
        placement = cycle_placement[head]
        if feeder != -1 and placement == -1:
            print(assigned_feeder)
            print(cycle_placement)

        pos.append([pcb_data.loc[placement]['x'] - head * head_interval + stopper_pos[0],
                    pcb_data.loc[placement]['y'] + stopper_pos[1]])

        feeder_set.add(feeder - head * interval_ratio)

    pos.insert(0, [slotf1_pos[0] + ((min(list(feeder_set)) + max(list(feeder_set))) / 2 - 1) * slot_interval,
                   slotf1_pos[1]])

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
            # 终点j需在当前考虑节点集合s内
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

    ans_dist = float('inf')
    ans_path = []
    # 求最终最短哈密顿回路
    for i in range(1, num_pos):
        if min_dist[(1 << num_pos) - 1][i] + dist[i][0] < ans_dist:
            # 更新，回路化
            ans_path = min_path[s][i]
            ans_dist = min_dist[(1 << num_pos) - 1][i] + dist[i][0]

    for element in ans_path:
        head_sequence.append(head_set[element - 1])

    start_head, end_head = head_sequence[0], head_sequence[-1]
    if pcb_data.loc[cycle_placement[start_head]]['x'] - start_head * head_interval > \
            pcb_data.loc[cycle_placement[end_head]]['x'] - end_head * head_interval:
        head_sequence = list(reversed(head_sequence))
    return head_sequence


@timer_wrapper
def greedy_placement_route_generation(component_data, pcb_data, component_result, cycle_result, feeder_slot_result):
    placement_result, head_sequence_result = [], []
    mount_point_index = [[] for _ in range(len(component_data))]
    mount_point_pos = [[] for _ in range(len(component_data))]

    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index[component_index].append(i)
        mount_point_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

    search_dir = 0  # 0：自左向右搜索  1：自右向左搜索
    for cycle_set in range(len(component_result)):
        floor_cycle, ceil_cycle = sum(cycle_result[:cycle_set]), sum(cycle_result[:(cycle_set + 1)])
        for cycle in range(floor_cycle, ceil_cycle):
            search_dir = 1 - search_dir
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
                        if search_dir:
                            index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                        else:
                            index = np.argmin(mount_point_pos[component_index], axis=0)[0]
                    else:
                        for next_head in head_range:
                            component_index = component_result[cycle_set][next_head]
                            if assigned_placement[next_head] == -1 and component_index != -1:
                                num_points = len(mount_point_pos[component_index])
                                index = np.argmin(
                                    [abs(mount_point_pos[component_index][i][0] - way_point[0]) * .1 + abs(
                                        mount_point_pos[component_index][i][1] - way_point[1]) for i in
                                     range(num_points)])
                                head = next_head
                                break
                    # index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                    assigned_placement[head] = mount_point_index[component_index][index]

                    # 记录路标点
                    way_point = mount_point_pos[component_index][index]
                    way_point[0] += (max_head_index - head - 1) * head_interval if search_dir else -head * head_interval

                    mount_point_index[component_index].pop(index)
                    mount_point_pos[component_index].pop(index)
                else:
                    head_index, point_index = -1, -1
                    min_cheby_distance, min_euler_distance = np.inf, np.inf
                    for next_head in range(max_head_index):
                        if assigned_placement[next_head] != -1 or component_result[cycle_set][next_head] == -1:
                            continue
                        next_comp_index = component_result[cycle_set][next_head]
                        for counter in range(len(mount_point_pos[next_comp_index])):
                            if search_dir:
                                delta_x = abs(mount_point_pos[next_comp_index][counter][0] - way_point[0]
                                              + (max_head_index - next_head - 1) * head_interval)
                            else:
                                delta_x = abs(mount_point_pos[next_comp_index][counter][0] - way_point[0]
                                              - next_head * head_interval)

                            delta_y = abs(mount_point_pos[next_comp_index][counter][1] - way_point[1])

                            euler_distance = pow(axis_moving_time(delta_x, 0), 2) + pow(axis_moving_time(delta_y, 1), 2)
                            cheby_distance = max(axis_moving_time(delta_x, 0),
                                                 axis_moving_time(delta_y, 1)) + 5e-2 * euler_distance
                            if cheby_distance < min_cheby_distance or (abs(cheby_distance - min_cheby_distance) < 1e-9
                                                                       and euler_distance < min_euler_distance):
                            # if euler_distance < min_euler_distance:
                                min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                head_index, point_index = next_head, counter

                    component_index = component_result[cycle_set][head_index]
                    assert (0 <= head_index < max_head_index)

                    assigned_placement[head_index] = mount_point_index[component_index][point_index]
                    way_point = mount_point_pos[component_index][point_index]
                    way_point[0] += (max_head_index - head_index - 1) * head_interval if search_dir \
                        else -head_index * head_interval

                    mount_point_index[component_index].pop(point_index)
                    mount_point_pos[component_index].pop(point_index)

            placement_result.append(assigned_placement)  # 各个头上贴装的元件类型
            head_sequence_result.append(
                dynamic_programming_cycle_path(pcb_data, assigned_placement, feeder_slot_result[cycle_set]))

    return placement_result, head_sequence_result


@timer_wrapper
def beam_search_for_route_generation(component_data, pcb_data, component_result, cycle_result, feeder_slot_result):
    beam_width = 4    # 集束宽度
    base_points = [float('inf'), float('inf')]

    mount_point_index = [[] for _ in range(len(component_data))]
    mount_point_pos = [[] for _ in range(len(component_data))]

    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]

        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index[component_index].append(i)
        mount_point_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

        # 记录最左下角坐标
        if mount_point_pos[component_index][-1][0] < base_points[0]:
            base_points[0] = mount_point_pos[component_index][-1][0]
        if mount_point_pos[component_index][-1][1] < base_points[1]:
            base_points[1] = mount_point_pos[component_index][-1][1]

    beam_placement_sequence, beam_head_sequence = [], []
    beam_mount_point_index, beam_mount_point_pos = [], []

    for beam_counter in range(beam_width):
        beam_mount_point_index.append(copy.deepcopy(mount_point_index))
        beam_mount_point_pos.append(copy.deepcopy(mount_point_pos))

        beam_placement_sequence.append([])
        beam_head_sequence.append([])

    beam_distance = [0 for _ in range(beam_width)]  # 记录当前集束搜索点的点数
    def argpartition(list, kth):
        if kth < len(list):
            return np.argpartition(list, kth)
        else:
            index, indexes = 0, []
            while len(indexes) < kth:
                indexes.append(index)
                index += 1
                if index >= len(list):
                    index = 0
            return np.array(indexes)

    with tqdm(total=100) as pbar:
        search_dir = 0
        pbar.set_description('route schedule')
        for cycle_set in range(len(component_result)):
            floor_cycle, ceil_cycle = sum(cycle_result[:cycle_set]), sum(cycle_result[:(cycle_set + 1)])

            for cycle in range(floor_cycle, ceil_cycle):
                # search_dir = 1 - search_dir
                beam_way_point = None
                for beam_counter in range(beam_width):
                    beam_placement_sequence[beam_counter].append([-1 for _ in range(max_head_index)])

                head_range = range(max_head_index - 1, -1, -1) if search_dir else range(max_head_index)
                for head in head_range:
                    component_index = component_result[cycle_set][head]
                    if component_index == -1:
                        continue

                    if beam_way_point is None:
                        # 首个贴装点的选取，距离基准点最近的beam_width个点
                        beam_way_point = [[0, 0]] * beam_width

                        for beam_counter in range(beam_width):
                            num_points = len(beam_mount_point_index[beam_counter][component_index])
                            index = 0
                            for i in range(1, num_points):
                                posA = beam_mount_point_pos[beam_counter][component_index][i]
                                posB = beam_mount_point_pos[beam_counter][component_index][index]
                                if search_dir == 1:
                                    if (posA[1] < posB[1] and abs(posA[1] - posB[1]) > 1) or (
                                            abs(posA[1] - posB[1]) < 1 and posA[0] > posB[0]):
                                        index = i
                                else:
                                    if (posA[1] < posB[1] and abs(posA[1] - posB[1]) > 1) or (
                                            abs(posA[1] - posB[1]) < 1 and posA[0] < posB[0]):
                                        index = i

                            beam_placement_sequence[beam_counter][-1][head] = beam_mount_point_index[beam_counter][component_index][index]

                            beam_way_point[beam_counter] = beam_mount_point_pos[beam_counter][component_index][index]
                            beam_way_point[beam_counter][0] += (max_head_index - head - 1) * head_interval if \
                                search_dir else -head * head_interval

                            beam_mount_point_index[beam_counter][component_index].pop(index)
                            beam_mount_point_pos[beam_counter][component_index].pop(index)
                    else:
                        # 后续贴装点
                        search_beam_distance = []
                        search_beam_index = [0] * (beam_width ** 2)
                        for beam_counter in range(beam_width ** 2):
                            search_beam_distance.append(beam_distance[beam_counter // beam_width])

                        for beam_counter in range(beam_width):
                            # 对于集束beam_counter + 1最近的beam_width个点
                            num_points = len(beam_mount_point_pos[beam_counter][component_index])

                            dist = []
                            for i in range(num_points):
                                if search_dir:
                                    delta_x = axis_moving_time(
                                        beam_mount_point_pos[beam_counter][component_index][i][0] -
                                        beam_way_point[beam_counter][0] + (max_head_index - head - 1) * head_interval,
                                        0)
                                else:
                                    delta_x = axis_moving_time(
                                        beam_mount_point_pos[beam_counter][component_index][i][0] -
                                        beam_way_point[beam_counter][0] - head * head_interval, 0)

                                delta_y = axis_moving_time(beam_mount_point_pos[beam_counter][component_index][i][1] -
                                                           beam_way_point[beam_counter][1], 1)

                                dist.append(max(delta_x, delta_y))
                                if delta_y > 0.02:
                                    dist[-1] += 10 * delta_y
                            indexes = argpartition(dist, kth=beam_width)[:beam_width]

                            # 记录中间信息
                            for i, index in enumerate(indexes):
                                search_beam_distance[i + beam_counter * beam_width] += dist[index]
                                search_beam_index[i + beam_counter * beam_width] = index

                        indexes = np.argsort(search_beam_distance)

                        beam_mount_point_pos_cpy = copy.deepcopy(beam_mount_point_pos)
                        beam_mount_point_index_cpy = copy.deepcopy(beam_mount_point_index)

                        beam_placement_sequence_cpy = copy.deepcopy(beam_placement_sequence)
                        beam_head_sequence_cpy = copy.deepcopy(beam_head_sequence)
                        beam_counter = 0
                        assigned_placement = []
                        for i, index in enumerate(indexes):
                            # 拷贝原始集束数据
                            beam_mount_point_pos[beam_counter] = copy.deepcopy(beam_mount_point_pos_cpy[index // beam_width])
                            beam_mount_point_index[beam_counter] = copy.deepcopy(beam_mount_point_index_cpy[index // beam_width])
                            beam_placement_sequence[beam_counter] = copy.deepcopy(beam_placement_sequence_cpy[index // beam_width])
                            beam_head_sequence[beam_counter] = copy.deepcopy(beam_head_sequence_cpy[index // beam_width])

                            # 更新各集束最新扫描的的贴装点
                            component_index = component_result[cycle_set][head]

                            beam_placement_sequence[beam_counter][-1][head] = \
                                beam_mount_point_index[beam_counter][component_index][search_beam_index[index]]

                            if beam_placement_sequence[beam_counter][
                                -1] in assigned_placement and beam_width - beam_counter < len(indexes) - i:
                                continue

                            assigned_placement.append(beam_placement_sequence[beam_counter][-1])

                            # 更新参考基准点
                            beam_way_point[beam_counter] = beam_mount_point_pos[beam_counter][component_index][search_beam_index[index]]
                            beam_way_point[beam_counter][0] += (max_head_index - head - 1) * head_interval if \
                                search_dir else -head * head_interval

                            # 更新各集束贴装路径长度，移除各集束已分配的贴装点
                            beam_distance[beam_counter] = search_beam_distance[index]

                            beam_mount_point_pos[beam_counter][component_index].pop(search_beam_index[index])
                            beam_mount_point_index[beam_counter][component_index].pop(search_beam_index[index])

                            beam_counter += 1

                            if beam_counter >= beam_width:
                                break
                        assert(beam_counter >= beam_width)

                # 更新头贴装顺序
                for beam_counter in range(beam_width):
                    beam_head_sequence[beam_counter].append(
                        dynamic_programming_cycle_path(pcb_data, beam_placement_sequence[beam_counter][-1],
                                                       feeder_slot_result[cycle_set]))

                pbar.update(1 / sum(cycle_result) * 100)

    index = np.argmin(beam_distance)
    return beam_placement_sequence[index], beam_head_sequence[index]
