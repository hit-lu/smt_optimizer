import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from dataloader import *
from common_function import *
from optimizer_celldivision import *


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

    min_dist = [[np.inf for i in range(num_pos)] for s in range(1 << num_pos)]
    min_path = [[[] for i in range(num_pos)] for s in range(1 << num_pos)]

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

@timer_warper
def genetic_based_cluster_route_generation(pcb_data, component_data, component_result, cycle_result):
    num_points = len(pcb_data)
    pop_size = 50

    head_counter = [67, 67, 67, 67, 66, 66]
    head_accumulate = []
    for idx, _ in enumerate(head_counter):
        if idx == 0:
            head_accumulate.append(head_counter[idx])
        else:
            head_accumulate.append(head_accumulate[-1] + head_counter[idx])

    def pop_convert(mount_index):
        mount_pos, head_index = [], 0
        for i, pcb_idx in enumerate(mount_index):
            for head in range(max_head_index):
                if i == head_accumulate[head]:
                    head_index = head + 1
                    break
            mount_pos.append(
                [pcb_data.loc[pcb_idx]['x'] - head_index * head_interval, pcb_data.loc[pcb_idx]['y'], head_index])

        cluster_pos, random_index, cluster_pos_index = [], [], []
        num_cluster = sum(cycle_result)
        for cycle in range(num_cluster):
            while True:
                num = np.random.randint(0, len(pcb_data), dtype=np.int)
                if num not in random_index:
                    break
            random_index.append(num)
            cluster_pos_index.append([num])
            cluster_pos.append([mount_pos[num][0], mount_pos[num][1]])  # 确定若干个中心位置

        cluster_pos = np.array(cluster_pos)
        center_pos = copy.deepcopy(cluster_pos)

        # clustering for pcb data
        cluster_counter, cluster_head = [1 for _ in range(num_cluster)], [[mount_pos[index[0]][2]] for index in
                                                                          cluster_pos_index]
        while True:
            for index, pos in enumerate(mount_pos):
                if index in random_index:
                    continue

                distance = [pow(center[0] - pos[0], 2) + pow(center[1] - pos[1], 2) for center in center_pos]
                while True:
                    cluster_index = np.argmin(distance)
                    if cluster_counter[cluster_index] >= max_head_index or mount_pos[index][2] in cluster_head[
                        cluster_index]:
                        distance[cluster_index] = np.inf
                    else:
                        break

                cluster_counter[cluster_index] += 1
                cluster_head[cluster_index].append(mount_pos[index][2])
                # === 递推更新中心位置参数 ===
                center_pos[cluster_index][0] = (1 - 1 / cluster_counter[cluster_index]) * center_pos[cluster_index][
                    0] + pos[0] / cluster_counter[cluster_index]
                center_pos[cluster_index][1] = (1 - 1 / cluster_counter[cluster_index]) * center_pos[cluster_index][
                    1] + pos[1] / cluster_counter[cluster_index]

                if cluster_pos_index[cluster_index] is None:
                    cluster_pos_index[cluster_index] = [index]
                else:
                    cluster_pos_index[cluster_index].append(index)

            if np.abs(np.sum(center_pos - cluster_pos)) < 1e-9:  # 临时决定的值，后期完善聚类标准后再修改
                break
            else:
                cluster_pos = copy.deepcopy(center_pos)
                cluster_counter = [0 for _ in range(num_cluster)]
                random_index, cluster_pos_index = [], [None] * num_cluster
                cluster_head = [[] for _ in range(num_cluster)]

        pop_placement_result, pop_head_sequence_result = [], []
        for cycle_pos_index in cluster_pos_index:
            cycle_placement = [-1 for _ in range(max_head_index)]
            for index in cycle_pos_index:
                cycle_placement[mount_pos[index][2]] = index
            head_sequence = dynamic_programming_cycle_path(pcb_data, cycle_placement)

            pop_placement_result.append(cycle_placement)
            pop_head_sequence_result.append(head_sequence)
        return pop_placement_result, pop_head_sequence_result

    def cal_pop_val(mount_index):
        distance = 0
        pop_placement_result, pop_head_sequence_result = pop_convert(mount_index)
        for cycle in range(len(pop_placement_result)):
            cycle_placement = pop_placement_result[cycle]
            for i in range(1, len(pop_head_sequence_result[cycle])):
                hd1, hd2 = pop_head_sequence_result[cycle][i - 1], pop_head_sequence_result[cycle][i]
                interval = (hd1 - hd2) * head_interval

                delta_x = abs(
                    pcb_data.loc[cycle_placement[hd1]]['x'] - pcb_data.loc[cycle_placement[hd2]]['x'] - interval)
                delta_y = abs(pcb_data.loc[cycle_placement[hd1]]['y'] - pcb_data.loc[cycle_placement[hd2]]['y'])
                distance += max(delta_x, delta_y)

        return 1 / distance

    crossover_rate, mutation_rate = .8, .2
    iteration_count = 5
    pops, pop_val = [], [0] * pop_size
    for _ in range(pop_size):
        pops.append(np.random.permutation(num_points))

    best_pop_val, best_pop = 0, []
    iter_ = 0
    while True:
        for pop_counter in range(pop_size):
            pop_val[pop_counter] = cal_pop_val(pops[pop_counter])
        iter_ += 1

        if iter_ > iteration_count:
            break

        max_idx = np.argmax(pop_val)
        if pop_val[max_idx] > best_pop_val:
            best_pop_val = copy.deepcopy(pop_val[max_idx])
            best_pop = copy.deepcopy(pops[max_idx])
            # placement_result, head_sequence_result = copy.deepcopy(pop_convert(pops[max_idx]))

        print('------------- current iter :   ' + str(iter_) + ' , total iter :   ' + str(
            iteration_count) + ' , current pop value : ' + str(best_pop_val) + '   -------------')

        # 选择
        new_pops = []
        top_k_index = get_top_k_value(pop_val, int(pop_size * 0.3))
        for index in top_k_index:
            new_pops.append(pops[index])
        index = [i for i in range(pop_size)]
        select_index = random.choices(index, weights = pop_val, k = int(pop_size * 0.7))
        for index in select_index:
            new_pops.append(pops[index])
        pops = new_pops

        # 交叉与交换
        for pop in range(pop_size):
            if pop % 2 == 0 and np.random.random() < crossover_rate:
                index1, index2 = selection(pop_val), -1
                while True:
                    index2 = selection(pop_val)
                    if index1 != index2:
                        break

                # 两点交叉算子
                pops[index1], pops[index2] = crossover(pops[index1], pops[index2])


            if np.random.random() < mutation_rate:
                index_ = selection(pop_val)
                swap_mutation(pops[index_])

    return pop_convert(best_pop)


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

            assigned_placement = [-1 for _ in range(max_head_index)]
            # 最近邻确定
            way_point = None
            head_range = range(max_head_index - 1, -1, -1) if search_dir else range(max_head_index)
            for head in head_range:
                if component_result[cycle_set][head] == -1:
                    continue

                component_index = component_result[cycle_set][head]
                if way_point is None:
                    index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                    assigned_placement[head] = mount_point_index[component_index][index]
                    way_point = mount_point_pos[component_index][index]  # 记录路标点

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
