import copy
import math
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dataloader import *
from common_function import *
from optimizer_hierarchy import dynamic_programming_cycle_path
def optimizer_route_generation(pcb_data, component_data, render = True):

    n_cycle = math.ceil(len(pcb_data) / max_head_index)
    n_points = len(pcb_data)

    # cluster_pos = []
    # cluster_counter = [1 for _ in range(max_head_index)]
    # max_cluster_count = [67, 67, 67, 67, 66, 66]  # TODO: 此处先临时指定，暂不考虑上限的影响
    #
    # random_index = []
    # cluster_memindex = []
    # for head in range(max_head_index):
    #     while True:
    #         num = np.random.randint(0, len(pcb_data), dtype=np.int)
    #         if num not in random_index:
    #             break
    #     random_index.append(num)
    #     cluster_memindex.append([num])
    #     cluster_pos.append(pcb_data.loc[num].x)  # 确定若干个中心位置
    #
    # cluster_pos = np.array(cluster_pos)
    # center_pos = copy.deepcopy(cluster_pos)
    #
    # # clustering for pcb data
    # while True:
    #     for index, row in pcb_data.iterrows():
    #         if index in random_index:
    #             continue
    #
    #         feature_differ = abs(row['x'] - center_pos)
    #         while True:
    #             cluster_index = np.argmin(feature_differ)
    #             if cluster_counter[cluster_index] >= max_cluster_count[cluster_index]:
    #                 feature_differ[cluster_index] = np.inf
    #             else:
    #                 break
    #         cluster_counter[cluster_index] += 1
    #
    #         # === 递推更新中心位置参数 ===
    #         center_pos[cluster_index] = (1 - 1 / cluster_counter[cluster_index]) * center_pos[cluster_index] \
    #                                     + row['x'] / cluster_counter[cluster_index]
    #
    #         if cluster_memindex[cluster_index] is None:
    #             cluster_memindex[cluster_index] = [index]
    #         else:
    #             cluster_memindex[cluster_index].append(index)
    #
    #     if np.abs(np.sum(center_pos - cluster_pos)) < 1e-9:  # 临时决定的值，后期完善聚类标准后再修改
    #         break
    #     else:
    #         cluster_pos = copy.deepcopy(center_pos)
    #         cluster_counter = [0 for _ in range(max_head_index)]
    #         random_index, cluster_memindex = [], [None] * max_head_index

    # flag = ['o', '*', 'v', '.', '<', '^']
    # color_ = ['r', 'b', 'k', 'y', 'c', 'm']
    # for i in range(max_head_index):
    #     for index in cluster_memindex[i]:
    #         x, y = pcb_data.iloc[index].x, pcb_data.iloc[index].y
    #         plt.plot(x, y, flag[i], color=color_[i])

    # head_index = np.argsort(cluster_pos)
    # index_2_cluster = {}
    # for head in head_index:
    #     for index in cluster_memindex[head]:
    #         index_2_cluster[index] = head
    #
    #         # pcb_data.iloc[index].x -= head_interval * i  # 坐标转换
    index_2_cluster = {}
    for i in range(n_points):
        index_2_cluster[i] = 1

    # 选择点0作为基准点(供料器取料位置)，点1~len(pcb_data)表示实际贴装点坐标
    arc_group = []
    for i in range(n_points):
        arc_group.append([-1, i, -1])

    save_value = -np.ones((n_points, n_points)) * np.inf
    base_point = [100, 0]        # 基准点

    # 初始化C-W矩阵
    mount_pos, pcb_pos = [], []
    for idx, row in pcb_data.iterrows():
        mount_pos.append([row['x'] - index_2_cluster[idx] * head_interval, row['y']])
        pcb_pos.append([row['x'], row['y']])

    for i in range(n_points):
        for j in range(i + 1, n_points):

            # c1 = pow(mount_pos[i][0] - base_point[0], 2) + pow(mount_pos[i][1] - base_point[1], 2)
            # c2 = pow(mount_pos[j][0] - base_point[0], 2) + pow(mount_pos[j][1] - base_point[1], 2)
            # c3 = pow(mount_pos[i][0] - mount_pos[j][0], 2) + pow(mount_pos[i][1] - mount_pos[j][1], 2)
            #
            # save_value[i][j] = np.sqrt(c1) + np.sqrt(c2) - np.sqrt(c3)

            c1 = max(abs(mount_pos[i][0] - base_point[0]), abs(mount_pos[i][1] - base_point[1]))
            c2 = max(abs(mount_pos[j][0] - base_point[0]), abs(mount_pos[j][1] - base_point[1]))
            c3 = max(abs(mount_pos[i][0] - mount_pos[j][0]), abs(mount_pos[i][1] - mount_pos[j][1]))

            save_value[i][j] = c1 + c2 - c3

            save_value[j][i] = save_value[i][j]

    # C-W 方法主循环
    while len(arc_group) > n_cycle:
        print(len(arc_group))

        min_index = np.argmax(save_value)
        row, col = min_index // n_points, min_index % n_points

        save_value[row][col] = -np.inf

        group_index1 = 0
        while row not in arc_group[group_index1]:
            group_index1 += 1

        index1 = arc_group[group_index1].index(row)

        if index1 != 1 and index1 != len(arc_group[group_index1]) - 2:
            continue

        group_index2 = 0
        while col not in arc_group[group_index2]:
            group_index2 += 1

        index2 = arc_group[group_index2].index(col)

        if index2 != 1 and index2 != len(arc_group[group_index2]) - 2:
            continue

        to_continue, head_list = True, []
        for idx in range(1, len(arc_group[group_index1]) - 1):
            head = index_2_cluster[arc_group[group_index1][idx]]
            if head in head_list:
                to_continue = False
                break
            head_list.append(head)

        for idx in range(1, len(arc_group[group_index2]) - 1):
            head = index_2_cluster[arc_group[group_index2][idx]]
            if head in head_list:
                to_continue = False
                break
            head_list.append(head)

        if not to_continue:
            continue

        if index1 == 1:
            reversed(arc_group[group_index1])

        if index2 != 1:
            reversed(arc_group[group_index2])

        if group_index1 == group_index2 or len(arc_group[group_index1]) + len(arc_group[group_index2]) > max_head_index + 4:
            continue

        arc_group[group_index1] = arc_group[group_index1][:-1] + arc_group[group_index2][1:]
        arc_group.pop(group_index2)

    # TODO: 2-opt/DP改进周期内贴装路径
    fig = 0
    component_result, cycle_result, feederslot_result = [], [], []
    placement_result, head_sequence = [], []
    for arc in arc_group:
        fig += 1
        assign_part = [-1 for _ in range(max_head_index)]
        assign_slot = [-1 for _ in range(max_head_index)]
        assign_point = [-1 for _ in range(max_head_index)]
        assign_sequence = []

        for node in arc:
            if node == -1:
                continue

            part = pcb_data.loc[node]['part']
            head = index_2_cluster[node]

            assign_point[head] = node

            part_index = component_data[component_data['part'] == part].index.tolist()[0]
            assign_part[head] = part_index
            assign_sequence.append(head)

            # TODO: assign_slot

        if len(component_result) == 0 or component_result[-1] != assign_part:
            component_result.append(assign_part)
            cycle_result.append(1)
        else:
            cycle_result[-1] += 1

        feederslot_result.append(assign_slot)
        placement_result.append(assign_point)
        head_sequence.append(assign_sequence)

        if fig > 1:
            continue

        if render:
            plt.figure(fig)
            plt.scatter([pos[0] for pos in pcb_pos], [pos[1] for pos in pcb_pos], c='b', s=20)
            plt.scatter([pos[0] for pos in mount_pos], [pos[1] for pos in mount_pos], c='y', s=20)
            plt.scatter([pcb_pos[i][0] for i in arc[1:-1]], [pcb_pos[i][1] for i in arc[1:-1]], c='r', marker='x')

            for i in range(1, len(arc) - 1):
                plt.text(mount_pos[arc[i]][0], mount_pos[arc[i]][1], 'HD' + str(assign_sequence[i - 1] + 1))

            plt.plot([base_point[0], mount_pos[arc[1]][0]], [base_point[1], mount_pos[arc[1]][1]], 'r')
            for i in range(2, len(arc) - 1):
                plt.plot([mount_pos[arc[i - 1]][0], mount_pos[arc[i]][0]], [mount_pos[arc[i - 1]][1], mount_pos[arc[i]][1]], 'r')

            plt.plot([base_point[0], mount_pos[arc[-2]][0]], [base_point[1], mount_pos[arc[-2]][1]], 'r')

    if render:
        plt.show()

    return component_result, cycle_result, feederslot_result, placement_result, head_sequence

def dynamic_based_cluster_route(cluster_pos):
    pass

@timer_warper
def cluster_based_route_generation(component_data, pcb_data, compoenent_result, cycle_result, feederslot_result):

    random.seed(0)
    placement_result, head_sequence = [], []

    mount_point = []
    for idx, row in pcb_data.iterrows():
        mount_point.append([row['x'], row['y'], idx])

    num_points, num_cluster = len(mount_point), sum(cycle_result)
    random_index = random.sample(list(range(len(mount_point))), num_cluster)
    # random_head = [random.randint(0, max_head_index) for _ in range(num_cluster)]

    # 计算聚类的中心，移除已选择的点
    cluster_center, cluster_prev_center = [], []
    cluster_members = [[] for _ in range(num_cluster)]
    for rdm_idx, pcb_idx in enumerate(random_index):
        # 聚类中心
        # cluster_center.append([mount_point[pcb_idx][0] - random_head[rdm_idx] * head_interval, mount_point[pcb_idx][1]])
        cluster_center.append([mount_point[pcb_idx][0], mount_point[pcb_idx][1]])

    counter, inf = 0, 1e10
    while True:
        counter += 1
        for pcb_idx, point in enumerate(mount_point):
            distance = []
            for cluster in range(num_cluster):
                for head in range(max_head_index):
                    delta_x, delta_y = abs(point[0] - head * head_interval - cluster_center[cluster][0]), abs(
                        point[1] - cluster_center[cluster][1])
                    # distance.append(max(delta_x, delta_y))
                    distance.append(pow(delta_x, 2) + pow(delta_y, 2))

            while True:
                dist_idx = distance.index(min(distance))
                cluster_idx, head_idx = dist_idx // max_head_index, dist_idx % max_head_index
                if len(cluster_members[cluster_idx]) >= max_head_index or head_idx in [member[1] for member in cluster_members[cluster_idx]]:
                    distance[dist_idx] = inf
                else:
                    cluster_members[cluster_idx].append([pcb_idx, head_idx])
                    break

        # 重新计算聚类的中心
        cluster_prev_center = copy.deepcopy(cluster_center)

        cluster_center = []
        deviation = 0
        for members in cluster_members:
            center = [0, 0]
            for pcb_idx, head_idx in members:
                center[0] += pcb_data.loc[pcb_idx]['x']
                # center[0] += pcb_data.loc[pcb_idx]['x'] - head_idx * head_interval
                center[1] += pcb_data.loc[pcb_idx]['y']
            cluster_center.append([center[0] / len(members), center[1] / len(members)])

            for pcb_idx, head_idx in members:
                deviation += math.sqrt(pow(center[0] - pcb_data.loc[pcb_idx]['x'], 2) + pow(
                    center[1] - pcb_data.loc[pcb_idx]['y'], 2))

        if np.sum(np.abs(np.array(cluster_prev_center) - np.array(cluster_center))) < 1e-9 or counter > 200:
            break
        else:
            print(' --- currernt cluster iteration: ', counter, ', deviation: ', deviation, ' ---')
            cluster_members = [[] for _ in range(num_cluster)]

    # 以下为临时测试代码：
    for members in cluster_members:
        pos_x, pos_y = [], []
        for i in range(num_points):

            pos_x.append(pcb_data.loc[i]['x'])
            pos_y.append(pcb_data.loc[i]['y'])

        plt.scatter(pos_x, pos_y, s=8)

        mount_pos_x, mount_pos_y = [], []
        for pcb_idx, head_idx in members:
            mount_pos_x.append(pcb_data.loc[pcb_idx]['x'] - head_idx * head_interval)
            mount_pos_y.append(pcb_data.loc[pcb_idx]['y'])
            plt.scatter(pcb_data.loc[pcb_idx]['x'], pcb_data.loc[pcb_idx]['y'], marker = 'x', color = 'r')
            plt.text(pcb_data.loc[pcb_idx]['x'], pcb_data.loc[pcb_idx]['y'], head_idx + 1)
        plt.show()

    # 转换为输出结果
    for members in cluster_members:
        cycle_placement = [-1 for _ in range(max_head_index)]
        for pcb_idx, head_idx in members:
            cycle_placement[head_idx] = pcb_idx
        placement_result.append(cycle_placement)
        head_sequence.append(dynamic_programming_cycle_path(pcb_data, cycle_placement))
    return placement_result, head_sequence

