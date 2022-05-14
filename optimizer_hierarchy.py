import matplotlib.pyplot as plt

from dataloader import *
from common_function import *

def dynamic_programming_cycle_path(pcb_data, cycle_placement):
    head_sequence = []
    num_pos = sum([placement != -1 for placement in cycle_placement]) + 1

    pos, head_set = [], []
    average_pos_x, counter = 0, 1
    for head, placement in enumerate(cycle_placement):
        if placement == -1:
            continue
        head_set.append(head)
        pos.append([pcb_data.loc[placement]['x'] - head * head_interval + stopper_pos[0], pcb_data.loc[placement]['y'] + stopper_pos[1]])
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

def greedy_placement_route_generation(pcb_data, component_data, component_result, cycle_result):
    placement_result, headsequence_result = [], []

    mountpoint_index = [[] for _ in range(len(component_data))]
    mountpoint_pos = [[] for _ in range(len(component_data))]

    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mountpoint_index[component_index].append(i)
        mountpoint_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

    for cycle_set in range(len(component_result)):
        floor_cycle, ceil_cycle = sum(cycle_result[:cycle_set]), sum(cycle_result[:(cycle_set + 1)])
        for cycle in range(floor_cycle, ceil_cycle):
            assigned_placement = [-1 for _ in range(max_head_index)]
            # 最近邻确定
            way_point = []
            for head in range(max_head_index):
                if component_result[cycle_set][head] == -1:
                    continue

                component_index = component_result[cycle_set][head]
                if head == 0:
                    index = np.argmin(mountpoint_pos[component_index], axis=0)[0]
                    assigned_placement[head] = mountpoint_index[component_index][index]
                    way_point.append(mountpoint_pos[component_index][index])  # 记录路标点

                    mountpoint_index[component_index].pop(index)
                    mountpoint_pos[component_index].pop(index)
                else:
                    head_index, point_index = -1, -1
                    min_cheby_distance, min_euler_distance = np.inf, np.inf
                    for next_head in range(1, max_head_index):
                        if assigned_placement[next_head] != -1:
                            continue

                        component_index = component_result[cycle_set][next_head]
                        for counter1 in range(len(mountpoint_pos[component_index])):
                            for counter2 in range(len(way_point)):
                                delta_x = abs(mountpoint_pos[component_index][counter1][0] - way_point[counter2][
                                    0] - head_index * slot_interval)
                                delta_y = abs(mountpoint_pos[component_index][counter1][1] - way_point[counter2][1])
                                cheby_distance = max(delta_x, delta_y)
                                euler_distance = pow(delta_x, 2) + pow(delta_y, 2)
                                if cheby_distance < min_cheby_distance or \
                                        (abs(cheby_distance - min_cheby_distance) < 10 and euler_distance < min_euler_distance):
                                    min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                    head_index, point_index = next_head, counter1

                    component_index = component_result[cycle_set][head_index]
                    assigned_placement[head_index] = mountpoint_index[component_index][point_index]

                    way_point.append(mountpoint_pos[component_index][point_index])  # 记录路标点(相对于吸杆1的位置)
                    way_point[-1][0] -= head_index * slot_interval

                    mountpoint_index[component_index].pop(point_index)
                    mountpoint_pos[component_index].pop(point_index)

            placement_result.append(assigned_placement)     # 各个头上贴装的元件类型
            headsequence_result.append(dynamic_programming_cycle_path(pcb_data, assigned_placement))

    return placement_result, headsequence_result
