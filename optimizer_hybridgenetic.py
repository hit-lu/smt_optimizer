import copy
import random
import numpy as np
import pandas as pd

from optimizer_common import *
from collections import defaultdict
# from itertools import pairwise


def roulette_weel_selection(pop_val):
    cumsum_pop_val = np.array(pop_val)
    cumsum_pop_val = np.divide(cumsum_pop_val, np.sum(cumsum_pop_val))
    cumsum_pop_val = cumsum_pop_val.cumsum()

    random_eval = np.random.random()
    index = 0
    while index < len(pop_val):
        if random_eval > cumsum_pop_val[index]:
            index += 1
        else:
            break
    return index


def directed_edge_recombination_crossover(c, individual1, individual2):
    assert len(individual1) == len(individual2)
    left_edge_list, right_edge_list = defaultdict(list), defaultdict(list)

    for index in range(len(individual1) - 1):
        elem1, elem2 = individual1[index], individual1[index + 1]
        right_edge_list[elem1].append(elem2)
        left_edge_list[elem2].append(elem1)

    for index in range(len(individual2) - 1):
        elem1, elem2 = individual2[index], individual2[index + 1]
        right_edge_list[elem1].append(elem2)
        left_edge_list[elem2].append(elem1)

    offspring = []
    while len(offspring) != len(individual1):
        while True:
            center_element = np.random.choice(individual1)
            if center_element not in offspring:        # 避免重复选取
                break
        direction, candidate = 1, [center_element]
        element = center_element
        for edge_list in left_edge_list.values():
            while element in edge_list:
                edge_list.remove(element)

        for edge_list in right_edge_list.values():
            while element in edge_list:
                edge_list.remove(element)

        while True:
            max_len, max_len_neighbor = -1, 0
            if direction == 1:
                if len(right_edge_list[element]) == 0:
                    direction, element = -1, center_element
                    continue
                for neighbor in right_edge_list[element]:
                    if max_len < len(right_edge_list[neighbor]):
                        max_len_neighbor = neighbor
                        max_len = len(right_edge_list[neighbor])
                candidate.append(max_len_neighbor)
                element = max_len_neighbor
            elif direction == -1:
                if len(left_edge_list[element]) == 0:
                    direction, element = 0, center_element
                    continue
                for neighbor in left_edge_list[element]:
                    if max_len < len(left_edge_list[neighbor]):
                        max_len_neighbor = neighbor
                        max_len = len(left_edge_list[neighbor])
                candidate.insert(0, max_len_neighbor)
                element = max_len_neighbor
            else:
                break

            # 移除重复元素
            for edge_list in left_edge_list.values():
                while max_len_neighbor in edge_list:
                    edge_list.remove(max_len_neighbor)

            for edge_list in right_edge_list.values():
                while max_len_neighbor in edge_list:
                    edge_list.remove(max_len_neighbor)

        offspring += candidate

    return offspring


def mutation(individual):
    range_ = np.random.randint(0, len(individual), 2)
    individual[range_[0]], individual[range_[1]] = individual[range_[1]], individual[range_[0]]
    return individual


def dynamic_programming_cycle_path(cycle_placement, cycle_points):
    head_sequence = []
    num_pos = sum([placement != -1 for placement in cycle_placement]) + 1

    pos, head_set = [], []
    average_pos_x, counter = 0, 1
    for head, placement in enumerate(cycle_placement):
        if placement == -1:
            continue
        head_set.append(head)
        pos.append([cycle_points[head][0], cycle_points[head][1]])
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


def pickup_group_combination(component_nozzle, designated_nozzle, supply, supply_cycle, demand, demand_cycle):

    combination, combination_cycle = demand.copy(), demand_cycle.copy()
    supply_cpy = supply.copy()

    while True:
        supply_cpy_bits = max_head_index - supply_cpy.count(None)
        if supply_cpy_bits == 0:
            break
        max_match_offset,  max_match_counter = 0, 0
        supply_cpy_index = [idx for idx, part in enumerate(supply_cpy) if part]     # 加快搜索速度
        for offset in range(-supply_cpy_index[-1], max_head_index - supply_cpy_index[0]):
            match_counter = 0
            for idx, part in enumerate(supply_cpy):
                if 0 <= idx + offset < max_head_index:
                    if part is None:
                        continue
                    if combination[idx + offset] is None and designated_nozzle[idx + offset] == designated_nozzle[idx]:
                        match_counter += 1
            if match_counter > max_match_counter:
                max_match_counter = match_counter
                max_match_offset = offset
                if match_counter == supply_cpy_bits:
                    break

        for idx, part in enumerate(supply_cpy):
            if 0 <= idx + max_match_offset < max_head_index:
                if part is None:
                    continue

                if demand[idx + max_match_offset] is None:
                    combination[idx + max_match_offset] = part
                    combination_cycle[idx + max_match_offset] = supply_cycle[idx]
                    supply_cpy[idx] = None

    return combination, combination_cycle


def cal_individual_val(component_nozzle, component_point_pos, designated_nozzle, pickup_group, pickup_group_cycle,
                       pair_group, feeder_part_arrange, individual):

    place_time, pick_time = 0.234, 0.4
    x_moving_speed, y_moving_speed = 300, 300  # mm/s

    prev_pair_index = None
    sequenced_pickup_group, sequenced_pickup_cycle = [], []
    for gene in individual:
        pickup = pickup_group[gene]

        pair_index = None
        for idx, pair in enumerate(pair_group):
            if gene in pair:
                pair_index = idx
                break

        if pair_index is not None and pair_index == prev_pair_index:
            for idx, component in enumerate(pickup):
                sequenced_pickup_group[-1][idx] = component
        else:
            sequenced_pickup_group.append(pickup.copy())
            sequenced_pickup_cycle.append(pickup_group_cycle[gene])

    V = [float('inf') for _ in range(len(sequenced_pickup_group) + 1)]      # Node Value
    V[0] = 0
    V_SNode = [-1 for _ in range(len(sequenced_pickup_group) + 1)]

    nozzle_assigned_heads = defaultdict(int)
    for nozzle in designated_nozzle:
        nozzle_assigned_heads[nozzle] += 1

    pickup_result, pickup_cycle_result = [[] for _ in range(len(V))], [[] for _ in range(len(V))]
    component_point_index = defaultdict(int)
    for i in range(1, len(V)):
        cost, t0 = 0, 0
        load = defaultdict(int)
        Pd, Pd_cycle = [None for _ in range(max_head_index)], [0 for _ in range(max_head_index)]  # demand pickup
        j = i
        while j < len(V):
            Ps, Ps_cycle = sequenced_pickup_group[j - 1], [sequenced_pickup_cycle[j - 1] for _ in
                                                       range(max_head_index)]  # supply pickup and its cycle
            for part in Ps:
                if part:
                    load[component_nozzle[part]] += 1

            is_combinable = True
            for nozzle, counter in load.items():
                if counter > nozzle_assigned_heads[nozzle]:
                    is_combinable = False

            if is_combinable:
                cost = cost - t0
                # combine sequenced pickup ρb and ps into ρu(union pickup)
                Pu, Pu_cycle = pickup_group_combination(component_nozzle, designated_nozzle, Ps, Ps_cycle, Pd, Pd_cycle)

                # decide the placement cluster and sequencing of pickup ρu
                pickup_action_counter, place_action_counter = 0, max_head_index - Pu.count(None)
                right_most_slot, left_most_slot = 0, max_slot_index // 2        # most left and right pickup slot

                # === TODO: 机械限位、后槽位分配未处理 ===
                for head in range(max_head_index):
                    if not Pu[head]:
                        continue
                    assert Pu[head] in feeder_part_arrange.keys()
                    for slot in feeder_part_arrange[Pu[head]]:
                        left_most_slot = min(slot - head * interval_ratio, left_most_slot)
                        right_most_slot = max(slot - head * interval_ratio, right_most_slot)

                # calculate forward, backward, pick and place traveling time
                t_FW, t_BW, t_PL, t_PU = 0, 0, 0, 0
                cycle = 0
                while cycle < max(Pu_cycle):
                    mount_points = []
                    for head, part in enumerate(Pu):
                        if part is None or cycle > Pu_cycle[head]:
                            continue
                        idx = component_point_index[part]
                        mount_points.append([component_point_pos[part][idx][0] - head * head_interval + stopper_pos[0],
                                             component_point_pos[part][idx][0] + stopper_pos[1]])
                    assert len(mount_points) > 0

                    # calculate cycle moving distance
                    mount_points.sort(key=lambda x: x[0])
                    t_FW += max(
                        abs(slotf1_pos[0] + (left_most_slot - 1) * slot_interval - mount_points[0][0]) / x_moving_speed,
                        abs(slotf1_pos[1] - mount_points[0][1]) / y_moving_speed)
                    t_BW += max(
                        abs(slotf1_pos[0] + (right_most_slot - 1) * slot_interval - mount_points[-1][0]) / x_moving_speed,
                        abs(slotf1_pos[1] - mount_points[-1][1]) / y_moving_speed)
                    # pick up moving time
                    t_PU += (right_most_slot - left_most_slot) * slot_interval / x_moving_speed
                    # place moving time
                    for idx_points in range(len(mount_points) - 1):
                        t_PL += max(abs(mount_points[idx_points][0] - mount_points[idx_points + 1][0]) / x_moving_speed,
                                    abs(mount_points[idx_points][1] - mount_points[idx_points + 1][1]) / y_moving_speed)
                    cycle += 1

                t0 = t_FW + (t_PL + place_action_counter * place_time) + t_BW
                cost += (t_PU + pickup_action_counter * pick_time) + t0

                if V[i - 1] + cost < V[j]:
                    pickup_result[j], pickup_cycle_result[j] = Pu, Pu_cycle
                    V_SNode[j] = i - 1
                    V[j] = V[i - 1] + cost

                Pd, Pd_cycle = Pu, Pu_cycle
                j += 1
            else:
                break

    node = len(V) - 1
    while True:
        prev_node = V_SNode[node]
        if prev_node == -1:
            break
        for k in range(prev_node + 1, node):
            pickup_result[k], pickup_cycle_result[k] = [], []
        node = prev_node
    return V[-1], pickup_result, pickup_cycle_result


def convert_individual_2_result(component_data, component_point_pos, designated_nozzle, pickup_group, pickup_group_cycle,
                       pair_group, feeder_lane, individual):
    component_result, cycle_result, feeder_slot_result = [], [], []
    placement_result, head_sequence_result = [], []

    # === 记录不同元件对应的槽位 ===
    feeder_part_arrange = defaultdict(list)
    for slot in range(1, max_slot_index // 2 + 1):
        if feeder_lane[slot]:
            feeder_part_arrange[feeder_lane[slot]].append(slot)

    # === 记录不同元件的注册吸嘴类型 ===
    component_nozzle = defaultdict(str)
    for pickup in pickup_group:
        for part in pickup:
            if part is None or part in component_nozzle.keys():
                continue
            component_nozzle[part] = component_data[component_data['part'] == part]['nz1'].tolist()[0]

    # initial result
    _, pickup_result, pickup_cycle_result = cal_individual_val(component_nozzle, component_point_pos, designated_nozzle,
                                                               pickup_group, pickup_group_cycle,
                                                               pair_group, feeder_part_arrange, individual)

    for idx, pickup in enumerate(pickup_result):
        while pickup and max(pickup_cycle_result[idx]) != 0:
            cycle = min([cycle_ for cycle_ in pickup_cycle_result[idx] if cycle_ > 0])
            feeder_part_arrange_index = defaultdict(int)
            component_result.append([-1 for _ in range(max_head_index)])
            feeder_slot_result.append([-1 for _ in range(max_head_index)])
            cycle_result.append(cycle)
            for head, part in enumerate(pickup):
                if part is None or pickup_cycle_result[idx][head] == 0:
                    continue
                    
                part_index = component_data[component_data['part'] == part].index.tolist()[0]
                component_result[-1][head] = part_index
                feeder_slot_result[-1][head] = feeder_part_arrange[part][feeder_part_arrange_index[part]]
                feeder_part_arrange_index[part] += 1
                if feeder_part_arrange_index[part] >= len(feeder_part_arrange[part]):
                    feeder_part_arrange_index[part] = 0

                pickup_cycle_result[idx][head] -= cycle

    component_point_index = defaultdict(int)
    for cycle_set in range(len(cycle_result)):
        for cycle in range(cycle_result[cycle_set]):
            placement_result.append([-1 for _ in range(max_head_index)])
            mount_point = [[0, 0] for _ in range(max_head_index)]
            for head in range(max_head_index):
                part_index = component_result[cycle_set][head]
                if part_index == -1:
                    continue

                part = component_data.iloc[part_index]['part']
                point_info = component_point_pos[part][component_point_index[part]]
                
                placement_result[-1][head] = point_info[2]
                mount_point[head] = point_info[0:2]

                component_point_index[part] += 1
            head_sequence_result.append(dynamic_programming_cycle_path(placement_result[-1], mount_point))

    return component_result, cycle_result, feeder_slot_result, placement_result, head_sequence_result


def get_top_k_value(pop_val, k: int):
    res = []
    pop_val_cpy = pop_val.copy()
    pop_val_cpy.sort(reverse=True)

    for i in range(min(len(pop_val_cpy), k)):
        for j in range(len(pop_val)):
            if abs(pop_val_cpy[i] - pop_val[j]) < 1e-9 and j not in res:
                res.append(j)
                break
    return res


@timer_wrapper
def optimizer_hybrid_genetic(pcb_data, component_data, hinter=True):
    random.seed(0)
    np.random.seed(0)
    designated_nozzle = optimal_nozzle_assignment(component_data, pcb_data)

    # === component assignment ===
    component_points, nozzle_components = defaultdict(int), defaultdict(list)   # 元件贴装点数，吸嘴-元件对应关系
    component_feeder_limit, component_divided_points = defaultdict(int), defaultdict(list)
    for step in pcb_data.iterrows():
        part = step[1]['part']
        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz1']

        component_feeder_limit[part] = component_data.loc[idx]['feeder-limit']
        component_points[part] += 1
        if nozzle_components[nozzle].count(part) < component_feeder_limit[part]:
            nozzle_components[nozzle].append(part)

    for part, feeder_limit in component_feeder_limit.items():
        for _ in range(feeder_limit):
            component_divided_points[part].append(component_points[part] // feeder_limit)

    for part, divided_points in component_divided_points.items():
        index = 0
        while sum(divided_points) < component_points[part]:
            divided_points[index] += 1
            index += 1

    CT_Group, CT_Points = [], []    # CT: Component Type
    while sum(len(nozzle_components[nozzle]) for nozzle in nozzle_components.keys()) != 0:

        CT_Group.append([None for _ in range(max_head_index)])
        CT_Points.append([0 for _ in range(max_head_index)])

        for head_index in range(max_head_index):
            nozzle = designated_nozzle[head_index]      # 分配的吸嘴
            if len(nozzle_components[nozzle]) == 0:      # 无可用元件
                continue

            max_points, designated_part = 0, None
            for part in nozzle_components[nozzle]:
                if component_points[part] > max_points:
                    max_points = component_points[part]
                    designated_part = part

            component_points[designated_part] -= component_divided_points[designated_part][-1]

            CT_Group[-1][head_index] = designated_part
            CT_Points[-1][head_index] = component_divided_points[designated_part][-1]

            component_divided_points[designated_part].pop()
            nozzle_components[nozzle].remove(designated_part)

    # === assign CT group to feeder slot ===
    point_num = len(pcb_data)
    component_point_pos = defaultdict(list)
    for point_cnt in range(point_num):
        part = pcb_data.loc[point_cnt, 'part']
        component_point_pos[part].append(
            [pcb_data.loc[point_cnt, 'x'] + stopper_pos[0], pcb_data.loc[point_cnt, 'y'] + stopper_pos[1], point_cnt])

    for pos_list in component_point_pos.values():
        pos_list.sort(key=lambda x: (x[0], x[1]))

    CT_Group_slot = [-1] * len(CT_Group)
    feeder_lane = [None] * max_slot_index     # 供料器基座上已分配的元件类型
    for CTIdx, pickup in enumerate(CT_Group):
        best_slot = []
        for cp_index, component in enumerate(pickup):
            if component is None:
                continue
            best_slot.append(round((sum(pos[0] for pos in component_point_pos[component]) / len(
                component_point_pos[component]) - slotf1_pos[0]) / slot_interval) + 1 - cp_index * interval_ratio)
        best_slot = round(sum(best_slot) / len(best_slot))

        search_dir, step = 0, 0  # dir: 1-向右, 0-向左
        prev_assign_available = True
        while True:
            assign_slot = best_slot + step if search_dir else best_slot - step
            if assign_slot + (len(pickup) - 1) * interval_ratio >= max_slot_index / 2 or assign_slot < 0:
                if not prev_assign_available:
                    raise Exception('feeder assign error!')
                # prev_assign_available = False
                search_dir = 1 - search_dir
                if search_dir == 1:
                    step += 1
                continue

            prev_assign_available = True
            assign_available = True

            # 分配对应槽位
            for slot in range(assign_slot, assign_slot + interval_ratio * len(pickup), interval_ratio):
                pickup_index = int((slot - assign_slot) / interval_ratio)
                if feeder_lane[slot] is not None and pickup[pickup_index] is not None:
                    assign_available = False
                    break

            if assign_available:
                for idx, component in enumerate(pickup):
                    if component is not None:
                        feeder_lane[assign_slot + idx * interval_ratio] = component
                CT_Group_slot[CTIdx] = assign_slot
                break

            search_dir = 1 - search_dir
            if search_dir == 1:
                step += 1

    # === Initial Pickup Group ===
    initial_pickup, initial_pickup_cycle = [], []
    for index, CT in enumerate(CT_Group):
        while True:
            if CT_Points[index].count(0) == max_head_index:
                break
            min_element = min([Points for Points in CT_Points[index] if Points > 0])

            initial_pickup.append(copy.deepcopy(CT_Group[index]))
            initial_pickup_cycle.append(min_element)
            for head in range(max_head_index):
                if CT_Points[index][head] >= min_element:
                    CT_Points[index][head] -= min_element
                if CT_Points[index][head] == 0:
                    CT_Group[index][head] = None

    # pickup partition rule
    partition_probability = 0.1
    pickup_group, pair_group = [], []        # pair_group:  pickups from same initial group
    pickup_group_cycle = []
    for idx, Pickup in enumerate(initial_pickup):
        pickup_num = len([element for element in Pickup if element is not None])
        if 2 <= pickup_num <= max_head_index / 3 or (
                max_head_index / 3 <= pickup_num <= max_head_index / 2 and np.random.rand() < partition_probability):
            # partitioned into single component pickups
            # or partition the potentially inefficient initial pickups with a small probability
            pair_index = []
            for index, CT in enumerate(Pickup):
                if CT is not None:
                    pair_index.append(len(pickup_group))
                    pickup_group.append([None for _ in range(max_head_index)])
                    pickup_group[-1][index] = CT
                    pickup_group_cycle.append(initial_pickup_cycle[idx])
            pair_group.append(pair_index)
        else:
            pickup_group.append(Pickup)
            pickup_group_cycle.append(initial_pickup_cycle[idx])

    # basic parameter
    # crossover rate & mutation rate: 80% & 10%
    # population size: 200
    # the number of generation: 500
    crossover_rate, mutation_rate = 0.8, 0.1
    population_size, n_generations = 200, 500

    # initial solution
    population = []
    for _ in range(population_size):
        pop_permutation = list(range(len(pickup_group)))
        np.random.shuffle(pop_permutation)
        population.append(pop_permutation)

    best_individual, best_pop_val = [], float('inf')
    generation_counter = 0

    # === 记录不同元件对应的槽位 ===
    feeder_part_arrange = defaultdict(list)
    for slot in range(1, max_slot_index // 2 + 1):
        if feeder_lane[slot]:
            feeder_part_arrange[feeder_lane[slot]].append(slot)

    # === 记录不同元件的注册吸嘴类型 ===
    component_nozzle = defaultdict(str)
    for pickup in pickup_group:
        for part in pickup:
            if part is None or part in component_nozzle.keys():
                continue
            component_nozzle[part] = component_data[component_data['part'] == part]['nz1'].tolist()[0]

    with tqdm(total=n_generations) as pbar:
        pbar.set_description('hybrid genetic process')

        while generation_counter < n_generations:
            # calculate fitness value
            pop_val = []
            for pop_idx, individual in enumerate(population):
                val, _, _ = cal_individual_val(component_nozzle, component_point_pos, designated_nozzle, pickup_group,
                                                  pickup_group_cycle, pair_group, feeder_part_arrange, individual)
                pop_val.append(val)

            if min(pop_val) < best_pop_val:
                best_pop_val = min(pop_val)
                index = pop_val.index(best_pop_val)
                best_individual = copy.deepcopy(population[index])

            # min-max convert
            max_val = 1.5 * max(pop_val)
            pop_val = list(map(lambda v: max_val - v, pop_val))

            # selection
            new_population, new_pop_val = [], []
            top_k_index = get_top_k_value(pop_val, int(population_size * 0.3))
            for index in top_k_index:
                new_population.append(population[index])
                new_pop_val.append(pop_val[index])

            index = [i for i in range(population_size)]

            # crossover and mutation
            select_index = random.choices(index, weights=pop_val, k=population_size - int(population_size * 0.3))
            for index in select_index:
                new_population.append(population[index])
                new_pop_val.append(pop_val[index])
            population, pop_val = new_population, new_pop_val
            c = 0
            for pop in range(population_size):
                if pop % 2 == 0 and np.random.random() < crossover_rate:
                    index1, index2 = roulette_weel_selection(pop_val), -1
                    while True:
                        index2 = roulette_weel_selection(pop_val)
                        if index1 != index2:
                            break
                    # 两点交叉算子
                    population[index1] = directed_edge_recombination_crossover(c,population[index1], population[index2])
                    c += 1
                    population[index2] = directed_edge_recombination_crossover(c,population[index2], population[index1])
                    c += 1
                if np.random.random() < mutation_rate:
                    index_ = roulette_weel_selection(pop_val)
                    mutation(population[index_])

            generation_counter += 1
            pbar.update(1)

    return convert_individual_2_result(component_data, component_point_pos, designated_nozzle, pickup_group,
                                       pickup_group_cycle, pair_group, feeder_lane, best_individual)
