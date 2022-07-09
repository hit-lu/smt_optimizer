import copy
import random
import numpy as np
import pandas as pd

from common_function import *
# from itertools import pairwise
from collections import defaultdict


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


def pickup_group_combination(component_data, designated_nozzle, supply, supply_cycle, demand, demand_cycle):
    combination = copy.deepcopy(demand)
    combination_cycle = copy.deepcopy(demand_cycle)

    supply_cpy = copy.deepcopy(supply)
    while len([part for part in supply_cpy if part is not None]) != 0:
        max_match_offset,  max_match_counter = 0, 0
        for offset in range(-max_head_index + 1, max_head_index):
            match_counter = 0
            for idx, part in enumerate(supply_cpy):
                if 0 <= idx + offset < max_head_index:
                    if part is None:
                        continue
                    nozzle = component_data[component_data['part'] == part]['nz1'].tolist()[0]
                    if combination[idx + offset] is None and designated_nozzle[idx] == nozzle:
                        match_counter += 1
            if match_counter > max_match_counter:
                max_match_counter = match_counter
                max_match_offset = offset

        for idx, part in enumerate(supply_cpy):
            if 0 <= idx + max_match_offset < max_head_index:
                if part is None:
                    continue
                nozzle = component_data[component_data['part'] == part]['nz1'].tolist()[0]
                if demand[idx + max_match_offset] is None and designated_nozzle[idx] == nozzle:
                    combination[idx + max_match_offset] = part
                    combination_cycle[idx + max_match_offset] = supply_cycle[idx]
                    supply_cpy[idx] = None

    return combination, combination_cycle

@timer_warper
def cal_individual_val(component_data, component_point_pos, designated_nozzle, pickup_group, pickup_group_cycle,
                       pair_group, feeder_lane, individual):
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
            sequenced_pickup_group.append(copy.deepcopy(pickup))
            sequenced_pickup_cycle.append(pickup_group_cycle[gene])

    V = [float('inf') for _ in range(len(sequenced_pickup_group) + 1)]      # Node Value
    V[0] = 0
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
                if part is not None:
                    nozzle = component_data[component_data['part'] == part]['nz1'].tolist()[0]
                    load[nozzle] += 1

            is_combinable = True
            for nozzle, counter in load.items():
                if counter > nozzle_assigned_heads[nozzle]:
                    is_combinable = False

            if is_combinable:
                cost = cost - t0
                # combine sequenced pickup ρb and ps into ρu
                Pu, Pu_cycle = pickup_group_combination(component_data, designated_nozzle, Ps, Ps_cycle, Pd, Pd_cycle)           # union pickup

                # decide the placement cluster and sequencing of pickup ρu
                pickup_action_counter, place_action_counter = 0, sum([1 for part in Pu if part is not None])
                right_most_slot, left_most_slot = 0, max_slot_index // 2        # most left and right pickup slot

                # TODO: 机械限位、后槽位
                for slot in range(max_slot_index // 2):
                    pick_action = False
                    for head in range(max_head_index):
                        if feeder_lane[slot + head * interval_ratio] is None:
                            continue
                        if feeder_lane[slot + head * interval_ratio] == Pu[head]:
                            pick_action = True
                            if slot < left_most_slot:
                                left_most_slot = slot
                            if slot > right_most_slot:
                                right_most_slot = slot
                    if pick_action:
                        pickup_action_counter += 1

                assert pickup_action_counter > 0

                # calculate forward and backward traveling time
                t_FW, t_BW, t_PL, t_PU = 0, 0, 0, 0  # represent forward, backward, place and pick moving time respectively
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
                    mount_points.sort(key = lambda x: x[0])
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
                    pickup_result[i - 1] = Pu
                    pickup_cycle_result[i - 1] = Pu_cycle

                    V[j] = V[i - 1] + cost
                Pd = Pu
                j += 1
            else:
                break

    return V[-1], pickup_result, pickup_cycle_result


def convert_individual_2_result(component_data, component_point_pos, designated_nozzle, pickup_group, pickup_group_cycle,
                       pair_group, feeder_lane, individual):
    component_result, cycle_result, feederslot_result = [], [], []
    placement_result, headsequence_result = [], []

    # initial result
    _, pickup_result, pickup_cycle_result = cal_individual_val(component_data, component_point_pos, designated_nozzle,
                                                               pickup_group, pickup_group_cycle,
                                                               pair_group, feeder_lane, individual)
    part_slot = defaultdict(int)
    for slot, part in enumerate(feeder_lane):
        if part is not None:
            part_slot[part] = slot

    for idx, pickup in enumerate(pickup_result):
        while pickup and max(pickup_cycle_result[idx]) != 0:
            cycle = min([cycle_ for cycle_ in pickup_cycle_result[idx] if cycle_ > 0])

            component_result.append([-1 for _ in range(max_head_index)])
            feederslot_result.append([-1 for _ in range(max_head_index)])
            cycle_result.append(cycle)
            for head, part in enumerate(pickup):
                if part is None or pickup_cycle_result[idx][head] == 0:
                    continue
                    
                component_index = component_data[component_data['part'] == part].index.tolist()[0]
                component_result[-1][head] = component_index
                component_result[-1][head] = part_slot[part]
                pickup_cycle_result[idx][head] -= cycle

    return component_result, cycle_result, feederslot_result, placement_result, headsequence_result


def get_top_k_value(pop_val, k: int):
    res = []
    pop_val_cpy = copy.deepcopy(pop_val)
    pop_val_cpy.sort(reverse = True)

    for i in range(min(len(pop_val_cpy), k)):
        for j in range(len(pop_val)):
            if abs(pop_val_cpy[i] - pop_val[j]) < 1e-9 and j not in res:
                res.append(j)
                break
    return res


@timer_warper
def optimizer_hybridgenetic(pcb_data, component_data):
    random.seed(0)
    np.random.seed(0)

    # === Nozzle Assignment ===
    nozzle_points = {}  # number of points for nozzle
    nozzle_assigned_heads = {}  # number of heads for nozzle
    for step in pcb_data.iterrows():
        part = step[1]['part']
        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz1']
        if nozzle not in nozzle_points.keys():
            nozzle_points[nozzle] = 0
            nozzle_assigned_heads[nozzle] = 0

        nozzle_points[nozzle] += 1

    assert (len(nozzle_points.keys()) > max_head_index, "nozzle type number should no more than the head num")
    total_points, available_head = len(pcb_data), max_head_index
    S1, S2, S3 = [], [], []

    for nozzle in nozzle_points.keys():     # Phase 1
        if nozzle_points[nozzle] * max_head_index < total_points:
            nozzle_assigned_heads[nozzle] = 1
            available_head -= 1
            total_points -= nozzle_points[nozzle]

            S1.append(nozzle)
        else:
            S2.append(nozzle)

    available_head_ = available_head        # Phase 2
    for nozzle in S2:
        nozzle_assigned_heads[nozzle] = math.floor(available_head * nozzle_points[nozzle] / total_points)
        available_head_ = available_head_ - nozzle_assigned_heads[nozzle]

    S2.sort(key = lambda x: nozzle_points[x] / nozzle_assigned_heads[x], reverse = True)
    while available_head_ > 0:
        nozzle = S2[0]
        nozzle_assigned_heads[nozzle] += 1

        S2.remove(nozzle)
        S3.append(nozzle)
        available_head_ -= 1

    while len(S2) != 0:                     # Phase 3
        nozzle_i_val, nozzle_j_val = None, None
        nozzle_i, nozzle_j = 0, 0
        for nozzle in S2:
            if nozzle_i_val is None or nozzle_points[nozzle] / nozzle_assigned_heads[nozzle] > nozzle_i_val:
                nozzle_i_val = nozzle_points[nozzle] / nozzle_assigned_heads[nozzle]
                nozzle_i = nozzle

            if nozzle_j_val is None or nozzle_points[nozzle] / (nozzle_assigned_heads[nozzle] - 1) < nozzle_j_val:
                nozzle_j_val = nozzle_points[nozzle] / (nozzle_assigned_heads[nozzle] - 1)
                nozzle_j = nozzle

        if nozzle_points[nozzle_j] / (nozzle_assigned_heads[nozzle_j] - 1) < nozzle_points[nozzle_i] / nozzle_assigned_heads[nozzle_i]:
            nozzle_points[nozzle_j] -= 1
            nozzle_points[nozzle_i] += 1
            S2.remove(nozzle_i)
            S3.append(nozzle_i)
        else:
            break

    # nozzle assignment result:
    designated_nozzle = [[] for _ in range(max_head_index)]
    head_index = 0
    for nozzle, num in nozzle_assigned_heads.items():
        while num > 0:
            designated_nozzle[head_index] = nozzle
            head_index += 1
            num -= 1

    # === component assignment ===
    component_points, nozzle_components = {}, {}        # 元件贴装点数，吸嘴-元件对应关系
    for step in pcb_data.iterrows():
        part = step[1]['part']
        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz1']

        if part not in component_points.keys():
            component_points[part] = 0
        if nozzle not in nozzle_components.keys():
            nozzle_components[nozzle] = []

        component_points[part] += 1
        if part not in nozzle_components[nozzle]:
            nozzle_components[nozzle].append(part)

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

            CT_Group[-1][head_index] = designated_part
            CT_Points[-1][head_index] = max_points

            nozzle_components[nozzle].remove(designated_part)

    # === assign CT group to feeder slot ===
    point_num = len(pcb_data)
    component_point_pos = defaultdict(list)
    for point_cnt in range(point_num):
        part = pcb_data.loc[point_cnt, 'part']
        component_point_pos[part].append([pcb_data.loc[point_cnt, 'x'] + stopper_pos[0], pcb_data.loc[point_cnt, 'y'] + stopper_pos[1]])

    for pos_list in component_point_pos.values():
        pos_list.sort(key = lambda x: (x[0], x[1]))

    CT_Group_slot = [-1] * len(CT_Group)
    feeder_lane = [None] * max_slot_index     # 供料器基座上已分配的元件类型
    for CTIdx, pickup in enumerate(CT_Group):
        best_slot = []
        for cp_index, component in enumerate(pickup):
            if component is None:
                continue
            best_slot.append(round((sum(pos[0] for pos in component_point_pos[component]) / len(component_point_pos[component]) - slotf1_pos[
                0]) / slot_interval) + 1 - cp_index * interval_ratio)
        best_slot = round(np.mean(best_slot))

        dir, step = 0, 0  # dir: 1-向右, 0-向左
        prev_assign_available = True
        while True:
            assign_slot = best_slot + step if dir else best_slot - step
            if assign_slot + (len(pickup) - 1) * interval_ratio >= max_slot_index / 2 or assign_slot < 0:
                if not prev_assign_available:
                    raise Exception('pickup group assign error!')
                prev_assign_available = False
                dir = 1 - dir
                if dir == 0:
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

            dir = 1 - dir
            if dir == 0:
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

    # === main search ===

    # basic parameter
    crossover_rate, mutation_rate = 0.8, 0.1
    population_size, n_generations = 20, 1

    # initial solution
    population = []
    for _ in range(population_size):
        pop_permutation = list(range(len(pickup_group)))
        np.random.shuffle(pop_permutation)
        population.append(pop_permutation)

    best_individual, best_pop_val = [], float('inf')
    generation_counter = 0
    while generation_counter < n_generations:
        print('---- current generation: ' + str(generation_counter) + ' ---- ')
        # calculate fitness value
        pop_val = []
        for pop_idx, individual in enumerate(population):
            val, _, _ = cal_individual_val(component_data, component_point_pos, designated_nozzle, pickup_group,
                                              pickup_group_cycle, pair_group, feeder_lane, individual)
            pop_val.append(val)

        if min(pop_val) < best_pop_val:
            best_pop_val = min(pop_val)
            index = pop_val.index(best_pop_val)
            best_individual = copy.deepcopy(population[index])

        # min-max convert
        max_val = 1.5 * max(pop_val)
        pop_val = list(map(lambda val: max_val - val, pop_val))

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

    return convert_individual_2_result(component_data, component_point_pos, designated_nozzle, pickup_group,
                                       pickup_group_cycle, pair_group, feeder_lane, best_individual)
