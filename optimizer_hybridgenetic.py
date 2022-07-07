import numpy as np
import pandas as pd

from common_function import *


@timer_warper
def optimizer_hybridgenetic(pcb_data, component_data):
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

    assert (len(nozzle_points) <= max_head_index, "nozzle type number should no more than the head num")
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
    component_pos = {}
    for point_cnt in range(point_num):
        part = pcb_data.loc[point_cnt, 'part']
        if part not in component_pos:
            component_pos[part] = []
        component_pos[part].append(pcb_data.loc[point_cnt, 'x'] + stopper_pos[0])

    CT_Group_slot = [-1] * len(CT_Group)
    feeder_lane_state = [0] * max_slot_index  # 0表示空，1表示已占有
    for index, pickup in enumerate(CT_Group):
        best_slot = []
        for cp_index, component in enumerate(pickup):
            if component is None:
                continue
            best_slot.append(round((sum(component_pos[component]) / len(component_pos[component]) - slotf1_pos[
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
            slot_set = []
            for slot in range(assign_slot, assign_slot + interval_ratio * len(pickup), interval_ratio):
                pickup_index = int((slot - assign_slot) / interval_ratio)
                if feeder_lane_state[slot] == 1 and pickup[pickup_index] is not None:
                    assign_available = False
                    break
                if pickup[pickup_index] is not None:
                    slot_set.append(slot)

            if assign_available:
                for slot in slot_set:
                    feeder_lane_state[slot] = 1
                CT_Group_slot[index] = slot_set[0]
                break

            dir = 1 - dir
            if dir == 0:
                step += 1

    # === Initial Pickup Group ===
    InitialPickup = []
    for index, CT in enumerate(CT_Group):
        while True:
            if CT_Points[index].count(0) == max_head_index:
                break
            min_element = min([Points for Points in CT_Points[index] if Points > 0])

            InitialPickup.append(copy.deepcopy(CT_Group[index]))
            for head in range(max_head_index):
                if CT_Points[index][head] >= min_element:
                    CT_Points[index][head] -= min_element
                if CT_Points[index][head] == 0:
                    CT_Group[index][head] = None

    # pickup partition rule
    partition_probability = 0.1
    pickup_group, pair_group = [], []        # pair_group:  pickups from same initial group
    for Pickup in InitialPickup:
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
            pair_group.append(pair_index)
        else:
            pickup_group.append(Pickup)

    # === main search ===

    # basic parameter
    place_time, pick_time = 0.234, 0.4
    moving_speed = 300  # mm/s
    crossover_rate, mutation_rate = 0.8, 0.1
    population_size, n_generations = 200, 500

    # initial solution
    population = []
    for _ in range(population_size):
        pop_permutation = list(range(len(pickup_group)))
        np.random.shuffle(pop_permutation)
        population.append(pop_permutation)

    while n_generations >= 0:
        # calculate fitness value
        pop_val = []
        for pop_idx, individual in enumerate(population):

            pickup_group_cpy = copy.deepcopy(pickup_group)

            pop_val.append(0)

        # selection


        pass

    print('')