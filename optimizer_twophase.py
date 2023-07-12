import copy
import random

import matplotlib.pyplot as plt

from optimizer_common import *
from collections import defaultdict
from gurobipy import *


def list_range(start, end=None):
    return list(range(start)) if end is None else list(range(start, end))


def gurobi_optimizer(pcb_data, component_data, feeder_data, initial=False, hinter=True):
    # data preparation: convert data to index
    component_list, nozzle_list = defaultdict(int), defaultdict(int)
    cpidx_2_part, nzidx_2_nozzle, cpidx_2_nzidx = {}, {}, {}

    for idx, data in component_data.iterrows():
        part, nozzle = data.part, data.nz

        cpidx_2_part[idx] = part
        nz_key = [key for key, val in nzidx_2_nozzle.items() if val == nozzle]

        nz_idx = len(nzidx_2_nozzle) if len(nz_key) == 0 else nz_key[0]
        nzidx_2_nozzle[nz_idx] = nozzle

        component_list[part] = 0
        cpidx_2_nzidx[idx] = nz_idx

    for _, data in pcb_data.iterrows():
        idx = component_data[component_data.part == data.part].index.tolist()[0]
        nozzle = component_data.loc[idx].nz

        nozzle_list[nozzle] += 1
        component_list[data.part] += 1

    part_feederbase = defaultdict(int)
    if feeder_data:
        for slot, part in feeder_data.items():
            idx = -1
            for idx, part_ in cpidx_2_part.items():
                if part == part_:
                    break
            assert idx != -1
            part_feederbase[idx] = slot  # part index - slot

    r = 1
    I, J = len(cpidx_2_part.keys()), len(nzidx_2_nozzle.keys())

    # === determine the hyper-parameter of L ===
    # first phase: calculate the number of heads for each type of nozzle
    nozzle_heads = defaultdict(int)
    for nozzle in nozzle_list.keys():
        nozzle_heads[nozzle] = 1

    while sum(nozzle_heads.values()) != max_head_index:
        max_cycle_nozzle = None

        for nozzle, head_num in nozzle_heads.items():
            if max_cycle_nozzle is None or nozzle_list[nozzle] / head_num > nozzle_list[max_cycle_nozzle] / \
                    nozzle_heads[max_cycle_nozzle]:
                max_cycle_nozzle = nozzle

        assert max_cycle_nozzle is not None
        nozzle_heads[max_cycle_nozzle] += 1

    nozzle_comp_points = defaultdict(list)
    for part, points in component_list.items():
        idx = component_data[component_data.part == part].index.tolist()[0]
        nozzle = component_data.loc[idx].nz
        nozzle_comp_points[nozzle].append([part, points])

    level = 1 if len(component_list) == 1 and len(component_list) % max_head_index == 0 else 2
    part_assignment, cycle_assignment = [], []

    def aux_func(info):
        return max(map(lambda points: max([p[1] for p in points]), info))

    def recursive_assign(assign_points, nozzle_compo_points, cur_level, total_level) -> int:
        def func(points):
            return map(lambda points: max([p[1] for p in points]), points)

        if cur_level > total_level and sum(func(nozzle_compo_points.values())) == 0:
            return 0
        elif assign_points <= 0 and cur_level == 1:
            return -1  # backtrack
        elif assign_points <= 0 or cur_level > total_level:
            return 1  # fail

        nozzle_compo_points_cpy = copy.deepcopy(nozzle_compo_points)
        prev_assign = 0
        for part in part_assignment[cur_level - 1]:
            if part != -1:
                prev_assign += 1

        head_idx = 0
        for nozzle, head in nozzle_heads.items():
            while head:
                min_idx = -1
                for idx, (part, points) in enumerate(nozzle_compo_points_cpy[nozzle]):
                    if points >= assign_points and (
                            min_idx == -1 or points < nozzle_compo_points_cpy[nozzle][min_idx][1]):
                        min_idx = idx
                part_assignment[cur_level - 1][head_idx] = -1 if min_idx == -1 else \
                    nozzle_compo_points_cpy[nozzle][min_idx][0]
                if min_idx != -1:
                    nozzle_compo_points_cpy[nozzle][min_idx][1] -= assign_points
                head -= 1
                head_idx += 1

        cycle_assignment[cur_level - 1] = assign_points
        for part in part_assignment[cur_level - 1]:
            if part != -1:
                prev_assign -= 1

        if prev_assign == 0:
            res = 1
        else:
            points = min(len(pcb_data) // max_head_index + 1, aux_func(nozzle_compo_points_cpy.values()))
            res = recursive_assign(points, nozzle_compo_points_cpy, cur_level + 1, total_level)
        if res == 0:
            return 0
        elif res == 1:
            # All cycles have been completed, but there are still points left to be allocated
            return recursive_assign(assign_points - 1, nozzle_compo_points, cur_level, total_level)

    # second phase: (greedy) recursive search to assign points for each cycle set and obtain an initial solution
    while True:
        part_assignment = [[-1 for _ in range(max_head_index)] for _ in range(level)]
        cycle_assignment = [-1 for _ in range(level)]
        points = min(len(pcb_data) // max_head_index + 1, max(component_list.values()))
        if recursive_assign(points, nozzle_comp_points, 1, level) == 0:
            break
        level += 1

    weight_cycle, weight_nz_change, weight_pick = 2, 3, 1

    L = len(cycle_assignment)
    S = r * I  # the available feeder num
    M = len(pcb_data)  # a sufficiently large number (number of placement points)
    HC = [[0 for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            HC[i][j] = 1 if cpidx_2_nzidx[i] == j else 0

    mdl = Model('SMT')
    mdl.setParam('Seed', 0)
    mdl.setParam('OutputFlag', hinter)  # set whether output the debug information
    mdl.setParam('TimeLimit', 600)
    # mdl.setParam('MIPFocus', 2)
    # mdl.setParam("Heuristics", 0.5)

    # Use only if other methods, including exploring the tree with the default settings, do not yield a viable solution
    # mdl.setParam("ZeroObjNodes", 100)

    # === Decision Variables ===
    y = mdl.addVars(list_range(I), list_range(max_head_index), list_range(L), vtype=GRB.BINARY, name='y')
    w = mdl.addVars(list_range(S), list_range(max_head_index), list_range(L), vtype=GRB.BINARY, name='w')

    c = mdl.addVars(list_range(I), list_range(max_head_index), list_range(L), vtype=GRB.INTEGER,
                    ub=len(pcb_data) // max_head_index + 1, name='c')

    # todo: the condition for upper limits of feeders exceed 1
    f = {}
    for i in range(I):
        for s in range(S):
            if i in part_feederbase.keys():
                f[s, i] = 1 if part_feederbase[i] == s // r else 0
            else:
                f[s, i] = mdl.addVar(vtype=GRB.BINARY, name='f_' + str(s) + '_' + str(i))

    u = mdl.addVars(list_range(L), vtype=GRB.INTEGER)
    p = mdl.addVars(list_range(-(max_head_index - 1) * r, S), list_range(L), vtype=GRB.BINARY, name='p')
    z = mdl.addVars(list_range(J), list_range(max_head_index), list_range(L), vtype=GRB.BINARY)

    d = mdl.addVars(list_range(L - 1), list_range(max_head_index), vtype=GRB.CONTINUOUS, name='d')
    d_plus = mdl.addVars(list_range(J), list_range(max_head_index), list_range(L - 1), vtype=GRB.CONTINUOUS,
                         name='d_plus')
    d_minus = mdl.addVars(list_range(J), list_range(max_head_index), list_range(L - 1), vtype=GRB.CONTINUOUS,
                          name='d_minus')

    PU = mdl.addVars(list_range(-(max_head_index - 1) * r, S), list_range(L), vtype=GRB.INTEGER, name='PU')
    WL = mdl.addVars(list_range(L), vtype=GRB.INTEGER, ub=len(pcb_data) // max_head_index + 1, name='WL')
    NC = mdl.addVars(list_range(max_head_index), vtype=GRB.CONTINUOUS, name='NC')

    part_2_cpidx = defaultdict(int)
    for idx, part in cpidx_2_part.items():
        part_2_cpidx[part] = idx
    if initial:
        # initial some variables to speed up the search process
        part_list = []
        for cycle_part in part_assignment:
            for part in cycle_part:
                if part != -1 and part not in part_list:
                    part_list.append(part)
        slot = 0
        for part in part_list:
            if feeder_data:
                while slot in feeder_data.keys():
                    slot += 1  # skip assigned feeder slot

            if part_2_cpidx[part] in part_feederbase.keys():
                continue

            part_feederbase[part_2_cpidx[part]] = slot
            slot += 1

        # ensure the priority of the workload assignment
        cycle_index = sorted(range(len(cycle_assignment)), key=lambda k: cycle_assignment[k], reverse=True)
        for idx, cycle in enumerate(cycle_index):
            WL[idx].Start = cycle_assignment[cycle]

    # === Objective ===
    mdl.setObjective(weight_cycle * quicksum(WL[l] for l in range(L)) + weight_nz_change * quicksum(
        NC[h] for h in range(max_head_index)) + weight_pick * quicksum(
        PU[s, l] for s in range(-(max_head_index - 1) * r, S) for l in range(L)) + 0.001 * quicksum(
        u[l] for l in range(L)))

    mdl.addConstrs(u[l] >= s1 * p[s1, l] - s2 * p[s2, l] for s1 in range(-(max_head_index - 1) * r, S) for s2 in
                   range(-(max_head_index - 1) * r, S) for l in range(L))

    # === Constraint ===
    # work completion
    mdl.addConstrs(c[i, h, l] == WL[l] * y[i, h, l] for i in range(I) for h in range(max_head_index) for l in range(L))
    mdl.addConstrs(
        quicksum(c[i, h, l] for h in range(max_head_index) for l in range(L)) == component_list[cpidx_2_part[i]] for i
        in range(I))

    # variable constraint
    mdl.addConstrs(
        quicksum(y[i, h, l] * w[s, h, l] for i in range(I) for s in range(S)) <= 1 for h in range(max_head_index) for l
        in range(L))

    mdl.addConstrs(
        quicksum(WL[l] * y[i, h, l] for h in range(max_head_index) for l in range(L)) == component_list[cpidx_2_part[i]]
        for i in range(I))

    # simultaneous pick
    for s in range(-(max_head_index - 1) * r, S):
        rng = list(range(max(0, -math.floor(s / r)), min(max_head_index, math.ceil((S - s) / r))))
        for l in range(L):
            mdl.addConstr(quicksum(w[s + h * r, h, l] for h in rng) <= M * p[s, l])
            mdl.addConstr(quicksum(w[s + h * r, h, l] for h in rng) >= p[s, l])

    mdl.addConstrs(PU[s, l] == p[s, l] * WL[l] for s in range(-(max_head_index - 1) * r, S) for l in range(L))
    # nozzle change
    mdl.addConstrs(
        z[j, h, l] - z[j, h, l + 1] == d_plus[j, h, l] - d_minus[j, h, l] for l in range(L - 1) for j in range(J) for h
        in range(max_head_index))

    mdl.addConstrs(
        2 * d[l, h] == quicksum(d_plus[j, h, l] for j in range(J)) + quicksum(d_minus[j, h, l] for j in range(J)) for l
        in range(L - 1) for h in range(max_head_index))

    mdl.addConstrs(NC[h] == quicksum(d[l, h] for l in range(L - 1)) for h in range(max_head_index))

    # nozzle-component compatibility
    mdl.addConstrs(
        y[i, h, l] <= quicksum(HC[i][j] * z[j, h, l] for j in range(J)) for i in range(I) for h in range(max_head_index)
        for l in range(L))

    # available number of feeder
    mdl.addConstrs(quicksum(f[s, i] for s in range(S)) <= 1 for i in range(I))

    # available number of nozzle
    mdl.addConstrs(quicksum(z[j, h, l] for h in range(max_head_index)) <= max_head_index for j in range(J) for l in range(L))

    # upper limit for occupation for feeder slot
    mdl.addConstrs(quicksum(f[s, i] for i in range(I)) <= 1 for s in range(S))
    mdl.addConstrs(
        quicksum(w[s, h, l] for s in range(S)) >= quicksum(y[i, h, l] for i in range(I)) for h in range(max_head_index)
        for l in range(L))

    # others
    mdl.addConstrs(quicksum(z[j, h, l] for j in range(J)) <= 1 for h in range(max_head_index) for l in range(L))
    mdl.addConstrs(
        quicksum(y[i, h, l] * w[s, h, l] for h in range(max_head_index) for l in range(L)) >= f[s, i] for i in range(I)
        for s in range(S))
    mdl.addConstrs(
        quicksum(y[i, h, l] * w[s, h, l] for h in range(max_head_index) for l in range(L)) <= M * f[s, i] for i in
        range(I) for s in range(S))

    # the constraints to speed up the search process
    mdl.addConstrs(quicksum(z[j, h, l] for j in range(J) for h in range(max_head_index)) >= quicksum(
        z[j, h, l + 1] for j in range(J) for h in range(max_head_index)) for l in range(L - 1))

    mdl.addConstr(quicksum(WL[l] for l in range(L)) >= math.ceil(len(pcb_data) // max_head_index))
    mdl.addConstrs(WL[l] >= WL[l + 1] for l in range(L - 1))

    # === search process ===
    mdl.update()
    # mdl.write('mdl.lp')
    print('num of constrs: ', str(len(mdl.getConstrs())), ', num of vars: ', str(len(mdl.getVars())))

    mdl.Params.PoolSearchMode = 2
    mdl.Params.PoolSolutions = 30
    mdl.Params.PoolGap = 1e-4
    mdl.optimize()

    # === result generation ===
    nozzle_assign, component_assign = [], []
    feeder_assign, cycle_assign = [], []
    if mdl.Status == GRB.OPTIMAL or mdl.Status == GRB.INTERRUPTED or GRB.TIME_LIMIT:
        # === selection from solution pool ===
        component_pos, component_avg_pos = defaultdict(list), defaultdict(list)
        for _, data in pcb_data.iterrows():
            component_index = component_data[component_data.part == data.part].index.tolist()[0]
            component_pos[component_index].append([data.x, data.y])

        for i in component_pos.keys():
            component_pos[i] = sorted(component_pos[i], key=lambda pos: (pos[0], pos[1]))
            component_avg_pos[i] = [sum(map(lambda pos: pos[0], component_pos[i])) / len(component_pos[i]),
                                    sum(map(lambda pos: pos[1], component_pos[i])) / len(component_pos[i])]

        min_dist, solution_number = None, -1
        for sol_counter in range(mdl.SolCount):
            mdl.Params.SolutionNumber = sol_counter
            pos_counter = defaultdict(int)

            dist = 0
            cycle_placement, cycle_points = defaultdict(list), defaultdict(list)
            for l in range(L):
                if abs(WL[l].Xn) <= 1e-10:
                    continue
                cycle_placement[l], cycle_points[l] = [-1] * max_head_index, [None] * max_head_index

            for h in range(max_head_index):
                for l in range(L):
                    if abs(WL[l].Xn) <= 1e-10:
                        continue

                    pos_list = []

                    for i in range(I):
                        if abs(y[i, h, l].Xn) <= 1e-10:
                            continue

                        for _ in range(round(WL[l].Xn)):
                            pos_list.append(component_pos[i][pos_counter[i]])
                            pos_counter[i] += 1

                        cycle_placement[l][h] = i
                        cycle_points[l][h] = [sum(map(lambda pos: pos[0], pos_list)) / len(pos_list),
                                              sum(map(lambda pos: pos[1], pos_list)) / len(pos_list)]
            for l in range(L):
                if abs(WL[l].Xn) <= 1e-10:
                    continue
                dist += dynamic_programming_cycle_path(cycle_placement[l], cycle_points[l])[0]

            if min_dist is None or dist < min_dist:
                min_dist = dist
                solution_number = sol_counter

        mdl.Params.SolutionNumber = solution_number
        # === 更新吸嘴、元件、周期数优化结果 ===
        for l in range(L):
            if abs(WL[l].Xn) <= 1e-10:
                continue

            nozzle_assign.append([-1 for _ in range(max_head_index)])
            component_assign.append([-1 for _ in range(max_head_index)])
            feeder_assign.append([-1 for _ in range(max_head_index)])

            cycle_assign.append(round(WL[l].Xn))

            for h in range(max_head_index):
                for i in range(I):
                    if abs(y[i, h, l].Xn - 1) < 1e-10:
                        component_assign[-1][h] = i

                        for j in range(J):
                            if HC[i][j]:
                                nozzle_assign[-1][h] = j
                for s in range(S):
                    if abs(w[s, h, l].Xn - 1) < 1e-10 and component_assign[l][h] != -1:
                        feeder_assign[l][h] = s * interval_ratio

        # === 更新供料器分配结果 ==
        component_head = defaultdict(int)
        for i in range(I):
            cycle_num = 0
            for l, component_cycle in enumerate(component_assign):
                for head, component in enumerate(component_cycle):
                    if component == i:
                        component_head[i] += cycle_assign[l] * head
                        cycle_num += cycle_assign[l]
            component_head[i] /= cycle_num      # 不同元件的加权拾取贴装头

        average_pos = 0
        for _, data in pcb_data.iterrows():
            average_pos += (data.x - component_head[part_2_cpidx[data.part]] * head_interval)

        average_pos /= len(pcb_data)    # 实际贴装位置的加权平均
        average_slot = 0
        for l in range(L):
            if abs(WL[l].Xn) <= 1e-10:
                continue
            cycle_min_slot, cycle_max_slot = None, None
            for head in range(max_head_index):
                if abs(WL[l].Xn) <= 1e-10:
                    continue
                if feeder_assign[l][head] == -1:
                    continue
                slot = feeder_assign[l][head] - head * interval_ratio
                if cycle_min_slot is None or slot < cycle_min_slot:
                    cycle_min_slot = slot
                if cycle_max_slot is None or slot > cycle_max_slot:
                    cycle_max_slot = slot
            average_slot += (cycle_max_slot - cycle_min_slot) * cycle_assign[l]
        average_slot /= sum(cycle_assign)
        start_slot = round((average_pos + stopper_pos[0] - slotf1_pos[0]) / slot_interval + average_slot / 2) + 1

        for l in range(L):
            if abs(WL[l].Xn) <= 1e-10:
                continue

            for h in range(max_head_index):
                for s in range(S):
                    if abs(w[s, h, l].Xn - 1) < 1e-10 and component_assign[l][h] != -1:
                        feeder_assign[l][h] = start_slot + s * interval_ratio

        if hinter:
            print('total cost = {}'.format(mdl.objval))
            print('cycle = {}, nozzle change = {}, pick up = {}'.format(quicksum(WL[l].Xn for l in range(L)), quicksum(
                NC[h].Xn for h in range(max_head_index)), quicksum(
                PU[s, l].Xn for s in range(-(max_head_index - 1) * r, S) for l in range(L))))

            print('workload: ')
            for l in range(L):
                print(WL[l].Xn, end=', ')

            print('')
            print('result')
            print('nozzle assignment: ', nozzle_assign)
            print('component assignment: ', component_assign)
            print('feeder assignment: ', feeder_assign)
            print('cycle assignment: ', cycle_assign)

    return component_assign, feeder_assign, cycle_assign


@timer_wrapper
def scan_based_placement_route_generation(component_data, pcb_data, component_assign, cycle_assign):
    placement_result, head_sequence_result = [], []

    mount_point_pos, mount_point_index, mount_point_angle, mount_point_part = [], [], [], []
    for i, data in pcb_data.iterrows():
        component_index = component_data[component_data.part == data.part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index.append(i)
        mount_point_pos.append([data.x + stopper_pos[0], data.y + stopper_pos[1]])
        mount_point_angle.append(data.r)

        mount_point_part.append(component_index)

    lBoundary, rBoundary = min(mount_point_pos, key=lambda x: x[0])[0], max(mount_point_pos, key=lambda x: x[0])[0]
    search_step = max((rBoundary - lBoundary) / max_head_index / 2, 0)

    ref_pos_y = min(mount_point_pos, key=lambda x: x[1])[1]
    for cycle_index, component_cycle in enumerate(component_assign):
        for _ in range(cycle_assign[cycle_index]):
            min_dist = None
            tmp_assigned_placement, tmp_assigned_head_seq = [], []
            tmp_mount_point_pos, tmp_mount_point_index = [], []
            for search_dir in range(3):     # 不同的搜索方向，贴装头和起始点的选取方法各不相同
                if search_dir == 0:
                    # 从左向右搜索
                    searchPoints = np.arange(lBoundary, (lBoundary + rBoundary) / 2, search_step)
                    head_range = list(range(max_head_index))
                elif search_dir == 1:
                    # 从右向左搜索
                    searchPoints = np.arange(rBoundary + 1e-3, (lBoundary + rBoundary) / 2, -search_step)
                    head_range = list(range(max_head_index - 1, -1, -1))
                else:
                    # 从中间向两边搜索
                    searchPoints = np.arange(lBoundary, rBoundary, search_step / 2)
                    head_range, head_index = [], (max_head_index - 1) // 2
                    while head_index >= 0:
                        if 2 * head_index != max_head_index - 1:
                            head_range.append(max_head_index - 1 - head_index)
                        head_range.append(head_index)
                        head_index -= 1

                for startPoint in searchPoints:
                    mount_point_pos_cpy, mount_point_index_cpy = copy.deepcopy(mount_point_pos), copy.deepcopy(
                        mount_point_index)
                    mount_point_angle_cpy = copy.deepcopy(mount_point_angle)

                    assigned_placement = [-1] * max_head_index
                    assigned_mount_point = [[0, 0]] * max_head_index
                    assigned_mount_angle = [0] * max_head_index
                    head_counter, point_index = 0, -1
                    for head_index in head_range:
                        if head_counter == 0:
                            component_index = component_assign[cycle_index][head_index]

                            if component_index == -1:
                                continue

                            min_horizontal_distance = None
                            for index, mount_index in enumerate(mount_point_index_cpy):
                                if mount_point_part[mount_index] != component_index:
                                    continue
                                horizontal_distance = abs(mount_point_pos_cpy[index][0] - startPoint) + 1e-3 * abs(
                                    mount_point_pos_cpy[index][1] - ref_pos_y)

                                if min_horizontal_distance is None or horizontal_distance < min_horizontal_distance:
                                    min_horizontal_distance = horizontal_distance
                                    point_index = index
                        else:
                            point_index = -1
                            min_cheby_distance = None

                            next_comp_index = component_assign[cycle_index][head_index]
                            if assigned_placement[head_index] != -1 or next_comp_index == -1:
                                continue
                            for index, mount_index in enumerate(mount_point_index_cpy):
                                if mount_point_part[mount_index] != next_comp_index:
                                    continue

                                point_pos = [[mount_point_pos_cpy[index][0] - head_index * head_interval,
                                              mount_point_pos_cpy[index][1]]]

                                cheby_distance, euler_distance = 0, 0
                                for next_head in range(max_head_index):
                                    if assigned_placement[next_head] == -1:
                                        continue
                                    point_pos.append(assigned_mount_point[next_head].copy())
                                    point_pos[-1][0] -= next_head * head_interval

                                point_pos = sorted(point_pos, key=lambda x: x[0])
                                for mount_seq in range(len(point_pos) - 1):
                                    cheby_distance += max(abs(point_pos[mount_seq][0] - point_pos[mount_seq + 1][0]),
                                                          abs(point_pos[mount_seq][1] - point_pos[mount_seq + 1][1]))
                                    euler_distance += math.sqrt(
                                        (point_pos[mount_seq][0] - point_pos[mount_seq + 1][0]) ** 2 + (
                                                    point_pos[mount_seq][1] - point_pos[mount_seq + 1][1]) ** 2)

                                cheby_distance += 0.01 * euler_distance
                                if min_cheby_distance is None or cheby_distance < min_cheby_distance:
                                    min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                    point_index = index

                        if point_index == -1:
                            continue

                        head_counter += 1

                        assigned_placement[head_index] = mount_point_index_cpy[point_index]
                        assigned_mount_point[head_index] = mount_point_pos_cpy[point_index].copy()
                        assigned_mount_angle[head_index] = mount_point_angle_cpy[point_index]

                        mount_point_index_cpy.pop(point_index)
                        mount_point_pos_cpy.pop(point_index)
                        mount_point_angle_cpy.pop(point_index)

                    dist, head_seq = dynamic_programming_cycle_path(assigned_placement, assigned_mount_point,
                                                                    assigned_mount_angle)

                    if min_dist is None or dist < min_dist:
                        tmp_mount_point_pos, tmp_mount_point_index = mount_point_pos_cpy, mount_point_index_cpy
                        tmp_assigned_placement, tmp_assigned_head_seq = assigned_placement, head_seq
                        min_dist = dist

            mount_point_pos, mount_point_index = tmp_mount_point_pos, tmp_mount_point_index

            placement_result.append(tmp_assigned_placement)
            head_sequence_result.append(tmp_assigned_head_seq)

    # return placement_result, head_sequence_result
    return placement_route_relink_heuristic(component_data, pcb_data, placement_result, head_sequence_result)


def placement_route_relink_heuristic(component_data, pcb_data, placement_result, head_sequence_result):
    mount_point_pos, mount_point_angle, mount_point_index, mount_point_part = [], [], [], []
    for i, data in pcb_data.iterrows():
        component_index = component_data[component_data.part == data.part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index.append(i)
        mount_point_pos.append([data.x + stopper_pos[0], data.y + stopper_pos[1]])
        mount_point_angle.append(data.r)

        mount_point_part.append(component_index)

    cycle_length, cycle_average_pos = [], []
    for cycle, placement in enumerate(placement_result):
        prev_pos, prev_angle = None, None
        cycle_pos_list = []
        cycle_length.append(0)
        for idx, head in enumerate(head_sequence_result[cycle]):
            point_index = placement[head]
            if point_index == -1:
                continue
            pos = [mount_point_pos[point_index][0] - head * head_interval, mount_point_pos[point_index][1]]
            angle = mount_point_angle[point_index]
            cycle_pos_list.append(pos)
            if prev_pos is not None:
                if head_sequence_result[cycle][idx - 1] // 2 == head_sequence_result[cycle][idx] // 2:  # 同轴
                    rotary_angle = prev_angle - angle
                else:
                    rotary_angle = 0

                cycle_length[-1] += max(axis_moving_time(prev_pos[0] - pos[0], 0),
                                        axis_moving_time(prev_pos[1] - pos[1], 1), head_rotary_time(rotary_angle))
            prev_pos, prev_angle = pos, angle

        cycle_average_pos.append([sum(map(lambda pos: pos[0], cycle_pos_list)) / len(cycle_pos_list),
                                  sum(map(lambda pos: pos[1], cycle_pos_list)) / len(cycle_pos_list)])

    best_placement_result, best_head_sequence_result = copy.deepcopy(placement_result), copy.deepcopy(
        head_sequence_result)

    best_cycle_length, best_cycle_average_pos = copy.deepcopy(cycle_length), copy.deepcopy(cycle_average_pos)

    n_runningtime, n_iteration = 10, 0
    start_time = time.time()
    with tqdm(total=n_runningtime) as pbar:
        pbar.set_description('swap heuristic process')
        prev_time = start_time
        while True:
            n_iteration += 1

            placement_result, head_sequence_result = copy.deepcopy(best_placement_result), copy.deepcopy(
                best_head_sequence_result)
            cycle_length = best_cycle_length.copy()
            cycle_average_pos = copy.deepcopy(best_cycle_average_pos)

            cycle_index = roulette_wheel_selection(cycle_length)  # 根据周期加权移动距离随机选择周期

            point_dist = []     # 周期内各贴装点距离中心位置的切氏距离
            for head in head_sequence_result[cycle_index]:
                point_index = placement_result[cycle_index][head]
                _delta_x = abs(mount_point_pos[point_index][0] - head * head_interval - cycle_average_pos[cycle_index][0])
                _delta_y = abs(mount_point_pos[point_index][1] - cycle_average_pos[cycle_index][1])
                point_dist.append(max(_delta_x, _delta_y))

            # 随机选择一个异常点
            head_index = head_sequence_result[cycle_index][roulette_wheel_selection(point_dist)]
            point_index = placement_result[cycle_index][head_index]

            # 找距离该异常点最近的周期
            min_dist = None
            chg_cycle_index = -1
            for idx in range(len(cycle_average_pos)):
                if idx == cycle_index:
                    continue
                dist_ = 0
                component_type_check = False
                for head in head_sequence_result[idx]:
                    dist_ += max(abs(mount_point_pos[placement_result[idx][head]][0] - mount_point_pos[point_index][0]),
                                 abs(mount_point_pos[placement_result[idx][head]][1] - mount_point_pos[point_index][1]))
                    if mount_point_part[placement_result[idx][head]] == mount_point_part[point_index]:
                        component_type_check = True

                if (min_dist is None or dist_ < min_dist) and component_type_check:
                    min_dist = dist_
                    chg_cycle_index = idx

            assert chg_cycle_index != -1

            chg_head, min_chg_dist = None, None
            chg_cycle_point = []
            for head in head_sequence_result[chg_cycle_index]:
                index = placement_result[chg_cycle_index][head]
                chg_cycle_point.append([mount_point_pos[index][0] - head * head_interval, mount_point_pos[index][1]])

            for idx, head in enumerate(head_sequence_result[chg_cycle_index]):
                chg_cycle_point_cpy = copy.deepcopy(chg_cycle_point)
                index = placement_result[chg_cycle_index][head]
                if mount_point_part[index] != mount_point_part[point_index]:
                    continue
                chg_cycle_point_cpy[idx][0] = (mount_point_pos[index][0]) - head * head_interval

                chg_dist = 0
                aver_chg_pos = [sum(map(lambda x: x[0], chg_cycle_point_cpy)) / len(chg_cycle_point_cpy),
                                sum(map(lambda x: x[1], chg_cycle_point_cpy)) / len(chg_cycle_point_cpy)]

                for pos in chg_cycle_point_cpy:
                    chg_dist += max(abs(aver_chg_pos[0] - pos[0]), abs(aver_chg_pos[1] - pos[1]))

                # 更换后各点距离中心更近
                if min_chg_dist is None or chg_dist < min_chg_dist:
                    chg_head = head
                    min_chg_dist = chg_dist

            assert chg_head is not None

            # === 第一轮，变更周期chg_cycle_index的贴装点重排 ===
            chg_placement_res = placement_result[chg_cycle_index].copy()
            chg_placement_res[chg_head] = point_index

            cycle_point_list = defaultdict(list)
            for head, point in enumerate(chg_placement_res):
                if point == -1:
                    continue
                cycle_point_list[mount_point_part[point]].append(point)

            for key, point_list in cycle_point_list.items():
                cycle_point_list[key] = sorted(point_list, key=lambda p: mount_point_pos[p][0])

            chg_placement_res, chg_point_assign_res = [], [[0, 0]] * max_head_index
            chg_angle_res = [0] * max_head_index
            for head, point_index in enumerate(placement_result[chg_cycle_index]):
                if point_index == -1:
                    chg_placement_res.append(-1)
                else:
                    part = mount_point_part[point_index]
                    chg_placement_res.append(cycle_point_list[part][0])
                    chg_point_assign_res[head] = mount_point_pos[cycle_point_list[part][0]].copy()
                    chg_angle_res[head] = mount_point_angle[cycle_point_list[part][0]]
                    cycle_point_list[part].pop(0)

            chg_place_moving, chg_head_res = dynamic_programming_cycle_path(chg_placement_res, chg_point_assign_res, chg_angle_res)

            # === 第二轮，原始周期cycle_index的贴装点重排 ===
            placement_res = placement_result[cycle_index].copy()
            placement_res[head_index] = placement_result[chg_cycle_index][chg_head]

            for point in placement_res:
                if point == -1:
                    continue
                cycle_point_list[mount_point_part[point]].append(point)

            for key, point_list in cycle_point_list.items():
                cycle_point_list[key] = sorted(point_list, key=lambda p: mount_point_pos[p][0])

            placement_res, point_assign_res = [], [[0, 0]] * max_head_index
            angle_assign_res = [0] * max_head_index
            for head, point_index in enumerate(placement_result[cycle_index]):
                if point_index == -1:
                    placement_res.append(-1)
                else:
                    part = mount_point_part[point_index]
                    placement_res.append(cycle_point_list[part][0])
                    point_assign_res[head] = mount_point_pos[cycle_point_list[part][0]].copy()
                    angle_assign_res[head] = mount_point_angle[cycle_point_list[part][0]]
                    cycle_point_list[part].pop(0)

            place_moving, place_head_res = dynamic_programming_cycle_path(placement_res, point_assign_res, angle_assign_res)

            # 更新贴装顺序分配结果
            placement_result[cycle_index], head_sequence_result[cycle_index] = placement_res, place_head_res
            placement_result[chg_cycle_index], head_sequence_result[chg_cycle_index] = chg_placement_res, chg_head_res

            # 更新移动路径
            cycle_length[cycle_index], cycle_length[chg_cycle_index] = place_moving, chg_place_moving

            # 更新平均坐标和最大偏离点索引
            point_list, point_index_list = [], []
            for head in head_sequence_result[cycle_index]:
                point_index_list.append(placement_result[cycle_index][head])
                point_pos = mount_point_pos[point_index_list[-1]].copy()
                point_pos[0] -= head * head_interval
                point_list.append(point_pos)

            cycle_average_pos[cycle_index] = [sum(map(lambda x: x[0], point_list)) / len(point_list),
                                              sum(map(lambda x: x[1], point_list)) / len(point_list)]

            point_list, point_index_list = [], []
            for head in head_sequence_result[chg_cycle_index]:
                point_index_list.append(placement_result[chg_cycle_index][head])
                point_pos = mount_point_pos[point_index_list[-1]].copy()
                point_pos[0] -= head * head_interval
                point_list.append(point_pos)

            cycle_average_pos[chg_cycle_index] = [sum(map(lambda x: x[0], point_list)) / len(point_list),
                                                  sum(map(lambda x: x[1], point_list)) / len(point_list)]

            if sum(cycle_length) < sum(best_cycle_length):
                best_cycle_length = cycle_length.copy()
                best_cycle_average_pos = copy.deepcopy(cycle_average_pos)
                best_placement_result, best_head_sequence_result = copy.deepcopy(placement_result), copy.deepcopy(
                    head_sequence_result)

            cur_time = time.time()
            if cur_time - start_time > n_runningtime:
                break

            pbar.update(cur_time - prev_time)
            prev_time = cur_time

    print("number of iteration: ", n_iteration)
    return best_placement_result, best_head_sequence_result