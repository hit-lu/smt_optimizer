from optimizer_common import *
from collections import defaultdict
from gurobipy import *


def list_range(start, end=None):
    return list(range(start)) if end is None else list(range(start, end))


def gurobi_optimizer(pcb_data, component_data, feeder_data, initial=False, hinter=True):
    # data preparation: convert data to index
    component_list, nozzle_list = defaultdict(int), defaultdict(int)
    cpidx_2_part, nzidx_2_nozzle, cpidx_2_nzidx = {}, {}, {}

    average_pos = sum(map(lambda data: data[1]['x'], pcb_data.iterrows())) / len(pcb_data)
    slot_start = int(round(average_pos / len(pcb_data) + stopper_pos[0] - slotf1_pos[0]) / slot_interval) + 1

    for idx, data in component_data.iterrows():
        part, nozzle = data['part'], data['nz']

        cpidx_2_part[idx] = part
        nz_key = [key for key, val in nzidx_2_nozzle.items() if val == nozzle]

        nz_idx = len(nzidx_2_nozzle) if len(nz_key) == 0 else nz_key[0]
        nzidx_2_nozzle[nz_idx] = nozzle

        component_list[part] = 0
        cpidx_2_nzidx[idx] = nz_idx

    for _, data in pcb_data.iterrows():
        part = data['part']

        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz']

        nozzle_list[nozzle] += 1
        component_list[part] += 1

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
        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz']
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

    weight_cycle, weight_nz_change, weight_pick = 2, 6, 1

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
    mdl.setParam('TimeLimit', 3600)
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
    PT = mdl.addVars(list_range(L), vtype=GRB.BINARY, name='PT')  # pick-and-place task

    if initial:
        # initial some variables to speed up the search process
        part_2_cpidx = defaultdict(int)
        for idx, part in cpidx_2_part.items():
            part_2_cpidx[part] = idx

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
            # for h in range(max_head_index):
            #     part = part_assignment[cycle][h]
            #     if part == -1:
            #         continue
            #     slot = part_feederbase[part_2_cpidx[part]]
            #     x[part_2_cpidx[part], slot, h, idx].Start = 1
            #     if type(f[slot, part_2_cpidx[part]]) == gurobipy.Var:
            #         f[slot, part_2_cpidx[part]].Start = 1

    # === Objective ===
    mdl.setObjective(weight_cycle * quicksum(WL[l] for l in range(L)) + weight_nz_change * quicksum(
        NC[h] for h in range(max_head_index)) + weight_pick * quicksum(
        PU[s, l] for s in range(-(max_head_index - 1) * r, S) for l in range(L)) + 0.01 * quicksum(
        PT[l]  for l in range(L)))

    # === Constraint ===
    # work completion
    mdl.addConstrs(c[i, h, l] == WL[l] * y[i, h, l] for i in range(I) for h in range(max_head_index) for l in range(L))
    mdl.addConstrs(
        quicksum(c[i, h, l] for h in range(max_head_index) for l in range(L)) == component_list[cpidx_2_part[i]] for i
        in range(I))

    mdl.addConstrs(WL[l] <= M * PT[l] for l in range(L))

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
    mdl.optimize()

    # === result generation ===
    nozzle_assign, component_assign = [], []
    feeder_assign, cycle_assign = [], []

    if mdl.Status == GRB.OPTIMAL or mdl.Status == GRB.INTERRUPTED or GRB.TIME_LIMIT:
        for l in range(L):
            if abs(WL[l].x) <= 1e-10:
                continue

            nozzle_assign.append([-1 for _ in range(max_head_index)])
            component_assign.append([-1 for _ in range(max_head_index)])
            feeder_assign.append([-1 for _ in range(max_head_index)])
            cycle_assign.append(int(WL[l].x))

            for h in range(max_head_index):
                for i in range(I):
                    if abs(y[i, h, l].x - 1) < 1e-10:
                        component_assign[-1][h] = i

                        for j in range(J):
                            if HC[i][j]:
                                nozzle_assign[-1][h] = j

                for s in range(S):
                    if abs(w[s, h, l].x - 1) < 1e-10 and component_assign[-1][h] != -1:
                        feeder_assign[-1][h] = slot_start + s * interval_ratio - 1

        if hinter:
            print('total cost = {}'.format(mdl.objval))
            print('cycle = {}, nozzle change = {}, pick up = {}'.format(quicksum(WL[l].x for l in range(L)), quicksum(
                NC[h].x for h in range(max_head_index)), quicksum(
                PU[s, l].x for s in range(-(max_head_index - 1) * r, S) for l in range(L))))

            print('workload: ')
            for l in range(L):
                print(WL[l].x, end=', ')

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

    mount_point_pos, mount_point_index, mount_point_part = [], [], []
    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index.append(i)
        mount_point_pos.append([pcb_data.loc[i]['x'] + stopper_pos[0], pcb_data.loc[i]['y'] + stopper_pos[1]])

        mount_point_part.append(component_index)

    lBoundary, rBoundary = min(mount_point_pos, key=lambda x: x[0])[0], max(mount_point_pos, key=lambda x: x[0])[0]
    search_step = max((rBoundary - lBoundary) / max_head_index / 2, 10)

    min_dist = None
    for search_dir in range(3):
        if search_dir == 0:
            # 从中间向两边搜索
            searchPoints = np.arange((3 * lBoundary + rBoundary) / 4, (3 * rBoundary + lBoundary) / 4, search_step)
            head_range, head_index = [], (max_head_index - 1) // 2
            while head_index >= 0:
                if 2 * head_index != max_head_index - 1:
                    head_range.append(max_head_index - 1 - head_index)
                head_range.append(head_index)
                head_index -= 1
        elif search_dir == 1:
            # 从右向左搜索
            searchPoints = np.arange(rBoundary + 1e-3, (lBoundary + rBoundary) / 2, -search_step)
            head_range = list(range(max_head_index - 1, -1, -1))
        else:
            # 从左向右搜索
            searchPoints = np.arange(lBoundary, (lBoundary + rBoundary) / 2, search_step)
            head_range = list(range(max_head_index))

        for startPoint in searchPoints:
            tmp_placement_result, tmp_head_sequence_result = [], []
            total_dist = 0
            mount_point_pos_cpy, mount_point_index_cpy = copy.deepcopy(mount_point_pos), copy.deepcopy(mount_point_index)
            for cycle_index, component_cycle in enumerate(component_assign):
                for _ in range(cycle_assign[cycle_index]):
                    assigned_placement = [-1] * max_head_index
                    assigned_mount_point = [[0, 0]] * max_head_index
                    min_pos_x, max_pos_x = min(mount_point_pos_cpy, key=lambda x: x[0])[0], \
                                           max(mount_point_pos_cpy, key=lambda x: x[0])[0]
                    search_interval = min((max_pos_x - min_pos_x) / (max_head_index - 1), head_interval)

                    # 最近邻确定
                    way_point = [startPoint, min(mount_point_pos_cpy, key=lambda x: x[1])[1]]
                    point_index = -1
                    for head_counter, head in enumerate(head_range):
                        if head_counter == 0:
                            unassigned_head = []
                            for next_head in range(max_head_index):
                                if assigned_placement[next_head] == -1 and component_assign[cycle_index][next_head] != -1:
                                    unassigned_head.append(next_head)

                            if len(unassigned_head) == 0:
                                continue

                            if rBoundary < startPoint:
                                head_index = unassigned_head[-1]
                            else:
                                head_index = unassigned_head[int(math.floor(
                                    (startPoint - lBoundary) / (rBoundary - lBoundary) * len(unassigned_head)))]

                            component_index = component_assign[cycle_index][head_index]
                            min_horizontal_distance = None
                            for index, mount_index in enumerate(mount_point_index_cpy):
                                if mount_point_part[mount_index] != component_index:
                                    continue
                                horizontal_distance = abs(mount_point_pos_cpy[index][0] - startPoint) + 1e-5 * abs(
                                    mount_point_pos_cpy[index][1] - way_point[1])

                                if min_horizontal_distance is None or horizontal_distance < min_horizontal_distance:
                                    min_horizontal_distance = horizontal_distance
                                    point_index = index
                        else:
                            head_index, point_index = -1, -1
                            min_cheby_distance, min_euler_distance = float('inf'), float('inf')
                            for next_head in range(max_head_index):
                                next_comp_index = component_assign[cycle_index][next_head]
                                if assigned_placement[next_head] != -1 or next_comp_index == -1:
                                    continue
                                for counter, mount_index in enumerate(mount_point_index_cpy):
                                    if mount_point_part[mount_index] != next_comp_index:
                                        continue

                                    if search_dir == 0:
                                        delta_x = abs(mount_point_pos_cpy[counter][0] - way_point[0]
                                                      + (max_head_index // 2 - 0.5 - next_head) * search_interval)
                                    elif search_dir == 1:
                                        delta_x = abs(mount_point_pos_cpy[counter][0] - way_point[0]
                                                      + (max_head_index - next_head - 1) * search_interval)
                                    else:
                                        delta_x = abs(mount_point_pos_cpy[counter][0] - way_point[0]
                                                      - next_head * search_interval)
                                    delta_x = delta_x * 0.01
                                    delta_y = abs(mount_point_pos_cpy[counter][1] - way_point[1])

                                    euler_distance = pow(axis_moving_time(delta_x, 0), 2) + pow(axis_moving_time(delta_y, 1), 2)
                                    cheby_distance = max(axis_moving_time(delta_x, 0),
                                                         axis_moving_time(delta_y, 1)) + 0.1 * euler_distance

                                    if cheby_distance < min_cheby_distance or (
                                            abs(cheby_distance - min_cheby_distance) < 1e-9
                                            and euler_distance < min_euler_distance):
                                        # if euler_distance < min_euler_distance:
                                        min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                        head_index, point_index = next_head, counter

                            if head_index == -1:
                                continue

                        assert point_index != -1

                        assigned_placement[head_index] = mount_point_index_cpy[point_index]
                        way_point = mount_point_pos_cpy[point_index]
                        assigned_mount_point[head_index] = way_point.copy()

                        mount_point_index_cpy.pop(point_index)
                        mount_point_pos_cpy.pop(point_index)

                        if search_dir == 0:
                            way_point[0] += (max_head_index // 2 - 0.5 - head_index) * search_interval
                        elif search_dir == 1:
                            way_point[0] += (max_head_index - head_index - 1) * search_interval
                        else:
                            way_point[0] -= head_index * search_interval

                    dist, head_seq = dynamic_programming_cycle_path(assigned_placement, assigned_mount_point)

                    total_dist += dist
                    tmp_head_sequence_result.append(head_seq)
                    tmp_placement_result.append(assigned_placement)  # 各个头上贴装的元件类型

            if min_dist is None or total_dist < min_dist:
                placement_result, head_sequence_result = tmp_placement_result, tmp_head_sequence_result
                min_dist = total_dist

    cycle_length, cycle_average_pos = [], []
    for cycle, placement in enumerate(placement_result):
        prev_pos = None
        pos_list = []
        movement = 0
        for head in head_sequence_result[cycle]:
            point_index = placement[head]
            if point_index == -1:
                continue
            pos = mount_point_pos[point_index].copy()
            pos[0] -= head * head_interval
            pos_list.append(pos)
            if prev_pos is not None:
                movement += max(abs(prev_pos[0] - pos[0]), abs(prev_pos[1] - pos[1]))
            prev_pos = pos

        cycle_average_pos.append([sum(map(lambda pos: pos[0], pos_list)) / len(pos_list),
                                  sum(map(lambda pos: pos[1], pos_list)) / len(pos_list)])
        cycle_length.append(movement)

    best_placement_result, best_head_sequence_result = copy.deepcopy(placement_result), copy.deepcopy(
        head_sequence_result)

    best_cycle_length, best_cycle_average_pos = cycle_length.copy(), copy.deepcopy(cycle_average_pos)
    def dfs(step, dist, point_assign, head_assign):
        nonlocal min_dist
        nonlocal point_assign_res, head_assign_res
        if min_dist is not None and dist > min_dist:
            return

        if step == len(point_list):
            if min_dist is None or dist < min_dist:
                point_assign_res, head_assign_res = point_assign.copy(), head_assign.copy()
                min_dist = dist
            return

        for _head in range(max_head_index):
            if head_component[_head] == -1 or _head in head_assign:
                continue

            head_assign.append(_head)
            for _point in point_list:
                if head_component[_head] != mount_point_part[_point] or _point in point_assign:
                    continue
                point_assign.append(_point)
                if len(point_assign) >= 2:
                    _delta_x = mount_point_pos[point_assign[-1]][0] - head_assign[-1] * head_interval - \
                              mount_point_pos[point_assign[-2]][0] + head_assign[-2] * head_interval
                    _delta_y = mount_point_pos[point_assign[-1]][1] - mount_point_pos[point_assign[-1]][1]
                else:
                    _delta_x, _delta_y = 0, 0
                dfs(step + 1, dist + max(abs(_delta_x), abs(_delta_y)), point_assign, head_assign)
                point_assign.pop(-1)
            head_assign.pop(-1)
        return

    n_generation = 0
    with tqdm(total=n_generation) as pbar:
        pbar.set_description('swap heuristic process')

        for _ in range(n_generation):
            # print('current generation:  ', n_generation, ', current total length :  ', sum(best_cycle_length))

            placement_result, head_sequence_result = copy.deepcopy(best_placement_result), copy.deepcopy(
                    best_head_sequence_result)
            cycle_length = best_cycle_length.copy()
            cycle_average_pos = copy.deepcopy(best_cycle_average_pos)

            cycle_index = roulette_wheel_selection(cycle_length)        # 根据周期加权移动距离随机选择周期

            point_dist = []
            for head in head_sequence_result[cycle_index]:
                point_index = placement_result[cycle_index][head]
                point_dist.append(
                    max(abs(mount_point_pos[point_index][0] - head * head_interval - cycle_average_pos[cycle_index][0]),
                        abs(mount_point_pos[point_index][1] - cycle_average_pos[cycle_index][1])))

            max_point_dist = max(point_dist)
            for i, dist in enumerate(point_dist):
                point_dist[i] = 2 * max_point_dist - dist

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
                for head in head_sequence_result[idx]:
                    dist_ += max(abs(mount_point_pos[placement_result[idx][head]][0] - mount_point_pos[point_index][0]),
                                 abs(mount_point_pos[placement_result[idx][head]][1] - mount_point_pos[point_index][1]))

                if min_dist is None or dist_ < min_dist:
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
                chg_cycle_point_cpy[idx][0] = mount_point_pos[index][0] - head * head_interval

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

            tmp_chg_placement_res, tmp_chg_head_res = placement_result[chg_cycle_index].copy(), head_sequence_result[
                chg_cycle_index].copy()
            tmp_chg_placement_res[chg_head] = point_index
            point_list, head_component = [], [-1 for _ in range(max_head_index)]
            for head, point in enumerate(tmp_chg_placement_res):
                if point == -1:
                    continue
                point_list.append(point)
                head_component[head] = mount_point_part[point]

            point_assign_res, head_assign_res = [], []
            min_dist = None

            dfs(0, 0, [], [])
            prev_pos = None
            tmp_chg_moving = 0
            for idx, head in enumerate(head_assign_res):
                tmp_chg_placement_res[head] = point_assign_res[idx]
                pos = mount_point_pos[point_assign_res[idx]].copy()
                pos[0] -= head * head_interval
                if prev_pos is not None:
                    tmp_chg_moving += max(abs(prev_pos[0] - pos[0]), abs(prev_pos[1] - pos[1]))
                prev_pos = pos

            tmp_chg_head_res = head_assign_res.copy()

            # 同样的方案重复一次
            tmp_placement_res, tmp_place_head_res = placement_result[cycle_index].copy(), head_sequence_result[
                cycle_index].copy()
            tmp_placement_res[head_index] = placement_result[chg_cycle_index][chg_head]

            point_list, head_component = [], [-1 for _ in range(max_head_index)]
            for head, point in enumerate(tmp_placement_res):
                if point == -1:
                    continue
                point_list.append(point)
                head_component[head] = mount_point_part[point]

            point_assign_res, head_assign_res = [], []
            min_dist = None

            dfs(0, 0, [], [])
            prev_pos = None
            tmp_place_moving = 0
            tmp_placement_res = [-1 for _ in range(max_head_index)]
            for idx, head in enumerate(head_assign_res):
                tmp_placement_res[head] = point_assign_res[idx]
                pos = mount_point_pos[point_assign_res[idx]].copy()
                pos[0] -= head * head_interval
                if prev_pos is not None:
                    tmp_place_moving += max(abs(prev_pos[0] - pos[0]), abs(prev_pos[1] - pos[1]))
                prev_pos = pos
            tmp_place_head_res = head_assign_res.copy()

            # 更新贴装顺序分配结果
            placement_result[cycle_index] = tmp_placement_res
            placement_result[chg_cycle_index] = tmp_chg_placement_res

            head_sequence_result[cycle_index] = tmp_place_head_res
            head_sequence_result[chg_cycle_index] = tmp_chg_head_res

            # 更新移动路径
            cycle_length[cycle_index], cycle_length[chg_cycle_index] = tmp_place_moving, tmp_chg_moving

            # 更新平均坐标和最大偏离点索引
            point_list, point_index_list = [], []
            for head in head_sequence_result[cycle_index]:
                point_index_list.append(tmp_placement_res[head])
                point_pos = mount_point_pos[point_index_list[-1]].copy()
                point_pos[0] -= head * head_interval
                point_list.append(point_pos)

            cycle_average_pos[cycle_index] = [sum(map(lambda x: x[0], point_list)) / len(point_list),
                                              sum(map(lambda x: x[1], point_list)) / len(point_list)]

            point_list, point_index_list = [], []
            for head in head_sequence_result[chg_cycle_index]:
                point_index_list.append(tmp_chg_placement_res[head])
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

            pbar.update(1)
    return best_placement_result, best_head_sequence_result