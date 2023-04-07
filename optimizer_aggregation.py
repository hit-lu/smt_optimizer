from optimizer_common import *

from gurobipy import *
from collections import defaultdict


def list_range(start, end=None):
    return list(range(start)) if end is None else list(range(start, end))


@timer_wrapper
def optimizer_aggregation(component_data, pcb_data):
    # === phase 0: data preparation ===
    M = 1000                                # a sufficient large number
    a, b = 1, 6                             # coefficient

    component_list, nozzle_list = defaultdict(int), defaultdict(int)
    cpidx_2_part, nzidx_2_nozzle = {}, {}
    for _, data in pcb_data.iterrows():
        part = data['part']
        if part not in cpidx_2_part.values():
            cpidx_2_part[len(cpidx_2_part)] = part

        component_list[part] += 1

        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz']
        if nozzle not in nzidx_2_nozzle.values():
            nzidx_2_nozzle[len(nzidx_2_nozzle)] = nozzle
        nozzle_list[nozzle] += 1

    I, J = len(component_list.keys()), len(nozzle_list.keys())  # the maximum number of component types and nozzle types
    L = I + 1                                                   # the maximum number of batch level
    K = max_head_index                                          # the maximum number of heads
    HC = [[M for _ in range(J)] for _ in range(I)]              # represent the nozzle-component compatibility

    for i in range(I):
        for _, item in enumerate(cpidx_2_part.items()):
            index, part = item
            cp_idx = component_data[component_data['part'] == part].index.tolist()[0]
            nozzle = component_data.loc[cp_idx]['nz']

            for j in range(J):
                if nzidx_2_nozzle[j] == nozzle:
                    HC[index][j] = 0

    # === phase 1: mathematical model solver ===
    mdl = Model('SMT')
    mdl.setParam('OutputFlag', 1)

    # === Decision Variables ===
    # the number of components of type i that are placed by nozzle type j on placement head k
    X = mdl.addVars(list_range(I), list_range(J), list_range(K), vtype=GRB.INTEGER, ub=max(component_list.values()))

    # the total number of nozzle changes on placement head k
    N = mdl.addVars(list_range(K), vtype=GRB.INTEGER)

    # the largest workload of all placement heads
    WL = mdl.addVar(vtype=GRB.INTEGER, lb=0, ub=len(pcb_data))

    # whether batch Xijk is placed on level l
    Z = mdl.addVars(list_range(I), list_range(J), list_range(L), list_range(K), vtype=GRB.BINARY)

    # Dlk := 2 if a change of nozzles in the level l + 1 on placement head k
    # Dlk := 1 if there are no batches placed on levels higher than l
    # Dlk := 0 otherwise
    D = mdl.addVars(list_range(L), list_range(K), vtype=GRB.BINARY, ub=2)
    D_plus = mdl.addVars(list_range(L), list_range(J), list_range(K), vtype=GRB.INTEGER)
    D_minus = mdl.addVars(list_range(L), list_range(J), list_range(K), vtype=GRB.INTEGER)

    # == Objective function ===
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(a * WL + b * quicksum(N[k] for k in range(K)))

    # === Constraint ===
    mdl.addConstrs(
        quicksum(X[i, j, k] for j in range(J) for k in range(K)) == component_list[cpidx_2_part[i]] for i in range(I))

    mdl.addConstrs(quicksum(X[i, j, k] for i in range(I) for j in range(J)) <= WL for k in range(K))

    mdl.addConstrs(
        X[i, j, k] <= M * quicksum(Z[i, j, l, k] for l in range(L)) for i in range(I) for j in range(J) for k in
        range(K))

    mdl.addConstrs(quicksum(Z[i, j, l, k] for l in range(L)) <= 1 for i in range(I) for j in range(J) for k in range(K))
    mdl.addConstrs(
        quicksum(Z[i, j, l, k] for l in range(L)) <= X[i, j, k] for i in range(I) for j in range(J) for k in range(K))

    mdl.addConstrs(quicksum(Z[i, j, l, k] for j in range(J) for i in range(I)) >= quicksum(
        Z[i, j, l + 1, k] for j in range(J) for i in range(I)) for k in range(K) for l in range(L - 1))

    mdl.addConstrs(quicksum(Z[i, j, l, k] for i in range(I) for j in range(J)) <= 1 for k in range(K) for l in range(L))
    mdl.addConstrs(D_plus[l, j, k] - D_minus[l, j, k] == quicksum(Z[i, j, l, k] for i in range(I)) - quicksum(
        Z[i, j, l + 1, k] for i in range(I)) for l in range(L - 1) for j in range(J) for k in range(K))

    mdl.addConstrs(
        D[l, k] == quicksum((D_plus[l, j, k] + D_minus[l, j, k]) for j in range(J)) for k in range(K) for l in
        range(L))

    mdl.addConstrs(2 * N[k] == quicksum(D[l, k] for l in range(L)) - 1 for k in range(K))
    mdl.addConstrs(
        0 >= quicksum(HC[i][j] * Z[i, j, l, k] for i in range(I) for j in range(J)) for l in range(L) for k in range(K))

    # === Main Process ===
    component_result, cycle_result = [], []
    feeder_slot_result, placement_result, head_sequence = [], [], []
    mdl.setParam("TimeLimit", 100)

    mdl.optimize()

    if mdl.Status == GRB.OPTIMAL:
        print('total cost = {}'.format(mdl.objval))

        # convert cp model solution to standard output
        model_cycle_result, model_component_result = [], []
        for l in range(L):
            model_component_result.append([None for _ in range(K)])
            model_cycle_result.append([0 for _ in range(K)])
            for k in range(K):
                for i in range(I):
                    for j in range(J):
                        if abs(Z[i, j, l, k].x - 1) <= 1e-3:
                            model_component_result[-1][k] = cpidx_2_part[i]
                            model_cycle_result[-1][k] = round(X[i, j, k].x)

            # remove redundant term
            if sum(model_cycle_result[-1]) == 0:
                model_component_result.pop()
                model_cycle_result.pop()

        head_component_index = [0 for _ in range(max_head_index)]
        while True:
            head_cycle = []
            for head, index in enumerate(head_component_index):
                head_cycle.append(model_cycle_result[index][head])

            if len([cycle for cycle in head_cycle if cycle > 0]) == 0:
                break

            component_result.append([None for _ in range(max_head_index)])
            min_cycle = min([cycle for cycle in head_cycle if cycle > 0])
            for head, index in enumerate(head_component_index):
                if model_cycle_result[index][head] != 0:
                    component_result[-1][head] = model_component_result[index][head]
                else:
                    continue

                model_cycle_result[index][head] -= min_cycle
                if model_cycle_result[index][head] == 0 and index + 1 < len(model_cycle_result):
                    head_component_index[head] += 1

            cycle_result.append(min_cycle)

        part_2_index = {}
        for index, data in component_data.iterrows():
            part_2_index[data['part']] = index

        for cycle in range(len(component_result)):
            for head in range(max_head_index):
                part = component_result[cycle][head]
                component_result[cycle][head] = -1 if part is None else part_2_index[part]

        feeder_slot_result = feeder_assignment(component_data, pcb_data, component_result, cycle_result)

        # === phase 2: heuristic method ===
        mount_point_pos = defaultdict(list)
        for pcb_idx, data in pcb_data.iterrows():
            part = data['part']
            part_index = component_data[component_data['part'] == part].index.tolist()[0]
            mount_point_pos[part_index].append([data['x'], data['y'], pcb_idx])

        for index_ in mount_point_pos.keys():
            mount_point_pos[index_].sort(key=lambda x: (x[1], x[0]))

        for cycle_idx, _ in enumerate(cycle_result):
            for _ in range(cycle_result[cycle_idx]):
                placement_result.append([-1 for _ in range(max_head_index)])
                for head in range(max_head_index):
                    if component_result[cycle_idx][head] == -1:
                        continue
                    index_ = component_result[cycle_idx][head]
                    placement_result[-1][head] = mount_point_pos[index_][-1][2]
                    mount_point_pos[index_].pop()
                head_sequence.append(dynamic_programming_cycle_path(pcb_data, placement_result[-1], feeder_slot_result[cycle_idx]))

    else:
        warnings.warn('No solution found!', UserWarning)

    return component_result, cycle_result, feeder_slot_result, placement_result, head_sequence
