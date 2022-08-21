from optimizer_common import *

from ortools.sat.python import cp_model
from collections import defaultdict


@timer_wrapper
def optimizer_aggregation(component_data, pcb_data):
    # === phase 0: data preparation ===
    M = 1000                                # a sufficient large number
    a, b = 1, 6                             # coefficient
    K, I, J, L = max_head_index, 0, 0, 0    # the maximum number of heads, component types, nozzle types and batch level

    component_list, nozzle_list = defaultdict(int), defaultdict(int)
    cpidx_2_part, nzidx_2_nozzle = {}, {}
    for _, data in pcb_data.iterrows():
        part = data['part']
        if part not in cpidx_2_part.values():
            cpidx_2_part[len(cpidx_2_part)] = part

        component_list[part] += 1

        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz1']
        if nozzle not in nzidx_2_nozzle.values():
            nzidx_2_nozzle[len(nzidx_2_nozzle)] = nozzle
        nozzle_list[nozzle] += 1

    I, J = len(component_list.keys()), len(nozzle_list.keys())
    L = I + 1
    HC = [[M for _ in range(J)] for _ in range(I)]      # the handing class when component i is handled by nozzle type j
                                                        # represent the nozzle-component compatibility
    for i in range(I):
        for _, item in enumerate(cpidx_2_part.items()):
            index, part = item
            cp_idx = component_data[component_data['part'] == part].index.tolist()[0]
            nozzle = component_data.loc[cp_idx]['nz1']

            for j in range(J):
                if nzidx_2_nozzle[j] == nozzle:
                    HC[index][j] = 0

    # === phase 1: mathematical model solver ===
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # === Decision Variables ===
    # the number of components of type i that are placed by nozzle type j on placement head k
    X = {}
    for i in range(I):
        for j in range(J):
            for k in range(K):
                X[i, j, k] = model.NewIntVar(0, component_list[cpidx_2_part[i]], 'X_{}_{}_{}'.format(i, j, k))

    # the total number of nozzle changes on placement head k
    N = {}
    for k in range(K):
        N[k] = model.NewIntVar(0, J, 'N_{}'.format(k))

    # the largest workload of all placement heads
    WL = model.NewIntVar(0, len(pcb_data), 'WL')

    # whether batch Xijk is placed on level l
    Z = {}
    for i in range(I):
        for j in range(J):
            for l in range(L):
                for k in range(K):
                    Z[i, j, l, k] = model.NewBoolVar('Z_{}_{}_{}_{}'.format(i, j, l, k))

    # Dlk := 2 if a change of nozzles in the level l + 1 on placement head k
    # Dlk := 1 if there are no batches placed on levels higher than l
    D = {}
    for l in range(L):
        for k in range(K):
            D[l, k] = model.NewIntVar(0, 2, 'D_{}_{}'.format(l, k))

    D_abs = {}
    for l in range(L):
        for j in range(J):
            for k in range(K):
                D_abs[l, j, k] = model.NewIntVar(0, M, 'D_abs_{}_{}_{}'.format(l, j, k))

    # == Objective function ===
    model.Minimize(a * WL + b * sum(N[k] for k in range(K)))

    # === Constraint ===
    for i in range(I):
        model.Add(sum(X[i, j, k] for j in range(J) for k in range(K)) == component_list[cpidx_2_part[i]])

    for k in range(K):
        model.Add(sum(X[i, j, k] for i in range(I) for j in range(J)) <= WL)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                model.Add(X[i, j, k] <= M * sum(Z[i, j, l, k] for l in range(L)))

    for i in range(I):
        for j in range(J):
            for k in range(K):
                model.Add(sum(Z[i, j, l, k] for l in range(L)) <= 1)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                model.Add(sum(Z[i, j, l, k] for l in range(L)) <= X[i, j, k])

    for k in range(K):
        for l in range(L - 1):
            model.Add(sum(Z[i, j, l, k] for j in range(J) for i in range(I)) >= sum(
                Z[i, j, l + 1, k] for j in range(J) for i in range(I)))

    for l in range(I):
        for k in range(K):
            model.Add(sum(Z[i, j, l, k] for i in range(I) for j in range(J)) <= 1)

    for l in range(L - 1):
        for j in range(J):
            for k in range(K):
                model.AddAbsEquality(D_abs[l, j, k],
                                     sum(Z[i, j, l, k] for i in range(I)) - sum(Z[i, j, l + 1, k] for i in range(I)))

    for k in range(K):
        for l in range(L):
            model.Add(D[l, k] == sum(D_abs[l, j, k] for j in range(J)))

    for k in range(K):
        model.Add(N[k] == sum(D[l, k] for l in range(L)) - 1)

    for l in range(L):
        for k in range(K):
            model.Add(0 >= sum(HC[i][j] * Z[i, j, l, k] for i in range(I) for j in range(J)))

    # === Main Process ===
    component_result, cycle_result = [], []
    feeder_slot_result, placement_result, head_sequence = [], [], []

    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('total cost = {}'.format(solver.ObjectiveValue()))

        # convert cp model solution to standard output
        model_cycle_result, model_component_result = [], []
        for l in range(L):
            model_component_result.append([None for _ in range(K)])
            model_cycle_result.append([0 for _ in range(K)])
            for k in range(K):
                for i in range(I):
                    for j in range(J):
                        if solver.BooleanValue(Z[i, j, l, k]) != 0:
                            model_component_result[-1][k] = cpidx_2_part[i]
                            model_cycle_result[-1][k] = solver.Value(X[i, j, k])

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

        feeder_limit = [1 for _ in range(len(component_data))]      # feeder available limit: no more than 1
        feeder_slot_result = feeder_assignment(component_data, pcb_data, component_result, cycle_result, feeder_limit)

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
        print('no solution found')

    return component_result, cycle_result, feeder_slot_result, placement_result, head_sequence
