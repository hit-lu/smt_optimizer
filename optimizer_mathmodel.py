import math

import matplotlib.pyplot as plt
import numpy as np

from optimizer_common import *
from gurobipy import *


def list_range(start, end=None):
    return list(range(start)) if end is None else list(range(start, end))


def head_task_model(component_data, pcb_data, hinter=True):

    mdl = Model('pick_route')
    mdl.setParam('Seed', 0)
    mdl.setParam('OutputFlag', hinter)          # set whether output the debug information
    mdl.setParam('TimeLimit', 50)

    H = max_head_index
    I = len(component_data)
    S = len(component_data)
    K = len(pcb_data)

    nozzle_type, component_type = [], []
    for _, data in component_data.iterrows():
        if not data['nz'] in nozzle_type:
            nozzle_type.append(data['nz'])
        component_type.append(data['part'])

    average_pos = 0
    for _, data in pcb_data.iterrows():
        average_pos += data['x']
    slot_start = int(round(average_pos / len(pcb_data) + stopper_pos[0] - slotf1_pos[0]) / slot_interval) + 1

    r = 1
    J = len(nozzle_type)
    M = 10000
    CompOfNozzle = [[0 for _ in range(J)] for _ in range(I)]    # Compatibility

    component_point = [0 for _ in range(I)]
    for _, data in pcb_data.iterrows():
        idx = component_data[component_data['part'] == data['part']].index.tolist()[0]
        nozzle = component_data.iloc[idx]['nz']
        CompOfNozzle[idx][nozzle_type.index(nozzle)] = 1
        component_point[idx] += 1

    # objective related
    g = mdl.addVars(list_range(K), vtype=GRB.BINARY)
    d = mdl.addVars(list_range(K - 1), list_range(H), vtype=GRB.CONTINUOUS)
    u = mdl.addVars(list_range(K), vtype=GRB.INTEGER)

    d_plus = mdl.addVars(list_range(J), list_range(H), list_range(K - 1), vtype=GRB.CONTINUOUS)
    d_minus = mdl.addVars(list_range(J), list_range(H), list_range(K - 1), vtype=GRB.CONTINUOUS)

    e = mdl.addVars(list_range(-(H - 1) * r, S), list_range(K), vtype=GRB.BINARY)
    f = mdl.addVars(list_range(S), list_range(I), vtype=GRB.BINARY, name='')
    x = mdl.addVars(list_range(I), list_range(S), list_range(K), list_range(H), vtype=GRB.BINARY)
    n = mdl.addVars(list_range(H), vtype=GRB.CONTINUOUS)

    mdl.addConstrs(g[k] <= g[k + 1] for k in range(K - 1))

    mdl.addConstrs(
        quicksum(x[i, s, k, h] for i in range(I) for s in range(S)) <= g[k] for k in range(K) for h in range(H))

    # nozzle no more than 1 for head h and cycle k
    mdl.addConstrs(
        quicksum(CompOfNozzle[i][j] * x[i, s, k, h] for i in range(I) for s in range(S) for j in range(J)) <= 1 for k in
        range(K) for h in range(H))

    # nozzle available number constraint
    mdl.addConstrs(
        quicksum(CompOfNozzle[i][j] * x[i, s, k, h] for i in range(I) for s in range(S) for h in range(H)) <= H for k in
        range(K) for j in range(J))

    # work completion
    mdl.addConstrs(
        quicksum(x[i, s, k, h] for s in range(S) for k in range(K) for h in range(H)) == component_point[i] for i in
        range(I))

    # nozzle change
    mdl.addConstrs(quicksum(CompOfNozzle[i][j] * x[i, s, k, h] for i in range(I) for s in range(S)) - quicksum(
        CompOfNozzle[i][j] * x[i, s, k + 1, h] for i in range(I) for s in range(S)) == d_plus[j, h, k] - d_minus[
                       j, h, k] for k in range(K - 1) for j in range(J) for h in range(H))

    mdl.addConstrs(
        2 * d[k, h] == quicksum(d_plus[j, h, k] for j in range(J)) + quicksum(d_minus[j, h, k] for j in range(J)) for k
        in range(K - 1) for h in range(H))

    mdl.addConstrs(n[h] == quicksum(d[k, h] for k in range(K - 1)) - 0.5 for h in range(H))

    # simultaneous pick
    for s in range(-(H - 1) * r, S):
        rng = list(range(max(0, -math.floor(s / r)), min(H, math.ceil((S - s) / r))))
        for k in range(K):
            mdl.addConstr(quicksum(x[i, s + h * r, k, h] for h in rng for i in range(I)) <= M * e[s, k], name='')
            mdl.addConstr(quicksum(x[i, s + h * r, k, h] for h in rng for i in range(I)) >= e[s, k], name='')
    # pickup movement
    mdl.addConstrs(
        u[k] >= s1 * e[s1, k] - s2 * e[s2, k] for s1 in range(-(H - 1) * r, S) for s2 in range(-(H - 1) * r, S) for k in
        range(K))

    # feeder related
    mdl.addConstrs(quicksum(f[s, i] for s in range(S)) <= 1 for i in range(I))
    mdl.addConstrs(quicksum(f[s, i] for i in range(I)) <= 1 for s in range(S))
    mdl.addConstrs(
        quicksum(x[i, s, k, h] for h in range(H) for k in range(K)) >= f[s, i] for i in range(I) for s in range(S))
    mdl.addConstrs(
        quicksum(x[i, s, k, h] for h in range(H) for k in range(K)) <= M * f[s, i] for i in range(I) for s in
        range(S))

    # objective
    t_c, t_n, t_p, t_m = 2, 6, 1, 0.2
    mdl.setObjective(t_c * quicksum(g[k] for k in range(K)) + t_n * quicksum(
        d[k, h] for h in range(H) for k in range(K - 1)) + t_p * quicksum(
        e[s, k] for s in range(-(H - 1) * r, S) for k in range(K))+ t_m * quicksum(u[k] for k in range(K)),
                     GRB.MINIMIZE)

    mdl.optimize()

    component_result, cycle_result, feeder_slot_result = [], [], []
    for k in range(K):
        if abs(g[k].x) < 1e-6:
            continue

        component_result.append([-1 for _ in range(H)])
        feeder_slot_result.append([-1 for _ in range(H)])
        cycle_result.append(1)
        for h in range(H):
            for i in range(I):
                for s in range(S):
                    if abs(x[i, s, k, h].x) > 1e-6:
                        component_result[-1][h] = i
                        feeder_slot_result[-1][h] = slot_start + s * interval_ratio - 1

    print(component_result)
    print(feeder_slot_result)

    print('')
    for h in range(H):
        val = 0
        for k in range(K - 1):
            val += d[k, h].x
        print(val)
    return component_result, cycle_result, feeder_slot_result


def place_route_model(component_data, pcb_data, component_result, feeder_slot_result, figure=False, hinter=True):
    mdl = Model('place_route')
    mdl.setParam('Seed', 0)
    mdl.setParam('OutputFlag', hinter)  # set whether output the debug information
    # mdl.setParam('TimeLimit', 20)

    component_type = []
    for _, data in component_data.iterrows():
        component_type.append(data['part'])

    pos = []
    for _, data in pcb_data.iterrows():
        pos.append([data['x'] + stopper_pos[0], data['y'] + stopper_pos[1]])
        # pos.append([data['x'], data['y']])

    I, P, H = len(component_data), len(pcb_data), max_head_index
    A = []
    for h1 in range(H):
        for h2 in range(H):
            if h1 == h2:
                continue
            A.append([h1, h2])
    K = len(component_result)

    CompOfPoint = [[0 for _ in range(P)] for _ in range(I)]
    for row, data in pcb_data.iterrows():
        idx = component_type.index(data['part'])
        CompOfPoint[idx][row] = 1

    d_FW, d_PL, d_BW = np.zeros([P, K, H]), np.zeros([P, P, len(A)]), np.zeros([P, K, H])
    for k in range(K):
        min_slot, max_slot = float('inf'), float('-inf')
        for h in range(H):
            if feeder_slot_result[k][h] == -1:
                continue
            min_slot = min(min_slot, feeder_slot_result[k][h] - h * interval_ratio)
            max_slot = max(max_slot, feeder_slot_result[k][h] - h * interval_ratio)

        for p in range(P):
            for h in range(H):
                d_FW[p, k, h] = max(
                    abs(slotf1_pos[0] + (max_slot - 1) * slot_interval - pos[p][0] + h * head_interval),
                    abs(slotf1_pos[1] - pos[p][1]))

                d_BW[p, k, h] = max(
                    abs(slotf1_pos[0] + (min_slot - 1) * slot_interval - pos[p][0] + h * head_interval),
                    abs(slotf1_pos[1] - pos[p][1]))

    for p in range(P):
        for q in range(P):
            for idx, arc in enumerate(A):
                h1, h2 = arc
                d_PL[p, q, idx] = max(abs(pos[p][0] - pos[q][0] - (h1 - h2) * head_interval), abs(pos[p][1] - pos[q][1]))

    w = mdl.addVars(list_range(P), list_range(P), list_range(K), list_range(len(A)), vtype=GRB.BINARY)
    y = mdl.addVars(list_range(P), list_range(K), list_range(H), vtype=GRB.BINARY)
    z = mdl.addVars(list_range(P), list_range(K), list_range(H), vtype=GRB.BINARY)

    def A_from(h):
        res = []
        for idx, arc in enumerate(A):
            if arc[0] == h:
                res.append(idx)
        return res

    def A_to(h):
        res = []
        for idx, arc in enumerate(A):
            if arc[1] == h:
                res.append(idx)
        return res

    def A_contain(h):
        res = []
        for idx, arc in enumerate(A):
            if h in arc:
                res.append(idx)
        return res

    # constraints on component assignment type, assigned points cannot conflict with the corresponding component type
    for k in range(K):
        for h in range(H):
            if component_result[k][h] == -1:
                # no components on the head
                mdl.addConstr(quicksum(w[p, q, k, a] for a in A_contain(h) for q in range(P) for p in range(P)) == 0)
            else:
                # there are components on the head
                mdl.addConstrs((quicksum(w[p, q, k, a] for a in A_from(h) for q in range(P)) + quicksum(
                    w[q, p, k, a] for a in A_to(h) for q in range(P))) / 2 <= CompOfPoint[component_result[k][h]][p] for
                               p in range(P))

    # each head corresponds to a maximum of one point in each cycle
    mdl.addConstrs(
        quicksum(w[p, q, k, a] for p in range(P) for q in range(P) for a in A_contain(h)) <= 2 for k in range(K) for h
        in range(H))

    mdl.addConstrs(
        quicksum(y[p, k, h] for p in range(P)) + quicksum(z[p, k, h] for p in range(P)) <= 1 for k in range(K) for h in
        range(H))

    # task continuity (for the same point the entering head and the leaving head should be same)
    mdl.addConstrs(quicksum(w[p, q, k, a] for p in range(P) for a in A_to(h)) + y[q, k, h] == quicksum(
        w[q, p, k, a] for p in range(P) for a in A_from(h)) + z[q, k, h] for k in range(K) for h in range(H) for q in
                   range(P))

    mdl.addConstrs(
        y[p, k, h] <= quicksum(w[p, q, k, a] for q in range(P) for a in A_from(h)) for h in range(H) for p in
        range(P) for k in range(K))

    mdl.addConstrs(
        z[p, k, h] <= quicksum(w[q, p, k, a] for q in range(P) for a in A_to(h)) for h in range(H) for p in
        range(P) for k in range(K))

    # one arrival point per cycle
    mdl.addConstrs(quicksum(y[p, k, h] for p in range(P) for h in range(H)) == 1 for k in range(K))

    # one departure point per cycle
    mdl.addConstrs(quicksum(z[p, k, h] for p in range(P) for h in range(H)) == 1 for k in range(K))

    # one enter edge per point
    mdl.addConstrs(quicksum(y[q, k, h] for h in range(H) for k in range(K)) + quicksum(
        w[p, q, k, a] for p in range(P) for a in range(len(A)) for k in range(K)) == 1 for q in range(P))

    # one leaving edge per point
    mdl.addConstrs(quicksum(z[q, k, h] for h in range(H) for k in range(K)) + quicksum(
        w[q, p, k, a] for p in range(P) for a in range(len(A)) for k in range(K)) == 1 for q in range(P))

    # subtour eliminate constraint
    n = mdl.addVars(list_range(P), vtype=GRB.CONTINUOUS)
    m = mdl.addVars(list_range(P), vtype=GRB.CONTINUOUS)
    v = mdl.addVars(list_range(P), list_range(P), vtype=GRB.CONTINUOUS)

    mdl.addConstrs(
        m[p] + quicksum(v[p, q] for q in range(P)) - n[p] - quicksum(v[q, p] for q in range(P)) == 1 for p in range(P))

    mdl.addConstrs(
        v[p, q] <= (P - K + 1) * quicksum(w[p, q, k, a] for a in range(len(A)) for k in range(K)) for p in range(P) for
        q in range(P))

    mdl.addConstrs(n[p] <= (P - K + 1) * quicksum(y[p, k, h] for h in range(H) for k in range(K)) for p in range(P))
    mdl.addConstrs(m[p] <= (P - K + 1) * quicksum(z[p, k, h] for h in range(H) for k in range(K)) for p in range(P))

    # objective
    mdl.setObjective(
        quicksum(d_FW[p, k, h] * y[p, k, h] for p in range(P) for k in range(K) for h in range(H)) + quicksum(
            d_PL[p, q, a] * w[p, q, k, a] for k in range(K) for p in range(P) for q in range(P) for a in
            range(len(A))) + quicksum(d_BW[p, k, h] * z[p, k, h] for p in range(P) for k in range(K) for h in range(H)),
        GRB.MINIMIZE)

    mdl.optimize()
    if figure:
        for k in range(K):
            plt.scatter([p[0] for p in pos[0:8]], [p[1] for p in pos[0:8]], color='red')
            plt.scatter([p[0] for p in pos[8:]], [p[1] for p in pos[8:]], color='blue')
            line_counter = 0
            for p in range(P):
                for q in range(P):
                    for idx, arc in enumerate(A):
                        if abs(w[p, q, k, idx].x) > 1e-6:
                            h1, h2 = arc
                            plt.plot([pos[p][0] - h1 * head_interval, pos[q][0] - h2 * head_interval],
                                     [pos[p][1], pos[q][1]], linestyle='-.', color='black', linewidth=1)
                            plt.text(pos[p][0] - h1 * head_interval, pos[p][1], 'H%d' % (h1 + 1), ha='center',
                                     va='bottom', size=10)

                            print(p, q, h1, h2, idx, d_PL[p, q, idx])
                            line_counter += 1

                for h in range(H):
                    if abs(y[p, k, h].x) > 1e-6:
                        print('y:', p, h)
                        plt.plot([pos[p][0] - h * head_interval, 500], [pos[p][1], 100], linestyle='-.', color='black',
                                 linewidth=1)
                        plt.text(pos[p][0] - h * head_interval, pos[p][1], 'H%d' % (h + 1), ha='center', va='bottom',
                                 size=10)
                        line_counter += 1

                for h in range(H):
                    if abs(z[p, k, h].x) > 1e-6:
                        print('z:', p, h)
                        plt.plot([pos[p][0] - h * head_interval, 900], [pos[p][1], 100], linestyle='-.', color='black',
                                 linewidth=1)
                        plt.text(pos[p][0] - h * head_interval, pos[p][1], 'H%d' % (h + 1), ha='center', va='bottom',
                                 size=10)
                        line_counter += 1

            print('num of line: ', line_counter)
            plt.show()

    # convert model result into standard form
    placement_result, head_sequence = [[-1 for _ in range(H)] for _ in range(K)], [[] for _ in
                                                                                   range(K)]
    for k in range(K):
        arc_list = []
        for p in range(P):
            for q in range(P):
                for idx, arc in enumerate(A):
                    if abs(w[p, q, k, idx].x) > 1e-6:
                        plt.plot([pos[p][0], pos[q][0]], [pos[p][1], pos[q][1]], linestyle='-.', color='black',
                                 linewidth=1)
                        placement_result[k][arc[0]], placement_result[k][arc[1]] = p, q
                        arc_list.append(arc)

        head, idx = -1, 0
        for p in range(P):
            for h in range(H):
                if abs(y[p, k, h].x) > 1e-6:
                    head = h

        while idx < len(arc_list):
            for i, arc in enumerate(arc_list):
                if arc[0] == head:
                    head_sequence[k].append(head)
                    head = arc[1]
                    idx += 1
                    break
        head_sequence[k].append(head)

    return placement_result, head_sequence


@timer_wrapper
def optimizer_mathmodel(component_data, pcb_data, hinter=True):

    component_result, cycle_result, feeder_slot_result = head_task_model(component_data, pcb_data, hinter)
    placement_result, head_sequence = place_route_model(component_data, pcb_data, component_result, feeder_slot_result)

    return component_result, cycle_result, feeder_slot_result, placement_result, head_sequence
