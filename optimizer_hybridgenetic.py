import numpy
import pandas

from common_function import *

def optimizer_hybridgenetic(pcb_data, component_data):

    # === 统计各类型吸嘴的贴装点数 ===
    nozzle_points = {}      # 吸嘴对应的贴装点数
    nozzle_assigned_num = {}   # 吸嘴分配的头数
    for step in pcb_data.iterrows():
        part = step[1]['part']
        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz1']
        if nozzle not in nozzle_points.keys():
            nozzle_points[nozzle] = 0
            nozzle_assigned_num[nozzle] = 0

        nozzle_points[nozzle] += 1

    assert(len(nozzle_points) <= max_head_index, "nozzle type number should no more than the head num")
    total_points, available_head = len(pcb_data), max_head_index
    S1, S2, S3 = [], [], []

    # === 分配吸嘴 ===
    for nozzle in nozzle_points.keys():     # Phase 1
        if nozzle_points[nozzle] * max_head_index < total_points:
            nozzle_assigned_num[nozzle] = 1
            available_head -= 1
            total_points -= nozzle_points[nozzle]

            S1.append(nozzle)
        else:
            S2.append(nozzle)

    available_head_ = available_head        # Phase 2
    for nozzle in S2:
        nozzle_assigned_num[nozzle] = math.floor(available_head * nozzle_points[nozzle] / total_points)
        available_head_ = available_head_ - nozzle_assigned_num[nozzle]

    S2.sort(key = lambda x: nozzle_points[x] / nozzle_assigned_num[x], reverse = True)
    while available_head_ > 0:
        nozzle = S2[0]
        nozzle_assigned_num[nozzle] += 1

        S2.remove(nozzle)
        S3.append(nozzle)
        available_head_ -= 1

    while len(S2) != 0:                     # Phase 3
        nozzle_i_val, nozzle_j_val = None, None
        nozzle_i, nozzle_j = 0, 0
        for nozzle in S2:
            if nozzle_i_val is None or nozzle_points[nozzle] / nozzle_assigned_num[nozzle] > nozzle_i_val:
                nozzle_i_val = nozzle_points[nozzle] / nozzle_assigned_num[nozzle]
                nozzle_i = nozzle

            if nozzle_j_val is None or nozzle_points[nozzle] / (nozzle_assigned_num[nozzle] - 1) < nozzle_j_val:
                nozzle_j_val = nozzle_points[nozzle] / (nozzle_assigned_num[nozzle] - 1)
                nozzle_j = nozzle

        if nozzle_points[nozzle_j] / (nozzle_assigned_num[nozzle_j] - 1) < nozzle_points[nozzle_i] / nozzle_assigned_num[nozzle_i]:
            nozzle_points[nozzle_j] -= 1
            nozzle_points[nozzle_i] += 1
            S2.remove(nozzle_i)
            S3.append(nozzle_i)
        else:
            break

    # 吸嘴分配结果
    designated_nozzle = [[] for _ in range(max_head_index)]
    head_index = 0
    for nozzle, num in nozzle_assigned_num.items():
        while num > 0:
            designated_nozzle[head_index] = nozzle
            head_index += 1
            num -= 1

    # === 元件分配 ===
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

    CT_Group, CT_Points = [], []
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

    # 拾取组合并