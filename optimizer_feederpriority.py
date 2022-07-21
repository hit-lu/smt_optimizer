import copy
from collections import defaultdict

import numpy as np

from dataloader import *
from optimizer_common import *


@timer_wrapper
def feeder_allocate(component_data, pcb_data, feeder_data, figure):
    feeder_points, mount_center_pos = {}, {}  # 供料器贴装点数
    feeder_base, feeder_state = [-1] * (max_slot_index // 2), [True] * len(component_data)  # feeder_state: 已安装在供料器基座上

    for data in pcb_data.iterrows():
        pos, part = data[1]['x'] + stopper_pos[0], data[1]['part']
        part_index = component_data[component_data['part'] == part].index.tolist()[0]
        feeder_state[part_index] = False
        if part_index not in feeder_points.keys():
            feeder_points[part_index] = 0
            mount_center_pos[part_index] = 0

        feeder_points[part_index] += 1
        mount_center_pos[part_index] += ((pos - mount_center_pos[part_index]) / feeder_points[part_index])

    if feeder_data is not None:
        for feeder in feeder_data.iterrows():
            slot, part = feeder[1]['slot'], feeder[1]['part']
            part_index = component_data[component_data['part'] == part].index.tolist()[0]

            feeder_base[slot] = part_index
            feeder_state[part_index] = True

    while sum(feeder is False for feeder in feeder_state) != 0:
        best_assign = []
        best_assign_slot, best_assign_value = -1, -np.Inf
        best_assign_points = []

        for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
            feeder_assign, feeder_assign_points = [], []
            tmp_feeder_state, tmp_feeder_points = feeder_state.copy(), feeder_points.copy()
            nozzle_change_counter = 0       # 吸嘴更换次数

            # 记录扫描到的已安装的供料器元件类型
            for head in range(max_head_index):
                feeder_assign.append(feeder_base[slot + head * interval_ratio])
                if feeder_assign[-1] != -1:
                    feeder_assign_points.append(feeder_points[feeder_assign[-1]])
                else:
                    feeder_assign_points.append(0)

            if -1 not in feeder_assign:
                continue

            # 分配新的供料器
            for idx, feeder in enumerate(feeder_assign):
                if feeder != -1:
                    continue

                while True:
                    # 选取未贴装元件中对应点数最多的元件
                    # TODO: 此处分配应兼顾多个原则：可用吸嘴数不应超过上限，当前已知最小贴装点数，则贴装点数多于它的具有同等地位
                    part = max(tmp_feeder_points.keys(), key = lambda x: tmp_feeder_points[x])

                    if tmp_feeder_points[part] == 0:
                        break

                    # 未分配且吸嘴类型保持一致时
                    if tmp_feeder_state[part] is False:
                        break
                    else:
                        # 重新选择
                        tmp_feeder_points[part] = 0

                # 待分配的供料器存在需要贴装的点
                if tmp_feeder_points[part] != 0:
                    feeder_assign[idx], feeder_assign_points[idx] = part, tmp_feeder_points[part]
                    tmp_feeder_state[part] = True

            assign_value = min(feeder_assign_points) - nozzle_change_counter * 2
            if assign_value >= best_assign_value:
                if assign_value == best_assign_value and abs(slot - 48) > abs(best_assign_slot - 48):
                    continue

                best_assign_value = assign_value
                best_assign = feeder_assign.copy()
                best_assign_slot = slot
                best_assign_points = feeder_assign_points

        for idx, feeder in enumerate(best_assign):
            if feeder == -1:
                continue

            # 更新供料器基座信息
            feeder_base[best_assign_slot + idx * interval_ratio] = feeder

            feeder_points[feeder] -= min(best_assign_points)
            feeder_state[feeder] = True

    for slot, feeder in enumerate(feeder_base):
        if feeder == -1 or component_data.loc[feeder]['part'] in feeder_data['part'].values:
            continue
        part = component_data.loc[feeder]['part']

        feeder_data.loc[len(feeder_data.index)] = [slot, part, 'None', 'SM8', '1 Times', '0', '0', '0', '0', '0', '0',
                                                   '1 System Dump', 'ON', 0]

    if figure:
        # 绘制供料器位置布局
        for slot in range(max_slot_index // 2):
            plt.scatter(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1], marker='x', s=12, color='black', alpha=0.5)
            plt.text(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1] - 45, slot + 1, ha='center', va='bottom',
                     size=8)

        for feeder in feeder_data.iterrows():
            slot, part = feeder[1]['slot'], feeder[1]['part']
            part_index = component_data[component_data['part'] == part].index.tolist()[0]

            plt.text(slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1] + 12,
                     part + ': ' + str(feeder_points[part_index]), ha='center', size=7, rotation=90)
            rec_x = [slotf1_pos[0] + slot_interval * (slot - 1) - slot_interval / 2,
                     slotf1_pos[0] + slot_interval * (slot - 1) + slot_interval / 2,
                     slotf1_pos[0] + slot_interval * (slot - 1) + slot_interval / 2,
                     slotf1_pos[0] + slot_interval * (slot - 1) - slot_interval / 2]
            rec_y = [slotf1_pos[1] - 40, slotf1_pos[1] - 40, slotf1_pos[1] + 10, slotf1_pos[1] + 10]

            c = 'red' if feeder[1]['arg'] == 1 else 'yellow'        # 红色绘制已分配，黄色绘制未分配
            # component_index = component_data[component_data['part'] == part].index.tolist()[0]
            # c = 'red' if component_data.loc[component_index]['nz1'] == 'CN065' else 'yellow'

            plt.fill(rec_x, rec_y, facecolor=c, alpha=0.4)

        plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
                 [slotf1_pos[1] + 10, slotf1_pos[1] + 10], color='black')
        plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
                 [slotf1_pos[1] - 40, slotf1_pos[1] - 40], color='black')

        for counter in range(max_slot_index // 2 + 1):
            pos = slotf1_pos[0] + (counter - 0.5) * slot_interval
            plt.plot([pos, pos], [slotf1_pos[1] + 10, slotf1_pos[1] - 40], color='black', linewidth=1)

        plt.ylim(-10, 100)
        plt.show()


@timer_wrapper
def feeder_base_scan(component_data, pcb_data, feeder_data):
    component_points = [0] * len(component_data)
    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]

        component_points[component_index] += 1

    feeder_part = pd.Series(-np.ones(max_slot_index), np.arange(max_slot_index), dtype=np.int16)
    for feeder in feeder_data.iterrows():
        part, slot = feeder[1]['part'], feeder[1]['slot']
        component_index = component_data[component_data['part'] == part].index.tolist()
        if len(component_index) != 1:
            print('unregistered component: ', part, ' in slot', slot)
            continue
        component_index = component_index[0]
        feeder_part[slot] = component_index

    component_result, cycle_result, feeder_slot_result = [], [], []  # 贴装点索引和拾取槽位优化结果
    nozzle_mode, nozzle_cycle_set = [], []

    while True:
        # === 周期内循环 ===
        assigned_head = [-1 for _ in range(max_head_index)]  # 当前扫描到的头分配元件信息
        assigned_cycle = [0 for _ in range(max_head_index)]  # 当前扫描到的元件最大分配次数
        assigned_slot = [-1 for _ in range(max_head_index)]

        max_eval_func = -float('inf')
        # === 前供料器基座扫描 ===
        best_scan_slot, best_cycle_insert_index = -1, -1
        for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
            cycle_insert_index = 0
            scan_cycle, scan_assigned_head = [0] * max_head_index, [-1] * max_head_index
            component_counter, nozzle_counter = 0, 0
            for head in range(max_head_index):
                part = feeder_part[slot + head * interval_ratio]
                if scan_assigned_head[head] == -1 and part != -1 and component_points[part] > 0:
                    component_counter += 1
                    scan_assigned_head[head] = feeder_part[slot + head * interval_ratio]
                    scan_cycle[head] = component_points[part]

            nozzle_counter = max_head_index * 2
            for cycle, nozzle_cycle in enumerate(nozzle_mode):
                tmp_counter = 0
                # 上一周期
                for head, nozzle in enumerate(nozzle_cycle):
                    if scan_assigned_head[head] == -1:
                        continue
                    if component_data.loc[scan_assigned_head[head]]['nz1'] != nozzle:
                        tmp_counter += 1 if nozzle != '' else 2         # 之前没有吸嘴，记为更换1次，否则记为更换2次（装/卸各1次）

                # 下一周期（额外增加的吸嘴更换次数）
                if cycle + 1 < len(nozzle_mode):
                    for head, nozzle in enumerate(nozzle_mode[cycle + 1]):
                        if scan_assigned_head[head] == -1:
                            continue
                        prev_counter, new_counter = 0, 0
                        if nozzle_mode[cycle][head] != nozzle:
                            prev_counter += 1 if nozzle != '' else 2
                        if component_data.loc[scan_assigned_head[head]]['nz1'] != nozzle:
                            new_counter += 1 if nozzle != '' else 2
                        tmp_counter += new_counter - prev_counter

                if tmp_counter < nozzle_counter:
                    nozzle_counter, cycle_insert_index = tmp_counter, cycle

            if len(np.nonzero(scan_cycle)[0]) == 0:
                continue

            # 计算扫描后的代价函数,记录扫描后的最优解
            cycle = min(filter(lambda x: x > 0, scan_cycle))

            eval_func = factor_simultaneous_pick * component_counter * cycle - factor_nozzle_change * nozzle_counter
            if eval_func > max_eval_func:
                max_eval_func = eval_func
                assigned_head, assigned_cycle = scan_assigned_head.copy(), scan_cycle.copy()
                best_scan_slot, best_cycle_insert_index = slot, cycle_insert_index

        # TODO: 后供料器基座扫描
        if best_scan_slot != -1:
            for head in range(max_head_index):
                if assigned_head[head] == -1:
                    continue

                head_nozzle[head] = component_data.loc[assigned_head[head]]['nz1']
                assigned_slot[head] = best_scan_slot + head * interval_ratio
        else:
            break     # 退出扫描过程

        cycle_nozzle = ['' for _ in range(max_head_index)]
        for head, component in enumerate(assigned_head):
            if component == -1:
                continue
            cycle_nozzle[head] = component_data.loc[component]['nz1']

        nozzle_cycle_counter = 0
        insert_index = sum(nozzle_cycle_set[:best_cycle_insert_index + 1])       # 扫描到的结果插入的位置
        while True:
            if len([x for x in assigned_cycle if x > 0]) == 0:
                break

            cycle = min(filter(lambda x: x > 0, assigned_cycle))
            nozzle_cycle_counter += 1

            component_result.insert(insert_index, copy.deepcopy(assigned_head))
            cycle_result.insert(insert_index, copy.deepcopy(cycle))
            feeder_slot_result.insert(insert_index, copy.deepcopy(assigned_slot))
            insert_index += 1

            for head in range(max_head_index):
                slot = best_scan_slot + head * interval_ratio
                if assigned_cycle[head] == 0:
                    continue
                component_points[feeder_part[slot]] -= cycle
                assigned_slot[head] = slot
                assigned_cycle[head] -= cycle
                if assigned_cycle[head] == 0:
                    assigned_slot[head], assigned_head[head] = -1, -1

        nozzle_mode.insert(best_cycle_insert_index + 1, cycle_nozzle)
        nozzle_cycle_set.insert(best_cycle_insert_index + 1, nozzle_cycle_counter)

        if sum([points != 0 for points in component_points]) == 0:
            break

    # === TODO: 供料器合并 ===
    nozzle_result = []
    for idx, components in enumerate(component_result):
        nozzle_cycle = ['' for _ in range(max_head_index)]
        for hd, component in enumerate(components):
            if component == -1:
                continue
            nozzle_cycle[hd] = component_data.loc[component]['nz1']
        nozzle_result.append(nozzle_cycle)

    component_result = [[2, 9, 19, 5, 8, 17], [2, -1, -1, -1, 8, -1], [-1, -1, -1, -1, -1, 32], [-1, -1, -1, -1, -1, 27],
                        [25, 31, 39, 26, 28, -1], [12, 24, 13, 30, -1, -1], [12, -1, 13, -1, -1, -1], [37, 38, 16, 23, 22, -1],
                        [1, 4, 3, 36, 11, 7], [1, 4, 3, 36, -1, 7], [1, -1, 3, -1, -1, -1],
                        [1, -1, -1, -1, -1, -1], [20, 41, 29, 33, 40, 37], [20, -1, 29, -1, 40, -1], [20, -1, -1, -1, -1, -1],
                        [-1, -1, -1, 35, -1, -1], [-1, -1, -1, 14, 10, 21], [-1, -1, -1, -1, -1, 34], [-1, -1, -1, 6, 18, 15],
                        [-1, -1, -1, 6, -1, 15]]
    cycle_result = [45, 5, 10, 20,
                    20, 30, 15, 30,
                    45, 5, 10,
                    30, 15, 5, 20,
                    20, 45, 15, 40,
                    5]
    feeder_slot_result = [[41, 43, 45, 47, 49, 51], [41, -1, -1, -1, 49, -1], [-1, -1, -1, -1, -1, 20], [-1, -1, -1, -1, -1, 54],
                          [23, 25, 27, 29, 31, -1], [33, 35, 37, 39, -1, -1], [33, -1, 37, -1, -1, -1], [32, 34, 36, 38, 40, 42],
                          [42, 44, 46, 48, 50, 52], [42, 44, 46, 48, -1, 52], [42, -1, 46, -1, -1, -1],
                          [42, -1, -1, -1, -1, -1], [22, 24, 26, 28, 30, 32], [22, -1, 26, -1, 30, -1], [22, -1, -1, -1, -1, -1],
                          [-1, -1, -1, 6, -1, -1], [-1, -1, -1, 14, 16, 18], [-1, -1, -1, -1, -1, 59], [-1, -1, -1, 53, 55, 57],
                          [-1, -1, -1, 53, -1, 57]]

    return component_result, cycle_result, feeder_slot_result

