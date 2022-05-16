from dataloader import *
from common_function import *


def feeder_allocate(component_data, pcb_data, feeder_data, figure):

    feeder_points = {}         # 供料器贴装点数
    feeder_base, feeder_state = [-1] * (max_slot_index // 2), [True] * len(component_data)  # feeder_state: 已安装在供料器基座上

    for data in pcb_data.iterrows():
        pos, part = data[1]['x'] + stopper_pos[0], data[1]['part']
        part_index = component_data[component_data['part'] == part].index.tolist()[0]
        feeder_state[part_index] = False
        if part_index not in feeder_points.keys():
            feeder_points[part_index] = 0

        feeder_points[part_index] += 1

    # TODO 1: 如何处理不同类型吸嘴的情形（增加吸嘴分配）
    # TODO 2: 如何处理不同宽度喂料器
    # TODO 3: 如何处理供料器位置分配的边界条件（占位数量≈可用槽位数）
    # TODO 4: 如何平衡供料器前基座和后基座之间的分配元件数量
    # TODO 5: 如何进一步提升效能
    # TODO 6: 扩大测试范围
    if feeder_data is not None:
        for feeder in feeder_data.iterrows():
            slot, part = feeder[1]['slot'], feeder[1]['part']
            part_index = component_data[component_data['part'] == part].index.tolist()[0]

            feeder_base[slot] = part_index
            feeder_state[part_index] = True

    while sum(feeder is False for feeder in feeder_state) != 0:
        best_assign = []
        best_simupick_slot, best_simupick_value = -1, 0

        for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
            feeder_assign, feeder_assign_points = [], []
            tmp_feeder_state, tmp_feeder_points = feeder_state.copy(), feeder_points.copy()

            # 记录扫描到的已安装的供料器元件类型
            for head in range(max_head_index):
                feeder_assign.append(feeder_base[slot + head * interval_ratio])
                if feeder_assign[-1] != -1:
                    feeder_assign_points.append(feeder_points[feeder_assign[-1]])
                else:
                    feeder_assign_points.append(0)

            if -1 not in feeder_assign:
                continue

            for idx, feeder in enumerate(feeder_assign):
                if feeder != -1:
                    continue

                while True:
                    part = max(tmp_feeder_points.keys(), key = (lambda x: tmp_feeder_points[x]))
                    if tmp_feeder_points[part] == 0:
                        break

                    if tmp_feeder_state[part] is False:
                        break
                    else:
                        tmp_feeder_points[part] = 0

                feeder_assign[idx], feeder_assign_points[idx] = part, tmp_feeder_points[part]
                tmp_feeder_state[part] = True

            if min(feeder_assign_points) >= best_simupick_value:
                # TODO: 中心位置判别需要完善，目前设置为42效果比较好（自主分配时选择45效果比较好）
                if min(feeder_assign_points) == best_simupick_value and abs(slot - 45) > abs(best_simupick_slot - 45):
                    continue
                best_simupick_value = min(feeder_assign_points)
                best_assign = feeder_assign.copy()
                best_simupick_slot = slot

        for idx, feeder in enumerate(best_assign):
            feeder_base[best_simupick_slot + idx * interval_ratio] = feeder
            feeder_points[feeder] -= best_simupick_value
            feeder_state[feeder] = True


    for slot, feeder in enumerate(feeder_base):
        if feeder == -1 or component_data.loc[feeder]['part'] in feeder_data['part'].values:
            continue
        part = component_data.loc[feeder]['part']

        feeder_data.loc[len(feeder_data.index)] = [slot, part, 'None', 'SM8', '1 Times', '0', '0', '0', '0', '0', '0', '1 System Dump', 'ON', 0]

    if figure:
        # 绘制供料器位置布局
        for slot in range(max_slot_index // 2):
            plt.scatter(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1], marker='x', s=12, color='black', alpha = 0.5)
            plt.text(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1] - 45, slot + 1, ha='center', va='bottom', size=8)

        for feeder in feeder_data.iterrows():
            slot, part = feeder[1]['slot'],feeder[1]['part']
            plt.text(slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1] + 12,
                     part, ha = 'center', size = 7, rotation = 90)
            rec_x = [slotf1_pos[0] + slot_interval * (slot - 1) - slot_interval / 2, slotf1_pos[0] + slot_interval * (slot - 1) + slot_interval / 2,
                     slotf1_pos[0] + slot_interval * (slot - 1) + slot_interval / 2, slotf1_pos[0] + slot_interval * (slot - 1) - slot_interval / 2]
            rec_y = [slotf1_pos[1] - 40, slotf1_pos[1] - 40, slotf1_pos[1] + 10, slotf1_pos[1] + 10]
            c = 'red' if feeder[1]['arg'] == 1 else 'yellow'
            plt.fill(rec_x, rec_y, facecolor = c, alpha = 0.4)

        plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
                        [slotf1_pos[1] + 10, slotf1_pos[1] + 10], color = 'black')
        plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
                        [slotf1_pos[1] - 40, slotf1_pos[1] - 40], color = 'black')

        for counter in range(max_slot_index // 2 + 1):
            pos = slotf1_pos[0] + (counter - 0.5) * slot_interval
            plt.plot([pos, pos], [slotf1_pos[1] + 10, slotf1_pos[1] - 40], color='black', linewidth = 1)

        plt.ylim(-10, 100)
        plt.show()

def feederbase_scan(component_data, pcb_data, feeder_data):
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

    component_result, cycle_result = [], []
    feederslot_result = []  # 贴装点索引和拾取槽位优化结果
    while True:
        # === 周期内循环 ===
        assigned_head = [-1 for _ in range(max_head_index)]  # 当前扫描到的头分配元件信息
        assigned_cycle = [0 for _ in range(max_head_index)]  # 当前扫描到的元件最大分配次数
        assigned_slot = [-1 for _ in range(max_head_index)]

        while True:
            max_eval_func = -np.inf
            # 前供料器基座扫描
            # TODO: 扫描过程中要兼顾机械限位的影响，优先满足机械限位，可能会有效降低拾贴周期数
            best_scan_assigned_head, best_scan_cycle = [], []
            best_scan_slot = -1
            for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
                scan_cycle = [0 for _ in range(max_head_index)]
                scan_assigned_head = assigned_head.copy()
                component_counter, nozzle_counter = 0, 0
                for head in range(max_head_index):
                    # TODO: 可用吸嘴数限制
                    part = feeder_part[slot + head * interval_ratio]
                    if scan_assigned_head[head] == -1 and part != -1 and component_points[part] > 0:
                        component_counter += 1
                        scan_assigned_head[head] = feeder_part[slot + head * interval_ratio]
                        if component_data.loc[scan_assigned_head[head]]['nz1'] != head_nozzle[head]:
                            nozzle_counter += 1
                            if head_nozzle[head] != '':
                                nozzle_counter += 1
                        scan_cycle[head] = component_points[part]

                if len(np.nonzero(scan_cycle)[0]) == 0:
                    continue

                # 计算扫描后的代价函数,记录扫描后的最优解
                cycle = min(filter(lambda x: x > 0, scan_cycle))
                # TODO: 同时拾取计算时，考虑不同供料器宽度的影响
                eval_func = factor_simultaneous_pick * component_counter * cycle - factor_nozzle_change * nozzle_counter
                if eval_func > max_eval_func:
                    max_eval_func = eval_func
                    best_scan_assigned_head, best_scan_cycle = scan_assigned_head.copy(), scan_cycle.copy()
                    best_scan_slot = slot

            if best_scan_slot != -1:
                # 根据扫描后的周期数，更新供料器槽位布局信息
                if len(np.nonzero(assigned_cycle)[0]) != 0:
                    cycle_prev, cycle_new = min(filter(lambda x: x > 0, assigned_cycle)), min(
                        filter(lambda x: x > 0, best_scan_cycle))

                    for head in range(max_head_index):
                        if cycle_prev <= cycle_new:
                            if best_scan_cycle[head] != 0:
                                best_scan_cycle[head] = cycle_prev
                        else:
                            if assigned_cycle[head] != 0:
                                assigned_cycle[head] = cycle_new

                                component_points[feeder_part[assigned_slot[head]]] += cycle_prev - cycle_new

                for head in range(max_head_index):
                    if best_scan_cycle[head] == 0:
                        continue

                    assigned_head[head] = best_scan_assigned_head[head]
                    assigned_cycle[head] = best_scan_cycle[head]
            else:
                break

            # 从供料器基座中移除对应数量的贴装点
            cycle = min(filter(lambda x: x > 0, assigned_cycle))
            for head in range(max_head_index):
                slot = best_scan_slot + head * interval_ratio
                if best_scan_cycle[head] == 0:
                    continue
                component_points[feeder_part[slot]] -= cycle
                assigned_slot[head] = slot

            if best_scan_slot != -1 and (not -1 in assigned_head or sum([points != 0 for points in component_points]) == 0):
                break

        component_result.append(assigned_head)
        cycle_result.append(min(filter(lambda x: x > 0, assigned_cycle)))
        feederslot_result.append(assigned_slot)

        if sum([points != 0 for points in component_points]) == 0:
            break

    return component_result, cycle_result, feederslot_result