from optimizer_common import *


def feeder_allocate(component_data, pcb_data, feeder_data, figure):
    feeder_points, mount_center_pos = defaultdict(int), defaultdict(int)  # 供料器贴装点数
    feeder_base, feeder_state = [-1] * (max_slot_index // 2), [True] * len(component_data)  # feeder_state: 已安装在供料器基座上

    for data in pcb_data.iterrows():
        pos, part = data[1]['x'] + stopper_pos[0], data[1]['part']
        part_index = component_data[component_data['part'] == part].index.tolist()[0]
        feeder_state[part_index] = False

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

            assign_value = min(feeder_assign_points) - nozzle_change_counter * factor_nozzle_change
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
    nozzle_mode, nozzle_mode_cycle = [['' for _ in range(max_head_index)]], [0]        # 吸嘴匹配模式
    while True:
        # === 周期内循环 ===
        assigned_part = [-1 for _ in range(max_head_index)]  # 当前扫描到的头分配元件信息
        assigned_cycle = [0 for _ in range(max_head_index)]  # 当前扫描到的元件最大分配次数
        assigned_slot = [-1 for _ in range(max_head_index)]  # 当前扫描到的供料器分配信息

        best_assigned_eval_func = -float('inf')
        nozzle_insert_cycle = 0
        for cycle_index, nozzle_cycle in enumerate(nozzle_mode):
            scan_eval_func_list = []        # 若干次扫描得到的最优解

            # nozzle_cycle 吸嘴模式下，已扫描到的最优结果
            cur_scan_part = [-1 for _ in range(max_head_index)]
            cur_scan_cycle = [0 for _ in range(max_head_index)]
            cur_scan_slot = [-1 for _ in range(max_head_index)]

            while True:
                best_scan_part, best_scan_cycle = [-1 for _ in range(max_head_index)], [-1 for _ in
                                                                                        range(max_head_index)]
                best_scan_slot = [-1 for _ in range(max_head_index)]
                scan_eval_func, search_break = -float('inf'), True

                # 前供料器基座扫描
                for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
                    scan_cycle, scan_part, scan_slot = cur_scan_cycle.copy(), cur_scan_part.copy(), cur_scan_slot.copy()
                    component_counter = 0
                    for head in range(max_head_index):
                        part = feeder_part[slot + head * interval_ratio]
                        # 1.匹配条件满足: 贴装头和拾取槽位满足对应关系
                        if scan_part[head] == -1 and part != -1 and component_points[part] > 0 and scan_part.count(
                                part) < component_points[part]:

                            # 2.增量条件满足: 引入新的元件类型不会使代价函数的值减少
                            if scan_cycle.count(0) == max_head_index:
                                gang_pick_change = component_points[part]
                            else:
                                prev_cycle = min(filter(lambda x: x > 0, scan_cycle))
                                prev_head = len([part for part in scan_part if part != -1])

                                # 同时拾取数的提升
                                gang_pick_change = min(prev_cycle,
                                                       component_points[part] // (scan_part.count(part) + 1)) * (
                                                               prev_head + 1) - prev_cycle * prev_head

                            # 3.拾取移动距离条件满足: 邻近元件进行同时抓取，降低移动路径长度
                            reference_slot = -1
                            for head_, slot_ in enumerate(scan_slot):
                                if slot_ != -1:
                                    reference_slot = slot_ - head_ * interval_ratio
                            if reference_slot != -1 and abs(reference_slot - slot) > len(component_result) * 2:
                                continue

                            # 吸嘴更换次数的下降
                            nozzle_change = 2 * (component_data.loc[part]['nz1'] != nozzle_cycle[head])

                            val = factor_simultaneous_pick * gang_pick_change - factor_nozzle_change * nozzle_change
                            if val < 0:
                                continue

                            component_counter += 1

                            scan_part[head] = part
                            scan_cycle[head] = component_points[part]
                            scan_slot[head] = slot + head * interval_ratio

                            # 避免重复分配，调整周期数
                            for head_ in range(max_head_index):
                                if scan_part[head_] == part:
                                    scan_cycle[head_] = component_points[part] // scan_part.count(part)

                    nozzle_counter = 0          # 吸嘴更换次数
                    # 上一周期
                    for head, nozzle in enumerate(nozzle_cycle):
                        if scan_part[head] == -1:
                            continue
                        if component_data.loc[scan_part[head]]['nz1'] != nozzle:
                            nozzle_counter += 1 if nozzle != '' else 2  # 之前没有吸嘴，记为更换1次，否则记为更换2次（装/卸各1次）

                    # 下一周期（额外增加的吸嘴更换次数）
                    # if cycle_index + 1 < len(nozzle_mode):
                    #     for head, nozzle in enumerate(nozzle_mode[cycle_index + 1]):
                    #         if scan_part[head] == -1:
                    #             continue
                    #         prev_counter, new_counter = 0, 0
                    #         if nozzle_mode[cycle_index][head] != nozzle:
                    #             prev_counter += 1 if nozzle != '' else 2
                    #         if component_data.loc[scan_part[head]]['nz1'] != nozzle:
                    #             new_counter += 1 if nozzle != '' else 2
                    #         nozzle_counter += new_counter - prev_counter

                    if component_counter == 0:
                        continue

                    search_break = False

                    scan_part_head = defaultdict(list)
                    for head, part in enumerate(scan_part):
                        if part == -1:
                            continue
                        scan_part_head[part].append(head)

                    for part, heads in scan_part_head.items():
                        part_cycle = component_points[part] // len(heads)
                        for head in heads:
                            scan_cycle[head] = part_cycle

                    # 计算扫描后的代价函数,记录扫描后的最优解
                    cycle = min(filter(lambda x: x > 0, scan_cycle))

                    eval_func = factor_simultaneous_pick * component_counter * cycle - factor_nozzle_change * nozzle_counter
                    if eval_func > scan_eval_func:
                        scan_eval_func = eval_func
                        best_scan_part, best_scan_cycle = scan_part.copy(), scan_cycle.copy()
                        best_scan_slot = scan_slot.copy()

                if search_break or scan_eval_func < 0:
                    break

                scan_eval_func_list.append(scan_eval_func)

                cur_scan_part = best_scan_part.copy()
                cur_scan_slot = best_scan_slot.copy()
                cur_scan_cycle = best_scan_cycle.copy()

            if sum(scan_eval_func_list) > best_assigned_eval_func:
                best_assigned_eval_func = sum(scan_eval_func_list)

                assigned_part = cur_scan_part.copy()
                assigned_slot = cur_scan_slot.copy()
                assigned_cycle = cur_scan_cycle.copy()

                nozzle_insert_cycle = cycle_index

        # 从供料器基座中移除对应数量的贴装点
        cycle = min(filter(lambda x: x > 0, assigned_cycle))
        for head, slot in enumerate(assigned_slot):
            if assigned_part[head] == -1:
                continue
            component_points[feeder_part[slot]] -= cycle

        component_result.insert(nozzle_insert_cycle, assigned_part)
        cycle_result.insert(nozzle_insert_cycle, cycle)
        feeder_slot_result.insert(nozzle_insert_cycle, assigned_slot)

        # 更新吸嘴匹配模式
        cycle_nozzle = nozzle_mode[nozzle_insert_cycle].copy()
        for head, component in enumerate(assigned_part):
            if component == -1:
                continue
            cycle_nozzle[head] = component_data.loc[component]['nz1']

        nozzle_mode.insert(nozzle_insert_cycle + 1, cycle_nozzle)
        if sum(component_points) == 0:
            break

    nozzle_result = []
    for idx, components in enumerate(component_result):
        nozzle_cycle = ['' for _ in range(max_head_index)]
        for hd, component in enumerate(components):
            if component == -1:
                continue
            nozzle_cycle[hd] = component_data.loc[component]['nz1']
        nozzle_result.append(nozzle_cycle)

    return component_result, cycle_result, feeder_slot_result

