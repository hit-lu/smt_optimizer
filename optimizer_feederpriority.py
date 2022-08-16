from optimizer_common import *


def feeder_allocate(component_data, pcb_data, feeder_data, figure):
    feeder_points, mount_center_pos = defaultdict(int), defaultdict(int)  # 供料器贴装点数
    feeder_limit, feeder_arrange = defaultdict(int), defaultdict(int)

    feeder_base = [-1] * (max_slot_index // 2)   # feeder_state: 已安装在供料器基座上

    for data in pcb_data.iterrows():
        pos, part = data[1]['x'] + stopper_pos[0], data[1]['part']

        part_index = component_data[component_data['part'] == part].index.tolist()[0]
        if part not in component_data:
            feeder_limit[part_index] = component_data.loc[part_index]['feeder-limit']
            feeder_arrange[part_index] = 0

        feeder_points[part_index] += 1
        mount_center_pos[part_index] += ((pos - mount_center_pos[part_index]) / feeder_points[part_index])

    if feeder_data is not None:
        for feeder in feeder_data.iterrows():
            slot, part = feeder[1]['slot'], feeder[1]['part']
            part_index = component_data[component_data['part'] == part].index.tolist()[0]

            feeder_base[slot] = part_index
            feeder_limit[part_index] -= 1
            feeder_arrange[part_index] += 1
            if feeder_limit[part_index] < 0:
                info = 'the number of arranged feeder for [' + part + '] exceeds the quantity limit'
                raise info

    while list(feeder_arrange.values()).count(0) != 0:         # 所有待贴装点元件供料器在基座上均有安装
        best_assign = []
        best_assign_slot, best_assign_value = -1, -np.Inf
        best_assign_points = []

        for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
            feeder_assign, feeder_assign_points = [], []
            tmp_feeder_limit, tmp_feeder_points = feeder_limit.copy(), feeder_points.copy()

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

                part = max(tmp_feeder_points.keys(), key=lambda x: (tmp_feeder_limit[x], tmp_feeder_points[x]))
                tmp_feeder_limit[part] = max(0, tmp_feeder_limit[part] - 1)

                if tmp_feeder_points[part] != 0:
                    feeder_assign[idx], feeder_assign_points[idx] = part, tmp_feeder_points[part]

            assign_value = min(feeder_assign_points)
            average_slot = []
            for head, feeder_ in enumerate(feeder_assign):
                if feeder_ == 0:
                    continue
                average_slot.append(
                    (mount_center_pos[feeder_] - slotf1_pos[0]) / slot_interval + 1 - head * interval_ratio)

            average_slot = sum(average_slot) / len(average_slot)

            if assign_value >= best_assign_value:
                if assign_value == best_assign_value and abs(slot - average_slot) > abs(best_assign_slot - average_slot):
                    continue

                best_assign_value = assign_value
                best_assign = feeder_assign.copy()
                best_assign_slot = slot
                best_assign_points = feeder_assign_points

        for idx, part in enumerate(best_assign):
            if part == -1:
                continue

            # 更新供料器基座信息
            feeder_base[best_assign_slot + idx * interval_ratio] = part

            feeder_points[part] -= min(best_assign_points)
            feeder_limit[part] = max(0, feeder_limit[part] - 1)
            feeder_arrange[part] += 1

    for slot, feeder in enumerate(feeder_base):
        if feeder == -1:
            continue
        part = component_data.loc[feeder]['part']

        feeder_data.loc[len(feeder_data.index)] = [slot, part]

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

                    # 预扫描确定各类型元件拾取数目
                    preview_scan_part = defaultdict(int)
                    for head in range(max_head_index):
                        part = feeder_part[slot + head * interval_ratio]
                        # 贴装头和拾取槽位满足对应关系
                        if scan_part[head] == -1 and part != -1 and component_points[part] > 0 and scan_part.count(
                                part) < component_points[part]:
                            preview_scan_part[part] += 1

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
                                                       component_points[part] // (scan_part.count(part) + 1),
                                                       component_points[part] // preview_scan_part[part]) * (
                                                               prev_head + 1) - prev_cycle * prev_head

                            # 3.拾取移动距离条件满足: 邻近元件进行同时抓取，降低移动路径长度
                            reference_slot = -1
                            for head_, slot_ in enumerate(scan_slot):
                                if slot_ != -1:
                                    reference_slot = slot_ - head_ * interval_ratio
                            if reference_slot != -1 and abs(reference_slot - slot) > 10:
                                continue

                            # 4.同时拾取的增量 和 吸嘴更换次数比较
                            prev_nozzle_change = 0
                            if cycle_index + 1 < len(nozzle_mode):
                                prev_nozzle_change = 2 * (nozzle_cycle[head] != nozzle_mode[cycle_index + 1][head])
                            nozzle_change = 2 * (component_data.loc[part]['nz1'] != nozzle_cycle[head])
                            if cycle_index + 1 < len(nozzle_mode):
                                nozzle_change += 2 * (
                                            component_data.loc[part]['nz1'] != nozzle_mode[cycle_index + 1][head])
                            nozzle_change -= prev_nozzle_change

                            val = e_gang_pick * gang_pick_change - e_nz_change * nozzle_change
                            if val < 0:
                                continue

                            component_counter += 1

                            scan_part[head] = part
                            scan_cycle[head] = component_points[part] // preview_scan_part[part]
                            scan_slot[head] = slot + head * interval_ratio

                            # 避免重复分配，调整周期数
                            # for head_ in range(max_head_index):
                            #     if scan_part[head_] == part:
                            #         scan_cycle[head_] = component_points[part] // scan_part.count(part)

                    nozzle_counter = 0          # 吸嘴更换次数
                    # 上一周期
                    for head, nozzle in enumerate(nozzle_cycle):
                        if scan_part[head] == -1:
                            continue
                        if component_data.loc[scan_part[head]]['nz1'] != nozzle:
                            nozzle_counter += 1 if nozzle == '' else 2  # 之前没有吸嘴，记为更换1次，否则记为更换2次（装/卸各1次）
                            if cycle_index == 0:
                                nozzle_counter += 1

                    # 下一周期（额外增加的吸嘴更换次数）
                    if cycle_index + 1 < len(nozzle_mode):
                        for head, nozzle in enumerate(nozzle_mode[cycle_index + 1]):
                            if scan_part[head] == -1:
                                continue
                            prev_counter, new_counter = 0, 0
                            if nozzle_cycle[head] != nozzle:
                                prev_counter += 2 if nozzle == '' else 1
                            if component_data.loc[scan_part[head]]['nz1'] != nozzle:
                                new_counter += 2 if nozzle == '' else 1
                            nozzle_counter += new_counter - prev_counter

                    if component_counter == 0:      # 当前情形下未扫描到任何元件
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

                    eval_func = e_gang_pick * (component_counter - 1) * cycle - e_nz_change * nozzle_counter
                    if eval_func > scan_eval_func:
                        scan_eval_func = eval_func
                        best_scan_part, best_scan_cycle = scan_part.copy(), scan_cycle.copy()
                        best_scan_slot = scan_slot.copy()

                # if search_break or scan_eval_func < 0:
                if search_break:
                    break

                scan_eval_func_list.append(scan_eval_func)

                cur_scan_part = best_scan_part.copy()
                cur_scan_slot = best_scan_slot.copy()
                cur_scan_cycle = best_scan_cycle.copy()

            if len(scan_eval_func_list) != 0 and sum(scan_eval_func_list) > best_assigned_eval_func:

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

    return component_result, cycle_result, feeder_slot_result

