from optimizer_common import *


@timer_wrapper
def feeder_allocate(component_data, pcb_data, feeder_data, figure=False):

    feeder_points, feeder_division_points = defaultdict(int), defaultdict(int)   # 供料器贴装点数
    mount_center_pos = defaultdict(int)

    feeder_limit, feeder_arrange = defaultdict(int), defaultdict(int)
    feeder_nozzle = defaultdict(str)

    feeder_base = [-1] * max_slot_index   # 已安装在供料器基座上的元件
    feeder_base_points = [0] * max_slot_index   # 供料器基座结余贴装点数量

    for data in pcb_data.iterrows():
        pos, part = data[1]['x'] + stopper_pos[0], data[1]['part']

        part_index = component_data[component_data['part'] == part].index.tolist()[0]
        if part not in component_data:
            feeder_limit[part_index] = component_data.loc[part_index]['feeder-limit']
            feeder_arrange[part_index] = 0

        feeder_points[part_index] += 1
        mount_center_pos[part_index] += ((pos - mount_center_pos[part_index]) / feeder_points[part_index])
        feeder_nozzle[part_index] = component_data.loc[part_index]['nz1']

    for part_index, points in feeder_points.items():
        feeder_division_points[part_index] = points // feeder_limit[part_index]

    nozzle_component, nozzle_component_points = defaultdict(list), defaultdict(list)
    for part, nozzle in feeder_nozzle.items():
        for _ in range(feeder_limit[part]):
            nozzle_component[nozzle].append(part)
            nozzle_component_points[nozzle].append(feeder_points[part])

    if feeder_data is not None:
        for _, feeder in feeder_data.iterrows():
            slot, part = feeder['slot'], feeder['part']
            part_index = component_data[component_data['part'] == part].index.tolist()[0]

            feeder_base[slot] = part_index
            feeder_limit[part_index] -= 1
            feeder_arrange[part_index] += 1
            if feeder_limit[part_index] < 0:
                info = 'the number of arranged feeder for [' + part + '] exceeds the quantity limit'
                raise ValueError(info)

    nozzle_pattern = optimal_nozzle_assignment(component_data, pcb_data)
    while list(feeder_arrange.values()).count(0) != 0:         # 所有待贴装点元件供料器在基座上均有安装
        best_assign, best_assign_points = [], []

        best_assign_slot, best_assign_value = -1, -np.Inf
        best_nozzle_component, best_nozzle_component_points = None, None
        for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
            feeder_assign, feeder_assign_points = [], []
            tmp_feeder_limit, tmp_feeder_points = feeder_limit.copy(), feeder_points.copy()
            tmp_nozzle_component, tmp_nozzle_component_points = copy.deepcopy(nozzle_component), copy.deepcopy(
                nozzle_component_points)

            # 记录扫描到的已安装的供料器元件类型
            for head in range(max_head_index):
                feeder_assign.append(feeder_base[slot + head * interval_ratio])
                if feeder_assign[-1] != -1:
                    feeder_assign_points.append(feeder_base_points[slot + head * interval_ratio])
                else:
                    feeder_assign_points.append(0)

            if -1 not in feeder_assign:
                continue

            assign_part_stack, assign_part_stack_points = [], []
            for idx, feeder in enumerate(feeder_assign):
                if feeder != -1:
                    continue

                part_nozzle = nozzle_pattern[idx]
                if len(tmp_nozzle_component[part_nozzle]) == 0:
                    # 当前头对应吸嘴类型无可用元件
                    part = max(tmp_feeder_points.keys(),
                               key=lambda x: tmp_feeder_points[x] / tmp_feeder_limit[x] if tmp_feeder_limit[
                                                                                               x] != 0 else 0)
                    for nozzle, component_list in tmp_nozzle_component.items():
                        if part in component_list:
                            part_nozzle = nozzle

                            assign_part_stack.append(part)
                            assign_part_stack_points.append(feeder_division_points[part])
                            break
                else:
                    index_ = tmp_nozzle_component_points[part_nozzle].index(
                        max(tmp_nozzle_component_points[part_nozzle]))
                    part = tmp_nozzle_component[part_nozzle][index_]
                    feeder_assign[idx], feeder_assign_points[idx] = part, feeder_division_points[part]

                if part in tmp_nozzle_component[part_nozzle]:
                    part_index = tmp_nozzle_component[part_nozzle].index(part)

                    tmp_nozzle_component[part_nozzle].pop(part_index)
                    tmp_nozzle_component_points[part_nozzle].pop(part_index)

                    tmp_feeder_limit[part] -= 1
                    tmp_feeder_points[part] -= feeder_division_points[part]

            # 在分配元件堆栈中，分配吸嘴类型一致的头
            for head, feeder in enumerate(feeder_assign):
                if feeder != -1:
                    continue
                for idx, part in enumerate(assign_part_stack):
                    if component_data.loc[part]['nz1'] == nozzle_pattern[head]:
                        feeder_assign[head], feeder_assign_points[head] = assign_part_stack[idx], \
                                                                          assign_part_stack_points[idx]

                        assign_part_stack.pop(idx)
                        assign_part_stack_points.pop(idx)
                        break

            # 分配元件堆栈中未分配的元件
            for head, feeder in enumerate(feeder_assign):
                if feeder != -1 or len(assign_part_stack) == 0:
                    continue

                feeder_assign[head], feeder_assign_points[head] = assign_part_stack[0], assign_part_stack_points[0]

                assign_part_stack.pop(0)
                assign_part_stack_points.pop(0)

            nozzle_change_counter = 0
            average_slot = []
            for head, feeder_ in enumerate(feeder_assign):
                if feeder_ == -1:
                    continue
                average_slot.append(
                    (mount_center_pos[feeder_] - slotf1_pos[0]) / slot_interval + 1 - head * interval_ratio)
                if component_data.loc[feeder_]['nz1'] != nozzle_pattern[head]:
                    # nozzle_change_counter += 1
                    pass

            average_slot = sum(average_slot) / len(average_slot)

            assign_value = 0
            feeder_assign_points_cpy = feeder_assign_points.copy()
            while True:
                points_filter = list(filter(lambda x: x > 0, feeder_assign_points_cpy))
                if not points_filter:
                    break
                assign_value += e_gang_pick * min(points_filter) * (len(points_filter) - 1)
                for head, _ in enumerate(feeder_assign_points_cpy):
                    if feeder_assign_points_cpy[head] == 0:
                        continue
                    feeder_assign_points_cpy[head] -= min(points_filter)

            assign_value -= e_nz_change * nozzle_change_counter
            # assign_value = e_gang_pick * min(filter(lambda x: x > 0, feeder_assign_points)) * (
            #             max_head_index - feeder_assign_points.count(0)) - e_nz_change * nozzle_change_counter

            if assign_value >= best_assign_value:
                if assign_value == best_assign_value and abs(slot - average_slot) > abs(best_assign_slot - average_slot):
                    continue

                best_assign_value = assign_value
                best_assign = feeder_assign.copy()
                best_assign_points = feeder_assign_points.copy()
                best_assign_slot = slot
                best_nozzle_component, best_nozzle_component_points = tmp_nozzle_component, tmp_nozzle_component_points

        for idx, part in enumerate(best_assign):
            if part == -1:
                continue

            # 除去分配给最大化同时拾取周期的项，保留结余项
            feeder_base_points[best_assign_slot + idx * interval_ratio] += (feeder_division_points[part] - min(best_assign_points))

            # 新安装的供料器
            if feeder_base[best_assign_slot + idx * interval_ratio] != part:
                feeder_points[part] -= feeder_division_points[part]
                feeder_limit[part] -= 1
                feeder_arrange[part] += 1

                if feeder_limit[part] == 0:
                    feeder_division_points[part] = 0

            # 更新供料器基座信息
            feeder_base[best_assign_slot + idx * interval_ratio] = part

            # 更新吸嘴信息
            nozzle_pattern[idx] = component_data.loc[part]['nz1']

        nozzle_component, nozzle_component_points = copy.deepcopy(best_nozzle_component), copy.deepcopy(
            best_nozzle_component_points)

    # 更新供料器占位信息
    for slot, feeder in enumerate(feeder_base):
        if feeder == -1:
            continue
        part = component_data.loc[feeder]['part']

        feeder_data.loc[len(feeder_data.index)] = [slot, part, 0]

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

    feeder_part = [-1] * max_slot_index
    for feeder in feeder_data.iterrows():
        part, slot = feeder[1]['part'], feeder[1]['slot']
        component_index = component_data[component_data['part'] == part].index.tolist()
        if len(component_index) != 1:
            print('unregistered component: ', part, ' in slot', slot)
            continue
        component_index = component_index[0]
        feeder_part[slot] = component_index

    component_result, cycle_result, feeder_slot_result = [], [], []  # 贴装点索引和拾取槽位优化结果

    nozzle_mode = [optimal_nozzle_assignment(component_data, pcb_data)]      # 吸嘴匹配模式
    with tqdm(total=len(pcb_data)) as pbar:
        pbar.set_description('feeder scan process')
        pbar_prev = 0
        value_increment_base = 0
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
                    for slot in range(1, max_slot_index // 2 - (max_head_index - 1) * interval_ratio + 1):
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

                                # 2.增量条件满足: 引入新的元件类型不会使代价函数的值减少(前瞻)
                                if scan_cycle.count(0) == max_head_index:
                                    gang_pick_change = component_points[part]
                                else:
                                    prev_cycle = min(filter(lambda x: x > 0, scan_cycle))

                                    # 同时拾取数的提升
                                    gang_pick_change = min(prev_cycle, component_points[part] // preview_scan_part[part])

                                # 3.拾取移动距离条件满足: 邻近元件进行同时抓取，降低移动路径长度
                                # reference_slot = -1
                                # for head_, slot_ in enumerate(scan_slot):
                                #     if slot_ != -1:
                                #         reference_slot = slot_ - head_ * interval_ratio
                                # if reference_slot != -1 and abs(reference_slot - slot) > 10:
                                #     continue

                                # 4.同时拾取的增量 和 吸嘴更换次数比较
                                prev_nozzle_change = 0
                                if cycle_index + 1 < len(nozzle_mode):
                                    prev_nozzle_change = 2 * (nozzle_cycle[head] != nozzle_mode[cycle_index + 1][head])

                                # 避免首个周期吸杆占用率低的问题
                                if nozzle_cycle[head] == '':
                                    nozzle_change = 0
                                else:
                                    nozzle_change = 2 * (component_data.loc[part]['nz1'] != nozzle_cycle[head])

                                if cycle_index + 1 < len(nozzle_mode):
                                    nozzle_change += 2 * (
                                                component_data.loc[part]['nz1'] != nozzle_mode[cycle_index + 1][head])
                                nozzle_change -= prev_nozzle_change

                                val = e_gang_pick * gang_pick_change - e_nz_change * nozzle_change
                                if val < value_increment_base:
                                    continue

                                component_counter += 1

                                scan_part[head] = part
                                scan_cycle[head] = component_points[part] // preview_scan_part[part]
                                scan_slot[head] = slot + head * interval_ratio

                        nozzle_counter = 0          # 吸嘴更换次数
                        # 上一周期
                        for head, nozzle in enumerate(nozzle_cycle):
                            if scan_part[head] == -1:
                                continue
                            if component_data.loc[scan_part[head]]['nz1'] != nozzle and nozzle != '':
                                nozzle_counter += 2

                        # 下一周期（额外增加的吸嘴更换次数）
                        if cycle_index + 1 < len(nozzle_mode):
                            for head, nozzle in enumerate(nozzle_mode[cycle_index + 1]):
                                if scan_part[head] == -1:
                                    continue
                                prev_counter, new_counter = 0, 0
                                if nozzle_cycle[head] != nozzle and nozzle_cycle[head] != '' and nozzle != '':
                                    prev_counter += 2
                                if component_data.loc[scan_part[head]]['nz1'] != nozzle and nozzle != '':
                                    new_counter += 2
                                nozzle_counter += new_counter - prev_counter
                        else:
                            for head, nozzle in enumerate(nozzle_mode[0]):
                                if scan_part[head] == -1:
                                    continue
                                prev_counter, new_counter = 0, 0
                                if nozzle_cycle[head] != nozzle and nozzle_cycle[head] != '' and nozzle != '':
                                    prev_counter += 2
                                if component_data.loc[scan_part[head]]['nz1'] != nozzle and nozzle != '':
                                    new_counter += 2
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
                        # 短期收益
                        cycle = min(filter(lambda x: x > 0, scan_cycle))
                        gang_pick_counter, gang_pick_slot_set = 0, set()
                        for head, pick_slot in enumerate(scan_slot):
                            gang_pick_slot_set.add(pick_slot - head * interval_ratio)

                        eval_func_short_term = e_gang_pick * (max_head_index - scan_slot.count(-1) - len(
                            gang_pick_slot_set)) * cycle - e_nz_change * nozzle_counter

                        # 长期收益
                        gang_pick_slot_dict = defaultdict(list)
                        for head, pick_slot in enumerate(scan_slot):
                            if pick_slot == -1:
                                continue
                            gang_pick_slot_dict[pick_slot - head * interval_ratio].append(scan_cycle[head])

                        eval_func_long_term = 0
                        for pick_cycle in gang_pick_slot_dict.values():
                            while pick_cycle:
                                min_cycle = min(pick_cycle)
                                eval_func_long_term += e_gang_pick * (len(pick_cycle) - 1) * min(pick_cycle)
                                pick_cycle = list(map(lambda c: c - min_cycle, pick_cycle))
                                pick_cycle = list(filter(lambda c: c > 0, pick_cycle))
                        eval_func_long_term -= e_nz_change * nozzle_counter

                        ratio = 0.5
                        eval_func = (1 - ratio) * eval_func_short_term + ratio * eval_func_long_term
                        if eval_func >= scan_eval_func:
                            scan_eval_func = eval_func
                            best_scan_part, best_scan_cycle = scan_part.copy(), scan_cycle.copy()
                            best_scan_slot = scan_slot.copy()

                    if search_break:
                        break

                    scan_eval_func_list.append(scan_eval_func)

                    cur_scan_part = best_scan_part.copy()
                    cur_scan_slot = best_scan_slot.copy()
                    cur_scan_cycle = best_scan_cycle.copy()

                if len(scan_eval_func_list) != 0:
                    if sum(scan_eval_func_list) >= best_assigned_eval_func:
                        best_assigned_eval_func = sum(scan_eval_func_list)

                        assigned_part = cur_scan_part.copy()
                        assigned_slot = cur_scan_slot.copy()
                        assigned_cycle = cur_scan_cycle.copy()

                        nozzle_insert_cycle = cycle_index

            # 从供料器基座中移除对应数量的贴装点
            nonzero_cycle = [cycle for cycle in assigned_cycle if cycle > 0]
            if not nonzero_cycle:
                value_increment_base -= max_head_index
                continue

            for head, slot in enumerate(assigned_slot):
                if assigned_part[head] == -1:
                    continue
                component_points[feeder_part[slot]] -= min(nonzero_cycle)

            component_result.insert(nozzle_insert_cycle, assigned_part)
            cycle_result.insert(nozzle_insert_cycle, min(nonzero_cycle))
            feeder_slot_result.insert(nozzle_insert_cycle, assigned_slot)

            # 更新吸嘴匹配模式
            cycle_nozzle = nozzle_mode[nozzle_insert_cycle].copy()
            for head, component in enumerate(assigned_part):
                if component == -1:
                    continue
                cycle_nozzle[head] = component_data.loc[component]['nz1']

            nozzle_mode.insert(nozzle_insert_cycle + 1, cycle_nozzle)

            pbar.update(len(pcb_data) - sum(component_points) - pbar_prev)
            pbar_prev = len(pcb_data) - sum(component_points)
            if sum(component_points) == 0:
                break

    return component_result, cycle_result, feeder_slot_result

