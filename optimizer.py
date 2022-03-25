from result_analysis import *
from dataloader import *

# 读取供料器基座信息
feeder_base = pd.Series(np.zeros(max_slot_index), np.arange(max_slot_index), dtype = np.int16)
feeder_part = pd.Series(-np.ones(max_slot_index), np.arange(max_slot_index), dtype = np.int16)

for point_cnt in range(point_num):
    slot = pcb_data.loc[point_cnt, 'fdr'].split(' ')[0]
    slot, part = int(slot[1:]) - 1, pcb_data.loc[point_cnt, 'fdr'].split(' ', 1)[1]
    feeder_base.loc[slot] += 1

    index = np.where(component_data['part'].values == part)
    if feeder_part[slot] == -1:
        feeder_part[slot] = index[0]

# feeder_base[:max_slot_index // 2].plot(kind = 'bar')
# plt.show()

# 扫描供料器基座分配元件
head_assigned_nozzle = ['' for _ in range(max_head_index)]    # 头上已经分配
factor_nozzle_change = 0.5
factor_simultaneous_pick = 1. / max_head_index

best_assigned_component = []
best_assigned_cycle = 0

# TODO: 供料器基座位置布局（目前采用已布局结果，需要研究不同供料器位置布局对结果的影响）

# === 供料器基座扫描 ===
component_result, cycle_result = [], []
placement_result, feederslot_result = [], []             # 贴装点索引和拾取槽位优化结果
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
                if scan_assigned_head[head] == -1 and feeder_part[slot + head * interval_ratio] != -1\
                        and feeder_base[slot + head * interval_ratio] > 0:
                    component_counter += 1
                    scan_assigned_head[head] = feeder_part[slot + head * interval_ratio]
                    if component_data.loc[scan_assigned_head[head]]['nz1'] != head_assigned_nozzle[head]:
                        nozzle_counter += 1
                        if head_assigned_nozzle[head] != '':
                            nozzle_counter += 1
                    scan_cycle[head] = feeder_base[slot + head * interval_ratio]

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
                cycle_prev, cycle_new = min(filter(lambda x: x > 0, assigned_cycle)), min(filter(lambda x: x > 0, best_scan_cycle))

                for head in range(max_head_index):
                    if cycle_prev <= cycle_new:
                        if best_scan_cycle[head] != 0:
                            best_scan_cycle[head] = cycle_prev
                    else:
                        if assigned_cycle[head] != 0:
                            assigned_cycle[head] = cycle_new
                            feeder_base[assigned_slot[head]] += cycle_prev - cycle_new

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
            feeder_base[slot] -= cycle
            assigned_slot[head] = slot

        if best_scan_slot != -1 and (not -1 in assigned_head or len(np.where(feeder_base.values > 0)[0]) == 0):
            break

    component_result.append(assigned_head)
    cycle_result.append(min(filter(lambda x: x > 0, assigned_cycle)))
    feederslot_result.append(assigned_slot)
    # feeder_base[:max_slot_index // 2].plot(kind = 'bar')
    # plt.show()
    if len(np.where(feeder_base.values > 0)[0]) == 0:
        break


# === 贴装路径规划 ===
mountpoint_index = [[] for _ in range(len(component_data))]
mountpoint_pos = [[] for _ in range(len(component_data))]

for i in range(len(pcb_data)):
    part = pcb_data.loc[i]['part']
    component_index = component_data[component_data['part'] == part].index.tolist()[0]
    # 记录贴装点序号索引和对应的位置坐标
    mountpoint_index[component_index].append(i)
    mountpoint_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

for cycle_set in range(len(component_result)):
    floor_cycle, ceil_cycle = sum(cycle_result[:cycle_set]), sum(cycle_result[:(cycle_set + 1)])
    for cycle in range(floor_cycle, ceil_cycle):
        assigned_placement = [-1 for _ in range(max_head_index)]
        way_point = []
        for head in range(max_head_index):
            if component_result[cycle_set][head] == -1:
                continue

            component_index = component_result[cycle_set][head]
            if head == 0:
                index = np.argmin(mountpoint_pos[component_index], axis = 0)[0]
                assigned_placement[head] = mountpoint_index[component_index][index]
                way_point.append(mountpoint_pos[component_index][index])        # 记录路标点

                mountpoint_index[component_index].pop(index)
                mountpoint_pos[component_index].pop(index)
            else:
                head_index, point_index = -1, -1
                min_cheby_distance, min_euler_distance = np.inf, np.inf
                for next_head in range(1, max_head_index):
                    if assigned_placement[next_head] != -1:
                        continue

                    component_index = component_result[cycle_set][next_head]
                    for counter1 in range(len(mountpoint_pos[component_index])):
                        for counter2 in range(len(way_point)):
                            delta_x = abs(mountpoint_pos[component_index][counter1][0] - way_point[counter2][0]  - head_index * slot_interval)
                            delta_y = abs(mountpoint_pos[component_index][counter1][1] - way_point[counter2][1])
                            cheby_distance = max(delta_x, delta_y)
                            euler_distance = pow(delta_x, 2) + pow(delta_y, 2)
                            if cheby_distance < min_cheby_distance or \
                                    (abs(cheby_distance - min_cheby_distance) < 10 and euler_distance < min_euler_distance):
                                min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                head_index, point_index = next_head, counter1

                component_index = component_result[cycle_set][head_index]
                if point_index >= len(mountpoint_index[component_index]) or point_index < 0:
                    print('')
                assigned_placement[head_index] = mountpoint_index[component_index][point_index]

                way_point.append(mountpoint_pos[component_index][point_index])  # 记录路标点(相对于吸杆1的位置)
                way_point[-1][0] -= head_index * slot_interval

                mountpoint_index[component_index].pop(point_index)
                mountpoint_pos[component_index].pop(point_index)

        placement_result.append(assigned_placement)

# 绘制各周期从供料器拾取的贴装点示意图
# pickup_cycle_schematic(feederslot_result, cycle_result)

# 绘制贴装路径图
# placement_route_schematic(component_result, cycle_result, feederslot_result, placement_result, 3)

# 估算贴装用时
placement_time_estimate(component_result, cycle_result, feederslot_result, placement_result)

print(component_result)
print(cycle_result)
print(feederslot_result)
print(placement_result)



