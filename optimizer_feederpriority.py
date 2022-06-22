import numpy as np

from dataloader import *
from common_function import *


# TODO 1: 如何处理不同类型吸嘴的情形（增加吸嘴分配），能否使用标准求解器求解（需要查阅相关文献是否进行过类似工作）-
# TODO 2: 如何处理不同宽度喂料器 ?
# TODO 3: 如何处理供料器位置分配的边界条件（占位数量≈可用槽位数） ×
# TODO 4: 如何平衡供料器前基座和后基座之间的分配元件数量（目前仅考虑前基座优化） ×
# TODO 5: 如何进一步提升效能
# TODO 6: 扩大测试范围，保存中间测试数据
# TODO 7: 可用供料器数目 > 1时的处理（以解决IPC9850为导向）
# TODO 8: 扫描过程考虑吸嘴安装任务 √
# TODO 9: 中心位置的选取（在处理拾取数相同时）
# TODO 10: 指定吸嘴功能 ×
# TODO 11: 算法效率提升，后期需要考虑扩大吸嘴搜索范围，如何对feeder_allocate函数进行提效（目标是5。03s压缩到1s内）
# TODO 12: 吸嘴->元件对分配结果进行分组 √
# TODO 13: 估计时间时考虑吸嘴更换等因素，降低估计时间和实际时间的差距 -
# TODO 14: 贴装路径的估计与优化（以分析解决IPC9850为导向,主要问题还是在于点聚集性问题，能否考虑使用C-W算法进行求解，以及如何适配于不同类型吸嘴） -
# TODO 15: branch and price解决贴装路径规划问题

def nozzle_assignment(component_data, pcb_data):
    nozzle_points = {}

    # 统计各类型吸嘴的贴装点数
    for step in pcb_data.iterrows():
        part = step[1]['part']
        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz1']
        if nozzle not in nozzle_points.keys():
            nozzle_points[nozzle] = 0

        nozzle_points[nozzle] += 1

    nozzle_result, nozzle_cycle = [], []

    # nozzle_result.append(['CN140', 'CN065', 'CN065', 'CN065', 'CN065', 'CN065'])
    # nozzle_result.append(['CN140', 'CN065', 'CN065', 'CN065', 'CN065', 'CN140'])
    # nozzle_result.append(['CN140', 'CN065', 'CN140', 'CN065', 'CN065', 'CN065'])
    # nozzle_result.append(['Empty', 'CN065', 'CN140', 'CN065', 'CN065', 'CN065'])
    #
    # nozzle_cycle.append(182)
    # nozzle_cycle.append(38)
    # nozzle_cycle.append(30)
    # nozzle_cycle.append(2)

    nozzle_result.append(['CN140', 'CN065', 'CN065', 'CN065', 'CN065', 'CN065'])
    nozzle_result.append(['CN140', 'CN065', 'CN065', 'CN065', 'CN065', 'CN140'])
    nozzle_result.append(['Empty', 'CN065', 'CN140', 'CN065', 'CN065', 'CN065'])

    nozzle_cycle.append(182)
    nozzle_cycle.append(68)
    nozzle_cycle.append(2)

    return nozzle_result, nozzle_cycle



@timer_warper
def feeder_allocate(component_data, pcb_data, feeder_data, nozzle_result, nozzle_cycle, figure):
    # 深拷贝用于后续计算
    nozzle_result, nozzle_cycle = nozzle_result.copy(), nozzle_cycle.copy()

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

        nozzle_idx = nozzle_cycle.index(max(nozzle_cycle))  # 当前匹配的吸嘴模式
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

            # 吸嘴模式匹配
            nozzle_unmatch_counter = 0
            for idx, feeder in enumerate(feeder_assign):
                if feeder == -1:
                    continue
                if component_data.loc[feeder]['nz1'] != nozzle_result[nozzle_idx][idx]:
                    nozzle_unmatch_counter += 1

            # 分配新的供料器
            for idx, feeder in enumerate(feeder_assign):
                if feeder != -1:
                    continue

                while True:
                    # 选取未贴装元件中对应点数最多的元件
                    part = max(tmp_feeder_points.keys(), key=(
                        lambda x: tmp_feeder_points[x] if nozzle_result[nozzle_idx][idx] == component_data.loc[x][
                            'nz1'] else 0))

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

            assign_value = min(feeder_assign_points) - nozzle_unmatch_counter * 100
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

        # 更新吸嘴匹配信息
        if best_assign_points:
            nozzle_cycle[nozzle_idx] -= min(best_assign_points)

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
            plt.text(slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1] + 12,
                     part, ha='center', size=7, rotation=90)
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


def feederbase_scan(component_data, pcb_data, feeder_data, nozzle_result, nozzle_cycle):
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

    component_result, cycle_result, feederslot_result = [], [], [] # 贴装点索引和拾取槽位优化结果
    nozzle_idx_list = []    # 用于调整当前扫描结果的插入位置
    while True:
        # === 周期内循环 ===
        assigned_head = [-1 for _ in range(max_head_index)]  # 当前扫描到的头分配元件信息
        assigned_cycle = [0 for _ in range(max_head_index)]  # 当前扫描到的元件最大分配次数
        assigned_slot = [-1 for _ in range(max_head_index)]

        nozzle_idx = nozzle_cycle.index(max(nozzle_cycle))
        while True:
            max_eval_func = -np.inf
            # 前供料器基座扫描，TODO:后供料器吸嘴扫描
            best_scan_assigned_head, best_scan_cycle = [], []
            best_scan_slot = -1
            for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
                scan_cycle = [0 for _ in range(max_head_index)]
                scan_assigned_head = assigned_head.copy()
                component_counter, nozzle_counter = 0, 0
                for head in range(max_head_index):
                    # TODO: 可用吸嘴数限制
                    part = feeder_part[slot + head * interval_ratio]
                    if scan_assigned_head[head] == -1 and part != -1 and component_points[part] > 0 \
                            and component_data.loc[part]['nz1'] == nozzle_result[nozzle_idx][head]:
                        component_counter += 1
                        scan_assigned_head[head] = feeder_part[slot + head * interval_ratio]
                        if component_data.loc[scan_assigned_head[head]]['nz1'] != head_nozzle[head]:
                            nozzle_counter += 1
                            if head_nozzle[head] != '':
                                nozzle_counter += 1
                        scan_cycle[head] = min(component_points[part], nozzle_cycle[nozzle_idx])

                if len(np.nonzero(scan_cycle)[0]) == 0:
                    continue

                # 计算扫描后的代价函数,记录扫描后的最优解
                cycle = min(filter(lambda x: x > 0, scan_cycle))

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
            cycle = min(cycle, nozzle_cycle[nozzle_idx])
            for head in range(max_head_index):
                slot = best_scan_slot + head * interval_ratio
                if best_scan_cycle[head] == 0:
                    continue
                component_points[feeder_part[slot]] -= cycle
                assigned_slot[head] = slot

            if best_scan_slot != -1 and (
                    not -1 in assigned_head or sum([points != 0 for points in component_points]) == 0):
                break

        insert_pos = len(nozzle_idx_list)
        for idx, val in enumerate(nozzle_idx_list):
            if val > nozzle_idx:
                insert_pos = idx - 1
                break

        cycle = min(filter(lambda x: x > 0, assigned_cycle))
        component_result.insert(insert_pos, assigned_head)
        cycle_result.insert(insert_pos, cycle)
        feederslot_result.insert(insert_pos, assigned_slot)

        nozzle_idx_list.insert(insert_pos, nozzle_idx)
        nozzle_cycle[nozzle_idx] -= cycle
        if sum([points != 0 for points in component_points]) == 0:
            break

    return component_result, cycle_result, feederslot_result

