import matplotlib.pyplot as plt
from dataloader import *
from common_function import *

# 将步骤列表中已有的数据转换为可计算格式
def convert_pcbdata_2_result():
    component_result, cycle_result, feederslot_result, placement_result = [], [], [], []

    assigned_part = [-1 for _ in range(max_head_index)]
    assigned_slot = [-1 for _ in range(max_head_index)]
    assigned_point = [-1 for _ in range(max_head_index)]
    for point_cnt in range(point_num + 1):

        cycle_start = 1 if point_cnt == point_num else pcb_data.loc[point_cnt, 'cs']
        if (cycle_start and point_cnt != 0) or not -1 in assigned_part:

            if len(component_result) != 0 and component_result[-1] == assigned_part:
                cycle_result[-1] += 1
            else:
                component_result.append(assigned_part)
                feederslot_result.append(assigned_slot)
                cycle_result.append(1)

            placement_result.append(assigned_point)
            assigned_part = [-1 for _ in range(max_head_index)]
            assigned_slot = [-1 for _ in range(max_head_index)]
            assigned_point = [-1 for _ in range(max_head_index)]
            if point_cnt == point_num:
                break

        slot = pcb_data.loc[point_cnt, 'fdr'].split(' ')[0]
        slot, part = int(slot[1:]), pcb_data.loc[point_cnt, 'fdr'].split(' ', 1)[1]
        head = pcb_data.loc[point_cnt, 'hd'] - 1


        component_index = component_data[component_data['part'] == part].index.tolist()[0]

        assigned_part[head] = component_index
        assigned_slot[head] = slot
        assigned_point[head] = point_cnt

    return component_result, cycle_result, feederslot_result, placement_result

# 绘制各周期从供料器周期拾取的元件位置
def pickup_cycle_schematic(feederslot_result, cycle_result):
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # data
    bar_width = .7
    # for cycle in range(len(feederslot_result)):
    feederpart = np.zeros((int)(max_slot_index / 2), dtype = np.int)
    for cycle in range(len(feederslot_result)):
        label_str = '周期' + str(cycle + 1)
        cur_feederpart = np.zeros((int)(max_slot_index / 2), dtype = np.int)
        for slot in feederslot_result[cycle]:
            if slot > 0:
                cur_feederpart[slot] += cycle_result[cycle]

        plt.bar(np.arange(max_slot_index/2), cur_feederpart, bar_width, \
                edgecolor = 'black', bottom = feederpart, label = label_str)

        for slot in feederslot_result[cycle]:
            if slot > 0:
                feederpart[slot] += cycle_result[cycle]

    plt.legend()
    plt.show()

# 绘制指定周期的拾贴路径图
def placement_route_schematic(component_result, cycle_result, feederslot_result, placement_result, cycle = 0):
    pos_x, pos_y = [], []
    for i in range(len(pcb_data)):
        pos_x.append(pcb_data.loc[i]['x'])
        pos_y.append(pcb_data.loc[i]['y'])
        # plt.text(pcb_data.loc[i]['x'], pcb_data.loc[i]['y'] + 0.1, '%d' % i, ha='center', va = 'bottom', size = 8)

    mount_pos = []
    for head in range(max_head_index):
        index = placement_result[cycle][head]
        plt.text(pcb_data.loc[index]['x'], pcb_data.loc[index]['y'] + 0.1, 'HD%d' % head, ha = 'center', va = 'bottom', size = 10)
        mount_pos.append([pcb_data.loc[index]['x'] - head * slot_interval * interval_ratio, pcb_data.loc[index]['y']])
        plt.plot(mount_pos[-1][0], mount_pos[-1][1], marker = '^', color = 'red', markerfacecolor = 'white')

    mount_pos = np.sort(mount_pos, axis = 1)
    for i in range(len(mount_pos) - 1):
        plt.plot([mount_pos[i][0], mount_pos[i + 1][0]], [mount_pos[i][1], mount_pos[i + 1][1]], color = 'blue', linewidth = 1)

    plt.scatter(pos_x, pos_y, s = 8)
    # TODO: 绘制供料器位置布局
    plt.show()

def component_assign_evaluate(component_result, cycle_result, feederslot_result) -> float:
    nozzle_change_counter = 0
    for head in range(max_head_index):
        nozzle = ''
        for cycle in range(len(component_result)):
            component_index = component_result[cycle][head]
            if component_index == -1:
                continue

            if cycle != 0 and nozzle != component_data.loc[component_index, 'nz1']:
                nozzle_change_counter += 1
            nozzle = component_data.loc[component_index, 'nz1']

    simupick_counter = [0 for _ in range(max_head_index)]
    for cycle in range(len(component_result)):
        head_group = [component_result[cycle][head] for head in range(max_head_index)]

        while not -1 in head_group:
            for feeder_group in feederslot_result:
                common_part = find_commonpart(head_group, feeder_group)
                common_length = 0
                for head_index in range(max_head_index):
                    if common_part[head_index] == -1:
                        continue

                    head_group[head_index] = -1
                    common_length += 1

                simupick_counter[common_length - 1] += cycle_result[cycle]

    # TODO: 打印同时拾取数
    for head in range(max_head_index):
        simupick_counter[head] *= (head + 1)
    return len(component_result) + factor_nozzle_change * nozzle_change_counter - factor_simultaneous_pick * sum(simupick_counter)

# TODO: 贴装时间预估函数
def placement_time_estimate(component_result, cycle_result, feederslot_result, placement_result) -> float:
    t_pick, t_place = .4, .4    # 贴装/拾取用时
    t_nozzle_change = 3.3       # 装卸忌嘴用时
    velocity = 0.3              # 移动速度
    total_distance = .0         # 总移动距离
    total_operation_time = .0   # 操作用时
    cur_pos, next_pos = [0, 0], [0, 0]       # 贴装头当前位置

    for cycle_set in range(len(component_result)):
        floor_cycle, ceil_cycle = sum(cycle_result[:cycle_set]), sum(cycle_result[:(cycle_set + 1)])
        for cycle in range(floor_cycle, ceil_cycle):
            pick_slot, mount_pos = [], []
            for head in range(max_head_index):
                if feederslot_result[cycle_set][head] != -1:
                    pick_slot.append(feederslot_result[cycle_set][head] - interval_ratio * head)

            pick_slot = list(set(pick_slot))
            sorted(pick_slot)

            # TODO: 更换吸嘴

            # 拾取路径
            for slot in pick_slot:
                if slot < max_slot_index // 2:
                    next_pos = [slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1]]
                else:
                    # TODO: 后槽位移动路径
                    pass
                total_operation_time += t_pick
                total_distance += max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
                cur_pos = next_pos

            # TODO: 固定相机检测

            # 贴装路径
            for head in range(max_head_index):
                index = placement_result[cycle][head]
                if index == -1:
                    continue
                mount_pos.append([pcb_data.loc[index]['x'] - head * slot_interval * interval_ratio, pcb_data.loc[index]['y']])

            mount_pos = np.sort(mount_pos, axis = 1)
            for pos in mount_pos:
                total_operation_time += t_place
                total_distance += max(abs(cur_pos[0] - pos[0]), abs(cur_pos[1] - pos[1]))
                cur_pos = pos


    print('预估贴装用时:  {} s'.format(total_distance / velocity + total_operation_time))
    return total_distance / velocity + total_operation_time
