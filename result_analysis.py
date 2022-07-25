import os

import matplotlib.pyplot as plt
from dataloader import *
from optimizer_common import *


# 将步骤列表中已有的数据转换为可计算格式
def convert_pcbdata_to_result(pcb_data, component_data):
    component_result, cycle_result, feeder_slot_result = [], [], []
    placement_result, head_sequence_result = [], []

    assigned_part = [-1 for _ in range(max_head_index)]
    assigned_slot = [-1 for _ in range(max_head_index)]
    assigned_point = [-1 for _ in range(max_head_index)]
    assigned_sequence = []

    point_num = len(pcb_data)           # total mount points num
    for point_cnt in range(point_num + 1):

        cycle_start = 1 if point_cnt == point_num else pcb_data.loc[point_cnt, 'cs']
        if (cycle_start and point_cnt != 0) or not -1 in assigned_part:

            if len(component_result) != 0 and component_result[-1] == assigned_part:
                cycle_result[-1] += 1
            else:
                component_result.append(assigned_part)
                feeder_slot_result.append(assigned_slot)
                cycle_result.append(1)

            placement_result.append(assigned_point)
            head_sequence_result.append(assigned_sequence)

            assigned_part = [-1 for _ in range(max_head_index)]
            assigned_slot = [-1 for _ in range(max_head_index)]
            assigned_point = [-1 for _ in range(max_head_index)]
            assigned_sequence = []

            if point_cnt == point_num:
                break

        slot = pcb_data.loc[point_cnt, 'fdr'].split(' ')[0]
        if slot == 'A':
            slot, part = 0, pcb_data.loc[point_cnt, 'part']
        else:

            slot, part = int(slot[1:]), pcb_data.loc[point_cnt, 'fdr'].split(' ', 1)[1]
        head = pcb_data.loc[point_cnt, 'hd'] - 1

        part_index = component_data[component_data['part'] == part].index.tolist()[0]

        assigned_part[head] = part_index
        assigned_slot[head] = slot
        assigned_point[head] = point_cnt
        assigned_sequence.append(head)

    return component_result, cycle_result, feeder_slot_result, placement_result, head_sequence_result


# 绘制各周期从供料器周期拾取的元件位置
def pickup_cycle_schematic(feeder_slot_result, cycle_result):
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # data
    bar_width = .7
    feeder_part = np.zeros((int)(max_slot_index / 2), dtype = np.int)
    for cycle in range(len(feeder_slot_result)):
        label_str = '周期' + str(cycle + 1)
        cur_feeder_part = np.zeros((int)(max_slot_index / 2), dtype = np.int)
        for slot in feeder_slot_result[cycle]:
            if slot > 0:
                cur_feeder_part[slot] += cycle_result[cycle]

        plt.bar(np.arange(max_slot_index / 2), cur_feeder_part, bar_width, edgecolor='black', bottom=feeder_part,
                label=label_str)

        for slot in feeder_slot_result[cycle]:
            if slot > 0:
                feeder_part[slot] += cycle_result[cycle]

    plt.legend()
    plt.show()


def placement_route_schematic(pcb_data, component_result, cycle_result, feeder_slot_result, placement_result,
                              head_sequence, cycle=-1):

    plt.figure('cycle {}'.format(cycle + 1))
    pos_x, pos_y = [], []
    for i in range(len(pcb_data)):
        pos_x.append(pcb_data.loc[i]['x'] + stopper_pos[0])
        pos_y.append(pcb_data.loc[i]['y'] + stopper_pos[1])
        # plt.text(pcb_data.loc[i]['x'], pcb_data.loc[i]['y'] + 0.1, '%d' % i, ha='center', va = 'bottom', size = 8)

    mount_pos = []
    for head in head_sequence[cycle]:
        index = placement_result[cycle][head]
        plt.text(pos_x[index], pos_y[index] + 0.1, 'HD%d' % (head + 1), ha = 'center', va = 'bottom', size = 10)
        plt.plot([pos_x[index], pos_x[index] - head * head_interval], [pos_y[index], pos_y[index]], linestyle='-.',
                 color='black', linewidth=1)
        mount_pos.append([pos_x[index] - head * head_interval, pos_y[index]])
        plt.plot(mount_pos[-1][0], mount_pos[-1][1], marker = '^', color = 'red', markerfacecolor = 'white')

        # plt.text(pos_x[index], pos_y[index], '%d' % (index + 1), size = 8)

    # 绘制贴装路径
    for i in range(len(mount_pos) - 1):
        plt.plot([mount_pos[i][0], mount_pos[i + 1][0]], [mount_pos[i][1], mount_pos[i + 1][1]], color = 'blue', linewidth = 1)

    draw_x, draw_y = [], []
    for c in range(cycle, len(placement_result)):
        for h in range(max_head_index):
            i = placement_result[c][h]
            if i == -1:
                continue
            draw_x.append(pcb_data.loc[i]['x'] + stopper_pos[0])
            draw_y.append(pcb_data.loc[i]['y'] + stopper_pos[1])

            # plt.text(draw_x[-1], draw_y[-1] - 5, '%d' % i, ha='center', va='bottom', size=10)

    plt.scatter(draw_x, draw_y, s = 8)

    # 绘制供料器位置布局
    for slot in range(max_slot_index // 2):
        plt.scatter(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1], marker = 'x', s = 12, color = 'green')
        plt.text(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1] - 50, slot + 1, ha = 'center', va = 'bottom', size = 8)

    feeder_part, feeder_counter = {}, {}
    placement_cycle = 0
    for cycle_, components in enumerate(component_result):
        for head, component in enumerate(components):
            if component == -1:
                continue
            placement = placement_result[placement_cycle][head]
            slot = feeder_slot_result[cycle_][head]
            feeder_part[slot] = pcb_data.loc[placement]['part']
            if slot not in feeder_counter.keys():
                feeder_counter[slot] = 0

            feeder_counter[slot] += cycle_result[cycle_]
        placement_cycle += cycle_result[cycle_]

    for slot, part in feeder_part.items():
        plt.text(slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1] + 15, part + ': ' + str(feeder_counter[slot]), ha = 'center', size = 7, rotation = 90)

    plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
                    [slotf1_pos[1] + 10, slotf1_pos[1] + 10], color = 'black')
    plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
                    [slotf1_pos[1] - 40, slotf1_pos[1] - 40], color = 'black')

    for counter in range(max_slot_index // 2 + 1):
        pos = slotf1_pos[0] + (counter - 0.5) * slot_interval
        plt.plot([pos, pos], [slotf1_pos[1] + 10, slotf1_pos[1] - 40], color='black', linewidth = 1)

    # 绘制拾取路径
    pick_slot = []
    cycle_group = 0
    while sum(cycle_result[0: cycle_group]) < cycle:
        cycle_group += 1
    for head, slot in enumerate(feeder_slot_result[cycle_group]):
        if slot == -1:
            continue
        pick_slot.append(slot - head * interval_ratio)
    pick_slot = list(set(pick_slot))
    sorted(pick_slot)

    plt.plot([mount_pos[0][0], slotf1_pos[0] + slot_interval * (pick_slot[0] - 1)], [mount_pos[0][1], slotf1_pos[1]], color = 'blue', linewidth = 1)
    plt.plot([mount_pos[-1][0], slotf1_pos[0] + slot_interval * (pick_slot[-1] - 1)], [mount_pos[-1][1], slotf1_pos[1]], color = 'blue', linewidth = 1)
    plt.plot([slotf1_pos[0] + slot_interval * (pick_slot[0] - 1), slotf1_pos[0] + slot_interval * (pick_slot[-1] - 1)],
                [slotf1_pos[1], slotf1_pos[1]], color = 'blue', linewidth = 1)

    plt.show()


def save_placement_route_figure(file_name, pcb_data, component_result, cycle_result, feederslot_result, placement_result, head_sequence):
    path = 'result/' + file_name[:file_name.find('.')]
    if not os.path.exists(path):
        os.mkdir(path)

    pos_x, pos_y = [], []
    for i in range(len(pcb_data)):
        pos_x.append(pcb_data.loc[i]['x'] + stopper_pos[0])
        pos_y.append(pcb_data.loc[i]['y'] + stopper_pos[1])
        # plt.text(pcb_data.loc[i]['x'], pcb_data.loc[i]['y'] + 0.1, '%d' % i, ha='center', va = 'bottom', size = 8)

    for cycle in range(len(placement_result)):
        plt.figure(cycle)

        mount_pos = []
        for head in head_sequence[cycle]:
            index = placement_result[cycle][head]
            plt.text(pos_x[index], pos_y[index] + 0.1, 'HD%d' % (head + 1), ha='center', va='bottom', size=10)
            plt.plot([pos_x[index], pos_x[index] - head * head_interval], [pos_y[index], pos_y[index]], linestyle='-.',
                     color='black', linewidth=1)
            mount_pos.append([pos_x[index] - head * head_interval, pos_y[index]])
            plt.plot(mount_pos[-1][0], mount_pos[-1][1], marker='^', color='red', markerfacecolor='white')

        # 绘制贴装路径
        for i in range(len(mount_pos) - 1):
            plt.plot([mount_pos[i][0], mount_pos[i + 1][0]], [mount_pos[i][1], mount_pos[i + 1][1]], color='blue',
                     linewidth=1)

        draw_x, draw_y = [], []
        for c in range(cycle, len(placement_result)):
            for h in range(max_head_index):
                i = placement_result[c][h]
                if i == -1:
                    continue
                draw_x.append(pcb_data.loc[i]['x'] + stopper_pos[0])
                draw_y.append(pcb_data.loc[i]['y'] + stopper_pos[1])

                # plt.text(draw_x[-1], draw_y[-1] - 5, '%d' % i, ha='center', va='bottom', size=10)

        plt.scatter(draw_x, draw_y, s=8)

        # plt.scatter(pos_x, pos_y, s=8)
        # # 绘制供料器位置布局
        # for slot in range(max_slot_index // 2):
        #     plt.scatter(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1], marker='x', s=12, color='green')
        #     plt.text(slotf1_pos[0] + slot_interval * slot, slotf1_pos[1] - 50, slot + 1, ha='center', va='bottom', size=8)
        #
        # feeder_part, feeder_counter = {}, {}
        # placement_cycle = 0
        # for cycle_, components in enumerate(component_result):
        #     for head, component in enumerate(components):
        #         if component == -1:
        #             continue
        #         placement = placement_result[placement_cycle][head]
        #         slot = feederslot_result[cycle_][head]
        #         feeder_part[slot] = pcb_data.loc[placement]['part']
        #         if slot not in feeder_counter.keys():
        #             feeder_counter[slot] = 0
        #
        #         feeder_counter[slot] += cycle_result[cycle_]
        #     placement_cycle += cycle_result[cycle_]
        #
        # for slot, part in feeder_part.items():
        #     plt.text(slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1] + 15,
        #              part + ': ' + str(feeder_counter[slot]), ha='center', size=7, rotation=90)
        #
        # plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
        #          [slotf1_pos[1] + 10, slotf1_pos[1] + 10], color='black')
        # plt.plot([slotf1_pos[0] - slot_interval / 2, slotf1_pos[0] + slot_interval * (max_slot_index // 2 - 1 + 0.5)],
        #          [slotf1_pos[1] - 40, slotf1_pos[1] - 40], color='black')
        #
        # for counter in range(max_slot_index // 2 + 1):
        #     pos = slotf1_pos[0] + (counter - 0.5) * slot_interval
        #     plt.plot([pos, pos], [slotf1_pos[1] + 10, slotf1_pos[1] - 40], color='black', linewidth=1)

        # # 绘制拾取路径
        # pick_slot = []
        # cycle_group = 0
        # while sum(cycle_result[0: cycle_group]) < cycle:
        #     cycle_group += 1
        # for head, slot in enumerate(feederslot_result[cycle_group]):
        #     if slot == -1:
        #         continue
        #     pick_slot.append(slot - head * interval_ratio)
        # pick_slot = list(set(pick_slot))
        # sorted(pick_slot)
        #
        # plt.plot([mount_pos[0][0], slotf1_pos[0] + slot_interval * (pick_slot[0] - 1)], [mount_pos[0][1], slotf1_pos[1]],
        #          color='blue', linewidth=1)
        # plt.plot([mount_pos[-1][0], slotf1_pos[0] + slot_interval * (pick_slot[-1] - 1)], [mount_pos[-1][1], slotf1_pos[1]],
        #          color='blue', linewidth=1)
        # plt.plot([slotf1_pos[0] + slot_interval * (pick_slot[0] - 1), slotf1_pos[0] + slot_interval * (pick_slot[-1] - 1)],
        #          [slotf1_pos[1], slotf1_pos[1]], color='blue', linewidth=1)

        plt.savefig(path + '/cycle_{}'.format(cycle + 1))

        plt.close(cycle)


def component_assign_evaluate(component_data, component_result, cycle_result, feeder_slot_result) -> float:
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

    gang_pick_counter = [0 for _ in range(max_head_index)]

    for cycle, feeder_slot in enumerate(feeder_slot_result):
        pick_slot = {}
        for head, slot in enumerate(feeder_slot):
            if slot == -1:
                continue
            if slot - head * head_interval not in pick_slot:
                pick_slot[slot - head * head_interval] = 0
            pick_slot[slot - head * head_interval] += 1
        for v in pick_slot.values():
            gang_pick_counter[v - 1] += cycle_result[cycle]

    for head in range(max_head_index):
        gang_pick_counter[head] *= (head + 1)

    return sum(cycle_result) + factor_nozzle_change * nozzle_change_counter - factor_simultaneous_pick * sum(
        gang_pick_counter)


def placement_time_estimate(component_data, pcb_data, component_result, cycle_result, feeder_slot_result,
                            placement_result, head_sequence, hinter=True) -> float:

    t_pick, t_place = .078, .125                  # 贴装/拾取用时
    t_nozzle_put, t_nozzle_pick = 0.9, 0.75       # 装卸吸嘴用时
    t_fix_camera_check = 0.22                     # 固定相机检测时间

    head_rotary_velocity = 8e-5                           # 贴装头R轴旋转时间
    x_max_velocity, y_max_velocity = 1.6, 1.5
    x_max_acceleration, y_max_acceleration = x_max_velocity / 0.079, y_max_velocity / 0.079

    def axis_moving_time_func(distance, axis=0):
        distance = abs(distance) * 1e-3
        Lamax = x_max_velocity ** 2 / x_max_acceleration if axis == 0 else y_max_velocity ** 2 / y_max_acceleration
        if axis == 0:
            return math.sqrt(distance / x_max_acceleration) if distance < Lamax else (distance - Lamax) / x_max_velocity
        else:
            return math.sqrt(distance / x_max_acceleration) if distance < Lamax else (distance - Lamax) / x_max_velocity

    def head_rotary_time_func(angle):
        while -180 > angle > 180:
            if angle > 180:
                angle -= 360
            else:
                angle += 360
        return abs(angle) * head_rotary_velocity

    total_moving_time = .0                          # 总移动用时
    total_operation_time = .0                       # 操作用时
    total_nozzle_change_counter = 0                 # 总吸嘴更换次数
    total_pick_counter = 0                          # 总拾取次数
    total_mount_distance, total_distance = .0, .0   # 贴装距离（临时使用）、总移动距离

    cur_pos, next_pos = anc_marker_pos, [0, 0]       # 贴装头当前位置

    # 初始化首个周期的吸嘴装配信息
    nozzle_assigned = []
    for components in component_result:
        for idx in components:
            if idx == -1:
                nozzle_assigned.append(['Empty'])
            else:
                nozzle_assigned.append(component_data.loc[idx]['nz1'])

    for cycle_set, _ in enumerate(component_result):
        floor_cycle, ceil_cycle = sum(cycle_result[:cycle_set]), sum(cycle_result[:(cycle_set + 1)])
        for cycle in range(floor_cycle, ceil_cycle):
            pick_slot, mount_pos, mount_angle = [], [], []
            nozzle_pick_counter, nozzle_put_counter = 0, 0  # 吸嘴更换次数统计（拾取/放置分别算一次）
            for head in range(max_head_index):
                if feeder_slot_result[cycle_set][head] != -1:
                    pick_slot.append(feeder_slot_result[cycle_set][head] - interval_ratio * head)
                if component_result[cycle_set][head] == -1:
                    continue
                nozzle = component_data.loc[component_result[cycle_set][head]]['nz1']
                if nozzle != nozzle_assigned[head]:
                    if nozzle_assigned[head] != 'Empty':
                        nozzle_put_counter += 1
                    nozzle_pick_counter += 1
                    nozzle_assigned[head] = nozzle

            # ANC处进行吸嘴更换
            if nozzle_pick_counter + nozzle_put_counter > 0:
                next_pos = anc_marker_pos
                total_moving_time += max(axis_moving_time_func(cur_pos[0] - next_pos[0], 0),
                                         axis_moving_time_func(cur_pos[1] - next_pos[1], 1))
                total_distance += max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
                cur_pos = next_pos

            pick_slot = list(set(pick_slot))
            sorted(pick_slot)

            # 拾取路径
            for slot in pick_slot:
                if slot < max_slot_index // 2:
                    next_pos = [slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1]]
                else:
                    next_pos = [slotr1_pos[0] - slot_interval * (max_slot_index - slot - 1), slotr1_pos[1]]
                total_operation_time += t_pick
                total_pick_counter += 1
                total_moving_time += max(axis_moving_time_func(cur_pos[0] - next_pos[0], 0),
                                         axis_moving_time_func(cur_pos[1] - next_pos[1], 1))
                total_distance += max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
                cur_pos = next_pos

            # 固定相机检测
            for head in range(max_head_index):
                if component_result[cycle_set][head] == -1:
                    continue
                camera = component_data.loc[component_result[cycle_set][head]]['camera']
                if camera == 'FIX_CAMERA':
                    next_pos = [fix_camera_pos[0] - head * head_interval, fix_camera_pos[1]]
                    total_moving_time += max(axis_moving_time_func(cur_pos[0] - next_pos[0], 0),
                                             axis_moving_time_func(cur_pos[1] - next_pos[1], 1))
                    total_distance += max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
                    total_operation_time += t_fix_camera_check
                    cur_pos = next_pos

            # 贴装路径
            for head in head_sequence[cycle]:
                index = placement_result[cycle][head]
                if index == -1:
                    continue
                mount_pos.append([pcb_data.loc[index]['x'] - head * head_interval + stopper_pos[0],
                                  pcb_data.loc[index]['y'] + stopper_pos[1]])
                mount_angle.append(pcb_data.loc[index]['r'])

            # 单独计算贴装路径
            for cntPoints in range(len(mount_pos) - 1):
                total_mount_distance += max(abs(mount_pos[cntPoints][0] - mount_pos[cntPoints + 1][0]),
                                            abs(mount_pos[cntPoints][1] - mount_pos[cntPoints + 1][1]))

            # 考虑R轴预旋转，补偿同轴角度转动带来的额外贴装用时
            total_operation_time += head_rotary_time_func(mount_angle[0])  # 补偿角度转动带来的额外贴装用时
            total_operation_time += t_nozzle_put * nozzle_put_counter + t_nozzle_pick * nozzle_pick_counter
            for pos in mount_pos:
                total_operation_time += t_place
                total_moving_time += max(axis_moving_time_func(cur_pos[0] - pos[0], 0),
                                         axis_moving_time_func(cur_pos[1] - pos[1], 1))
                total_distance += max(abs(cur_pos[0] - pos[0]), abs(cur_pos[1] - pos[1]))
                cur_pos = pos

            total_nozzle_change_counter += nozzle_put_counter + nozzle_pick_counter

    total_time = total_moving_time + total_operation_time
    minutes, seconds = int(total_time // 60), int(total_time) % 60
    millisecond = (total_time - minutes * 60 - seconds) * 60

    if hinter:
        print('Cycle counter: {}'.format(sum(cycle_result)))
        print('Nozzle change counter: {}'.format(total_nozzle_change_counter))
        print('Single and gang pick counter: {}'.format(total_pick_counter))

        print('Expected mounting tour length: {} mm'.format(total_mount_distance))
        print('Expected total tour length: {} mm'.format(total_distance))

        print('Expected total moving time: {} s'.format(total_moving_time))
        print('Expected total operation time: {} s'.format(total_operation_time))

        if minutes > 0:
            print('Mounting time estimation:  {:d} min {} s {:.4f}'.format(minutes, seconds, millisecond))
        else:
            print('Mounting time estimation:  {} s {:.4f}'.format(seconds, millisecond))
    return total_time
