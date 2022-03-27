from dataloader import *
from common_function import *

def placement_route_generation(component_result, cycle_result):
    placement_result = []

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
                    index = np.argmin(mountpoint_pos[component_index], axis=0)[0]
                    assigned_placement[head] = mountpoint_index[component_index][index]
                    way_point.append(mountpoint_pos[component_index][index])  # 记录路标点

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
                                delta_x = abs(mountpoint_pos[component_index][counter1][0] - way_point[counter2][
                                    0] - head_index * slot_interval)
                                delta_y = abs(mountpoint_pos[component_index][counter1][1] - way_point[counter2][1])
                                cheby_distance = max(delta_x, delta_y)
                                euler_distance = pow(delta_x, 2) + pow(delta_y, 2)
                                if cheby_distance < min_cheby_distance or \
                                        (
                                                abs(cheby_distance - min_cheby_distance) < 10 and euler_distance < min_euler_distance):
                                    min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                    head_index, point_index = next_head, counter1

                    component_index = component_result[cycle_set][head_index]
                    assigned_placement[head_index] = mountpoint_index[component_index][point_index]

                    way_point.append(mountpoint_pos[component_index][point_index])  # 记录路标点(相对于吸杆1的位置)
                    way_point[-1][0] -= head_index * slot_interval

                    mountpoint_index[component_index].pop(point_index)
                    mountpoint_pos[component_index].pop(point_index)

            placement_result.append(assigned_placement)

    return placement_result
