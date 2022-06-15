import copy
import time
import math
import matplotlib.pyplot as plt

from functools import wraps

# 机器参数
max_slot_index = 120
max_head_index = 6
interval_ratio = 2
slot_interval = 15
head_interval = slot_interval * interval_ratio
head_nozzle = ['' for _ in range(max_head_index)]    # 头上已经分配吸嘴


# 位置信息
slotf1_pos, slotr1_pos = [-31.267, 44.], [807., 810.545]     # F1(前基座最左侧)、R1(后基座最右侧)位置
stopper_pos = [620.000, 200]                        # 止档块位置

# 权重
factor_nozzle_change = 0.5
factor_simultaneous_pick = 1. / max_head_index


def find_commonpart(head_group, feeder_group):
    feeder_group_len = len(feeder_group)

    max_length, max_common_part = -1, []
    for offset in range(-max_head_index + 1, feeder_group_len - 1):
        # offset: head_group相对于feeder_group的偏移量
        length, common_part = 0, []
        for hd_index in range(max_head_index):
            fd_index = hd_index + offset
            if fd_index < 0 or fd_index >= feeder_group_len:
                common_part.append(-1)
                continue

            if head_group[hd_index] == feeder_group[fd_index] and head_group[hd_index] != -1:
                length += 1
                common_part.append(head_group[hd_index])
            else:
                common_part.append(-1)
        if length > max_length:
            max_length = length
            max_common_part = common_part

    return max_common_part


def timer_warper(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)

        print("function {} running time :  {} s".format(func.__name__, time.time() - start_time))
        return result
    return measure_time


def axis_moving_time(distance, axis = 0):
    distance = abs(distance) * 1e-3
    if axis == 0:
        v_max, Tamax = 1.6, 0.079
    else:
        v_max, Tamax = 1.5, 0.079

    a_max = v_max / Tamax
    Lamax = a_max * Tamax * Tamax

    return math.sqrt(distance / a_max) if distance < Lamax else (distance - Lamax) / v_max
