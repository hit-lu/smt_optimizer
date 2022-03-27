import copy
import time
import math

from functools import wraps

# 机器参数
max_slot_index = 120
max_head_index = 6
interval_ratio = 2
slot_interval = 15
head_interval = slot_interval * interval_ratio
head_nozzle = ['' for _ in range(max_head_index)]    # 头上已经分配吸嘴

# 位置信息
slotf1_pos, slotr1_pos = [-74., 151.], [807., 917.]     # F1(前基座最左侧)、R1(后基座最右侧)位置
stopper_pos = [640.325, 135.189]                        # 止档块位置

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

    # m = [[0 for i in range(max_head_index + 1)] for j in range(max_head_index + 1)]
    # max_length = 0  # 最长匹配的长度
    # p = 0  # 最长匹配对应在s1中的最后一位
    # for i in range(max_head_index):
    #     for j in range(max_head_index):
    #         if element1[i] == element2[j]:
    #             m[i + 1][j + 1] = m[i][j] + 1
    #             if m[i + 1][j + 1] > max_length:
    #                 max_length = m[i + 1][j + 1]
    #                 p = i + 1
    # # 返回最长子串、长度、起始匹配位置
    # return element1[p - max_length: p], max_length, start_index

def timer_warper(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)

        print("function {} running time :  {} s".format(func.__name__, time.time() - start_time))
        return result
    return measure_time
