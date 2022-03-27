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

# TODO: 供料器基座位置布局（目前采用已布局结果，需要研究不同供料器位置布局对结果的影响）

# 绘制各周期从供料器拾取的贴装点示意图
# pickup_cycle_schematic(feederslot_result, cycle_result)

# 绘制贴装路径图
# placement_route_schematic(component_result, cycle_result, feederslot_result, placement_result, 3)

# 估算贴装用时
# placement_time_estimate(component_result, cycle_result, feederslot_result, placement_result)




