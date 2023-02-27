import random

from optimizer_common import *


def load_data(filename: str, load_cp_data=True, load_feeder_data=True, component_register=False):
    # 锁定随机数种子
    random.seed(0)

    # 读取PCB数据
    filename = 'data/' + filename
    pcb_data = pd.DataFrame(pd.read_csv(filepath_or_buffer=filename, sep='\t', header=None))
    if len(pcb_data.columns) <= 17:
        step_col = ["ref", "x", "y", "z", "r", "part", "desc", "fdr", "nz", "hd", "cs", "cy", "sk", "bl", "ar",
                    "pl", "lv"]
    elif len(pcb_data.columns) <= 18:
        step_col = ["ref", "x", "y", "z", "r", "part", "desc", "fdr", "nz", "hd", "cs", "cy", "sk", "bl", "ar", "fid",
                    "pl", "lv"]
    else:
        step_col = ["ref", "x", "y", "z", "r", "part", "desc", "fdr", "nz", "hd", "cs", "cy", "sk", "bl", "ar", "fid",
                    "", "pl", "lv"]

    pcb_data.columns = step_col
    pcb_data = pcb_data.dropna(axis=1)

    # 坐标系处理
    # pcb_data = pcb_data.sort_values(by = ['x', 'y'], ascending = True)
    # pcb_data["x"] = pcb_data["x"].apply(lambda x: -x)

    # 注册元件检查
    component_data = None
    if load_cp_data:
        part_feeder_assign = defaultdict(set)
        part_col = ["part", "desc", "fdr", "nz", 'camera', 'group', 'feeder-limit']
        try:
            component_data = pd.DataFrame(pd.read_csv(filepath_or_buffer='component.txt', sep='\t', header=None), columns=part_col)
        except:
            component_data = pd.DataFrame(columns=part_col)
        for _, data in pcb_data.iterrows():
            part, nozzle = data.part, data.nz.split(' ')[1]
            slot = data['fdr'].split(' ')[0]

            if part not in component_data['part'].values:
                if not component_register:
                    raise Exception("unregistered component:  " + component_data['part'].values)
                else:
                    component_data = pd.concat([component_data,
                                                pd.DataFrame([part, '', 'SM8', nozzle, '飞行相机1', 'CHIP-Rect', 0],
                                                             index=part_col).T], ignore_index=True)
                    # warning_info = 'register component ' + part + ' with default feeder type'
                    # warnings.warn(warning_info, UserWarning)
            part_index = component_data[component_data['part'] == part].index.tolist()[0]
            part_feeder_assign[part].add(slot)

            if nozzle != 'A' and component_data.loc[part_index]['nz'] != nozzle:
                warning_info = 'the nozzle type of component ' + part + ' is not consistent with the pcb data'
                warnings.warn(warning_info, UserWarning)

        for idx, data in component_data.iterrows():
            if data['fdr'][0:3] == 'SME':       # 电动供料器和气动供料器参数一致
                component_data.at[idx, 'fdr'] = data['fdr'][0:2] + data['fdr'][3:]

        for part, slots in part_feeder_assign.items():
            part_index = component_data[component_data['part'] == part].index.tolist()[0]
            component_data.at[part_index, 'feeder-limit'] = max(len(slots), component_data.at[part_index, 'feeder-limit'])

    # 读取供料器基座数据
    feeder_data = pd.DataFrame(columns=range(3))
    if load_feeder_data:
        for data in pcb_data.iterrows():
            fdr = data[1]['fdr']
            slot, part = fdr.split(' ')
            if slot[0] != 'F' and slot[0] != 'R':
                continue
            slot = int(slot[1:]) if slot[0] == 'F' else int(slot[1:]) + max_slot_index // 2
            feeder_data = pd.concat([feeder_data, pd.DataFrame([slot, part, 1]).T])

        feeder_data.columns = ['slot', 'part', 'arg']  # arg表示是否为预分配，不表示分配数目
        feeder_data.drop_duplicates(subset='slot', inplace=True, ignore_index=True)
        # 随机移除部分已安装的供料器
        if load_feeder_data == 2:
            drop_index = random.sample(list(range(len(feeder_data))), len(feeder_data) // 2)
            feeder_data.drop(index=drop_index, inplace=True)

        feeder_data.sort_values(by='slot', ascending=True, inplace=True, ignore_index=True)
    else:
        feeder_data.columns = ['slot', 'part', 'arg']  # 同上

    return pcb_data, component_data, feeder_data
