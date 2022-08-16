from optimizer_common import *


def load_data(filename: str, load_cp_data=True, load_feeder_data=True, component_register=False):
    # 读取PCB数据
    filename = 'data/' + filename
    pcb_data = pd.DataFrame(pd.read_csv(filepath_or_buffer=filename, sep='\t', header=None))

    if len(pcb_data.columns) <= 18:
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
        warnings.simplefilter('always')
        part_col = ["part", "fdr", "nz1", "nz2", 'camera', 'feeder-limit']
        component_data = pd.DataFrame(pd.read_csv(filepath_or_buffer='component.txt', sep='\t', header=None))
        component_data.columns = part_col
        for i in range(len(pcb_data)):
            if not pcb_data.loc[i].part in component_data['part'].values:
                if not component_register:
                    raise Exception("unregistered component:  " + pcb_data.loc[i].part)
                else:
                    part, nozzle = pcb_data.loc[i].part, pcb_data.loc[i].nz.split(' ')[1]
                    new_component = pd.Series([part, 'SM8', nozzle, nozzle, 'FLY_CAMERA', 1], index=part_col)
                    component_data = component_data.append(new_component, ignore_index=True)
                    warning_info = 'register component ' + part + ' with default feeder type'
                    warnings.warn(warning_info, DeprecationWarning)

    # 读取供料器基座数据
    feeder_data = pd.DataFrame(columns=range(2))
    if load_feeder_data:
        for data in pcb_data.iterrows():
            fdr = data[1]['fdr']
            slot, part = fdr.split(' ')
            if slot[0] != 'F' and slot[0] != 'R':
                continue
            slot = int(slot[1:]) if slot[0] == 'F' else int(slot[1:]) + max_slot_index // 2
            feeder_data = pd.concat([feeder_data, pd.DataFrame([slot, part]).T])

    feeder_data.columns = ['slot', 'part']
    feeder_data.drop_duplicates(subset='slot', inplace=True)
    feeder_data.sort_values(by='slot', ascending=True, inplace=True, ignore_index=True)

    return pcb_data, component_data, feeder_data
