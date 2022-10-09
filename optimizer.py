import argparse
import traceback

from dataloader import *
from optimizer_celldivision import *
from optimizer_feederpriority import *
from optimizer_hybridgenetic import *
from optimizer_aggregation import *

from random_generator import *

# TODO: 贴装路径规划完善 + 随机数据生成 + 参考吸嘴模式如何同供料器既定安装位置联系起来
def optimizer(pcb_data, component_data, feeder_data=None, method='', hinter=True, figure=False, save=False, output=False, save_path=''):

    if method == 'cell_division':  # 基于元胞分裂的遗传算法
        component_result, cycle_result, feeder_slot_result = optimizer_celldivision(pcb_data, component_data, hinter)
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result, feeder_slot_result)
    elif method == 'feeder_priority':  # 基于基座扫描的供料器优先算法
        # 第1步：分配供料器位置
        nozzle_pattern = feeder_allocate(component_data, pcb_data, feeder_data, False)
        # 第2步：扫描供料器基座，确定元件拾取的先后顺序
        component_result, cycle_result, feeder_slot_result = feeder_base_scan(component_data, pcb_data, feeder_data, nozzle_pattern)

        # 第3步：贴装路径规划
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result, feeder_slot_result)
        # placement_result, head_sequence = beam_search_for_route_generation(component_data, pcb_data, component_result,
        #                                                                    cycle_result, feeder_slot_result)

    elif method == 'route_schedule':  # 路径规划测试
        component_result, cycle_result, feeder_slot_result, _, _ = convert_pcbdata_to_result(
            pcb_data, component_data)

        # placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
        #                                                                     cycle_result, feeder_slot_result)
        placement_result, head_sequence = beam_search_for_route_generation(component_data, pcb_data, component_result,
                                                                           cycle_result, feeder_slot_result)

    elif method == 'hybrid_genetic':  # 基于拾取组的混合遗传算法
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_hybrid_genetic(
            pcb_data, component_data, hinter=hinter)

    elif method == 'aggregation':  # 基于batch-level的整数规划 + 启发式算法
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_aggregation(
            component_data, pcb_data)
    else:
        raise 'method is not existed'

    if figure:
        # 绘制各周期从供料器拾取的贴装点示意图
        # pickup_cycle_schematic(feeder_slot_result, cycle_result)

        # 绘制贴装路径图
        for cycle in range(len(placement_result)):
            placement_route_schematic(pcb_data, component_result, cycle_result, feeder_slot_result, placement_result,
                                      head_sequence, cycle)

    if save:
        save_placement_route_figure(save_path, pcb_data, component_result, cycle_result, feeder_slot_result,
                                    placement_result, head_sequence)

    # 估算贴装用时
    if hinter:
        placement_time_estimate(component_data, pcb_data, component_result, cycle_result, feeder_slot_result,
                                placement_result, head_sequence)

    if output:
        output_optimize_result(params.filename, component_data, pcb_data, feeder_data, component_result, cycle_result,
                               feeder_slot_result, placement_result, head_sequence)

    return component_result, cycle_result, feeder_slot_result, placement_result, head_sequence


if __name__ == '__main__':
    # warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser(description='smt optimizer implementation')
    # parser.add_argument('--filename', default='PCB.txt', type=str, help='load pcb data')
    parser.add_argument('--filename', default='testlib/IPC9850-P400-C1-N1.txt', type=str, help='load pcb data')
    parser.add_argument('--mode', default=1, type=int, help='mode: 0 -directly load pcb data without optimization '
                                                            'for data analysis, 1 -optimize pcb data')
    parser.add_argument('--load_feeder', default=0, type=int,
                        help='load assigned feeder data: 0 - not load feeder data, 1 - load feeder data completely, '
                             '2- load feeder data partially')
    parser.add_argument('--optimize_method', default='feeder_priority', type=str, help='optimizer algorithm')
    parser.add_argument('--figure', default=0, type=int, help='plot mount process figure or not')
    parser.add_argument('--save', default=0, type=int, help='save the optimized result and figure')
    parser.add_argument('--output', default=0, type=int, help='output optimized result file')
    parser.add_argument('--auto_register', default=1, type=int, help='register the component according the pcb data')

    params = parser.parse_args()

    # 显示所有行和列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = [], [], [], [], []
    if params.mode == 0:
        # Load模式
        pcb_data, component_data, _ = load_data(params.filename, load_feeder_data=False,
                                                component_register=params.auto_register)  # 加载PCB数据
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = convert_pcbdata_to_result(
            pcb_data, component_data)

        if params.figure:
            # 绘制各周期从供料器拾取的贴装点示意图
            pickup_cycle_schematic(feeder_slot_result, cycle_result)

            # 绘制贴装路径图
            for cycle in range(len(placement_result)):
                placement_route_schematic(pcb_data, component_result, cycle_result, feeder_slot_result,
                                          placement_result, head_sequence, cycle)

        if params.save:
            save_placement_route_figure(params.filename, pcb_data, component_result, cycle_result, feeder_slot_result,
                                        placement_result, head_sequence)

        # 估算贴装用时
        placement_time_estimate(component_data, pcb_data, component_result, cycle_result, feeder_slot_result,
                                placement_result, head_sequence)

    elif params.mode == 1:
        # Debug模式
        pcb_data, component_data, feeder_data = load_data(params.filename, load_feeder_data=params.load_feeder,
                                                          component_register=params.auto_register)  # 加载PCB数据

        optimizer(pcb_data, component_data, feeder_data, params.optimize_method, hinter=True, figure=params.figure,
                  save=params.save, output=params.output, save_path=params.filename)

    elif params.mode == 2:
        # Test模式(根据data / testlib文件夹下的数据，测试比较不同算法性能)
        # optimize_method = ['standard', 'cell_division', 'feeder_priority', 'aggregation', 'hybrid_genetic']
        optimize_method = ['standard', 'feeder_priority']
        optimize_result = pd.DataFrame(columns=optimize_method)
        optimize_running_time = pd.DataFrame(columns=optimize_method)
        optimize_result.index.name, optimize_running_time.index.name = 'file', 'file'

        start_time = time.time()
        for file_index, file in enumerate(os.listdir('data/testlib')):
            print('--- (' + str(file_index + 1) + ') file ：  ' + file + ' --- ')
            pcb_data, component_data, feeder_data = load_data('testlib/' + file, load_feeder_data=params.load_feeder,
                                                              component_register=params.auto_register)   # 加载PCB数据
            optimize_result.loc[file] = [0 for _ in range(len(optimize_method))]
            for method in optimize_method:
                prev_time = time.time()
                try:
                    if method == 'standard':
                        # 转化为标准数据
                        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = \
                            convert_pcbdata_to_result(pcb_data, component_data)
                    else:
                        # 调用具体算法时，不显示、不绘图、不保存
                        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer(
                            pcb_data, component_data, feeder_data, method=method, hinter=False, figure=False, save=False)
                except:
                    traceback.print_exc()
                    warning_info = 'file: ' + file + ', method: ' + method + ': an unexpected error occurs'
                    warnings.warn(warning_info, SyntaxWarning)
                    continue

                placement_time = placement_time_estimate(component_data, pcb_data, component_result, cycle_result,
                                                         feeder_slot_result, placement_result, head_sequence,
                                                         hinter=False)
                optimize_result.loc[file, method] = placement_time if placement_time > 1e-10 else float('inf')
                optimize_running_time.loc[file, method] = time.time() - prev_time
                print('file: ' + file + ', method: ' + method + ', placement time: ' + str(placement_time) + 's')
            print('')

        print(optimize_result)
        print('result/opt_result_' + time.strftime('%Y%m%d%H%M', time.localtime()) + '.xlsx')

        writer = pd.ExcelWriter(r'result/opt_result_' + time.strftime('%Y%m%d%H%M', time.localtime()) + '.xlsx')

        optimize_result.to_excel(writer, sheet_name='result', float_format='%.3f', na_rep='')
        optimize_running_time.to_excel(writer, sheet_name='running_time', float_format='%.3f', na_rep='')

        writer.save()

        print("optimization process running time :  {} s".format(time.time() - start_time))


