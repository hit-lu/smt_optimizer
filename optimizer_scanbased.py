import itertools
from optimizer_common import *


@timer_wrapper
def optimizer_scanbased(component_data, pcb_data, hinter):

    population_size = 200  # 种群规模
    crossover_rate, mutation_rate = .4, .02
    n_generation = 5

    component_points = [0] * len(component_data)
    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        part_index = component_data[component_data['part'] == part].index.tolist()[0]

        component_points[part_index] += 1
        nozzle_type = component_data.loc[part_index]['nz']
        if nozzle_type not in nozzle_limit.keys() or nozzle_limit[nozzle_type] <= 0:
            info = 'there is no available nozzle [' + nozzle_type + '] for the assembly process'
            raise ValueError(info)

    # randomly generate permutations
    generation_ = np.array([i for i in range(max_slot_index // 2)])      # 仅考虑前基座
    pop_individual, pop_val = [], []
    for _ in range(population_size):
        np.random.shuffle(generation_)
        pop_individual.append(generation_.tolist())

        _, cycle_result, feeder_slot_result = convert_individual_2_result(component_points, pop_individual[-1])

        pop_val.append(feeder_arrange_evaluate(feeder_slot_result, cycle_result))

    with tqdm(total=n_generation) as pbar:
        pbar.set_description('hybrid genetic process')
        for _ in range(n_generation):
            # 交叉
            for pop in range(population_size):
                if pop % 2 == 0 and np.random.random() < crossover_rate:
                    index1, index2 = roulette_wheel_selection(pop_val), -1
                    while True:
                        index2 = roulette_wheel_selection(pop_val)
                        if index1 != index2:
                            break

                    # 两点交叉算子
                    offspring1, offspring2 = cycle_crossover(pop_individual[index1], pop_individual[index2])

                    _, cycle_result, feeder_slot_result = convert_individual_2_result(component_points, offspring1)
                    pop_val.append(feeder_arrange_evaluate(feeder_slot_result, cycle_result))
                    pop_individual.append(offspring1)

                    _, cycle_result, feeder_slot_result = convert_individual_2_result(component_points, offspring2)
                    pop_val.append(feeder_arrange_evaluate(feeder_slot_result, cycle_result))
                    pop_individual.append(offspring2)

                    sigma_scaling(pop_val, 1)

                # 变异
                if np.random.random() < mutation_rate:
                    index_ = roulette_wheel_selection(pop_val)
                    offspring = swap_mutation(pop_individual[index_])
                    _, cycle_result, feeder_slot_result = convert_individual_2_result(component_points, offspring)

                    pop_val.append(feeder_arrange_evaluate(feeder_slot_result, cycle_result))
                    pop_individual.append(offspring)

                    sigma_scaling(pop_val, 1)

            new_population, new_popval = [], []
            for index in get_top_k_value(pop_val, population_size):
                new_population.append(pop_individual[index])
                new_popval.append(pop_val[index])

            pop_individual, pop_val = new_population, new_popval

    # select the best individual
    pop = np.argmin(pop_val)
    component_result, cycle_result, feeder_slot_result = convert_individual_2_result(component_points, pop_individual[pop])

    placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                        cycle_result, feeder_slot_result)

    return component_result, cycle_result, feeder_slot_result, placement_result, head_sequence


def convert_individual_2_result(component_points, pop):
    component_result, cycle_result, feeder_slot_result = [], [], []

    feeder_part = [-1] * (max_slot_index // 2)  # 已安装在供料器基座上的元件（0: 未分配）
    feeder_base_points = [0] * (max_slot_index // 2)  # 供料器基座结余贴装点数量

    # 将基因信息转换为供料器基座安装结果
    for idx, gene in enumerate(pop):
        if idx >= len(component_points):
            break
        feeder_part[gene], feeder_base_points[gene] = idx, component_points[idx]

    # TODO: 暂时未考虑可用吸嘴数的限制
    # for _ in range(math.ceil(sum(component_points) / max_head_index)):
    while True:
        # === 周期内循环 ===
        assigned_part = [-1 for _ in range(max_head_index)]  # 当前扫描到的头分配元件信息
        assigned_slot = [-1 for _ in range(max_head_index)]  # 当前扫描到的供料器分配信息

        prev_scan_slot = len(feeder_part) // 2  # 前一轮扫描的位置
        while True:
            best_scan_part, best_scan_slot = [-1 for _ in range(max_head_index)], [-1 for _ in range(max_head_index)]
            best_slot_index = -1
            for slot in range(max_slot_index // 2 - (max_head_index - 1) * interval_ratio):
                scan_part, scan_slot = assigned_part.copy(), assigned_slot.copy()

                for head in range(max_head_index):
                    part = feeder_part[slot + head * interval_ratio]

                    # 贴装头和拾取槽位满足对应关系
                    if scan_part[head] == -1 and part != -1 and feeder_base_points[slot + head * interval_ratio] > 0:
                        scan_part[head], scan_slot[head] = part, slot + head * interval_ratio + 1

                if scan_part.count(-1) < best_scan_part.count(-1) or (scan_part.count(-1) == best_scan_part.count(-1)
                                                                      and abs(slot - prev_scan_slot) <
                                                                      abs(best_slot_index - prev_scan_slot)):
                    best_slot_index = slot
                    best_scan_part, best_scan_slot = scan_part.copy(), scan_slot.copy()

            assigned_points = 0
            for idx, slot in enumerate(best_scan_slot):
                if slot != -1 and assigned_slot[idx] == -1:
                    feeder_base_points[slot - 1] -= 1
                    assigned_points += 1

            assigned_part, assigned_slot = best_scan_part.copy(), best_scan_slot.copy()
            prev_scan_slot = best_slot_index

            if assigned_part.count(-1) == 0 or assigned_points == 0:
                break

        if len(cycle_result) == 0 or component_result[-1] != assigned_part:
            cycle_result.append(1)
            component_result.append(assigned_part)
            feeder_slot_result.append(assigned_slot)
        else:
            cycle_result[-1] += 1

        if sum(feeder_base_points) == 0:
            break

    return component_result, cycle_result, feeder_slot_result


def feeder_arrange_evaluate(feeder_slot_result, cycle_result):
    assert len(feeder_slot_result) == len(cycle_result)
    arrange_val = 0
    for cycle, feeder_slot in enumerate(feeder_slot_result):
        pick_slot = set()
        for head, slot in enumerate(feeder_slot):
            pick_slot.add(slot - head * interval_ratio)

        arrange_val += len(pick_slot) * t_pick * cycle_result[cycle]
        pick_slot = list(pick_slot)
        pick_slot.sort()
        arrange_val += axis_moving_time(pick_slot[0] - pick_slot[-1]) * cycle_result[cycle]

    return arrange_val

