import pandas as pd

from optimizer_common import *
from result_analysis import *

import random
import numpy as np


def crossover(element1, element2):
    range_ = np.random.randint(0, len(element1), 2)      # 前闭后开
    range_ = sorted(range_)

    element1_cpy, element2_cpy = [-1 for _ in range(len(element1))], [-1 for _ in range(len(element2))]

    element1_cpy[range_[0]: range_[1] + 1] = copy.deepcopy(element2[range_[0]: range_[1] + 1])
    element2_cpy[range_[0]: range_[1] + 1] = copy.deepcopy(element1[range_[0]: range_[1] + 1])

    for index in range(len(element1)):
        if range_[0] <= index <= range_[1]:
            continue

        cur_ptr, cur_elem = 0, element1[index]
        while True:
            element1_cpy[index] = cur_elem
            if element1_cpy.count(cur_elem) == 1:
                break
            element1_cpy[index] = -1

            if cur_ptr == 0:
                cur_ptr, cur_elem = 1, element2[index]
            else:
                index_ = element1_cpy.index(cur_elem)
                cur_elem = element2[index_]

    for index in range(len(element2)):
        if range_[0] <= index <= range_[1]:
            continue

        cur_ptr, cur_elem = 0, element2[index]
        while True:
            element2_cpy[index] = cur_elem
            if element2_cpy.count(cur_elem) == 1:
                break
            element2_cpy[index] = -1

            if cur_ptr == 0:
                cur_ptr, cur_elem = 1, element1[index]
            else:
                index_ = element2_cpy.index(cur_elem)
                cur_elem = element1[index_]

    return element1_cpy, element2_cpy


def swap_mutation(element):
    range_ = np.random.randint(0, len(element), 2)
    element[range_[0]], element[range_[1]] = element[range_[1]], element[range_[0]]
    return element


def selection(pop_eval):
    # Roulette wheel
    cumsum_pop_eval = np.array(pop_eval)
    cumsum_pop_eval = np.divide(cumsum_pop_eval, np.sum(cumsum_pop_eval))
    cumsum_pop_eval = cumsum_pop_eval.cumsum()

    random_eval = np.random.random()
    index = 0
    while index < len(pop_eval):
        if random_eval > cumsum_pop_eval[index]:
            index += 1
        else:
            break
    return index


def get_top_k_value(pop_val, k: int):
    res = []
    pop_val_cpy = copy.deepcopy(pop_val)
    pop_val_cpy.sort(reverse = True)

    for i in range(min(len(pop_val_cpy), k)):
        for j in range(len(pop_val)):
            if abs(pop_val_cpy[i] - pop_val[j]) < 1e-9 and j not in res:
                res.append(j)
                break
    return res


def convert_cell_2_result(pcb_data, component_data, component_cell, population):
    assert component_cell['points'].sum() == len(pcb_data)
    head_assignment = [[] for _ in range(max_head_index)]

    wl = [0 for _ in range(max_head_index)]     # workload

    e1, e2, e3 = 1, 0.5, 1. / 6

    component_result, cycle_result, feeder_slot_result = [], [], []
    for index in population:
        if component_cell.loc[index]['points'] == 0:
            continue
        # 元胞对应的元件类型和贴装点数
        component_type, component_points = component_cell.loc[index, 'index'], component_cell.loc[index, 'points']

        nozzle_change, maxwl = [0 for _ in range(max_head_index)], [0 for _ in range(max_head_index)]
        for head in range(max_head_index):
            if head_assignment[head]:
                assigned_part = head_assignment[head][-1][0]
                if component_data.loc[assigned_part]['nz'] != component_data.loc[component_type]['nz']:
                    nozzle_change[head] = 1
            wl1 = wl.copy()
            wl1[head] += component_points
            maxwl[head] = max(wl1) + e1 * nozzle_change[head]

        awl, wl2 = min(maxwl), wl.copy()
        for idx, val in enumerate(maxwl):
            if val > awl:
                wl2[idx] += e3
        head_ = wl2.index(min(wl2))
        wl[head_] += component_points
        head_assignment[head_].append([component_type, component_points])

    head_assignment_counter = [0 for _ in range(max_head_index)]
    while True:
        assigned_part, assigned_cycle = [-1 for _ in range(max_head_index)], [0 for _ in range(max_head_index)]
        for head in range(max_head_index):
            counter = head_assignment_counter[head]

            if head_assignment[head] and head_assignment[head][counter][1] > 0:
                assigned_part[head] = head_assignment[head][counter][0]
                assigned_cycle[head] = head_assignment[head][counter][1]

        nonzero_cycle = [cycle for cycle in assigned_cycle if cycle > 0]
        if not nonzero_cycle:
            break

        cycle = min(nonzero_cycle)
        cycle_result.append(cycle)
        component_result.append(assigned_part)

        for head in range(max_head_index):
            counter = head_assignment_counter[head]

            if head_assignment[head] and head_assignment[head][counter][1] > 0:
                head_assignment[head][counter][1] -= cycle_result[-1]
                if head_assignment[head][counter][1] == 0 and counter < len(head_assignment[head]) - 1:
                    head_assignment_counter[head] += 1

    feeder_slot_result = feeder_assignment(component_data, pcb_data, component_result, cycle_result)
    return component_result, cycle_result, feeder_slot_result


@timer_wrapper
def optimizer_celldivision(pcb_data, component_data, hinter=True):
    # Crossover method: Two-point crossover
    # Mutation method: Swap
    # Parent selection method: Roulette wheel
    # Termination condition: 20 successive non-improvement iterations
    population_size = 40       # 种群规模
    crossover_rate, mutation_rate = .6, .02
    golden_section = 0.618

    # 获取元件元胞
    point_num = len(pcb_data)
    component_cell = pd.DataFrame({'index': np.arange(len(component_data)), 'points': np.zeros(len(component_data), dtype=int)})
    for point_cnt in range(point_num):
        part = pcb_data.loc[point_cnt, 'fdr'].split(' ', 1)[1]
        index = np.where(component_data['part'].values == part)
        component_cell.loc[index[0], 'points'] += 1
    component_cell = component_cell[~component_cell['points'].isin([0])]

    # component_cell.sort_values(by = "points" , inplace = True, ascending = False)
    best_population, best_component_cell = [], []
    min_pop_val = float('inf')                               # 最优种群价值
    Div, Imp = 0, 0
    while True:
        # randomly generate permutations
        generation_ = np.array(component_cell.index)
        pop_generation = []
        for _ in range(population_size):
            np.random.shuffle(generation_)
            pop_generation.append(generation_.tolist())

        pop_val = []
        for pop in range(population_size):
            component_result, cycle_result, feeder_slot_result = convert_cell_2_result(pcb_data, component_data,
                                                                                       component_cell,
                                                                                       pop_generation[pop])
            pop_val.append(
                component_assign_evaluate(component_data, component_result, cycle_result, feeder_slot_result))


        # 初始化随机生成种群
        Upit = int(1.5 * np.sqrt(len(component_cell)))

        while Div < Upit:
            if hinter:
                print('----- current div :   ' + str(Div) + ' , total div :   ' + str(Upit) + '  -----')

            # 选择
            new_pop_generation, new_pop_val = [], []
            top_k_index = get_top_k_value(pop_val, int(population_size * 0.3))
            for index in top_k_index:
                new_pop_generation.append(pop_generation[index])
                new_pop_val.append(pop_val[index])
            index = [i for i in range(population_size)]

            select_index = random.choices(index, weights=pop_val, k=population_size - int(population_size * 0.3))
            for index in select_index:
                new_pop_generation.append(pop_generation[index])
                new_pop_val.append(pop_val[index])
            pop_generation, pop_val = new_pop_generation, new_pop_val

            # 交叉
            for pop in range(population_size):
                if pop % 2 == 0 and np.random.random() < crossover_rate:
                    index1, index2 = selection(pop_val), -1
                    while True:
                        index2 = selection(pop_val)
                        if index1 != index2:
                            break
                    # 两点交叉算子
                    pop_generation[index1], pop_generation[index2] = crossover(pop_generation[index1],
                                                                               pop_generation[index2])

                if np.random.random() < mutation_rate:
                    index_ = selection(pop_val)
                    swap_mutation(pop_generation[index_])

            # 将元件元胞分配到各个吸杆上，计算价值函数
            for pop in range(population_size):
                component_result, cycle_result, feeder_slot_result = convert_cell_2_result(pcb_data, component_data,
                                                                                           component_cell,
                                                                                           pop_generation[pop])
                pop_val[pop] = component_assign_evaluate(component_data, component_result, cycle_result,
                                                         feeder_slot_result)
                assert(pop_val[pop] > 0)

            if min(pop_val) < min_pop_val:
                min_pop_val = min(pop_val)
                best_population = copy.deepcopy(pop_generation[np.argmin(pop_val)])
                best_component_cell = copy.deepcopy(component_cell)
                Div, Imp = 0, 1
            else:
                Div += 1

        if Imp == 1:
            Div, Imp = 0, 0
            # Section: cell division operation
            if hinter:
                print(' -------------  cell division operation  ------------- ')
            division_component_cell = pd.DataFrame()
            for idx, rows in component_cell.iterrows():
                if component_cell.loc[idx, 'points'] <= 1:
                    division_component_cell = pd.concat([division_component_cell, pd.DataFrame([rows])],
                                                        ignore_index=True)
                else:
                    division_component_cell = pd.concat([division_component_cell, pd.DataFrame([rows] * 2)],
                                                        ignore_index=True)

                    rows_counter = len(division_component_cell)
                    division_points = int(max(np.ceil(division_component_cell.loc[rows_counter - 2,
                                                                                  'points'] * golden_section), 1))
                    # 避免出现空元胞的情形
                    if division_points == 0 or division_points == division_component_cell.loc[
                        rows_counter - 2, 'points']:
                        division_component_cell.loc[rows_counter - 2, 'points'] = 1
                    else:
                        division_component_cell.loc[rows_counter - 2, 'points'] = division_points

                    division_component_cell.loc[rows_counter - 1, 'points'] -= division_component_cell.loc[
                        rows_counter - 2, 'points']

                    if division_component_cell.loc[rows_counter - 2, 'points'] == 0 or division_component_cell.loc[
                        rows_counter - 1, 'points'] == 0:
                        raise ValueError

            component_cell = division_component_cell

            # 完成分裂后重新生成染色体组
            generation_ = np.array(range(len(component_cell)))
            pop_generation = []
            for _ in range(population_size):
                np.random.shuffle(generation_)
                pop_generation.append(generation_.tolist())
        else:
            break

    assert(len(best_component_cell) == len(best_population))
    return convert_cell_2_result(pcb_data, component_data, best_component_cell, best_population)
