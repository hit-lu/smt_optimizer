from dataloader import *
from result_analysis import *

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

def convert_cell_2_result(component_cell, population):
    head_counter = [0 for _ in range(max_head_index)]
    head_component = [-1 for _ in range(max_head_index)]

    component_result, cycle_result, feederslot_result = [], [], []
    for index in population:
        if component_cell.loc[index, 'points'] == 0:
            continue

        if min(head_counter) != 0:
            cycle = min(head_counter)
            cycle_result.append(cycle)
            component_result.append(copy.deepcopy(head_component))
            for head in range(max_head_index):
                if head_counter[head] == cycle:
                    head_component[head] = -1
            head_counter -= cycle

        head = np.argmin(np.array(head_counter))
        head_counter[head] += component_cell.loc[index, 'points']
        head_component[head] = component_cell.loc[index, 'index']

    while sum(head_component) != -max_head_index:
        cycle = min(filter(lambda x: x > 0, head_counter))
        cycle_result.append(cycle)
        component_result.append(copy.deepcopy(head_component))
        for head in range(max_head_index):
            if head_counter[head] == 0:
                continue
            head_counter[head] -= cycle
            if head_counter[head] == 0:
                head_component[head] = -1

    # TODO: 供料器分配结果
    return component_result, cycle_result, feederslot_result

def optimizer_celldivision():
    # Crossover method: Two-point crossover
    # Mutation method: Swap
    # Parent selection method: Roulette wheel
    # Termination condition: 20 successive non-improvement iterations
    population_size = 40       # 种群规模
    crossover_rate, mutation_rate = .6, .02
    golden_section = 0.618

    # 获取元件元胞
    component_cell = pd.DataFrame({'index': np.arange(len(component_data)), 'points': np.zeros(len(component_data), dtype = np.int)})
    for point_cnt in range(point_num):
        part = pcb_data.loc[point_cnt, 'fdr'].split(' ', 1)[1]
        index = np.where(component_data['part'].values == part)
        component_cell.loc[index[0], 'points'] += 1

    # component_cell.sort_values(by = "points" , inplace = True, ascending = False)
    best_population = []
    min_pop_eval = np.inf                               # 最优种群价值
    Div, Imp = 0, 0
    while True:
        pop_eval = [0 for _ in range(population_size)]          # 种群个体价值

        # 初始化随机生成种群
        iteration_count = np.ceil(1.5 * len(component_cell))
        while Div < iteration_count:
            generation_ = np.array(range(len(component_cell)))
            pop_generation = []
            for _ in range(population_size):
                np.random.shuffle(generation_)
                pop_generation.append(generation_.tolist())

            # TODO: 两点交叉算子
            crossover(pop_generation[0], pop_generation[1])

            # TODO: 交换变异算法
            swap_mutation(pop_generation[0])

            # 将元件元胞分配到各个吸杆上
            for pop in range(population_size):
                pop_eval[pop] = component_assign_evaluate(convert_cell_2_result(pop_generation[pop]))

            if min(pop_eval) < min_pop_eval:
                min_pop_eval = min(pop_eval)
                best_population = copy.deepcopy(pop_generation[np.argmin(pop_eval)])
                Div, Imp = 0, 1
            else:
                Div += 1
        if Imp == 1:
            # TODO: cell division operation
            pass
        else:
            break

    return convert_cell_2_result(best_population)

optimizer_celldivision()