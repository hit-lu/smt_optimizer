
import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

# readin
step_col = ["ref", "x", "y", "z", "r", "part", "fdr", "nz", "hd", "cs", "cy", "sk", "ar", "fid", "pl", "lv"]
part_col = ["part", "fdr", "nz1", "nz2"]

pcb_data = pd.DataFrame(pd.read_csv('PCB.txt', '\t', header=None)).dropna(axis=1)
pcb_data.columns = step_col

component_data = pd.DataFrame(pd.read_csv('Component.txt', '\t', header=None))
component_data.columns = part_col

# registed component check
for i in range(len(pcb_data)):
    if not pcb_data.loc[i].part in component_data['part'].values:
        raise Exception("unregisted component:  " + pcb_data.loc[i].part)

# output
# component_assign: 2-dimension array, its element represents the index in dataframe "component_data"
# subcyle: 1-dimension array, its element represents the number of repetitions for each row in "component_assign"
# feeder_assign: 1-dimension, feeder_assign[s] represents the component index in feeder slot s
# placement_result: 2-dimension array, placement_result[c, h] represents the placement index in "pcb_data"

# Paper Information : PCB assembly scheduling with alternative nozzle types for one component type [2011] [Shujuan Guo • Katsuhiko Takahashi • Katsumi Morikawa]
# 1.Nozzle assignment **** Genetic algorithm
# 2.Head allocation **** Greedy heuristic
# 3.CT grouping **** Greedy heuristic
# 4.Pickup clustering **** Agglomerative hierarchical clustering


# ***** Nozzle assignment **** Genetic algorithm ****
nt_list = component_data["nz1"].append(component_data["nz2"], ignore_index=True).drop_duplicates()
nt_list.index = range(len(nt_list))
print(nt_list)
# ct-nt matrix
ct_nt_mat = pd.DataFrame(0, index=range(len(nt_list)), columns=component_data["part"])
for part_index in range(len(component_data)):
    for nt_index in range(len(nt_list)):
        if component_data["nz1"].iloc[part_index] == nt_list.iloc[nt_index] or component_data["nz2"].iloc[part_index] == nt_list.iloc[nt_index]:
            ct_nt_mat[component_data["part"].iloc[part_index]].iloc[nt_index] = 1

ct_list = pd.DataFrame(0, index=range(len(component_data)), columns=["ct_cnt"])
for ct_index in range(len(component_data)):
    for step_index in range(len(pcb_data)):
        if pcb_data["part"].iloc[step_index] == component_data["part"].iloc[ct_index]:
            ct_list.iloc[ct_index] += 1
print(ct_list)
# Encoding
chromosome = pd.DataFrame(columns=['part', 'gene'])
gene_cnt = 0
for part_index in range(len(component_data)):
    part_name = component_data["part"].iloc[part_index]
    if ct_nt_mat[part_name].sum() <= 1:
        continue
    gene_len = ct_list.iloc[part_index] + ct_nt_mat[part_name].sum() - 1
    gene_1 = ct_nt_mat[part_name].sum() - 1
    gene = np.zeros(gene_len)
    while gene.sum() < gene_1:
        init_1 = np.random.randint(gene_len - 1)
        gene[init_1] = 1  # 随机初始化 1
    chromosome.loc[gene_cnt] = [part_name, gene]
    gene_cnt += 1
# print(chromosome)

# Decoding
for part_index in range(len(component_data)):
    part_name = component_data["part"].iloc[part_index]
    if ct_nt_mat[part_name].sum() <= 1:
        for nt_index in range(len(nt_list)):
            if ct_nt_mat.loc[nt_index, part_name] == 1:
                # 元件仅对应一种吸嘴，直接等于所有元件数
                ct_nt_mat.loc[nt_index, part_name] = ct_list.iloc[part_index]

for part_index_gene in range(len(chromosome)):
    part_name = chromosome["part"].iloc[part_index_gene]
    temp_gene = chromosome["gene"].iloc[part_index_gene]
    gene_len = len(temp_gene)
    gene_1 = np.zeros(int(ct_nt_mat[part_name].sum()))
    gene_1[int(ct_nt_mat[part_name].sum() - 1)] = gene_len
    gene_1_index = 0
    for gene_index in range(gene_len - 1):
        if temp_gene[gene_index] == 1:
            gene_1[gene_1_index] = gene_index
            gene_1_index = gene_1_index + 1
    # print(gene_1)
    nt_cnt = np.zeros(int(ct_nt_mat[part_name].sum()))
    for index in range(len(gene_1)):
        if index < 1:
            nt_cnt[index] = gene_1[index]
        else:
            nt_cnt[index] = gene_1[index] - gene_1[index - 1] - 1
    # print(nt_cnt)
    nt_cnt_index = 0
    for nt_index in range(len(nt_list)):
        if ct_nt_mat.loc[nt_index, part_name] == 1:
            # 元件对应多种吸嘴，解码赋值
            ct_nt_mat.loc[nt_index, part_name] = nt_cnt[nt_cnt_index]
            nt_cnt_index = nt_cnt_index + 1
print(ct_nt_mat)

# 适应度函数

# 轮盘赌选择

# 交叉

# 变异

# ***** Head allocation **** Greedy heuristic ****
max_head_num = 6
step_cnt = len(pcb_data)
head_allocation_list = pd.DataFrame(0, index=range(len(nt_list)), columns=["head"])
# S1.分配对应数量较少元件的吸嘴单个头
R_1 = max_head_num
P_1 = step_cnt
P_2 = P_1
for nt_index in range(len(nt_list)):
    if P_1 / R_1 > ct_nt_mat.iloc[nt_index].sum():
        head_allocation_list.loc[nt_index] = 1
        P_2 -= ct_nt_mat.iloc[nt_index].sum()
# S2.分配剩余数量吸嘴
R_2 = R_1 - head_allocation_list["head"].sum()
S_2 = []
if R_2 > 0:
    for nt_index in range(len(nt_list)):
        if P_1 / R_1 <= ct_nt_mat.iloc[nt_index].sum():
            head_allocation_list.loc[nt_index] = math.floor(R_2 * ct_nt_mat.iloc[nt_index].sum() / P_2)
            S_2.append(nt_index)
# S3.调整S2中分配吸嘴
R_3 = R_1 - head_allocation_list["head"].sum()
if R_3 > 0:
    max_index = 0
    for index in range(len(nt_list)):
        if head_allocation_list.iloc[max_index].values <= head_allocation_list.iloc[index].values:
            max_index = index
    head_allocation_list.iloc[max_index] += 1
    for i in range(0, len(S_2)):
        for j in range(i+1, len(S_2)):
            b_i = math.floor(R_2 * ct_nt_mat.iloc[i].sum() / P_2)
            b_j = math.floor(R_2 * ct_nt_mat.iloc[j].sum() / P_2)
            if ct_nt_mat.iloc[i].sum() / (b_i - 1) < ct_nt_mat.iloc[j].sum() / b_j and b_i > 1:
                head_allocation_list.iloc[i] -= 1
                head_allocation_list.iloc[j] += 1
            elif ct_nt_mat.iloc[i].sum() / (b_i - 1) > ct_nt_mat.iloc[j].sum() / b_j and b_j > 1:
                head_allocation_list.iloc[j] -= 1
                head_allocation_list.iloc[i] += 1
print(head_allocation_list)

# ***** CT grouping  *****  Greedy heuristic *****
ct_group_num = 0
for nt_index in range(len(nt_list)):
    cnt = 0
    for part_index in range(len(component_data)):
        part_name = component_data["part"].iloc[part_index]
        if ct_nt_mat.iloc[nt_index][part_name] > 0:
            cnt += 1
    if math.ceil(cnt / head_allocation_list.iloc[nt_index]) > ct_group_num and cnt > 0:
        ct_group_num = math.ceil(cnt / head_allocation_list.iloc[nt_index])

group_columns = []
for group_index in range(ct_group_num):
    group_columns.append(group_index)
ct_group_list = pd.DataFrame(" ", index=range(max_head_num), columns=group_columns)
ct_nt_temp = ct_nt_mat.copy(deep=True)
sum_head = 0
head_num = 0
for nt_index in range(len(nt_list)):
    temp_line = (ct_nt_temp.iloc[nt_index]).to_frame()
    temp_line.columns = ["num"]
    temp_line.sort_values(axis=0, by=["num"], ascending=False, inplace=True)  # 降序
    group_len = len(temp_line) - (temp_line["num"] == 0).astype(int).sum(axis=0)  # 非零个数
    sum_head += head_num
    head_num = head_allocation_list.iloc[nt_index]["head"]
    for i in range(math.ceil(group_len / head_num)):
        for j in range(head_num):
            temp_index = head_num*i+j
            if temp_index > len(temp_line) - 1:
                continue
            if temp_line.iloc[temp_index]["num"] > 0:
                ct_group_list.iloc[sum_head+j][i] = temp_line.iloc[temp_index].name
print(ct_group_list)

# ***** Pickup clustering **** Agglomerative hierarchical clustering *****
