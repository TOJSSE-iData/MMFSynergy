# %%
import os
import csv
import random
import re
import sys

from collections import defaultdict, Counter, OrderedDict
from itertools import combinations

import pandas as pd
import numpy as np
import sklearn
import joblib
import torch

from tqdm import tqdm
from rdkit import Chem

sys.path.insert(0, '..')

# %%
def create_dir(*args):
    """获取文件夹路径，如果文件夹不存在，则创建文件夹
    """
    dir_pth = os.path.join(*args)
    if not os.path.exists(dir_pth):
        os.makedirs(dir_pth)
        print(f"make dir: {dir_pth}")
    else:
        print(f"dir exists: {dir_pth}")
    return dir_pth

def pands_df_to_tsv(df, filepath):
    df.to_csv(filepath, sep='\t', index=False)

def get_prev_version():
    if VERSION == 'v00':
        raise ValueError("Already the first version")
    elif VERSION == 'v0':
        return 'v00'
    else:
        v = int(VERSION[1:])
        return f"v{v-1}"

# %% [markdown]
# # V00

# %%
VERSION = 'v00'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## DrugComb数据

# %%
DATASET = 'drug_comb'
data_dir_dataset = create_dir('raw', DATASET)

# %%
data_dir_output = create_dir(data_dir_dataset, 'study')
study_counter = Counter()

with open(os.path.join(data_dir_dataset, "summary_v_1_5.csv", 'r', newline="")) as csvfile:
    sample_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(sample_reader)
    col2idx = {x: i for i, x in enumerate(header)}
    for i, row in enumerate(sample_reader):
        study_counter[row[col2idx['study_name']]] += 1
for k, v in study_counter.items():
    print(k, v)

chunck = defaultdict(list)
for k in study_counter:
    with open(os.path.join(data_dir_output, f"{k.replace('/', '-')}.csv"), 'w') as f:
        f.write(','.join([f'"{x}"' for x in header])+'\n')

with open(os.path.join(data_dir_output, "summary_v_1_5.csv", 'r', newline="")) as csvfile:
    sample_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(sample_reader)
    for i, row in enumerate(sample_reader):
        study_name = row[col2idx['study_name']]
        chunck[study_name].append(row)
        if len(chunck[study_name]) == 10000:
            with open(os.path.join(data_dir_output, f"{study_name.replace('/', '-')}.csv"), "a") as f:
                for row in chunck[study_name]:
                    f.write(','.join([f'"{x}"' for x in header])+'\n')
            chunck[study_name] = []
for study_name, rows in chunck.items():
    with open(os.path.join(data_dir_output, f"{study_name.replace('/', '-')}.csv"), "a") as f:
        for row in rows:
            f.write(','.join([f'"{x}"' for x in header])+'\n')
del chunck

# %%
with open(os.path.join(data_dir_dataset, 'statistic.txt'), 'w') as f:
    f.write('\t'.join(['study', 'n_drugs', 'n_cell_lines', 'n_ddcs']) + '\n')
    for fn in sorted(os.listdir(os.path.join(data_dir_dataset, 'study'))):
        if not fn.endswith('csv'):
            continue
        df = pd.read_csv(os.path.join(data_dir_dataset, 'study', fn), usecols=['drug_row', 'drug_col', 'cell_line_name'])
        n_cells = df['cell_line_name'].nunique()
        all_drugs = set(df['drug_row'])
        df = df.dropna()
        df = df[df['drug_col'] != '\\N']
        all_drugs.update(df['drug_col'])
        n_drugs = len(all_drugs)
        n_dcs = df.drop_duplicates().shape[0]
        f.write('\t'.join(map(str, [fn.split('.')[0], n_drugs, n_cells, n_dcs])) + '\n')

# %% [markdown]
# ## Oneil数据

# %%
STUDY = 'oneil'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %%
synergy_file = os.path.join("raw", STUDY, "labels.csv")
synergy_samples = pd.read_csv(synergy_file, usecols=['drug_a_name','drug_b_name','cell_line','synergy','fold'])
synergy_samples = synergy_samples.groupby(['drug_a_name','drug_b_name','cell_line','fold']).mean().reset_index()
synergy_samples.columns = ['drug_row', 'drug_col', 'cell_line', 'fold', 'synergy_loewe']
print(synergy_samples.shape)
# synergy_samples.head()

o2dc = {}
for tp in ['drug', 'cell_line']:
    o2dc_tp = pd.read_csv(os.path.join(proc_dir_study, f"{tp}_name_map_oneil2dc.tsv"), sep='\t', usecols=['o_name','dc_name'], index_col='o_name')
    o2dc_tp = o2dc_tp['dc_name'].to_dict()
    o2dc[tp] = o2dc_tp
    print(tp, len(o2dc[tp]))

synergy_samples['drug_row'] = synergy_samples['drug_row'].apply(lambda x: o2dc['drug'][x])
synergy_samples['drug_col'] = synergy_samples['drug_col'].apply(lambda x: o2dc['drug'][x])
synergy_samples['cell_line'] = synergy_samples['cell_line'].apply(lambda x: o2dc['cell_line'][x])
# synergy_samples.head()

supply_file = os.path.join("raw", "drug_comb", "study", f"{STUDY.upper()}.csv")
supply_samples = pd.read_csv(supply_file, usecols=['drug_row', 'drug_col', 'cell_line_name', 'synergy_zip', 'synergy_hsa', 'synergy_bliss'])
supply_samples = supply_samples.groupby(['drug_row','drug_col','cell_line_name']).mean().reset_index()
supply_samples.columns = ['drug_row', 'drug_col', 'cell_line', 'synergy_zip', 'synergy_hsa', 'synergy_bliss']
print(supply_samples.shape)
# supply_samples.head()

synergy_samples = synergy_samples.join(
    supply_samples.set_index(['drug_row','drug_col','cell_line']), on=['drug_row','drug_col','cell_line'], how='inner'
)
print(synergy_samples.shape)
# synergy_samples.head()

synergy_samples.to_csv(os.path.join(proc_dir_study, "samples.tsv"), sep='\t', index=False)

# %%
synergy_file = os.path.join("raw", STUDY, "labels.csv")
synergy_samples = pd.read_csv(synergy_file, usecols=['drug_a_name','drug_b_name','cell_line','synergy','fold'])
synergy_samples = synergy_samples.groupby(['drug_a_name','drug_b_name','cell_line','fold']).mean().reset_index()
synergy_samples.columns = ['drug_row', 'drug_col', 'cell_line', 'fold', 'synergy_loewe']
print(synergy_samples.shape)
# synergy_samples.head()

o2dc = {}
for tp in ['drug', 'cell_line']:
    o2dc_tp = pd.read_csv(os.path.join("proc", tp, STUDY, "name_map_oneil2dc.tsv"), sep='\t', usecols=['o_name','dc_name'], index_col='o_name')
    o2dc_tp = o2dc_tp['dc_name'].to_dict()
    o2dc[tp] = o2dc_tp
    print(tp, len(o2dc[tp]))

synergy_samples['drug_row'] = synergy_samples['drug_row'].apply(lambda x: o2dc['drug'][x])
synergy_samples['drug_col'] = synergy_samples['drug_col'].apply(lambda x: o2dc['drug'][x])
synergy_samples['cell_line'] = synergy_samples['cell_line'].apply(lambda x: o2dc['cell_line'][x])
# synergy_samples.head()

supply_file = os.path.join("raw", "drug_comb", "study", f"{STUDY.upper()}.csv")
supply_samples = pd.read_csv(supply_file, usecols=['drug_row', 'drug_col', 'cell_line_name', 'synergy_zip', 'synergy_hsa', 'synergy_bliss'])
supply_samples = supply_samples.groupby(['drug_row','drug_col','cell_line_name']).mean().reset_index()
supply_samples.columns = ['drug_row', 'drug_col', 'cell_line', 'synergy_zip', 'synergy_hsa', 'synergy_bliss']
print(supply_samples.shape)
# supply_samples.head()

synergy_samples = synergy_samples.join(
    supply_samples.set_index(['drug_row','drug_col','cell_line']), on=['drug_row','drug_col','cell_line'], how='inner'
)
print(synergy_samples.shape)
# synergy_samples.head()

synergy_samples.to_csv(os.path.join("proc", "synergy", STUDY, "samples.tsv"), sep='\t', index=False)

# %%
all_drugs = set()
all_drugs.update(synergy_samples['drug_row'])
all_drugs.update(synergy_samples['drug_col'])
drug2idx = {}
with open(os.path.join(proc_dir_study, "drug2idx.tsv"), 'w') as f:
    f.write('drug\tidx\n')
    for i, e in enumerate(sorted(all_drugs)):
        f.write(f"{e}\t{i}\n")
        drug2idx[e] = i

all_cells = set(synergy_samples['cell_line'])
cell2idx = {}
with open(os.path.join(proc_dir_study, "cell_line2idx.tsv"), 'w') as f:
    f.write('cell_line\tidx\n')
    for i, e in enumerate(sorted(all_cells)):
        f.write(f"{e}\t{i}\n")
        cell2idx[e] = i
synergy_samples_idx = synergy_samples.rename(columns={'drug_row': 'drug_row_idx', 'drug_col': 'drug_col_idx', 'cell_line': 'cell_line_idx'})
synergy_samples_idx['drug_row_idx'] = synergy_samples_idx['drug_row_idx'].apply(lambda x: drug2idx[x])
synergy_samples_idx['drug_col_idx'] = synergy_samples_idx['drug_col_idx'].apply(lambda x: drug2idx[x])
synergy_samples_idx['cell_line_idx'] = synergy_samples_idx['cell_line_idx'].apply(lambda x: cell2idx[x])
synergy_samples_idx.to_csv(os.path.join(proc_dir_study, "samples_idx.tsv"), sep='\t', index=False)

# %% [markdown]
# ### CGMS数据-生成药物/细胞系特征

# %%
DATASET = 'cgms'

# %%
for entity in ['drug', 'cell_line']:
    ent_shorten = entity.split('_')[0]
    raw_data_dir = os.path.join('raw', DATASET)
    ent_shorten = entity.split('_')[0]
    entity2idx_old = pd.read_csv(os.path.join(raw_data_dir, f"dc2o-{ent_shorten}s.csv"), index_col=['dc-name'])
    entity2idx_old = entity2idx_old['idx'].to_dict()
    feat_old = np.load(os.path.join("raw", ent_shorten, DATASET, f"{ent_shorten}_feat_ae.npy"))
    entity2idx_new = pd.read_csv(os.path.join(proc_dir_study, f"{entity}2idx.tsv"), sep='\t', index_col=[entity])
    entity2idx_new = entity2idx_new['idx'].to_dict()
    feat_new = np.zeros((len(entity2idx_new), feat_old.shape[1]))
    print(feat_new.shape)
    for ent, id_new in entity2idx_new.items():
        id_old = entity2idx_old[ent]
        feat_new[id_new] = feat_old[id_old]
    np.save(os.path.join(proc_dir_study, f"{entity}_feat_cgms.npy"), feat_new)

# %% [markdown]
# # V0

# %%
VERSION = 'v0'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## Harmonizome数据探索

# %%
DATASET = 'harmonizome'
raw_data_dir = os.path.join('raw', DATASET)

# %%
def stat_gene_matrix(fp):
    gene_symbols = set()
    gene_ids = set()
    genes = set()
    n_genes = 0
    with open(fp, 'r') as f:
        cell_header = next(f).strip().split('\t')[3:]
        print(f"n raw cells: {len(cell_header)}")
        cells = set(cell_header)
        print(f"n dedup cells: {len(cells)}")
        next(f)
        next(f)
        for line in f:
            gca = line.strip().split('\t')
            gs, _, gid = gca[:3]
            n_genes += 1
            gene_symbols.add(gs)
            gene_ids.add(gid)
            genes.add((gs, gid))
        print(f"n raw genes: {n_genes}")
        print(f"n dedup gene symbols: {len(gene_symbols)}")
        print(f"n dedup gene ids: {len(gene_ids)}")
        print(f"n dedup genes: {len(genes)}")

# %%
# gene attribute matrix
# 官网介绍
"""
matrix with genes labeling the rows and attributes labeling the columns,
reporting positive [+1], negative [-1] (if applicable), 
and unobserved or non-significant [0] associations
"""
fp = os.path.join(raw_data_dir, "gene_attribute_matrix.txt")
print(fp)
stat_gene_matrix(fp)

gca_counter = Counter()
with open(fp, 'r') as f:
    for _ in range(3):
        next(f)
    for line in f:
        gca = list(map(int, map(float, line.strip().split('\t')[3:])))
        gca_counter.update(gca)
for k, v in gca_counter.items():
    print(k, v)
print(sum(gca_counter.values()))

# %%
# gene attribute edge list
# 官网介绍
"""
list of (gene, attribute, value) triplets,
reporting positive [+1] and negative [-1] (if applicable) associations
"""
fp = os.path.join(raw_data_dir, "gene_attribute_edges.txt")
print(fp)
df = pd.read_csv(fp, sep='\t', skiprows=1, usecols=['GeneSym', 'GeneID', 'CellLine', 'Tissue', 'weight'])
print(df.groupby('weight').count()['GeneID'])
df = df.drop_duplicates(subset=['GeneID', 'CellLine'])
print(df.shape[0])
print(df.groupby('weight').count()['GeneID'])

# %%
# gene_set_library_up_crisp.gmt
# 官网介绍
"""
list of gene sets, where each gene set is labelled with an attribute,
reporting positive associations only
"""
fp = os.path.join(raw_data_dir, "gene_set_library_up_crisp.gmt")
print(fp)
cell2up = defaultdict(set)
with open(fp, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(cell2up[pieces[0]]) > 0:
            print(f"dup cell line: {pieces[0]}")
            cur_genes = set(pieces[2:])
            print(f"n previous association: {len(cell2up[pieces[0]])}")
            print(f"n current  association: {len(cur_genes)}")
            print(f"n common: {len(cur_genes.intersection(cell2up[pieces[0]]))}")
        else:
            cell2up[pieces[0]].update(pieces[2:])
n_asss = sorted([len(x) for x in cell2up.values()])
print(n_asss[:10])
print(n_asss[-10:])

# %%
# gene_set_library_dn_crisp.gmt
# 官网介绍
"""
list of gene sets, where each gene set is labelled with an attribute,
reporting negative associations only
"""
fp = os.path.join(raw_data_dir, "gene_set_library_dn_crisp.gmt")
print(fp)
cell2dn = defaultdict(set)
with open(fp, 'r') as f:
    for line in f:
        pieces = line.strip().split('\t')
        if len(cell2dn[pieces[0]]) > 0:
            print(f"dup cell line: {pieces[0]}")
            cur_genes = set(pieces[2:])
            print(f"n previous association: {len(cell2dn[pieces[0]])}")
            print(f"n current  association: {len(cur_genes)}")
            print(f"n common: {len(cur_genes.intersection(cell2dn[pieces[0]]))}")
        else:
            cell2dn[pieces[0]].update(pieces[2:])
n_asss = sorted([len(x) for x in cell2dn.values()])
print(n_asss[:10])
print(n_asss[-10:])

# %%
# gene_list_terms.txt
# 官网介绍
"""
list of genes in the processed dataset
"""
fp = os.path.join(raw_data_dir, "gene_list_terms.txt")
df = pd.read_csv(fp, sep='\t', usecols=['GeneSym', 'GeneID'])
print(df.shape)
print(df.info())
df.head()

# %%
# attribute_list_entries.txt
# 官网介绍
"""
list of attributes in the processed dataset
"""
fp = os.path.join(raw_data_dir, "attribute_list_entries.txt")
df = pd.read_csv(fp, sep='\t', usecols=['CellLine', 'Tissue'])
print(df.shape)
print(df.info())
df = df.drop_duplicates()
print(df.shape)

# %%
# gene_attribute_matrix_cleaned.txt
# 官网介绍
"""
matrix with genes labeling the rows and attributes labeling the columns,
 after harmonizing gene or protein labels,
 filtering rows or columns with too many missing values (more than 5%),
 imputing remaining missing values, and averaging rows or columns with replicate measurements or samples
"""
fp = os.path.join(raw_data_dir, "gene_attribute_matrix_cleaned.txt")
print(fp)
stat_gene_matrix(fp)

# %%
# gene_attribute_matrix_standardized.txt
# 官网介绍
"""
matrix with genes labeling the rows and attributes labeling the columns,
 after standardizing scores for gene-biological entity associations
"""
fp = os.path.join(raw_data_dir, "gene_attribute_matrix_standardized.txt")
print(fp)
stat_gene_matrix(fp)

# %%
# 用来确认gene_matrix文件作用的代码

# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# with open(f"{base_data_dir}/gene_attribute_matrix_cleaned.txt", 'r') as f:
#     for _ in range(3):
#         next(f)
#     gca = next(f).strip().split('\t')
#     gs, _, gid = gca[:3]
#     print(gs, gid)
#     gel = list(map(float, gca[3:]))
#     gel = [[x] for x in gel]
#     gel_ss = ss.fit_transform(gel)
#     print(gel_ss[:30, 0])


# with open(f"{base_data_dir}/gene_attribute_matrix_standardized.txt", 'r') as f:
#     cells = next(f).strip().split('\t')[3:]
#     tgt_cidx = cells.index('1321N1')
#     gel = []
#     gss = []
#     print(tgt_cidx)
#     for _ in range(2):
#         next(f)
#     for line in f:
#         gca = line.strip().split('\t')
#         gs, _, gid = gca[:3]
#         gss.append(gs)
#         gel.append(float(gca[3+tgt_cidx]))
# print(len(gss))
# print(len(gel))
# gel[gss.index('MRI1')]

# %% [markdown]
# ## Oneil数据集

# %%
STUDY = 'oneil'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### 药物相关数据与V00相同

# %%
proc_dir_study_prev = os.path.join('proc', 'v00', STUDY)

# %%
command = f"ln {proc_dir_study_prev}/drug2idx.tsv {proc_dir_study}/drug2idx.tsv"
cmd_res = os.system(command)

command = f"ln {proc_dir_study_prev}/drug_feat_cgms.npy {proc_dir_study}/drug_feat_cgms.npy"
cmd_res = os.system(command)

# %% [markdown]
# ### 生成细胞系-基因关联
# 
# 使用Harmonizome数据中提供的基因上下调数据，提取蛋白质与细胞系的关联

# %%
DATASET = 'harmonizome'
raw_data_dir = os.path.join('raw', DATASET)
proc_dir_study_prev = os.path.join('proc', 'v00', STUDY)

# %% [markdown]
# #### 抽取存在基因上下调关系的细胞系

# %%
cell_lines = []
proc_dir_study_prev = os.path.join('proc', 'v00', STUDY)
with open(os.path.join(proc_dir_study_prev, "cell_line2idx.tsv"), 'r') as f:
    next(f)
    for line in f:
        cell_lines.append(line.split('\t')[0])
print(len(cell_lines))

hmz_cell_lines = pd.read_csv(os.path.join(raw_data_dir, "attribute_list_entries.txt"), sep='\t', usecols=['CellLine'])
hmz_cell_lines = set(hmz_cell_lines['CellLine'].tolist())
print(len(hmz_cell_lines))

ccle2dc = pd.read_csv(os.path.join(proc_dir_study, "cell_line_name_map_ccle2dc.tsv"), sep='\t', index_col=['ccle_name'])
ccle2dc = ccle2dc['dc_name'].to_dict()
dc2ccle = {v: k for k, v in ccle2dc.items()}
print(len(ccle2dc))

oneil_cell_in_ccle = []
for cc, dc in ccle2dc.items():
    if cc in hmz_cell_lines:
        oneil_cell_in_ccle.append(dc)
    else:
        print(dc)

with open(os.path.join(proc_dir_study, "cell_line2idx.tsv"), 'w') as f:
    f.write("cell_line\tidx\n")
    for idx, cc in enumerate(sorted(oneil_cell_in_ccle)):
        f.write(f"{cc}\t{idx}\n")

# %% [markdown]
# #### 生成相应的synergy数据

# %%
synergy_samples = pd.read_csv(os.path.join(proc_dir_study_prev, "samples.tsv"), sep='\t')
print(synergy_samples.shape[0])
# synergy_samples.head()

synergy_samples_hmz_34 = synergy_samples[synergy_samples['cell_line'].isin(oneil_cell_in_ccle)]
print(synergy_samples_hmz_34.shape[0])
print(synergy_samples_hmz_34.groupby('fold').count()['cell_line'])

synergy_samples_hmz_34.to_csv(os.path.join(proc_dir_study, "samples.tsv"), sep='\t', index=False)


# %%
drug2idx = pd.read_csv(os.path.join(proc_dir_study, "drug2idx.tsv"), sep='\t', index_col=['drug'])
drug2idx = drug2idx['idx'].to_dict()

cell2idx = pd.read_csv(os.path.join(proc_dir_study, "cell_line2idx.tsv"), sep='\t', index_col=['cell_line'])
cell2idx = cell2idx['idx'].to_dict()

synergy_samples_hmz_34_idx = synergy_samples_hmz_34.rename(columns={'drug_row': 'drug_row_idx', 'drug_col': 'drug_col_idx', 'cell_line': 'cell_line_idx'})
synergy_samples_hmz_34_idx['drug_row_idx'] = synergy_samples_hmz_34_idx['drug_row_idx'].apply(lambda x: drug2idx[x])
synergy_samples_hmz_34_idx['drug_col_idx'] = synergy_samples_hmz_34_idx['drug_col_idx'].apply(lambda x: drug2idx[x])
synergy_samples_hmz_34_idx['cell_line_idx'] = synergy_samples_hmz_34_idx['cell_line_idx'].apply(lambda x: cell2idx[x])
synergy_samples_hmz_34_idx.to_csv(os.path.join(proc_dir_study, "samples_idx.tsv"), sep='\t', index=False)

# %% [markdown]
# #### 生成细胞系-基因关联

# %%
def read_cell_gene_association(fp):
    cell2gene_ass = defaultdict(set)
    with open(fp, 'r') as f:
        for line in f:
            pieces = line.strip().split('\t')
            if len(cell2gene_ass[pieces[0]]) > 0:
                continue
            else:
                cell2gene_ass[pieces[0]].update(pieces[2:])
    return cell2gene_ass

cell2up = read_cell_gene_association(os.path.join(raw_data_dir, "gene_set_library_up_crisp.gmt"))
cell2dn = read_cell_gene_association(os.path.join(raw_data_dir, "gene_set_library_dn_crisp.gmt"))

# %%
cell2strength = {c: defaultdict(list) for c in oneil_cell_in_ccle}

with open(os.path.join(raw_data_dir, "gene_attribute_matrix_standardized.txt"), 'r') as f:
    cell_header = next(f).strip().split('\t')[3:]
    cell2idx = {c : i for i, c in enumerate(cell_header)}
    for _ in range(2):
        next(f)
    for line in f:
        gca = line.strip().split('\t')
        gs, _, gid = gca[:3]
        gca = gca[3:]
        for dc in oneil_cell_in_ccle:
            cc = dc2ccle[dc]
            if gs in cell2up[cc]:
                cell2strength[dc]['up'].append((gs, float(gca[cell2idx[cc]])))
            elif gs in cell2dn[cc]:
                cell2strength[dc]['dn'].append((gs, float(gca[cell2idx[cc]])))

# %%
N_TRUNCATE = 100
oneil_genes = set()
cell2strength_sml = {}
for dc in cell2strength:
    gs_up = sorted(cell2strength[dc]['up'], key=lambda x: x[1], reverse=True)[:N_TRUNCATE]
    gs_dn = sorted(cell2strength[dc]['dn'], key=lambda x: x[1], reverse=False)[:N_TRUNCATE]
    cell2strength_sml[dc] = dict()
    cell2strength_sml[dc]['up'] = gs_up
    cell2strength_sml[dc]['dn'] = gs_dn
    for gs, s in gs_up:
        oneil_genes.add(gs)
    for gs, s in gs_dn:
        oneil_genes.add(gs)
print(len(oneil_genes))

# %%
gene_symb2id = pd.read_csv(os.path.join(raw_data_dir, "gene_list_terms.txt"), sep='\t', usecols=['GeneSym', 'GeneID'], index_col=['GeneSym'])
gene_symb2id = gene_symb2id['GeneID'].to_dict()

# %%
with open(os.path.join(proc_dir_study, f"cell_line_gene_asso_up{N_TRUNCATE}dn{N_TRUNCATE}.tsv"), 'w') as f:
    f.write("cell_line\tgene_symbol\tgene_id\tweight\n")
    for dc in cell2strength_sml:
        for gs, s in cell2strength_sml[dc]['up']:
            f.write(f"{dc}\t{gs}\t{gene_symb2id[gs]}\t{s:.6f}\n")
        for gs, s in cell2strength_sml[dc]['dn']:
            f.write(f"{dc}\t{gs}\t{gene_symb2id[gs]}\t{s:.6f}\n")

# %% [markdown]
# ### 生成细胞系-蛋白质关联

# %%
N_TRUNCATE = 100

# %%
gene_asso = pd.read_csv(os.path.join(proc_dir_study, f"cell_line_gene_asso_up{N_TRUNCATE}dn{N_TRUNCATE}.tsv"), sep='\t', dtype={'gene_id': str})
print(gene_asso.shape[0])
print(gene_asso['gene_id'].nunique())
print(gene_asso['gene_symbol'].nunique())
print(gene_asso.drop_duplicates(subset=['gene_id', 'gene_symbol']).shape[0])
# # 上面三个的输出一致（5489），说明这里GeneID和GeneSymbol关系是 1-1
# gene_asso.head()

# %%
def comb_symbol_id(gene_symbols, gene_ids):
    gene_symbols = [x.strip() for x in gene_symbols.split(';')]
    if gene_ids != gene_ids:
        gene_ids = [-1] * len(gene_symbols)
    else:
        gene_ids = [int(x.strip()) for x in gene_ids.strip(';').split(';')]
    # 处理长度不一的情况
    if len(gene_symbols) != len(gene_ids):
        if len(gene_symbols) == 1:
            gene_ids = gene_ids[:1]
        elif len(gene_ids) == 1:
            gene_symbols = gene_symbols[:1]
        else:
            return []
    ret = [(s, i) for s, i in zip(gene_symbols, gene_ids)]
    return ret

raw_data_dir = os.path.join('raw', 'uniprot')
gene2prot = pd.read_csv(os.path.join(raw_data_dir, "uniprotkb_9606_reviewed_230722.tsv"), sep='\t', usecols=['Entry', 'Gene Names (primary)', 'GeneID'])
gene2prot.columns = ['protein', 'gene_symbol', 'gene_id']
print(gene2prot.shape[0])  # 20424
# 存在没有gene symbol的数据，这些数据的gene symbol和gene id都是空，删除
gene2prot = gene2prot[~gene2prot['gene_symbol'].isna()]
print(gene2prot.shape[0])  # 20268
# protein与gene是n-n的关系，展开
gene2prot['gene'] = gene2prot.apply(lambda row: comb_symbol_id(row['gene_symbol'], row['gene_id']), axis=1)
print(gene2prot.shape[0])  # 20268
gene2prot = gene2prot[gene2prot['gene'].apply(lambda x: len(x) > 0)]
print(gene2prot.shape[0]) # 20263, 删除了5条多对多的数据
gene2prot = gene2prot[['protein', 'gene']].explode('gene')
gene2prot['gene_symbol'] = gene2prot['gene'].apply(lambda x: x[0])
gene2prot['gene_id'] = gene2prot['gene'].apply(lambda x: x[1])
gene2prot = gene2prot.drop(columns=['gene'])
print(gene2prot.shape[0])  # 20372
print(gene2prot['gene_symbol'].nunique())  # 20286, protein和GeneSymbol 1-n
print(gene2prot[gene2prot['gene_id'] > 0].shape[0])  # 18861, gene id 缺失严重
print(gene2prot[gene2prot['gene_id'] > 0]['gene_id'].nunique())  # 18780, protein和GeneID n-1
# 综上，gene symbol更完整，且与蛋白质关系是n-1，更好处理，使用gene symbol做映射

gene2prot = gene2prot.set_index('gene_symbol')['protein'].to_dict()
gene_asso['protein'] = gene_asso['gene_symbol'].apply(lambda x: gene2prot.get(x, ''))
gene_protein_asso = gene_asso[gene_asso['protein'] != '']
print(gene_asso.shape[0], gene_protein_asso.shape[0])  # 6763 5982

# %%
protein2seq = pd.read_csv(
    os.path.join(raw_data_dir, "uniprotkb_9606_230722.tsv"),
    sep='\t', usecols=['Entry', 'Sequence'], index_col=['Entry']
)
protein2seq = protein2seq['Sequence'].to_dict()
print(len(protein2seq))  # 207981

gene_protein_asso = gene_protein_asso[gene_protein_asso['protein'].isin(protein2seq)]
print(gene_protein_asso.shape[0])  # 5982
gene_protein_asso.to_csv(os.path.join(proc_dir_study, f"cell_line_protein_asso_up{N_TRUNCATE}dn{N_TRUNCATE}.tsv"), sep='\t', index=False)

# %%
protein2idx = {}
with open(os.path.join(proc_dir_study, "protein2idx.tsv"), 'w') as f:
    f.write('protein\tidx\n')
    for i, protein in enumerate(sorted(gene_protein_asso['protein'].unique().tolist())):
        f.write(f"{protein}\t{i + 1}\n")
        protein2idx[protein] = i + 1

# %%
cell2idx = pd.read_csv(os.path.join(proc_dir_study, "cell_line2idx.tsv"), sep='\t', index_col=['cell_line'])
cell2idx = cell2idx['idx'].to_dict()

gene_protein_asso_idx = gene_protein_asso.rename(columns={'cell_line': 'cell_line_idx', 'protein': 'protein_idx'})
gene_protein_asso_idx['cell_line_idx'] = gene_protein_asso_idx['cell_line_idx'].apply(lambda x: cell2idx[x])
gene_protein_asso_idx['protein_idx'] = gene_protein_asso_idx['protein_idx'].apply(lambda x: protein2idx[x])
gene_protein_asso_idx.to_csv(os.path.join(proc_dir_study, 'cell_line_protein_asso_up100dn100_idx.tsv'), sep='\t', index=False)

# %% [markdown]
# ### 生成蛋白质特征

# %%
from data_utils import calculate_aa_composition, calculate_dipeptide_composition

# %%
protein2idx = pd.read_csv(os.path.join(proc_dir_study, "protein2idx.tsv"), sep='\t', index_col=['protein'])
protein2idx = protein2idx['idx'].to_dict()
protein_features_aac = np.zeros((len(protein2idx) + 1, 20))
protein_features_dpc = np.zeros((len(protein2idx) + 1, 400))

# %%
for p, i in protein2idx.items():
    protein_features_aac[i] = calculate_aa_composition(protein2seq[p])
    protein_features_dpc[i] = calculate_dipeptide_composition(protein2seq[p])
protein_feat = np.concatenate([protein_features_aac, protein_features_dpc], axis=1)
print(protein_feat.shape)
np.save(os.path.join(proc_dir_study, "protein_feat.npy"), protein_feat)

# %% [markdown]
# # V1
# 
# 主要是模型改动，没引入新的数据

# %%
VERSION = 'v1'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## Oneil数据集

# %%
STUDY = 'oneil'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %%
proc_dir_study_prev = os.path.join('proc', 'v0', STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# # V2

# %%
VERSION = 'v2'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## 预训练数据

# %%
STUDY = 'pretrain_protein'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### Uniprot数据

# %%
DATASET = 'uniprot'
raw_data_dir = os.path.join('raw', DATASET)

# %%
def split_sequences(aa_seq, n_gram):
    tokens = []
    for i in range(0, len(aa_seq), n_gram):
        tokens.append(aa_seq[i:i+n_gram].upper())
    return tokens

# %%
seq_data_fp = os.path.join(raw_data_dir, 'uniprotkb_9606_230722.tsv')

seq_data = pd.read_csv(seq_data_fp, sep='\t', usecols=['Entry', 'Length', 'Sequence'])
seq_data['Length'].describe([0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99])

# %%
# raw sequences
with open(os.path.join(proc_dir_study, 'aa_sequence.txt'), 'w') as f:
    for sentence in seq_data['Sequence'].tolist():
        f.write(sentence + '\n')

# %%
# split into train / valid / test
# total 200k
# train : valid : test ~= 8 : 1 : 1
n_train = int(seq_data.shape[0] * 0.9)
n_valid = int(seq_data.shape[0] * 0.05)
random.seed(42)
for N_GRAM in [3]:
    seq_data['sentence'] = seq_data['Sequence'].apply(lambda x: split_sequences(x, N_GRAM))
    sentences = seq_data['sentence'].tolist()
    random.shuffle(sentences)
    # train
    n = 0
    with open(os.path.join(proc_dir_study, f'aa_sequence_pieces_{N_GRAM}_train.txt'), 'w') as f:
        for sentence in sentences[:n_train]:
            f.write(' '.join(sentence) + '\n')
            n += 1
    print(f"n train: {n}")
    # valid
    n = 0
    with open(os.path.join(proc_dir_study, f'aa_sequence_pieces_{N_GRAM}_valid.txt'), 'w') as f:
        for sentence in sentences[n_train:n_train+n_valid]:
            f.write(' '.join(sentence) + '\n')
            n += 1
    print(f"n valid: {n}")
    # test
    n = 0
    with open(os.path.join(proc_dir_study, f'aa_sequence_pieces_{N_GRAM}_test.txt'), 'w') as f:
        for sentence in sentences[n_train+n_valid:]:
            f.write(' '.join(sentence) + '\n')
            n += 1
    print(f"n test: {n}")

# %% [markdown]
# ## Oneil 数据集

# %%
STUDY = 'oneil'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %%
# 除去蛋白质特征需要重新生成 其他都不变
proc_dir_study_prev = os.path.join('proc', 'v1', STUDY)
for fn in os.listdir(proc_dir_study_prev):
    if fn == 'protein_feat.npy':
        continue
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# ### 使用训练好的Protein编码器获得蛋白质编码

# %%
from my_config import BaseConfig
from models.models import BertWithoutSegEmb
from models.utils import convert_to_bert_config
from train_tokenizer import get_tokenizer
from train_encoder_simcse import get_default_config as get_prot_enc_default_cfg

# %%
# BEST_PROTEIN_ENCODER_DIR = '../output/v2/pretrain_protein/aa_encoder_lyr3_lr0.0002_hdsz256_ir0.03'
# BEST_CKPT = 'model_57000_6.2227.pt'
BEST_PROTEIN_ENCODER_DIR = '../output/v2/pretrain_protein/aa_encoder_simcse_lyr3_lr0.00003_hdsz256_ir0.03'
BEST_CKPT = 'model_16000_0.0187.pt'
DEVICE = 'cuda:7'

# %%
protein_encoder_cfg = get_prot_enc_default_cfg(os.path.join(BEST_PROTEIN_ENCODER_DIR, 'configs.yml'))
protein_encoder_cfg.model_dir = os.path.join('..', protein_encoder_cfg.model_dir)
protein_encoder_cfg.tokenizer.model_dir = os.path.join('..', protein_encoder_cfg.tokenizer.model_dir)

# load tokenizer
tokenizer, tokenizer_cfg = get_tokenizer(protein_encoder_cfg.tokenizer, False)
# create model
model_cfg = protein_encoder_cfg.model
if model_cfg.vocab_size is None:
    model_cfg.vocab_size = tokenizer_cfg['vocab_size']
model_cfg = convert_to_bert_config(model_cfg)
model = BertWithoutSegEmb(model_cfg, model_cfg.add_pooler)
if torch.cuda.is_available():
    model.to(DEVICE)
# load pre-trained weights
ckpt = torch.load(os.path.join(BEST_PROTEIN_ENCODER_DIR, BEST_CKPT), map_location=DEVICE)
renamed_model_state_dict = OrderedDict()
for k, v in ckpt['model_state_dict'].items():
    if k.startswith('bert.'):
        k = k[5:]
        renamed_model_state_dict[k] = v
model.load_state_dict(renamed_model_state_dict)


# %%
seq_data_fp = os.path.join('raw', 'uniprot', 'uniprotkb_9606_230722.tsv')
protein2seq = pd.read_csv(seq_data_fp, sep='\t', usecols=['Entry', 'Sequence'], index_col='Entry')['Sequence'].to_dict()
idx2protein = pd.read_csv(os.path.join(proc_dir_study, 'protein2idx.tsv'), sep='\t', index_col=['idx'])['protein'].to_dict()
idx2seq = []
for idx, protein in idx2protein.items():
    seq = protein2seq[protein]
    idx2seq.append((idx, seq))

# %%
def gen_batch(sequence, batch_size):
    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i+batch_size]

def prepare_batch_input(pad_token_id, batch_enc_res):
    max_len = -1
    for enc_res in batch_enc_res:
        max_len = max(max_len, len(enc_res.ids))
    batch_input_ids = []
    batch_attn_mask = []
    for enc_res in batch_enc_res:
        masked_input_ids = enc_res.ids[:max_len]
        attn_mask = enc_res.attention_mask[:max_len]
        if len(masked_input_ids) < max_len:
            n_pad = (max_len - len(masked_input_ids))
            masked_input_ids += [pad_token_id] * n_pad
            attn_mask += [0] * n_pad
        batch_input_ids.append(masked_input_ids)
        batch_attn_mask.append(attn_mask)
    ret_dict = {
        "input_ids": torch.LongTensor(batch_input_ids),
        "attention_mask": torch.LongTensor(batch_attn_mask)
    }
    return ret_dict

tokenizer.enable_truncation(512)
pad_token_id = tokenizer_cfg['special_tokens']['PAD_TOKEN']['id']
protein_feat = np.zeros((max(idx2protein.keys()) + 1, model_cfg.hidden_size), dtype=float)
with torch.no_grad():
    model.eval()
    for batch in gen_batch(idx2seq, batch_size=256):
        indices = [x[0] for x in batch]
        sequences = [x[1] for x in batch]
        batch_enc_res = tokenizer.encode_batch(sequences)
        input_dict = prepare_batch_input(pad_token_id, batch_enc_res)
        input_dict = {k: v.to(DEVICE) for k, v in input_dict.items()}
        # pool out via mean
        last_hidden_state = model(**input_dict)[0]  # batch * seq_len * hidden size
        attn_mask = input_dict['attention_mask'].unsqueeze(-1) # batch * seq_len * 1
        num_valid_token = input_dict['attention_mask'].sum(1, keepdim=True)
        pooled_feat = torch.sum(last_hidden_state * attn_mask, dim=1) / num_valid_token
        pooled_feat = pooled_feat.detach().cpu().float().numpy()
        for row, p_idx in enumerate(indices):
            protein_feat[p_idx, :] = pooled_feat[row]

# %%
print(protein_feat.shape)
np.save(os.path.join(proc_dir_study, "protein_feat_simcse.npy"), protein_feat)

# %% [markdown]
# # V3

# %%
VERSION = 'v3'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## 预训练数据-蛋白质

# %%
STUDY = 'pretrain_protein'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### Uniprot数据

# %%
proc_dir_study_prev = os.path.join('proc', get_prev_version(), STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# ## 预训练数据-药物

# %%
STUDY = 'pretrain_drug'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### ChEMBL

# %%
DATASET = 'chembl'
raw_data_dir = os.path.join('raw', 'drug', DATASET)

# %%
chembl_smiles = pd.read_csv(os.path.join(raw_data_dir, 'small_mol_230720.tsv'), usecols=['#RO5 Violations', 'Smiles'], sep='\t')
chembl_smiles = chembl_smiles.dropna()
chembl_smiles = chembl_smiles[chembl_smiles['#RO5 Violations'] != 'None']
chembl_smiles = chembl_smiles.drop_duplicates()
chembl_smiles['#RO5 Violations'] = chembl_smiles['#RO5 Violations'].apply(int)
chembl_smiles = chembl_smiles[chembl_smiles['#RO5 Violations'] < 3]  # refer to https://dev.drugbank.com/guides/terms/lipinski-s-rule-of-five
chembl_smiles = chembl_smiles.drop(columns='#RO5 Violations')
chembl_smiles.shape

# %%
chembl_smiles['length'] = chembl_smiles['Smiles'].apply(lambda x: len(x))
chembl_smiles['length'].describe([0.1, 0.2, 0.25, 0.5, 0.75, 0.9])

# %%
chembl_smiles = chembl_smiles[chembl_smiles['length'] >= 30]['Smiles'].tolist()

# %%
# raw sequences
with open(os.path.join(proc_dir_study, 'smiles.txt'), 'w') as f:
    for sentence in chembl_smiles:
        f.write(sentence + '\n')

# %%
def split_smiles(smiles):
    smiles = smiles.replace('(', ' ( ')
    smiles = smiles.replace(')', ' ) ')
    smiles = smiles.replace('[', ' [ ')
    smiles = smiles.replace(']', ' ] ')
    return smiles.strip()

split_smiles('O=C1[C@H](Cc2c[nH]c3ccccc23)CN(Cc2ccccc2)C[C@@H]1Cc1c[nH]c2ccccc12')

# %%
# split into train / valid / test
# total 2M
# train : valid : test ~= 8 : 1 : 1
n_train = int(len(chembl_smiles) * 0.9)
n_valid = int(len(chembl_smiles) * 0.05)
random.seed(42)
random.shuffle(chembl_smiles)
# train
n = 0
with open(os.path.join(proc_dir_study, f'smiles_train.txt'), 'w') as f:
    for sentence in chembl_smiles[:n_train]:
        f.write(split_smiles(sentence) + '\n')
        n += 1
print(f"n train: {n}")
# valid
n = 0
with open(os.path.join(proc_dir_study, f'smiles_valid.txt'), 'w') as f:
    for sentence in chembl_smiles[n_train:n_train+n_valid]:
        f.write(split_smiles(sentence) + '\n')
        n += 1
print(f"n valid: {n}")
# test
n = 0
with open(os.path.join(proc_dir_study, f'smiles_test.txt'), 'w') as f:
    for sentence in chembl_smiles[n_train+n_valid:]:
        f.write(split_smiles(sentence) + '\n')
        n += 1
print(f"n test: {n}")

# %% [markdown]
# ## Oneil 数据集

# %%
STUDY = 'oneil'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %%
# 除去药物特征需要重新生成 其他都不变
proc_dir_study_prev = os.path.join('proc', get_prev_version(), STUDY)
for fn in os.listdir(proc_dir_study_prev):
    if fn == 'drug_feat_cgms.npy':
        continue
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# ### 使用训练好的Smiles编码器获得药物编码

# %%
from my_config import BaseConfig
from models.models import BertWithoutSegEmb
from models.utils import convert_to_bert_config
from train_tokenizer import get_tokenizer
from train_encoder_simcse import get_default_config as get_drug_enc_default_cfg

# %%
# BEST_SMILES_ENCODER_DIR = '../output/v3/pretrain_drug/smiles_encoder_lyr3_lr0.0002_hdsz256_ir0.03'
# BEST_CKPT = 'model_121000_0.5660.pt'
BEST_SMILES_ENCODER_DIR = '../output/v3/pretrain_drug/smiles_encoder_simcse_lyr3_lr0.00003_hdsz256_ir0.03'
BEST_CKPT = 'model_24000_0.6788.pt'
DEVICE = 'cuda:7'

# %%
smiles_encoder_cfg = get_drug_enc_default_cfg(os.path.join(BEST_SMILES_ENCODER_DIR, 'configs.yml'))
smiles_encoder_cfg.model_dir = os.path.join('..', smiles_encoder_cfg.model_dir)
smiles_encoder_cfg.tokenizer.model_dir = os.path.join('..', smiles_encoder_cfg.tokenizer.model_dir)

# load tokenizer
tokenizer, tokenizer_cfg = get_tokenizer(smiles_encoder_cfg.tokenizer, False)

# %%
sequence = "FC1=CNC(=O)NC1=O"
enc_res = tokenizer.encode(sequence)
print(enc_res)
print("ids:", enc_res.ids)
print("tokens:", enc_res.tokens)
print("attention_mask:", enc_res.attention_mask)
print("special_tokens_mask:", enc_res.special_tokens_mask)
print("overflowing:", enc_res.overflowing)

# %%
smiles_encoder_cfg = get_drug_enc_default_cfg(os.path.join(BEST_SMILES_ENCODER_DIR, 'configs.yml'))
smiles_encoder_cfg.model_dir = os.path.join('..', smiles_encoder_cfg.model_dir)
smiles_encoder_cfg.tokenizer.model_dir = os.path.join('..', smiles_encoder_cfg.tokenizer.model_dir)

# load tokenizer
tokenizer, tokenizer_cfg = get_tokenizer(smiles_encoder_cfg.tokenizer, False)
# create model
model_cfg = smiles_encoder_cfg.model
if model_cfg.vocab_size is None:
    model_cfg.vocab_size = tokenizer_cfg['vocab_size']
model_cfg = convert_to_bert_config(model_cfg)
model = BertWithoutSegEmb(model_cfg, model_cfg.add_pooler)
if torch.cuda.is_available():
    model.to(DEVICE)
# load pre-trained weights
ckpt = torch.load(os.path.join(BEST_SMILES_ENCODER_DIR, BEST_CKPT), map_location=DEVICE)
renamed_model_state_dict = OrderedDict()
for k, v in ckpt['model_state_dict'].items():
    if k.startswith('bert.'):
        k = k[5:]
        renamed_model_state_dict[k] = v
model.load_state_dict(renamed_model_state_dict)


# %%
map_fn = 'drug_name_map_oneil2dc.tsv'
if not os.path.exists(os.path.join(proc_dir_study, map_fn)):
    command = f"ln {os.path.join('proc', 'v00', 'oneil', map_fn)} {os.path.join(proc_dir_study, map_fn)}"
    cmd_res = os.system(command)

drug_dc2o = {}
with open(os.path.join(proc_dir_study, map_fn), 'r') as f:
    next(f)
    for line in f:
        o_name, dc_name = line.strip().split('\t')
        drug_dc2o[dc_name] = o_name

# %%
from rdkit import Chem

seq_data_fp = os.path.join('raw', 'oneil', 'smiles.csv')
drug2seq = {}
with open(seq_data_fp, 'r') as f:
    for line in f:
        drug, seq = line.strip().split(',')
        drug2seq[drug] = Chem.CanonSmiles(seq)

idx2drug = pd.read_csv(os.path.join(proc_dir_study, 'drug2idx.tsv'), sep='\t', index_col=['idx'])['drug'].to_dict()
idx2seq = []
for idx, drug in idx2drug.items():
    drug = drug_dc2o[drug]
    seq = drug2seq[drug]
    idx2seq.append((idx, seq))

# %%
def gen_batch(sequence, batch_size):
    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i+batch_size]

def prepare_batch_input(pad_token_id, batch_enc_res):
    max_len = -1
    for enc_res in batch_enc_res:
        max_len = max(max_len, len(enc_res.ids))
    batch_input_ids = []
    batch_attn_mask = []
    for enc_res in batch_enc_res:
        masked_input_ids = enc_res.ids[:max_len]
        attn_mask = enc_res.attention_mask[:max_len]
        if len(masked_input_ids) < max_len:
            n_pad = (max_len - len(masked_input_ids))
            masked_input_ids += [pad_token_id] * n_pad
            attn_mask += [0] * n_pad
        batch_input_ids.append(masked_input_ids)
        batch_attn_mask.append(attn_mask)
    ret_dict = {
        "input_ids": torch.LongTensor(batch_input_ids),
        "attention_mask": torch.LongTensor(batch_attn_mask)
    }
    return ret_dict

tokenizer.enable_truncation(256)
pad_token_id = tokenizer_cfg['special_tokens']['PAD_TOKEN']['id']
drug_feat = np.zeros((max(idx2drug.keys()) + 1, model_cfg.hidden_size), dtype=float)
with torch.no_grad():
    model.eval()
    for batch in gen_batch(idx2seq, batch_size=256):
        indices = [x[0] for x in batch]
        sequences = [x[1] for x in batch]
        batch_enc_res = tokenizer.encode_batch(sequences)
        input_dict = prepare_batch_input(pad_token_id, batch_enc_res)
        input_dict = {k: v.to(DEVICE) for k, v in input_dict.items()}
        # pool out via mean
        last_hidden_state = model(**input_dict)[0]  # batch * seq_len * hidden size
        attn_mask = input_dict['attention_mask'].unsqueeze(-1) # batch * seq_len * 1
        num_valid_token = input_dict['attention_mask'].sum(1, keepdim=True)
        pooled_feat = torch.sum(last_hidden_state * attn_mask, dim=1) / num_valid_token
        pooled_feat = pooled_feat.detach().cpu().float().numpy()
        for row, p_idx in enumerate(indices):
            drug_feat[p_idx, :] = pooled_feat[row]

# %%
print(drug_feat.shape)
np.save(os.path.join(proc_dir_study, "drug_feat_simcse.npy"), drug_feat)

# %% [markdown]
# # V4

# %%
VERSION = 'v4'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## Macro网络

# %%
STUDY = 'pretrain_macro'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### DrugBank-DDI

# %%
DATASET = 'drug_bank'
raw_data_dir = os.path.join('raw', DATASET)

# %%
import xml.etree.ElementTree as ET

def parse_and_remove(filename, path, namespace):
    path_parts = [f"{{{namespace}}}{x}" for x in path.split('/')]
    doc = ET.iterparse(filename, ('start', 'end'))
    # Skip the root element
    next(doc)

    elem_stack = []
    for event, elem in doc:
        if event == 'start' and len(elem_stack) < len(path_parts) and elem.tag == path_parts[len(elem_stack)]:
            elem_stack.append(elem)
        elif event == 'end' and len(elem_stack) == len(path_parts) and id(elem) == id(elem_stack[-1]):
            yield elem
            elem_stack.pop()

# %%
DRUG_BANK_NS = "http://www.drugbank.ca"
def get_drugs_iter(drug_bank_full_db_fp):
    drugs = parse_and_remove(drug_bank_full_db_fp, 'drug', DRUG_BANK_NS)
    for drug in drugs:
        # get only small mol drug
        if drug.get('type', None) == "small molecule":
            yield drug

def add_ns(path):
    return '/'.join([f'{{{DRUG_BANK_NS}}}{x}' for x in path.split('/')])

def get_dbid(drug):
    for dbid_elem in drug.findall(add_ns("drugbank-id")):
        if dbid_elem.get("primary", None) == "true":
            return dbid_elem.text

def get_drug_groups(drug):
    groups = []
    for group in drug.findall(add_ns("groups/group")):
        groups.append(group.text)
    return groups

def get_drug_inters(drug):
    groups = []
    for group in drug.findall(add_ns("drug-interactions/drug-interaction/drugbank-id")):
        groups.append(group.text)
    return groups

def get_drug_smiles(drug):
    for prop in drug.findall(add_ns("calculated-properties/property")):
        if prop.find(add_ns("kind")).text == 'SMILES':
            return prop.find(add_ns("value")).text

drug_bank_full_db_fp = os.path.join(raw_data_dir, 'full_database.xml')

# %%
def is_invalid_drug(drug_groups):
    invalid_groups = ('illicit',) # 'nutraceutical', 'withdrawn', 'vet_approved')
    for g in drug_groups:
        if g in invalid_groups:
            return True
    return False

def is_valid_drug(drug_groups):
    valid_groups = ('approved', 'experimental', 'investigational')
    if 'approved' not in valid_groups:
        return False
    for g in drug_groups:
        if g not in valid_groups:
            return False
    return True

# %%
with open('proc/v4/pretrain_macro/valid_drugs_supplement_approved.tsv', 'r') as f:
    lines = f.readlines()
with open('proc/v4/pretrain_macro/valid_drugs_supplement_approved.tsv', 'a') as f:
    f.write('\n\n')
    f.write(lines[0])
    for line in lines[1:]:
        if 'approved' in line:
            f.write(line)



# %%
# find valid drugs
valid_drugs_fp = os.path.join(proc_dir_study, 'valid_drugs.tsv')
valid_drugs_supplement_fp = os.path.join(proc_dir_study, 'valid_drugs_supplement_approved.tsv')
valid_drugs_all_fp = os.path.join(proc_dir_study, 'valid_drugs_all.tsv')
drug2smiles = dict()
if not os.path.exists(valid_drugs_fp):
    drug2group = dict()
    drug2group_missing = dict()
    for drug in get_drugs_iter(drug_bank_full_db_fp):
        dbid = get_dbid(drug)
        if dbid is None:
            continue
        drug_groups = get_drug_groups(drug)
        is_invalid = False
        if is_invalid_drug(drug_groups):
            continue
        
        smi = get_drug_smiles(drug)
        if smi is None:
            if is_valid_drug(drug_groups):
                drug2group_missing[dbid] =  ','.join(drug_groups)
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            if is_valid_drug(drug_groups):
                drug2group_missing[dbid] =  ','.join(drug_groups)
            continue
        smi = Chem.MolToSmiles(m)
        drug2smiles[dbid] = smi
        drug2group[dbid] = ','.join(drug_groups)
    with open(valid_drugs_fp, 'w') as f:
        f.write("drug_bank_id\tgroups\tsmiles\tsource\n")
        for dbid in sorted(drug2group.keys()):
            f.write(f"{dbid}\t{drug2group[dbid]}\t{drug2smiles[dbid]}\tDrugBank\n")
    if len(drug2group_missing) > 0:
        if not os.path.exists(valid_drugs_supplement_fp):
            with open(valid_drugs_supplement_fp, 'w') as f:
                f.write("drug_bank_id\tgroups\tsmiles\tsource\n")
                for dbid in sorted(drug2group_missing.keys()):
                    f.write(f"{dbid}\t{drug2group_missing[dbid]}\t\t\n")
valid_drugs = set()
drug2smiles = dict()

with open(valid_drugs_all_fp, 'w') as vda_fp:
    with open(valid_drugs_fp, 'r') as f:
        for line in f:
            vda_fp.write(line)
    if os.path.exists(valid_drugs_supplement_fp):
        with open(valid_drugs_supplement_fp, 'r') as f:
            next(f)
            for line in f:
                _, _, smi, _ = line.strip('\n').split('\t')
                if len(smi) > 0:
                    vda_fp.write(line)

with open(valid_drugs_all_fp, 'r') as f:
    next(f)
    for line in f:
        dbid, _, smi, _ = line.strip('\n').split('\t')
        valid_drugs.add(dbid)
        drug2smiles[dbid] = smi
print(len(valid_drugs))


# %%
ddi_data = defaultdict(list)
for drug in get_drugs_iter(drug_bank_full_db_fp):
    dbid = get_dbid(drug)
    if dbid is None:
        continue
    if dbid not in valid_drugs:
        continue
    for int_dbid in get_drug_inters(drug):
        if int_dbid in valid_drugs:
            ddi_data[dbid].append(int_dbid)

pairs = set()
for da, dbs in ddi_data.items():
    for db in dbs:
        pairs.add(tuple(sorted([da, db])))

with open(os.path.join(proc_dir_study, 'ddi.tsv'), 'w') as f:
    f.write("drug_bank_id_row\tdrug_bank_id_col\n")
    for da, db in pairs:
        f.write(f"{da}\t{db}\n")

# %%
ddi = pd.read_csv(os.path.join(proc_dir_study, 'ddi.tsv'), dtype=str, sep='\t')
n_train = int(ddi.shape[0] * 0.9)
n_valid = int(ddi.shape[0] * 0.05)
folds = [0] * n_train + [1] * n_valid + [2] * (ddi.shape[0] - n_train - n_valid)
random.seed(42)
random.shuffle(folds)
ddi['folds'] = folds
ddi.to_csv(os.path.join(proc_dir_study, 'ddi.tsv'), index=False, sep='\t')

# %% [markdown]
# ### CFX

# %% [markdown]
# #### DTI

# %%
DATASET = 'cfx'
raw_data_dir = os.path.join('raw', DATASET)

# %%
dti = pd.read_excel(os.path.join(raw_data_dir, 'exp_val_dti.xlsx'), usecols='A:B')
dti.columns = ['drug_bank_id', 'entrez_id']
dti = dti[dti['drug_bank_id'].isin(valid_drugs)]
n_train = int(dti.shape[0] * 0.9)
n_valid = int(dti.shape[0] * 0.05)
folds = [0] * n_train + [1] * n_valid + [2] * (dti.shape[0] - n_train - n_valid)
random.seed(42)
random.shuffle(folds)
dti['folds'] = folds
dti.to_csv(os.path.join(proc_dir_study, 'dti.tsv'), sep='\t', index=False)

# %% [markdown]
# #### PPI

# %%
ppi = pd.read_excel(os.path.join(raw_data_dir, 'human_ppi.xlsx'), usecols='A:B')
ppi.columns = ['entrez_id_row', 'entrez_id_col']
n_train = int(ppi.shape[0] * 0.9)
n_valid = int(ppi.shape[0] * 0.05)
folds = [0] * n_train + [1] * n_valid + [2] * (ppi.shape[0] - n_train - n_valid)
random.seed(42)
random.shuffle(folds)
ppi['folds'] = folds
ppi.to_csv(os.path.join(proc_dir_study, 'ppi.tsv'), sep='\t', index=False)

# %% [markdown]
# ### Macro Network

# %%
all_drugs = set()
all_proteins = set()
# ddi
ddi = pd.read_csv(os.path.join(proc_dir_study, 'ddi.tsv'), dtype=str, sep='\t')
all_drugs.update(ddi[ddi.columns[0]])
all_drugs.update(ddi[ddi.columns[1]])
print('-'*20 + 'ddi')
print(f"# drugs: {len(all_drugs)}")
print(f"# ddi: {ddi.shape[0]}")
# ppi
ppi = pd.read_csv(os.path.join(proc_dir_study, 'ppi.tsv'), dtype=str, sep='\t')
all_proteins.update(ppi[ppi.columns[0]])
all_proteins.update(ppi[ppi.columns[1]])
print('-'*20 + 'ppi')
print(f"# proteins: {len(all_proteins)}")
print(f"# ppi: {ppi.shape[0]}")
# dti
dti = pd.read_csv(os.path.join(proc_dir_study, 'dti.tsv'), dtype=str, sep='\t')
all_drugs.update(dti[dti.columns[0]])
all_proteins.update(dti[dti.columns[1]])
print('-'*20 + 'dti')
print(f"# dti: {dti.shape[0]}")
print(f"# drugs: {dti[dti.columns[0]].nunique()}")
print(f"# proteins: {dti[dti.columns[1]].nunique()}")
print('-'*20, 'total')
print(f"# drugs: {len(all_drugs)}")
print(f"# proteins: {len(all_proteins)}")

# %%
drug2idx = {}
with open(os.path.join(proc_dir_study, 'drug2idx.tsv'), 'w') as f:
    f.write('drug\tidx\n')
    for i, d in enumerate(sorted(all_drugs)):
        f.write(f"{d}\t{i}\n")
        drug2idx[d] = i

protein2idx = {}
with open(os.path.join(proc_dir_study, 'protein2idx.tsv'), 'w') as f:
    f.write('protein\tidx\n')
    for i, p in enumerate(sorted(all_proteins)):
        f.write(f"{p}\t{i}\n")
        protein2idx[p] = i

# %%
rel2idx = {}
with open(os.path.join(proc_dir_study, 'relation2idx.tsv'), 'w') as f:
    f.write('relation\tidx\n')
    rels = ['drug2drug', 'protein2protein', 'drug2protein', 'protein2drug']
    for i, r in enumerate(rels):
        f.write(f"{r}\t{i}\n")
        rel2idx[r] = i

# %% [markdown]
# #### Drug特征

# %%
from rdkit.Chem import Descriptors

# %%
dbids = sorted(drug2idx.keys(), key=lambda x: drug2idx[x])
drug_feats = np.zeros((len(drug2idx), len(Descriptors._descList)), dtype=float)
print(drug_feats.shape)
for dbid in dbids:
    idx = drug2idx[dbid]
    smi = drug2smiles[dbid]
    mol = Chem.MolFromSmiles(smi)
    descs = Descriptors.CalcMolDescriptors(mol, missingVal=0, silent=True)
    for j, (desc, _) in enumerate(Descriptors._descList):
        drug_feats[idx, j] = descs[desc]
drug_feats[drug_feats == np.inf] = 0 

# %%
descriptors = pd.DataFrame(
    data=drug_feats,
    index=dbids,
    columns=[d for d, _ in Descriptors._descList]
)
descriptors.index.name = 'drug_bank_id'
descriptors.to_csv(os.path.join(proc_dir_study, 'descriptors.tsv'), sep='\t')

# %%
from sklearn.feature_selection import VarianceThreshold

drug_vt = VarianceThreshold()

drug_feats_var = drug_vt.fit_transform(drug_feats)
print(drug_feats_var.shape)

joblib.dump(drug_vt, os.path.join(proc_dir_study, 'drug_vt.joblib'))

# %%
from sklearn.preprocessing import StandardScaler

drug_zscaler = StandardScaler()

drug_feats_var_std = drug_zscaler.fit_transform(drug_feats_var)

joblib.dump(drug_zscaler, os.path.join(proc_dir_study, 'drug_zscaler.joblib'))

# %%
np.save(os.path.join(proc_dir_study, 'drug_feat.npy'), drug_feats_var_std)

# %% [markdown]
# #### Protein特征

# %%
np.random.seed(0)
n_proteins = len(protein2idx)
n_dim = 256
lim = np.sqrt(6./(n_proteins+n_dim))
print(lim)
protein_feats = np.random.uniform(-lim, lim, (n_proteins, n_dim))
protein_feats.shape
np.save(os.path.join(proc_dir_study, 'protein_feat.npy'), protein_feats)

# %% [markdown]
# #### Graph

# %%
from models.datasets import MacroNetDataset

macro_net_graph = MacroNetDataset('macro_net_dataset', 'proc/v4/pretrain_macro', verbose=True)
graph = macro_net_graph[0]

# %%
# def get_edge_mask(graph, dataset):
#     edges = {}
#     srcs = {}
#     dsts = {}
#     for etype in graph.canonical_etypes:
#         mask = graph.edges[etype].data[f'{dataset}_mask']
#         ds_idx = mask.nonzero(as_tuple=False).squeeze().type(torch.int32)
#         edges[etype] = ds_idx
#         src, dst = graph.edges(etype=etype)
#         srcs[etype] = src[ds_idx]
#         dsts[etype] = dst[ds_idx]
#     return edges, srcs, dsts

# # train graph
# train_edges, _, _ = get_edge_mask(graph, 'train')
# train_graph = graph.edge_subgraph(train_edges, relabel_nodes=False)
# node_feats = train_graph.ndata['feature']
# del train_graph.ndata['feature']
# # valid & test
# _, valid_srcs, valid_dsts = get_edge_mask(graph, 'valid')
# _, test_srcs, test_dsts = get_edge_mask(graph, 'test')

# %%
ns = 'http://www.drugbank.ca'
drugbank2chembl = dict()
drugs = parse_and_remove('raw/drug_bank/full_database.xml', 'drug', ns)
for drug in drugs:
    dbid = None
    for dbid_elem in drug.findall(f'{{{ns}}}drugbank-id'):
        if dbid_elem.get("primary", None) == "true":
            dbid = dbid_elem.text
        break
    chembl_id = None
    for ext_link_elem in drug.findall(f'{{{ns}}}external-identifiers/{{{ns}}}external-identifier'):
        for child in ext_link_elem:
            if child.tag.endswith('identifier') and child.text.startswith("CHEMBL"):
                chembl_id = child.text
                break
    if dbid is not None and chembl_id is not None:
        drugbank2chembl[dbid] = chembl_id
print(len(drugbank2chembl))

# %% [markdown]
# # V5

# %%
VERSION = 'v5'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## Oneil 数据集

# %%
STUDY = 'oneil'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %%
# 沿用v3特征，重跑实验
proc_dir_study_prev = os.path.join('proc', 'v3', STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# # V6
# 
# 主要解决超参敏感问题

# %%
VERSION = 'v6'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## Oneil

# %%
STUDY = 'oneil'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %%
# 药物/蛋白质特征需要重新生成 其他都不变
proc_dir_study_prev = os.path.join('proc', 'v3', STUDY)
for fn in os.listdir(proc_dir_study_prev):
    if fn == 'drug_feat.npy' or fn == 'protein_feat.npy':
        continue
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %%
drug_feat = np.load(os.path.join(proc_dir_study_prev, 'drug_feat_simcse.npy'))
p2 = np.linalg.norm(drug_feat, ord=2, axis=1, keepdims=True)
drug_feat /= p2
np.save(os.path.join(proc_dir_study, 'drug_feat.npy'), drug_feat)

# %%
protein_feat = np.load(os.path.join(proc_dir_study_prev, 'protein_feat_simcse.npy'))
p2 = np.linalg.norm(protein_feat, ord=2, axis=1, keepdims=True)
p2[0][0] = 1
protein_feat /= p2
np.save(os.path.join(proc_dir_study, 'protein_feat.npy'), protein_feat)

# %%


# %%


# %%


# %% [markdown]
# ## 预训练数据-蛋白质预训练

# %%
STUDY = 'pretrain_protein'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### Uniprot数据

# %%
proc_dir_study_prev = os.path.join('proc', 'v2', STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# ## 预训练数据-药物

# %%
STUDY = 'pretrain_drug'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### ChEMBL

# %%
proc_dir_study_prev = os.path.join('proc', 'v3', STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# ## Oneil

# %%
# 沿用v5特征，
proc_dir_study_prev = os.path.join('proc', 'v3', STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %%
valid_drugs = set(pd.read_csv('proc/v7/pretrain_macro/valid_drugs_all.tsv', sep='\t', usecols=['drug_bank_id'])['drug_bank_id'])
len(valid_drugs)

# %%
dbid2drug_o = pd.read_csv('proc/v4/oneil/drug_name_map_dc2drugbank.tsv', sep='\t', index_col='drug_bank_id')['dc_name'].to_dict()
for k, v in dbid2drug_o.items():
    if k not in all_drugs:
        print(k, v)

# %%
dbid2drug_n = pd.read_csv('proc/v4/almanac/drug_name_map_nci2drugbank.tsv', sep='\t', index_col='drug_bank_id')['nci_name2'].to_dict()
for k, v in dbid2drug_n.items():
    if k not in all_drugs:
        print(k, v)

# %% [markdown]
# # V7

# %%
VERSION = 'v7'
proc_dir_ver = create_dir('proc', VERSION)

# %% [markdown]
# ## 预训练数据-蛋白质

# %%
STUDY = 'pretrain_protein'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### Uniprot数据

# %%
proc_dir_study_prev = os.path.join('proc', get_prev_version(), STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# ## 预训练数据-药物

# %%
STUDY = 'pretrain_drug'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# #### ChEMBL

# %%
proc_dir_study_prev = os.path.join('proc', get_prev_version(), STUDY)
for fn in os.listdir(proc_dir_study_prev):
    command = f"ln {os.path.join(proc_dir_study_prev, fn)} {os.path.join(proc_dir_study, fn)}"
    cmd_res = os.system(command)

# %% [markdown]
# ## Macro网络

# %%
STUDY = 'pretrain_macro'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### DrugBank-DDI

# %%
DATASET = 'drug_bank'
raw_data_dir = os.path.join('raw', DATASET)

# %%
import xml.etree.ElementTree as ET

def parse_and_remove(filename, path, namespace):
    path_parts = [f"{{{namespace}}}{x}" for x in path.split('/')]
    doc = ET.iterparse(filename, ('start', 'end'))
    # Skip the root element
    next(doc)

    elem_stack = []
    for event, elem in doc:
        if event == 'start' and len(elem_stack) < len(path_parts) and elem.tag == path_parts[len(elem_stack)]:
            elem_stack.append(elem)
        elif event == 'end' and len(elem_stack) == len(path_parts) and id(elem) == id(elem_stack[-1]):
            yield elem
            elem_stack.pop()

# %%
DRUG_BANK_NS = "http://www.drugbank.ca"
def get_drugs_iter(drug_bank_full_db_fp):
    drugs = parse_and_remove(drug_bank_full_db_fp, 'drug', DRUG_BANK_NS)
    for drug in drugs:
        # get only small mol drug
        if drug.get('type', None) == "small molecule":
            yield drug

def add_ns(path):
    return '/'.join([f'{{{DRUG_BANK_NS}}}{x}' for x in path.split('/')])

def get_dbid(drug):
    for dbid_elem in drug.findall(add_ns("drugbank-id")):
        if dbid_elem.get("primary", None) == "true":
            return dbid_elem.text

def get_drug_groups(drug):
    groups = []
    for group in drug.findall(add_ns("groups/group")):
        groups.append(group.text)
    return groups

def get_drug_inters(drug):
    groups = []
    for group in drug.findall(add_ns("drug-interactions/drug-interaction/drugbank-id")):
        groups.append(group.text)
    return groups

def get_drug_smiles(drug):
    for prop in drug.findall(add_ns("calculated-properties/property")):
        if prop.find(add_ns("kind")).text == 'SMILES':
            return prop.find(add_ns("value")).text

drug_bank_full_db_fp = os.path.join(raw_data_dir, 'full_database.xml')

# %%
def is_invalid_drug(drug_groups):
    invalid_groups = ('illicit',) # 'nutraceutical', 'withdrawn', 'vet_approved')
    for g in drug_groups:
        if g in invalid_groups:
            return True
    return False

def is_valid_drug(drug_groups):
    valid_groups = ('approved', 'experimental', 'investigational')
    if 'approved' not in valid_groups:
        return False
    for g in drug_groups:
        if g not in valid_groups:
            return False
    return True

# %%
valid_drugs_supplement_fp = os.path.join(proc_dir_study, 'valid_drugs_supplement_approved.tsv')
command = f"ln {'proc/v4/pretrain_macro/valid_drugs_supplement_approved.tsv'} {valid_drugs_supplement_fp}"
cmd_res = os.system(command)

# %%
# find valid drugs
valid_drugs_fp = os.path.join(proc_dir_study, 'valid_drugs.tsv')
valid_drugs_supplement_fp = os.path.join(proc_dir_study, 'valid_drugs_supplement_approved.tsv')
valid_drugs_all_fp = os.path.join(proc_dir_study, 'valid_drugs_all.tsv')
drug2smiles = dict()
if not os.path.exists(valid_drugs_fp):
    drug2group = dict()
    drug2group_missing = dict()
    for drug in get_drugs_iter(drug_bank_full_db_fp):
        dbid = get_dbid(drug)
        if dbid is None:
            continue
        drug_groups = get_drug_groups(drug)
        is_invalid = False
        if is_invalid_drug(drug_groups):
            continue
        
        smi = get_drug_smiles(drug)
        if smi is None:
            if is_valid_drug(drug_groups):
                drug2group_missing[dbid] =  ','.join(drug_groups)
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            if is_valid_drug(drug_groups):
                drug2group_missing[dbid] =  ','.join(drug_groups)
            continue
        smi = Chem.MolToSmiles(m)
        drug2smiles[dbid] = smi
        drug2group[dbid] = ','.join(drug_groups)
    with open(valid_drugs_fp, 'w') as f:
        f.write("drug_bank_id\tgroups\tsmiles\tsource\n")
        for dbid in sorted(drug2group.keys()):
            f.write(f"{dbid}\t{drug2group[dbid]}\t{drug2smiles[dbid]}\tDrugBank\n")
    if len(drug2group_missing) > 0:
        if not os.path.exists(valid_drugs_supplement_fp):
            with open(valid_drugs_supplement_fp, 'w') as f:
                f.write("drug_bank_id\tgroups\tsmiles\tsource\n")
                for dbid in sorted(drug2group_missing.keys()):
                    f.write(f"{dbid}\t{drug2group_missing[dbid]}\t\t\n")
valid_drugs = set()
drug2smiles = dict()

with open(valid_drugs_all_fp, 'w') as vda_fp:
    with open(valid_drugs_fp, 'r') as f:
        for line in f:
            vda_fp.write(line)
    if os.path.exists(valid_drugs_supplement_fp):
        with open(valid_drugs_supplement_fp, 'r') as f:
            next(f)
            for line in f:
                _, _, smi, _ = line.strip('\n').split('\t')
                if len(smi) > 0:
                    vda_fp.write(line)

with open(valid_drugs_all_fp, 'r') as f:
    next(f)
    for line in f:
        dbid, _, smi, _ = line.strip('\n').split('\t')
        valid_drugs.add(dbid)
        drug2smiles[dbid] = smi
print(len(valid_drugs))


# %%
ddi_data = defaultdict(list)
for drug in get_drugs_iter(drug_bank_full_db_fp):
    dbid = get_dbid(drug)
    if dbid is None:
        continue
    if dbid not in valid_drugs:
        continue
    for int_dbid in get_drug_inters(drug):
        if int_dbid in valid_drugs:
            ddi_data[dbid].append(int_dbid)

pairs = set()
for da, dbs in ddi_data.items():
    for db in dbs:
        pairs.add(tuple(sorted([da, db])))

with open(os.path.join(proc_dir_study, 'ddi.tsv'), 'w') as f:
    f.write("drug_bank_id_row\tdrug_bank_id_col\n")
    for da, db in pairs:
        f.write(f"{da}\t{db}\n")

# %%
ddi = pd.read_csv(os.path.join(proc_dir_study, 'ddi.tsv'), dtype=str, sep='\t')
n_train = int(ddi.shape[0] * 0.9)
n_valid = int(ddi.shape[0] * 0.05)
folds = [0] * n_train + [1] * n_valid + [2] * (ddi.shape[0] - n_train - n_valid)
random.seed(42)
random.shuffle(folds)
ddi['folds'] = folds
ddi.to_csv(os.path.join(proc_dir_study, 'ddi.tsv'), index=False, sep='\t')

# %% [markdown]
# ### CFX

# %% [markdown]
# #### DTI

# %%
DATASET = 'cfx'
raw_data_dir = os.path.join('raw', DATASET)

# %%
dti = pd.read_excel(os.path.join(raw_data_dir, 'exp_val_dti.xlsx'), usecols='A:B')
dti.columns = ['drug_bank_id', 'entrez_id']
dti = dti[dti['drug_bank_id'].isin(valid_drugs)]
n_train = int(dti.shape[0] * 0.9)
n_valid = int(dti.shape[0] * 0.05)
folds = [0] * n_train + [1] * n_valid + [2] * (dti.shape[0] - n_train - n_valid)
random.seed(42)
random.shuffle(folds)
dti['folds'] = folds
dti.to_csv(os.path.join(proc_dir_study, 'dti.tsv'), sep='\t', index=False)

# %% [markdown]
# #### PPI

# %%
ppi = pd.read_excel(os.path.join(raw_data_dir, 'human_ppi.xlsx'), usecols='A:B')
ppi.columns = ['entrez_id_row', 'entrez_id_col']
n_train = int(ppi.shape[0] * 0.9)
n_valid = int(ppi.shape[0] * 0.05)
folds = [0] * n_train + [1] * n_valid + [2] * (ppi.shape[0] - n_train - n_valid)
random.seed(42)
random.shuffle(folds)
ppi['folds'] = folds
ppi.to_csv(os.path.join(proc_dir_study, 'ppi.tsv'), sep='\t', index=False)

# %% [markdown]
# ### Macro Network

# %%
all_drugs = set()
all_proteins = set()
# ddi
ddi = pd.read_csv(os.path.join(proc_dir_study, 'ddi.tsv'), dtype=str, sep='\t')
all_drugs.update(ddi[ddi.columns[0]])
all_drugs.update(ddi[ddi.columns[1]])
print('-'*20 + 'ddi')
print(f"# drugs: {len(all_drugs)}")
print(f"# ddi: {ddi.shape[0]}")
# ppi
ppi = pd.read_csv(os.path.join(proc_dir_study, 'ppi.tsv'), dtype=str, sep='\t')
all_proteins.update(ppi[ppi.columns[0]])
all_proteins.update(ppi[ppi.columns[1]])
print('-'*20 + 'ppi')
print(f"# proteins: {len(all_proteins)}")
print(f"# ppi: {ppi.shape[0]}")
# dti
dti = pd.read_csv(os.path.join(proc_dir_study, 'dti.tsv'), dtype=str, sep='\t')
all_drugs.update(dti[dti.columns[0]])
all_proteins.update(dti[dti.columns[1]])
print('-'*20 + 'dti')
print(f"# dti: {dti.shape[0]}")
print(f"# drugs: {dti[dti.columns[0]].nunique()}")
print(f"# proteins: {dti[dti.columns[1]].nunique()}")
print('-'*20, 'total')
print(f"# drugs: {len(all_drugs)}")
print(f"# proteins: {len(all_proteins)}")

# %%
drug2idx = {}
with open(os.path.join(proc_dir_study, 'drug2idx.tsv'), 'w') as f:
    f.write('drug\tidx\n')
    for i, d in enumerate(sorted(all_drugs)):
        f.write(f"{d}\t{i}\n")
        drug2idx[d] = i

protein2idx = {}
with open(os.path.join(proc_dir_study, 'protein2idx.tsv'), 'w') as f:
    f.write('protein\tidx\n')
    for i, p in enumerate(sorted(all_proteins)):
        f.write(f"{p}\t{i}\n")
        protein2idx[p] = i

# %%
rel2idx = {}
with open(os.path.join(proc_dir_study, 'relation2idx.tsv'), 'w') as f:
    f.write('relation\tidx\n')
    rels = ['drug2drug', 'protein2protein', 'drug2protein', 'protein2drug']
    for i, r in enumerate(rels):
        f.write(f"{r}\t{i}\n")
        rel2idx[r] = i

# %% [markdown]
# #### Drug特征

# %%
from rdkit.Chem import Descriptors

# %%
dbids = sorted(drug2idx.keys(), key=lambda x: drug2idx[x])
drug_feats = np.zeros((len(drug2idx), len(Descriptors._descList)), dtype=float)
print(drug_feats.shape)
for dbid in dbids:
    idx = drug2idx[dbid]
    smi = drug2smiles[dbid]
    mol = Chem.MolFromSmiles(smi)
    descs = Descriptors.CalcMolDescriptors(mol, missingVal=0, silent=True)
    for j, (desc, _) in enumerate(Descriptors._descList):
        drug_feats[idx, j] = descs[desc]
drug_feats[drug_feats == np.inf] = 0 

# %%
descriptors = pd.DataFrame(
    data=drug_feats,
    index=dbids,
    columns=[d for d, _ in Descriptors._descList]
)
descriptors.index.name = 'drug_bank_id'
descriptors.to_csv(os.path.join(proc_dir_study, 'descriptors.tsv'), sep='\t')

# %%
from sklearn.feature_selection import VarianceThreshold

drug_vt = VarianceThreshold()

drug_feats_var = drug_vt.fit_transform(drug_feats)
print(drug_feats_var.shape)

joblib.dump(drug_vt, os.path.join(proc_dir_study, 'drug_vt.joblib'))

# %%
from sklearn.preprocessing import StandardScaler

drug_zscaler = StandardScaler()

drug_feats_var_std = drug_zscaler.fit_transform(drug_feats_var)

joblib.dump(drug_zscaler, os.path.join(proc_dir_study, 'drug_zscaler.joblib'))

# %%
np.save(os.path.join(proc_dir_study, 'drug_feat.npy'), drug_feats_var_std)

# %% [markdown]
# #### Protein特征

# %%
np.random.seed(0)
n_proteins = len(protein2idx)
n_dim = 256
lim = np.sqrt(6./(n_proteins+n_dim))
print(lim)
protein_feats = np.random.uniform(-lim, lim, (n_proteins, n_dim))
protein_feats.shape
np.save(os.path.join(proc_dir_study, 'protein_feat.npy'), protein_feats)

# %% [markdown]
# #### Graph

# %%
from models.datasets import MacroNetDataset

macro_net_graph = MacroNetDataset('macro_net_dataset', f'proc/{VERSION}/pretrain_macro', verbose=True)
graph = macro_net_graph[0]

# %%
# def get_edge_mask(graph, dataset):
#     edges = {}
#     srcs = {}
#     dsts = {}
#     for etype in graph.canonical_etypes:
#         mask = graph.edges[etype].data[f'{dataset}_mask']
#         ds_idx = mask.nonzero(as_tuple=False).squeeze().type(torch.int32)
#         edges[etype] = ds_idx
#         src, dst = graph.edges(etype=etype)
#         srcs[etype] = src[ds_idx]
#         dsts[etype] = dst[ds_idx]
#     return edges, srcs, dsts

# # train graph
# train_edges, _, _ = get_edge_mask(graph, 'train')
# train_graph = graph.edge_subgraph(train_edges, relabel_nodes=False)
# node_feats = train_graph.ndata['feature']
# del train_graph.ndata['feature']
# # valid & test
# _, valid_srcs, valid_dsts = get_edge_mask(graph, 'valid')
# _, test_srcs, test_dsts = get_edge_mask(graph, 'test')

# %% [markdown]
# ## NCI

# %%
STUDY = 'almanac'
proc_dir_study = create_dir(proc_dir_ver, STUDY)

# %% [markdown]
# ### 分折

# %% [markdown]
# #### 原始数据处理

# %%
raw_data_dir = os.path.join('raw', 'drug_comb', 'study')
synergy_file = os.path.join(raw_data_dir, f"{STUDY.upper()}.csv")
synergy_samples = pd.read_csv(
    synergy_file, 
    usecols=['drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe', 'synergy_zip', 'synergy_hsa', 'synergy_bliss'],
)
invalid_cell_lines = ['MDA-N', 'MDA-MB-435', 'U251', 'NCI/ADR-RES']
invalid_drugs = ['NSC707389']
synergy_samples = synergy_samples[~synergy_samples['cell_line_name'].isin(invalid_cell_lines)]
synergy_samples = synergy_samples[~synergy_samples['drug_row'].isin(invalid_drugs)]
synergy_samples = synergy_samples[~synergy_samples['drug_col'].isin(invalid_drugs)]
synergy_samples['comb_key'] = synergy_samples.apply(
    lambda row: tuple(sorted([row['drug_row'], row['drug_col']])) + (row['cell_line_name'],),
    axis=1
)
synergy_samples = synergy_samples.drop(columns=['drug_row','drug_col','cell_line_name'])
synergy_samples = synergy_samples.groupby('comb_key').mean().reset_index()
synergy_samples['drug_row'] = synergy_samples['comb_key'].apply(lambda x: x[0])
synergy_samples['drug_col'] = synergy_samples['comb_key'].apply(lambda x: x[1])
synergy_samples['cell_line_name'] = synergy_samples['comb_key'].apply(lambda x: x[2])
synergy_samples = synergy_samples.drop(columns=['comb_key'])
print(synergy_samples.shape)
synergy_samples.head()

# %%
all_drugs = set()
all_drugs.update(synergy_samples['drug_row'])
all_drugs.update(synergy_samples['drug_col'])
print(len(all_drugs))

# %%
combs = set()
for _, row in synergy_samples.iterrows():
    combs.add((row['drug_row'], row['drug_col']))
print(len(combs))

n_folds = 5
n_combs_per_fold = [len(combs) // 5] * 4
n_combs_per_fold.append(len(combs)-sum(n_combs_per_fold))
print(n_combs_per_fold)
combs = list(combs)
random.seed(42)
random.shuffle(combs)

combs2fold = {}
start = 0
for fd in range(5):
    end = start + n_combs_per_fold[fd]
    for c in combs[start:end]:
        combs2fold[c] = fd
    start = end
print(len(combs2fold))

# %%
synergy_samples['fold'] = synergy_samples.apply(
    lambda row: combs2fold[(row['drug_row'], row['drug_col'])], axis=1
)
synergy_samples = synergy_samples[['drug_row','drug_col','cell_line_name', 'fold', 'synergy_loewe', 'synergy_zip', 'synergy_hsa', 'synergy_bliss']]
synergy_samples.to_csv('proc/almanac/scores.tsv', sep='\t', index=False)

# %%
# print(synergy_samples.groupby('fold').count()['drug_row'])
# print(synergy_samples.groupby('fold').agg(lambda x: x.nunique())['cell_line_name'])
# fd2drugs = defaultdict(set)
# for (d1, d2), fd in combs2fold.items():
#     fd2drugs[fd].add(d1)
#     fd2drugs[fd].add(d2)
# for k, v in fd2drugs.items():
#     print(k, len(v))

# %% [markdown]
# #### 去除无法获取相关特征的药物/细胞系

# %%
valid_drugs = set()
drug2smiles = dict()
with open(os.path.join(proc_dir_ver, 'pretrain_macro', 'valid_drugs_all.tsv'), 'r') as f:
    next(f)
    for line in f:
        dbid, _, smi, _ = line.strip('\n').split('\t')
        valid_drugs.add(dbid)
        drug2smiles[dbid] = smi
print(len(valid_drugs))

# %%
synergy_samples = pd.read_csv('proc/almanac/scores.tsv', sep='\t')
print(synergy_samples.shape)
synergy_samples.head()

# %%
# dbid2drug_n = pd.read_csv(os.path.join(proc_dir_study, 'drug_name_map_almanac2drugbank.tsv'), sep='\t', index_col='drug_bank_id')['nci_name2'].to_dict()
# for k, v in dbid2drug_n.items():
#     if k not in valid_drugs:
#         print(k, v)
# None
drugs = set()
drugs.update(synergy_samples['drug_row'])
drugs.update(synergy_samples['drug_col'])
print(len(drugs))
with open(os.path.join(proc_dir_study, 'drug2idx.tsv'), 'w') as f:
    f.write('drug\tidx\n')
    for i, drug in enumerate(sorted(drugs)):
        f.write(f"{drug}\t{i}\n")

# %%
has_hmz_cells = []
with open(os.path.join(proc_dir_study, 'cell_line_name_map_ccle2dc.tsv'), 'r') as f:
    next(f)
    for line in f:
        ccle, dc = line.rstrip('\n').split('\t')
        if len(ccle) > 0:
            has_hmz_cells.append(dc)

synergy_samples = synergy_samples[synergy_samples['cell_line_name'].isin(has_hmz_cells)]
print(synergy_samples['cell_line_name'].nunique())
print(synergy_samples.shape)

cells = set()
cells.update(synergy_samples['cell_line_name'])
print(len(cells))
with open(os.path.join(proc_dir_study, 'cell_line2idx.tsv'), 'w') as f:
    f.write('cell_line\tidx\n')
    for i, cell in enumerate(sorted(cells)):
        f.write(f"{cell}\t{i}\n")

# %%
synergy_samples.to_csv(os.path.join(proc_dir_study, 'samples.tsv'), sep='\t', index=False)

# %%
synergy_samples = pd.read_csv(os.path.join(proc_dir_study, 'samples.tsv'), sep='\t')
drug2idx = pd.read_csv(os.path.join(proc_dir_study, 'drug2idx.tsv'), sep='\t',index_col='drug')['idx'].to_dict()
cell_line2idx = pd.read_csv(os.path.join(proc_dir_study, 'cell_line2idx.tsv'), sep='\t',index_col='cell_line')['idx'].to_dict()
synergy_samples['drug_row'] = synergy_samples['drug_row'].apply(lambda x: drug2idx[x])
synergy_samples['drug_col'] = synergy_samples['drug_col'].apply(lambda x: drug2idx[x])
synergy_samples['cell_line_name'] = synergy_samples['cell_line_name'].apply(lambda x: cell_line2idx[x])
synergy_samples_idx = synergy_samples.rename(columns={ 'drug_row': 'drug_row_idx', 'drug_col': 'drug_col_idx', 'cell_line_name': 'cell_line_idx'})
synergy_samples_idx.to_csv(os.path.join(proc_dir_study, 'samples_idx.tsv'), sep='\t', index=False)


