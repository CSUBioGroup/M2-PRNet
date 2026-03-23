
from collections import defaultdict
import lmdb
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import sys

import tqdm
sys.path.append("/public/home/qiang/jkwang/EquiScore-main")
from utils.image_process import ImageProcessor
import utils.utils as utils 
import numpy as np
import torch
import random
import math
from scipy.spatial import distance_matrix
import pickle
import dgl
import dgl.data
from utils.ifp_construct import get_nonBond_pair
from utils.dataset_utils import *
import pandas as pd

random.seed(42)
class PDBEpochSampler(Sampler):
    """每个epoch只随机选取每个PDB一个样本的采样器"""
    
    def __init__(self, keys, batch_size, shuffle=True, random_state=None, samples_per_pdb=1):
        """
        Args:
            keys: 所有样本的key列表，如 ['1A4T_0_RNA_A_1A4T_0_PROT_B', ...]
            batch_size: 批次大小
            shuffle: 是否打乱batch
            random_state: 随机种子
            samples_per_pdb: 每个PDB在每个epoch中采样的样本数（默认1）
        """
        self.keys = keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples_per_pdb = samples_per_pdb
        self.rng = np.random.RandomState(random_state)
        
        # 根据PDB ID分组
        self.pdb_to_indices = defaultdict(list)
        for idx, key in enumerate(keys):
            pdb_id = key[:4]
            self.pdb_to_indices[pdb_id].append(idx)
        
        self.unique_pdbs = list(self.pdb_to_indices.keys())
        print(f"共发现 {len(self.unique_pdbs)} 个不同的PDB复合物。")

    def __iter__(self):
        # 1️⃣ 每个epoch随机选取每个PDB的若干样本
        selected_indices = []
        for pdb_id, indices in self.pdb_to_indices.items():
            chosen = self.rng.choice(indices, 
                                     size=min(self.samples_per_pdb, len(indices)), 
                                     replace=False)
            selected_indices.extend(chosen)
        
        # 2️⃣ 打乱全体样本
        if self.shuffle:
            self.rng.shuffle(selected_indices)
        
        # 3️⃣ 按batch划分
        for i in range(0, len(selected_indices), self.batch_size):
            yield from selected_indices[i:i+self.batch_size]

    def __len__(self):
        # 每个epoch的样本总数 = unique_pdbs数量 * samples_per_pdb
        return len(self.unique_pdbs) * self.samples_per_pdb
class PDBBalancedSampler(Sampler):
    """基于PDB ID的平衡采样器，确保同一个复合物的样本均匀分布"""
    
    def __init__(self, keys, batch_size, shuffle=True, random_state=42):
        """
        Args:
            keys: 所有样本的key列表，如 ['1A4T_0_RNA_A_1A4T_0_PROT_B', ...]
            batch_size: 批次大小
            shuffle: 是否打乱
        """
        self.keys = keys
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        
        # 提取PDB ID (key的前4个字符)
        self.pdb_ids = [key[:4] for key in keys]
        
        # 按PDB ID分组
        self.pdb_to_indices = defaultdict(list)
        for idx, pdb_id in enumerate(self.pdb_ids):
            self.pdb_to_indices[pdb_id].append(idx)
        
        self.unique_pdbs = list(self.pdb_to_indices.keys())
        print(f"发现 {len(self.unique_pdbs)} 个不同的PDB复合物")
        
        # 统计信息
        for pdb_id in list(self.unique_pdbs)[:5]:  # 打印前5个
            print(f"  {pdb_id}: {len(self.pdb_to_indices[pdb_id])} 个样本")
    
    def __iter__(self):
        rng = np.random.RandomState(self.random_state)
        
        # 为每个PDB创建样本池
        pdb_pools = {}
        for pdb_id in self.unique_pdbs:
            pool = self.pdb_to_indices[pdb_id].copy()
            if self.shuffle:
                rng.shuffle(pool)
            pdb_pools[pdb_id] = {
                'pool': pool,
                'pointer': 0
            }
        
        batches = []
        max_attempts = len(self.keys) * 2  # 防止无限循环
        
        for attempt in range(max_attempts):
            if len(batches) * self.batch_size >= len(self.keys):
                break
                
            current_batch = []
            used_pdbs = set()
            
            # 尽量选择不同的PDB
            available_pdbs = [pdb_id for pdb_id in self.unique_pdbs 
                            if (len(pdb_pools[pdb_id]['pool']) > pdb_pools[pdb_id]['pointer'] and 
                                pdb_id not in used_pdbs)]
            
            if self.shuffle:
                rng.shuffle(available_pdbs)
            
            for pdb_id in available_pdbs:
                if len(current_batch) >= self.batch_size:
                    break
                    
                pool_info = pdb_pools[pdb_id]
                if pool_info['pointer'] < len(pool_info['pool']):
                    idx = pool_info['pool'][pool_info['pointer']]
                    current_batch.append(idx)
                    pool_info['pointer'] += 1
                    used_pdbs.add(pdb_id)
            
            if len(current_batch) > 0:
                batches.append(current_batch)
            else:
                # 没有更多样本了
                break
        
        # 如果还有剩余样本，分配到已有batch中
        remaining_indices = []
        for pdb_id in self.unique_pdbs:
            pool_info = pdb_pools[pdb_id]
            remaining = pool_info['pool'][pool_info['pointer']:]
            remaining_indices.extend(remaining)
        
        if remaining_indices and self.shuffle:
            rng.shuffle(remaining_indices)
        
        # 将剩余样本分配到batch中
        idx = 0
        for batch in batches:
            while len(batch) < self.batch_size and idx < len(remaining_indices):
                batch.append(remaining_indices[idx])
                idx += 1
        
        # 创建最后一个batch（如果有剩余）
        if idx < len(remaining_indices):
            batches.append(remaining_indices[idx:])
        
        # 最终打乱batch顺序
        if self.shuffle:
            rng.shuffle(batches)
        
        # 验证每个batch的PDB分布
        # self._validate_batches(batches)
        
        for batch in batches:
            yield from batch
    
    def _validate_batches(self, batches):
        """验证每个batch中PDB的分布"""
        print("验证batch中PDB分布...")
        for i, batch in enumerate(batches[:3]):  # 只检查前3个batch
            pdb_counts = defaultdict(int)
            for idx in batch:
                pdb_id = self.pdb_ids[idx]
                pdb_counts[pdb_id] += 1
            
            max_count = max(pdb_counts.values()) if pdb_counts else 0
            unique_pdbs = len(pdb_counts)
            
            print(f"Batch {i}: {len(batch)} 样本, {unique_pdbs} 个PDB, 最大重复: {max_count}")
    
    def __len__(self):
        return len(self.keys)
class DTISampler(Sampler):
    """"
    weight based sampler for DTI dataset
    """
    def __init__(self, weights, num_samples, replacement=True):

        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    def __iter__(self):

        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())
    def __len__(self):
        return self.num_samples

class ESDataset_mut(Dataset):
    """
    每个样本返回一对 (突变结构, 原始结构, ddg)
    """
    def __init__(self, keys, ground_true, graph, allgraph, res_level, args, data_dir,image_process,debug = False):
        super(ESDataset_mut, self).__init__()
        # print(keys)
        self.keys = keys
        self.affinity = ground_true
        self.data_dir = data_dir
        self.debug = debug
        self.args = args
        self.graphs = []
        self.error = []
        
        
       
        self.res_level = res_level
        if args.data_set == 'PNA_keys.csv' or args.data_set == 'PNA_keys_201.csv':
            self.max_pro_len = 1370
            self.max_rna_len = 159
        else:
            self.max_pro_len = 2000
            self.max_rna_len = 200
        self.txn = None
        self.env  = None
        # 图像处理器
        self.image_processor = image_process
        self.image_features_cache = {}  # 缓存图像特征
        self.graph = graph
        self.allgraph = allgraph

    def __len__(self):
        return len(self.keys) if not self.debug else min(len(self.keys), 30000)

    def init_txn(self):
        if self.txn is None:
            self.env = lmdb.open(self.args.lmdb_cache, map_size=int(1e11), max_dbs=1, readonly=True)
            self.graph_db = self.env.open_db('data'.encode())
            self.txn = self.env.begin(buffers=True, write=False)

    def pad_array(self, arr, target_shape, pad_value=0):
        pad_width = [(0, t - s) for s, t in zip(arr.shape, target_shape)]
        return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)

    def _load_single_structure(self, key):
        """
        给定key，返回该结构的所有图/特征/padding数据
        """
        g, full_g = self.graph[key][1], self.allgraph[key][0]
        g.ndata['coors'] = full_g.ndata['coors']

        res = self.res_level[key]
        pro_feat, rna_feat = res['pro_feats'], res['rna_feats']
        mol_indicator, chain_indicator = res['mol_indicator'], res['chain_indicator']
        pro_coords, rna_coords = res['pro_coords'], res['rna_coords']
        prot_emb, rna_emb = res['prot_emb'], res['rna_emb']
        prot_len, rna_len = res['prot_len'], res['rna_len']
        neighbor_matrix = res['neighbor_matrix']
        pro_coords_use = pro_coords[:self.max_pro_len]
        rna_coords_use = rna_coords[:self.max_rna_len]
        # 如果某一方为空，构建形状正确的空数组
        if pro_coords_use.size == 0:
            pro_coords_use = np.zeros((0, 3))
        if rna_coords_use.size == 0:
            rna_coords_use = np.zeros((0, 3))
        coords_all = np.vstack([pro_coords_use, rna_coords_use]) 
      # 计算欧氏距离矩阵并按照阈值生成二值邻接矩阵
        if coords_all.shape[0] == 0:
           neighbor_matrix = np.zeros((0, 0), dtype=np.int8)
        else:
           dm = distance_matrix(coords_all, coords_all)
           neighbor_matrix = (dm < float(self.args.threshold)).astype(np.int8)
        prot_whole_emb, rna_whole_emb = res['prot_whole_emb'], res['rna_whole_emb']

        max_total = self.max_pro_len + self.max_rna_len
        # print("Max total length:", len(pro_feat)+len(rna_feat))
        # print(len(neighbor_matrix))
        # print(key)
        # padding
        pro_feat = np.array(pro_feat)
        rna_feat = np.array(rna_feat)
        pro_feat_padded = self.pad_array(np.array(pro_feat), (self.max_pro_len, pro_feat.shape[1]), 0)
        rna_feat_padded = self.pad_array(np.array(rna_feat), (self.max_rna_len, rna_feat.shape[1]), 0)
        mol_indicator_padded = self.pad_array(np.array(mol_indicator), (max_total, 2), 0)
        chain_indicator_padded = self.pad_array(np.array(chain_indicator), (max_total, 6), 0)
        pro_coords_padded = self.pad_array(np.array(pro_coords), (self.max_pro_len, 3), 0)
        rna_coords_padded = self.pad_array(np.array(rna_coords), (self.max_rna_len, 3), 0)
        neighbor_matrix_padded = self.pad_array(np.array(neighbor_matrix), (max_total, max_total), 0)

        emb_dim = prot_emb.shape[1]
        prot_emb_padded = self.pad_array(prot_emb, (self.max_pro_len, emb_dim))
        rna_emb_padded = self.pad_array(rna_emb, (self.max_rna_len, emb_dim))

        promask = [1] * len(pro_feat) + [0] * (self.max_pro_len - len(pro_feat))
        rnamask = [1] * len(rna_feat) + [0] * (self.max_rna_len - len(rna_feat))
        mask = promask + rnamask

        # 图像部分
        if self.args.image_network and self.image_processor:
            front_img, side_img, top_img = self.image_processor.get_image_views(key)
        else:
            front_img = side_img = top_img = torch.zeros(3, 224, 224)

        return dict(
            g=g, full_g=full_g, key=key,
            pro_feat_padded=pro_feat_padded,
            rna_feat_padded=rna_feat_padded,
            mol_indicator_padded=mol_indicator_padded,
            chain_indicator_padded=chain_indicator_padded,
            pro_coords_padded=pro_coords_padded,
            rna_coords_padded=rna_coords_padded,
            prot_emb_padded=prot_emb_padded,
            rna_emb_padded=rna_emb_padded,
            neighbor_matrix_padded=neighbor_matrix_padded,
            prot_len=prot_len, rna_len=rna_len, mask=mask,
            prot_whole_emb=prot_whole_emb, rna_whole_emb=rna_whole_emb,
            front_img=front_img, side_img=side_img, top_img=top_img
        )

    def __getitem__(self, idx):
        mut_key, orig_key, ddg = self.keys[idx], self.keys[idx][0:4] +'.pdb', self.affinity[idx]

        mut_data = self._load_single_structure(mut_key)
        orig_data = self._load_single_structure(orig_key)
        Y = float(ddg)

        return mut_data, orig_data, Y

    def collate(self, samples):
        """collate 一批 (mut_data, orig_data, ddg)"""
        muts, origs, ddgs = zip(*samples)

        def to_batch_dict(batch_list):
            g = [b['g'] for b in batch_list]
            full_g = [b['full_g'] for b in batch_list]
            batch_g = dgl.batch(g)
            batch_full_g = dgl.batch(full_g)

            to_tensor = lambda x: torch.from_numpy(np.stack(x)).float()
            return dict(
                g=batch_g,
                full_g=batch_full_g,
                pro_feat=to_tensor([b['pro_feat_padded'] for b in batch_list]),
                rna_feat=to_tensor([b['rna_feat_padded'] for b in batch_list]),
                mol_indicator=to_tensor([b['mol_indicator_padded'] for b in batch_list]),
                chain_indicator=to_tensor([b['chain_indicator_padded'] for b in batch_list]),
                pro_coords=to_tensor([b['pro_coords_padded'] for b in batch_list]),
                rna_coords=to_tensor([b['rna_coords_padded'] for b in batch_list]),
                prot_emb=to_tensor([b['prot_emb_padded'] for b in batch_list]),
                rna_emb=to_tensor([b['rna_emb_padded'] for b in batch_list]),
                neighbor_matrix=to_tensor([b['neighbor_matrix_padded'] for b in batch_list]),
                prot_len=[b['prot_len'] for b in batch_list],
                rna_len=[b['rna_len'] for b in batch_list],
                mask=torch.tensor([b['mask'] for b in batch_list]),
                prot_whole_emb=None,
                rna_whole_emb= None,
                front_imgs=torch.stack([b['front_img'] for b in batch_list]),
                side_imgs=torch.stack([b['side_img'] for b in batch_list]),
                top_imgs=torch.stack([b['top_img'] for b in batch_list]),
                keys=[b['key'] for b in batch_list],
            )

        mut_batch = to_batch_dict(muts)
        orig_batch = to_batch_dict(origs)
        ddg = torch.tensor(ddgs, dtype=torch.float)

        return mut_batch, orig_batch, ddg

        # self.init_txn()
        key = self.keys[idx]
        with lmdb.open(self.args.lmdb_cache, map_size=int(1e11), max_dbs=1, readonly=True, lock=False, readahead=False) as env:
            graph_db = env.open_db('data'.encode())
            with env.begin(buffers=True, write=False, db=graph_db) as txn:
                try:
                    g, full_g, Y = pickle.loads(txn.get(key.encode()))
                except Exception as e:
                    print(f"Error loading key {key}: {e}")
                    return None
        
        Y = self.affinity[idx]
        Y = float(Y)
       

        # a, b = g.edges()
        # # # construct geometric distance based graph 
        # g.ndata['coors'] = full_g.ndata['coors']
        # dm_all = distance_matrix(g.ndata['coors'].numpy(),g.ndata['coors'].numpy())#g.ndata['coors'].matmul(g.ndata['coors'].T)
        # edges_g = np.concatenate([a.reshape(-1,1).numpy(),b.reshape(-1,1).numpy()],axis = 1)
        # src,dst = np.where(dm_all < self.args.threshold)
        # # add covalent bond based edges and remove duplicated edges
        # edges_full = np.concatenate([src.reshape(-1,1),dst.reshape(-1,1)],axis = 1)
        # edges_full = np.unique(np.concatenate([edges_full,edges_g],axis = 0),axis = 0)
        # full_g = dgl.graph((edges_full[:,0],edges_full[:,1]))
        # full_g.ndata['coors'] = g.ndata['coors']
        # full_g.ndata['x'] = g.ndata['x']
        g.ndata['coors'] = full_g.ndata['coors']
        # g.ndata.pop('coors')

        return g,full_g,Y,key

    
from rdkit import Chem
from rdkit.Chem import AllChem


    
if __name__ == "__main__":
    import lmdb
    import argparse
    from rdkit import RDLogger
    from dgl import save_graphs, load_graphs
    import dgl
    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
    import time
    from multiprocessing import Pool, cpu_count
    import os
    RDLogger.DisableLog('rdApp.*')

    from utils.parsing import parse_train_args
    args = parse_train_args()

    # create lmdb database for map data and key,this step can help speed up training! Also ,can can skip this step too.
    
    env = lmdb.open(args.lmdb_cache, map_size=int(1e12), max_dbs=1)
    # create lmdb database
    dgl_graph_db = env.open_db('data'.encode())
    # read all data file path from pkl file  
    """ 
    Attention:
        you should change contain all data path in test_keys when you process data to LMDB database,
        also you can just specity a file path directly rather than passing it via args vatriable!
        
    """
    # with open (args.test_keys, 'rb') as fp:
    #     val_keys = pickle.load(fp)
    # keys =  val_keys 
    pdb = os.listdir("/public/home/qiang/jkwang/RNA_dock_score/data/data/")
    target_directory = "/public/home/qiang/jkwang/RNA_dock_score/data/errorFile"
    all_sdf = []
    for RNA in tqdm(pdb):
        directory = f"/public/home/qiang/jkwang/RNA_dock_score/data/data/{RNA}/docking/dockprep/"
        i = 1
        for sdf_file in sdf_files:
            rmsdfile = sdf_file.replace("docking.sdf","docking.rmsd")
            # 初始化一个空列表来存储RMSD值
            rmsd_list = []

            # 打开文件进行读取
            with open( rmsdfile, 'r') as file:
                for line in file:
                    # 分割每一行以获取RMSD值，假设每行的格式为 "序号 RMSD值"
                    _, rmsd = line.split()
                    # 将RMSD值转换为浮点数并添加到列表中
                    rmsd_list.append(float(rmsd))
            # exit()
            if os.path.getsize(sdf_file) == 0:continue

            i = split_sdf_and_save(sdf_file, i)
            exit()
            break
            # allsdf.append(sdf_files)
    ################ save processed data into database and then you can index data by the key in training_keys.pkl file ##########################
    def saveDB(key):
        with env.begin(write=True) as txn:
            try:
                g,y = ESDataset._GetGraph(key,args)
                txn.put(key.encode(), pickle.dumps((g,y)), db = dgl_graph_db)
            except:
                print('file: {} is not a valid file!'.format(key))
    all_keys = len(keys)
    with Pool(processes = 32) as pool:
        list(pool.imap(saveDB, keys))
    print('save done!')
    env.close()




        

       