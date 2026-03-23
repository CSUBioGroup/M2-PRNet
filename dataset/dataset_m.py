
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
        self.max_per_pdb = 20
        if self.max_per_pdb is not None:
            for pdb_id, inds in self.pdb_to_indices.items():
                if len(inds) > self.max_per_pdb:
                    self.pdb_to_indices[pdb_id] = inds[:self.max_per_pdb]
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

class ESDataset_m(Dataset):

    def __init__(self, keys, ground_true, graph, allgraph, res_level, args, data_dir,image_process,debug = False):
        super(ESDataset_m, self).__init__()
        # print(keys)
        self.keys = keys
        self.affinity = ground_true
        self.data_dir = data_dir
        self.debug = debug
        self.args = args
        self.graphs = []
        self.error = []
        
        
       
        self.res_level = res_level
        if args.data_set == 'PNA_keys.csv' or args.data_set == 'PNA_keys_201.csv' or args.data_set == 'MD_keys.csv' or args.data_set == 'data/case/case.csv':
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
    def init_load(self):
        print("..............preloading data..................")
        try:
            with open("PNAdata.pkl", "rb") as f:
                self.graph, self.allgraph, self.aff = pickle.load(f)
                
        except EOFError:
            print(len(self.graph))
            print("PNAdata.pkl 文件为空或损坏，请重新生成。")
    def precompute_image_features(self, image_network, device='cuda'):
        """预计算所有图像特征"""
        if self.image_processor is None:
            return
            
        print("Precomputing image features...")
        image_network.eval()
        image_network.to(device)
        
        with torch.no_grad():
            for i, key in enumerate(self.keys):
                if i % 100 == 0:
                    print(f"Processing {i}/{len(self.keys)}")
                
                # 获取三视图图像
                front, side, top = self.image_processor.get_image_views(key)
                
                # 添加batch维度
                front = front.unsqueeze(0).to(device)
                side = side.unsqueeze(0).to(device)
                top = top.unsqueeze(0).to(device)
                
                # 提取图像特征
                # image_feat = image_network(front, side, top)
                # self.image_features_cache[key] = image_feat.cpu().squeeze(0)

        print("Image features precomputation completed!")

    def init_txn(self):
        if self.txn is None:
            self.env = lmdb.open(self.args.lmdb_cache, map_size=int(1e11), max_dbs=1, readonly=True)
            self.graph_db = self.env .open_db('data'.encode()) # graph database
            self.txn = self.env .begin(buffers=True,write=False)
    def __len__(self):
        if self.debug:
            return 30000
        return len(self.keys)
    def pad_array(self, arr, target_shape, pad_value=0):
        """
        对 numpy 数组进行 padding。
        
        Args:
            arr: 原始 numpy 数组
            target_shape: 目标 shape，要求每一维 target_shape[i] >= arr.shape[i]
            pad_value: 填充值，默认为 0
        Returns:
            padded: 填充后的数组
        """
        pad_width = []
        for orig, target in zip(arr.shape, target_shape):
            pad_width.append((0, target - orig))
        # 对多余的轴（如果 target_shape 长度大于 arr 的轴数）也要 pad
        if len(target_shape) > arr.ndim:
            for _ in range(len(target_shape) - arr.ndim):
                pad_width.append((0, target_shape[len(pad_width)]))
        # print(pad_width)
        padded = np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
        return padded
    def collate(self, samples):

        """ 
        The input samples is a list of pairs (graph, label)
        collate function for building graph dataloader

        """
        samples = list(filter(lambda  x : x is not None,samples))
        g,full_g,Y,key,pro_feat_padded, rna_feat_padded, mol_indicator_padded, chain_indicator_padded, pro_coords_padded,rna_coords_padded,\
        prot_emb_padded, rna_emb_padded, neighbor_matrix_padded, prot_len, rna_len, mask,prot_whole_emb, rna_whole_emb,front_imgs, side_imgs, top_imgs = map(list, zip(*samples))
        # print(key)
        pro_feat = torch.from_numpy(np.stack(pro_feat_padded)).float()
        rna_feat = torch.from_numpy(np.stack(rna_feat_padded)).float()
        # pro_feat = torch.from_numpy(np.stack(pro_feat_padded)).float()
        # rna_feat = torch.from_numpy(np.stack(rna_feat_padded)).float()
        # mol_indicator = torch.tensor(mol_indicator_padded)
        mol_indicator = torch.from_numpy(np.stack(mol_indicator_padded)).float()
        chain_indicator = torch.from_numpy(np.stack(chain_indicator_padded)).float()
        # chain_indicator = torch.tensor(chain_indicator_padded)
        pro_coords = torch.from_numpy(np.stack(pro_coords_padded)).float()
        rna_coords = torch.from_numpy(np.stack(rna_coords_padded)).float()
        prot_emb = torch.from_numpy(np.stack(prot_emb_padded)).float()
        rna_emb = torch.from_numpy(np.stack(rna_emb_padded)).float()
        # for p in prot_whole_emb:
            # print(p.shape)
        neighbor_matrix_padded = torch.from_numpy(np.stack(neighbor_matrix_padded)).float()
        # neighbor_matrix_padded = None
        # prot_whole_emb = torch.from_numpy(np.stack(prot_whole_emb)).float()
        prot_whole_emb = None
        # rna_whole_emb = torch.from_numpy(np.stack(rna_whole_emb)).float()
        rna_whole_emb = None
        # pro_coords = torch.tensor(pro_coords_padded)
        # rna_coords = torch.tensor(rna_coords_padded)
        # data_type = torch.tensor(data_type)
        # prot_emb = torch.tensor(prot_emb_padded)
        # rna_emb = torch.tensor(rna_emb_padded)
        # neighbor_matrix_padded = torch.tensor(neighbor_matrix_padded)
        mask = torch.tensor(mask)
        # mask = torch.tensor(mask)

        batch_g = dgl.batch(g)
        batch_full_g = dgl.batch(full_g)
        if self.args.image_network is not None:
            # 组装batch图像数据
            # front_batch = torch.stack([img_data['front'] for img_data in image_data])
            # side_batch = torch.stack([img_data['side'] for img_data in image_data])
            # top_batch = torch.stack([img_data['top'] for img_data in image_data])
            front_batch = torch.stack(front_imgs)
            side_batch = torch.stack(side_imgs)
            top_batch = torch.stack(top_imgs)
        #     # 通过图像网络提取特征
        #     image_features = self.image_network(front_batch, side_batch, top_batch)
        else:
            front_batch = None
            side_batch = None
            top_batch = None
        #     image_features = torch.zeros(len(samples), 256)
        Y = torch.tensor(Y)
        return batch_g, batch_full_g,Y,key, pro_feat, rna_feat, mol_indicator, chain_indicator, pro_coords,rna_coords, prot_emb,\
        rna_emb, neighbor_matrix_padded, prot_len, rna_len, mask,prot_whole_emb, rna_whole_emb, front_batch, side_batch, top_batch

    def map_to_range(self, Y, minY, maxY):
        # 新区间的最小值和最大值
        newMin, newMax = 0, 5
        
        # 计算映射系数
        scale = (newMax - newMin) / (maxY - minY)
        
        # 应用映射规则
        newY = newMin + (Y - minY) * scale
        
        return newY
    def map_to_new_range(self,x, original_mean , original_std , new_min = 0, new_max = 5):
        # 计算标准化分数
        standardized = (x - original_mean) / original_std
        # 将标准化分数映射到新区间
        return new_min + (standardized * (new_max - new_min))
    def normalized(numbers):
        # 计算最小值和最大值
        min_value = min(numbers)
        max_value = max(numbers)

        # 归一化列表
        normalized_numbers = [10* (x - min_value) / (max_value - min_value) for x in numbers]
        return normalized_numbers

    def __getitem__(self, idx):
        key = self.keys[idx]
        # print(key)
        if self.args.data_set =='PNA_keys_201.csv' or  self.args.data_set =='PNA_keys.csv' :
            g, full_g  = self.graph[key],self.allgraph[key]
            pid = key[0:4]
            modelid = key.split('_')[1]
            full_key = f"{pid}_model{modelid}"
        elif self.args.data_set == 'MD_keys.csv':
            pid = key[0:4]
            modelid = key.split('_')[1]
            full_key = f"{pid}_model{modelid}.pdb"
            g, full_g  = self.graph[full_key],self.allgraph[full_key]
            key = full_key
        elif self.args.data_set == 'data/case/case.csv':
            pid = key.split('_')[0] if '_' in key else key[0:4]
            modelid = key.split('_')[1]
            full_key = f"{pid}_model_{modelid}.pdb"
            # print(self.graph.keys())
            # print(self.allgraph.keys())
            if '4BS2' not in key and '1URN' not in key:
                g, full_g  = self.graph[full_key],self.allgraph[full_key]
                key = full_key
            else:
                g, full_g  = self.graph[key[0:4]+'.pdb'],self.allgraph[key[0:4]+'.pdb']
                key = key[0:4]+'.pdb'
            
        else:
            g   = self.graph[key+'.pdb']
            full_g = self.allgraph[key+'.pdb']
            pid = key[0:4]
            modelid = key.split('model')[1]
            full_key = key
        # else:
            # g, full_g  = self.graph[key],self.allgraph[key]
        g.ndata['coors'] = full_g.ndata['coors']
        Y = self.affinity[idx]
        Y = float(Y)
        pro_feat = self.res_level[key]['pro_feats']
        # print(len(pro_feat))
        # pro_feat , mask = self.pad_sequences([pro_feat])
        # print(pro_feat.shape)
        rna_feat = self.res_level[key]['rna_feats']
        # rna_feat = self.pad_sequences([rna_feat])[0][0]
        mol_indicator = self.res_level[key]['mol_indicator']
        # mol_indicator = self.pad_sequences([mol_indicator])[0][0]
        chain_indicator = self.res_level[key]['chain_indicator']
        pro_coords = self.res_level[key]['pro_coords']
        rna_coords = self.res_level[key]['rna_coords']
        data_type = self.res_level[key]['data_type']
        prot_emb = self.res_level[key]['prot_emb']
        rna_emb = self.res_level[key]['rna_emb']
        prot_len = self.res_level[key]['prot_len']
        rna_len = self.res_level[key]['rna_len']
        neighbor_matrix = self.res_level[key]['neighbor_matrix']
        prot_whole_emb = self.res_level[key]['prot_whole_emb']
        rna_whole_emb = self.res_level[key]['rna_whole_emb']
        # self.init_txn()
        # 目标总节点数
        max_total = self.max_pro_len + self.max_rna_len
        pro_feat_padded = self.pad_array(np.array(pro_feat), (self.max_pro_len, len(pro_feat[0])), pad_value=0)
        rna_feat_padded = self.pad_array(np.array(rna_feat), (self.max_rna_len, len(rna_feat[0])), pad_value=0)
        # 对 mol_indicator 进行 padding 到 (max_total, 2)
        mol_indicator_padded = self.pad_array(np.array(mol_indicator), (max_total, 2), pad_value=0)

        # 对 chain_indicator 进行 padding 到 (max_total, 6)
        chain_indicator_padded = self.pad_array(np.array(chain_indicator), (max_total, 6), pad_value=0)

        # 对坐标进行 padding 到 (max_total, 3)
        # coords_padded = self.pad_array(np.array(coords), (max_total, 3), pad_value=0)
        pro_coords_padded = self.pad_array(np.array(pro_coords), (self.max_pro_len, 3), pad_value=0)
        rna_coords_padded = self.pad_array(np.array(rna_coords), (self.max_rna_len, 3), pad_value=0)

        # 对邻接矩阵 padding 到 (max_total, max_total)
        neighbor_matrix_padded = self.pad_array(np.array(neighbor_matrix), (max_total, max_total), pad_value=0)
        # pro_feats = self.pad_array(np.array(pro_feats), (self.max_pro_len, len(pro_feats[0])), pad_value=0)
        # rna_feats = self.pad_array(np.array(rna_feats), (self.max_rna_len, len(rna_feats[0])), pad_value=0)
        # 对蛋白质LLM嵌入 padding 到 (max_pro_len, emb_dim)
        # 假设你已经分别获得蛋白质和RNA的 LLM 嵌入列表：
        # prot_emb = np.array(concat_prot_emd)  # shape = (n_prot, emb_dim)
        # rna_emb = np.array(concat_rna_emd)     # shape = (n_rna, emb_dim)

        # 如果需要对两类嵌入分别进行 padding（假设目标长度 max_pro_len 和 max_rna_len 已知）：
        # prot_emb_padded = self.pad_array(prot_emb, (max_pro_len, prot_emb.shape[1]), pad_value=0)
        # rna_emb_padded = self.pad_array(rna_emb, (max_rna_len, rna_emb.shape[1]), pad_value=0)
        # final_llm_emb = np.concatenate([prot_emb_padded, rna_emb_padded], axis=0)
        # prot_emb = np.array(prot_emb)  # shape=(n_prot, emb_dim)
        emb_dim = prot_emb.shape[1]
        # print(prot_emb.shape)
        prot_emb_padded = self.pad_array(prot_emb, (self.max_pro_len, emb_dim), pad_value=0)

        # 对RNA LLM嵌入 padding 到 (max_rna_len, emb_dim)
        # rna_emb = np.array(rna_emb)  # shape=(n_rna, emb_dim)
        rna_emb_padded = self.pad_array(rna_emb, (self.max_rna_len, emb_dim), pad_value=0)

        promask = [1] * len(pro_feat) + [0] * (self.max_pro_len - len(pro_feat))
        rnamask = [1] * len(rna_feat) + [0] * (self.max_rna_len - len(rna_feat))
        mask = promask + rnamask
        # mask = [1] * (len(pro_feats) + len(rna_feats)) + [0] * (max_total - (len(pro_feats) + len(rna_feats)))
        # mask = [1] * (len(pro_feat) + len(rna_feat)) + [0] * (max_total - (len(pro_feat) + len(rna_feat)))
        matirx_mask = np.array(mask).reshape(1, -1) * np.array(mask).reshape(-1, 1)

       
        prot_len = len(pro_feat)
        rna_len = len(rna_feat)    
       # 图像特征 - 返回原始图像张量，在collate中批量处理
        if self.args.image_network:
            front_img, side_img, top_img = self.image_processor.get_image_views(full_key)
            # 这里返回原始图像，在collate中统一通过image_network提取特征
        #     image_data = {
        #         'front': front_img,
        #         'side': side_img, 
        #         'top': top_img,
        #         'has_image': True
        #     }
        else:
            front_img = torch.zeros(3, 224, 224)
            side_img = torch.zeros(3, 224, 224)
            top_img = torch.zeros(3, 224, 224)
        #     image_data = {
        #         'front': torch.zeros(3, 224, 224),
        #         'side': torch.zeros(3, 224, 224),
        #         'top': torch.zeros(3, 224, 224),
        #         'has_image': False
        #     }
        return g,full_g,Y,key, pro_feat_padded, rna_feat_padded, mol_indicator_padded, chain_indicator_padded, \
        pro_coords_padded,rna_coords_padded, prot_emb_padded, rna_emb_padded, neighbor_matrix_padded, prot_len, rna_len, mask, prot_whole_emb, rna_whole_emb,front_img, side_img, top_img
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

    @staticmethod
    def _GetGraph(self, rna_path, ligand_path, args):
        """
        construct structual graph based on covalent bond and non-bond interaction and save to LMDB database for speed up data loading
        Parameters
        ----------
        key : string 
            file path
        args : dictionary
            parameters 
	
		Returns
		-------
		(dgl.DGLGraph, Tensor)

        
        """
        # try:
        #     try:
        #         with open(key, 'rb') as f:
        #             m1,m2= pickle.load(f)
        #     except:
        #         with open(key, 'rb') as f:
        #             m1,m2,atompairs,iter_types= pickle.load(f)
        # except:
        #     return None
        m1 = Chem.MolFromPDBFile(rna_path, removeHs=True)
        # print(rna_path)
        # Check if the conversion was successful
        # if m1 is not None:
        #     print("Conversion successful.")
        #     # You can now manipulate or inspect the Mol object
        # else:
        #     print("Conversion failed.")
        # Create an SDMolSupplier object
        supplier = Chem.SDMolSupplier(ligand_path)
        # supplier = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        
     
        # for mol in supplier:
        #     if mol is not None:
        #         print("Molecule loaded successfully.")
        #         # You can now manipulate or inspect the Mol object
        #     else:
        #         print("Failed to load a molecule.")
        # for m2 in supplier:
        m2 = supplier[0]
        n1,d1,adj1 = get_mol_info(m1)
        n2,d2,adj2 = get_mol_info(m2)

        from rdkit.Chem.rdchem import Mol
        # print(SplitMolByPDBResidues(m1).values())

        # for atom in m2.GetAtoms():
        #     atom.SetUnsignedProp("mapindex", atom.GetIdx())
        #     print(atom.GetIdx())

        H1 = np.concatenate([get_atom_graphformer_feature(m1,FP = False) ,np.array([0]).reshape(1,-1).repeat(n1,axis = 0)],axis=1)
        H2 = np.concatenate([get_atom_graphformer_feature(m2,FP = False) ,np.array([1]).reshape(1,-1).repeat(n2,axis = 0)],axis=1)
        # if args.virtual_aromatic_atom:
        if True:
            adj1,H1,d1,n1 = add_atom_to_mol(m1,adj1,H1,d1,n1)
            adj2,H2,d2,n2 = add_atom_to_mol(m2,adj2,H2,d2,n2)
        H = torch.from_numpy(np.concatenate([H1, H2], 0))
        # get covalent bond edges
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        # get non-bond interactions based edges
        # if args.fingerprintEdge:

        if True:
            # slowly when trainging from raw data ,so we save the result in disk,for next time use
            if 'inter_types' not in vars().keys() and 'atompairs' not in vars().keys():
                try:
                    atompairs,iter_types = get_nonBond_pair(m2,m1)
                except:
                    atompairs,iter_types = [],[]
                    # print(key)
                # print("---1--")
                # with open("1AJU.pickle",'wb') as f:
                #     pickle.dump((m1,m2,atompairs,iter_types),f)
                # f.close()
            if len(atompairs) > 0:
                temp_fp= np.array(atompairs)
                u,v = list(temp_fp[:,0]) +  list((n1+ temp_fp[:,1])),list((n1+ temp_fp[:,1])) + list(temp_fp[:,0])
                agg_adj1[u,v] = 1

        agg_adj1 = torch.from_numpy(agg_adj1)
        adj_graph_1 = np.copy(agg_adj1)
        pocket = (m1,m2)
        item_1 = mol2graph(pocket,H,args,adj = adj_graph_1,n1 = n1,n2 = n2,\
            dm = (d1,d2) )
        g = preprocess_item(item_1, args,adj_graph_1)
        # print("---2--")
        g.ndata['coors'] = torch.from_numpy(np.concatenate([d1,d2],axis=0))
        valid = torch.zeros((n1+n2,))
        # set readout feature for task layer
        if args.pred_mode == 'ligand':
            valid[:n1] = 1
        elif args.pred_mode == 'protein':
            valid[n1:] = 1
        else:
            raise ValueError(f'not support this mode : {args.pred_mode} plz check the args')
        # path_parts = ligand_path.split(os.sep)
        # target_directory = "/public/home/qiang/jkwang/RNA_dock_score/data/Graph"
        # new_file_name = f"{path_parts[-6]}_{path_parts[-3]}_{path_parts[-2]}_{os.path.basename(ligand_path)}"
        # key = os.path.join(target_directory, new_file_name)
#             if args.loss_fn == 'mse_loss':

#                 Y = -float(key.split('-')[1].split('_')[0])
#             else:
#                 if '_active' in key.split('/')[-1]:
#                     Y = 1 
#                 else:
#                     Y = 0 
        Y = ligand_path.split('_')[-1]
        g.ndata['V'] = valid.float().reshape(-1,1)
        # path_parts = key.split(os.sep)
        # mid = f"{path_parts[-6]}_{path_parts[-3]}_{path_parts[-2]}_{os.path.basename(file)}"
        # with env.begin(write=True) as txn:
        #     txn.put(key.encode(), pickle.dumps((g,y)), db = dgl_graph_db)
        return g, Y
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




        

       