
import lmdb
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import sys

import tqdm
sys.path.append("/public/home/qiang/jkwang/EquiScore-main")
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

class ESDataset(Dataset):

    def __init__(self, keys, ground_true, graph, allgraph, aff, args, data_dir,debug = False):
        super(ESDataset, self).__init__()
        # print(keys)
        self.keys = keys
        self.affinity = ground_true
        self.data_dir = data_dir
        self.debug = debug
        self.args = args
        self.graphs = []
        self.error = []
        self.mean = 7.941685094681864
        self.mean_10 = 5.899208519653227
        self.var = 27.93067861983596
        self.var_10 = 7.1360770169446095
        self.gaff_dir = "Gaff.csv"
        self.rmsd_dir = "modified_file.csv"
        self.w_gaff = 0.1
        self.b_gaff = 473.58
        # 读取 CSV 文件
        # df = pd.read_csv(self.gaff_dir)
        # df_r = pd.read_csv(self.rmsd_dir)
        # print(df_r)
        # 创建一个新的字典
        self.sdf_gaff_dict = {}
        self.norm_rmsd = {}
        self.newRMSD = {}
        # 提取RNA的标识符（假设RNA标识符在路径中是固定位置）
        # df['RNA'] = df['sdf_file'].apply(lambda x: x.split('/')[-2])
        # df['rmsd'] = df['sdf_file'].apply(lambda x: float(x.split('/')[-1].split('_')[-1].split(".sdf")[0]))
        # # print(df['RNA'])
        # # 对每个RNA进行分段归一化
        # def normalize(group):
        #     min_gaff = group['GAFF_energy'].min()
        #     max_gaff = group['GAFF_energy'].max()
        #     group['normalized_GAFF'] = (group['GAFF_energy'] - min_gaff) / (max_gaff - min_gaff)
        #     min_rmsd = group['rmsd'].min()
        #     max_rmsd = group['rmsd'].max()
        #     group['normalized_RMSD'] = (group['rmsd'] - min_rmsd) / (max_rmsd - min_rmsd)
        #     return group

        # df = df.groupby('RNA').apply(normalize)
        
        self.txn = None
        self.env  = None
        # if not args.test:
            # load data from LMDB database
            # env = lmdb.open(args.lmdb_cache, map_size=int(1e11), max_dbs=1, readonly=True)
            # self.graph_db = env.open_db('data'.encode()) # graph database
            # self.txn = env.begin(buffers=True,write=False)
            # from tqdm import tqdm
            # for key in tqdm(self.keys):
            #     try:
            #         g,Y= pickle.loads(self.txn.get(key.encode(), db=self.graph_db))
            #     except:
            #         self.error.append(key)
            # with open('test_error.pkl', 'wb') as f:
            #     pickle.dump(self.error, f)
             # Iterate through self.keys and check if they exist in LMDB
            # valid_keys = []
            # for key in self.keys:
            #     if self.txn.get(key.encode()) is not None:
            #         valid_keys.append(key)
            
            # self.keys = valid_keys
            # print("keys", len(self.keys))
        # else:
            # pass
        self.graph = graph
        self.allgraph = allgraph
        self.aff = aff
        # if len(self.graph) == 0:
            # self.init_load()
    def init_load(self):
        print("..............preloading data..................")
        # for idx, key in tqdm.tqdm(enumerate(self.keys),total = len(self.keys)) :
        #     key = self.keys[idx]
        #     with lmdb.open(self.args.lmdb_cache, map_size=int(1e10), max_dbs=1, readonly=True, lock=False, readahead=False) as env:
        #         graph_db = env.open_db('data'.encode())
        #         with env.begin(buffers=True, write=False, db=graph_db) as txn:
        #             try:
        #                 g, full_g, Y = pickle.loads(txn.get(key.encode()))
        #             except Exception as e:
        #                 print(f"Error loading key {key}: {e}")
        #                 return None
        #     self.graph[key] = g
        #     self.allgraph[key] = full_g
        #     self.aff[key] = Y
        try:
            with open("PNAdata.pkl", "rb") as f:
                self.graph, self.allgraph, self.aff = pickle.load(f)
                
        except EOFError:
            print(len(self.graph))
            print("PNAdata.pkl 文件为空或损坏，请重新生成。")
            
    def init_txn(self):
        if self.txn is None:
            self.env = lmdb.open(self.args.lmdb_cache, map_size=int(1e11), max_dbs=1, readonly=True)
            self.graph_db = self.env .open_db('data'.encode()) # graph database
            self.txn = self.env .begin(buffers=True,write=False)
    def __len__(self):
        if self.debug:
            return 30000
        return len(self.keys)
    def collate(self, samples):

        """ 
        The input samples is a list of pairs (graph, label)
        collate function for building graph dataloader

        """
        samples = list(filter(lambda  x : x is not None,samples))
        g,full_g,Y,key = map(list, zip(*samples))
        batch_g = dgl.batch(g)
        batch_full_g = dgl.batch(full_g)
        Y = torch.tensor(Y)
        return batch_g, batch_full_g,Y,key
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
        g, full_g, Y  = self.graph[key],self.allgraph[key],self.aff[key]
        g.ndata['coors'] = full_g.ndata['coors']
        Y = self.affinity[idx]
        Y = float(Y)
        
        return  g, full_g, Y ,key
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




        

       