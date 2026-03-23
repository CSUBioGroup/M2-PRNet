from collections import Counter
import pickle
import time
import numpy as np
from utils.equiscore_utils import LogScaler
import utils.utils as utils
from utils.utils import *
from utils.loss_utils import *
# from dataset_utils import *
from utils.dist_utils import *
from dataset.dataset import *
from dataset.dataset_m import ESDataset_m
import torch.nn as nn
import torch
import time
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset.dataset_m import *
import argparse
import time
from torch.utils.data import DataLoader          
from prefetch_generator import BackgroundGenerator
from model.equiscore import EquiScore
# from equiformer.nets.graph_attention_transformer import GraphAttentionTransformer_dgl
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average'):
        return mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
            squared=False,
        )
import os
import glob
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr,pearsonr
# from model.IPA import IPAAffinityModel
# from model.GVP import GVPAffinityModel
# python Train_PNA.py --sampler --share_weights --image_network --contrastive --data_set newdata.csv
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())    
now = time.localtime()
from rdkit import RDLogger
# 直接使用这个简单的方案：
def safe_negative_log_transform(targets):
    """安全的负对数变换"""
    # 强制转换为numpy数组
    targets = np.array(targets, dtype=np.float32)
    epsilon = 1e-8
    return -np.log(-targets + epsilon)


def safe_inverse_negative_log(transformed):
    """安全的反向变换"""
    return -np.exp(-transformed)
def balance_training_data(train_data):
    """
    对训练数据进行重采样平衡
    train_data: [(key, affinity), ...]
    """
    if not train_data:
        return train_data
    
    # 提取PDB ID
    pdb_samples = defaultdict(list)
    for key, affinity in train_data:
        pdb_id = key.split('_')[0]  # 提取PDB ID
        pdb_samples[pdb_id].append((key, affinity))
    
    # 找到最大样本数
    max_count = max(len(samples) for samples in pdb_samples.values())
    
    balanced_data = []
    
    for pdb_id, samples in pdb_samples.items():
        current_count = len(samples)
        
        if current_count < max_count:
            # 重复采样到最大数量
            repeat_times = max_count // current_count + 1
            repeated_samples = samples * repeat_times
            # 随机选择max_count个样本
            balanced_data.extend(repeated_samples[:max_count])
        else:
            balanced_data.extend(samples)
    
    return balanced_data
def _load_chunks(output_file, mode="auto", duplicate="keep_first"):
    """
    读取 save_data_in_chunks 生成的数据（multi 或 append），合并为一个 dict。
    output_file: 保存时使用的基名（例如 "newdata_graphs.pkl"）
    mode: "auto"/"multi"/"append"
    duplicate: "keep_first" 或 "overwrite"
    """
    merged = {}
    base, ext = os.path.splitext(output_file)
    if ext == "":
        ext = ".pkl"
    part_pattern = f"{base}.part*{ext}"
    part_files = sorted(glob.glob(part_pattern))

    if mode == "auto":
        mode = "multi" if part_files else "append"

    if mode == "multi":
        if not part_files:
            raise FileNotFoundError(f"No part files found for pattern: {part_pattern}")
        print(f"[INFO] Found part files: {part_files}")
        for pf in tqdm(part_files):
            with open(pf, "rb") as f:
                try:
                    chunk = pickle.load(f)
                except Exception as e:
                    print(f"[WARN] 读取 {pf} 失败: {e}, 跳过")
                    continue
            if not isinstance(chunk, dict):
                print(f"[WARN] {pf} 内容不是 dict，跳过")
                continue
            for k, v in chunk.items():
                if k in merged and duplicate == "keep_first":
                    continue
                merged[k] = v
    else:  # append 单文件内有多个 pickle 对象
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"File not found: {output_file}")
        with open(output_file, "rb") as f:
            idx = 0
            while True:
                try:
                    chunk = pickle.load(f)
                except EOFError:
                    break
                except Exception as e:
                    print(f"[WARN] 读取块 {idx} 失败: {e}, 跳过")
                    idx += 1
                    continue
                if not isinstance(chunk, dict):
                    print(f"[WARN] 第 {idx} 块内容不是 dict，跳过")
                    idx += 1
                    continue
                for k, v in chunk.items():
                    if k in merged and duplicate == "keep_first":
                        continue
                    merged[k] = v
                idx += 1
    return merged

def load_saved_graphs(graphs_file="newdata_graphs.pkl", allgraphs_file="newdata_allgraphs.pkl",
                      mode="auto", duplicate="keep_first"):
    """
    加载并合并之前分块保存的 graphs 和 allgraphs。
    返回 (graphs_dict, allgraphs_dict)
    """
    graphs = _load_chunks(graphs_file, mode=mode, duplicate=duplicate)
    allgraphs = _load_chunks(allgraphs_file, mode=mode, duplicate=duplicate)
    print(f"[INFO] Loaded graphs: {len(graphs)}, allgraphs: {len(allgraphs)}")
    return graphs, allgraphs

RDLogger.DisableLog('rdApp.*')
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
# print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
def run(local_rank,args):
    args.local_rank = local_rank
    args.lmdb_cache = "lmdbs/score"
    config = {
            'rna_biochem_dim': 9,      # RNA biochemical feature dimension
            'rna_seq_dim': 1280,         # RNA sequence feature dimension
            'protein_biochem_dim': 8, #  protein biochemical feature dimension
            'protein_seq_dim': 1280,    #  protein sequence feature dimension
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.4,
            'num_gaussian_kernels': 32,
            'sigma_min': 0.1,
            'sigma_max': 10.0,
            'output_dim': 1  # 
        }
    args.llm_seq_dim = 1280
    # args.lmdb_cache = "lmdbs/affTest100_draw"
    # torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu)  # multi gpus training，'nccl' mode
    torch.cuda.set_device(args.local_rank) 
    seed_torch(seed = args.seed + args.local_rank)
    # use attentiveFP feature or not
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28
    num_epochs = args.epoch
    lr = args.lr
    save_dir = args.save_dir
    train_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    # make save dir if it doesn't exist
    if args.hot_start:
        if os.path.exists(args.save_model):
            best_name = args.save_model
            model_name = best_name.split('/')[-1]
            save_path = best_name.replace(model_name,'')
        else:
            raise ValueError('save_model is not a valid file check it again!')
    else:
        save_path = os.path.join(save_dir,args.model,train_time)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        # os.system('mkdir -p ' + save_path)
    log_path = save_path+'/logs' 
    args.log_path_step = save_path+'/logs_steps' 
    args.save_step_path = save_path
    args.fig_flag = True
    #read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.

    import random
    import json
    import csv
    
    # key,affinity,fold0,fold1,fold2,fold3,fold4
    # with open(graph_path, "rb") as f:
        # ADD_graph, ADD_allgraph = pickle.load(f)
    if args.data_set == 'PNA_keys.csv' or args.data_set == 'PNA_keys_201.csv':
        graph_path = 'PNAdata.pkl'
        with open(graph_path, "rb") as f:
            graph,allgraph,aff = pickle.load(f)
            # print(graph)
        with open('res_level.pkl', 'rb') as f:
            res_level = pickle.load(f)
        # with open('md150.csv', 'r') as f:
        #     reader = csv.DictReader(f)
        #     validpdb = []
        #     for row in reader:
        #         key = row['PDB']
        #         validpdb.append(key)
    elif args.data_set == 'MD_keys.csv':
        allgraph_path = 'MDdata_allgraphs.pkl'
        graph_path = 'MDdata_graph.pkl'
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        with open(allgraph_path, "rb") as f:
            allgraph = pickle.load(f)
        with open('MD_res_level.pkl', 'rb') as f:
            res_level = pickle.load(f)
    
        with open('md150.csv', 'r') as f:
            reader = csv.DictReader(f)
            validpdb = []
            for row in reader:
                key = row['PDB']
                validpdb.append(key)
    # args.data_set = 'MD_keys.csv' # test MD
    # print(graph)
    # print(len(graph))
    fold_files = 5
    fold_data = [defaultdict(list) for _ in range(fold_files)]  # [{train:[], val:[], test:[]}, ...]
    
    with open(args.data_set, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['key']
            # print(f"[INFO] Found row: {row}")
            affinity = float(row['affinity'])
            for i in range(fold_files):
                fold_type = row[f'fold{i}']
                
                fold_data[i][fold_type].append((key, affinity))
    # print(f"Loaded data for {fold_data} folds.")
    avgpcc = 0.0
    avgspcc = 0.0
    avg_r2 = 0.0
    avg_rmse = 0.0
    avg_mae = 0.0
    args.avgpcc = 0.0
    args.avgspcc = 0.0
    args.avgr2 = 0.0
    args_avgrmse = 0.0
    args.avgmae = 0.0
    scaler = LogScaler()
    image_processor = ImageProcessor(args.image_dir)
    # aff = {}
    for i, fold in enumerate(fold_data):
        # if i<2:
            # continue
        args.cv = i
        print(f"Fold {i}:")
        for split in ['train', 'val', 'test']:
            print(f"  {split}: {len(fold[split])} samples")
        fold_data[i]['train'] = balance_training_data(fold_data[i]['train'])

        balanced_train_size = len(fold_data[i]['train'])
        print(f"  平衡后训练集大小: {balanced_train_size}")    
        random.shuffle(fold)
        pdbids = list(set([k[:4] for k, _ in fold['train'] ]))
        random.shuffle(pdbids)
        train_pdb = pdbids[:int(0.9 * len(pdbids))]
        train_pdb = pdbids
        val_pdb =  pdbids[int(0.9 * len(pdbids)):]
        train_keys = [k for k, _ in fold['train'] if k[0:4] in train_pdb ]
        # collect candidate validation keys (only keep '_0_' anchors)
        # if args.data_set == 'PNA_keys.csv' or args.data_set == 'PNA_keys_201.csv':
        val_keys_raw = [k for k, _ in fold['train'] if k[0:4] in val_pdb ]

        # keep only one validation key per PDB (first occurrence), to remove duplicates from augmentation
        seen_pdb = set()
        val_keys = []
        for k in val_keys_raw:
            pdb_id = k[:4]
            if pdb_id not in seen_pdb:
                seen_pdb.add(pdb_id)
                val_keys.append(k)
        # val_keys = train_keys[:int(0.1*len(train_keys))]  # 10% of training for validation
        # train_keys = train_keys[int(0.1*len(train_keys)):]
        # val_keys = [k for k, _ in fold['val'] ]
        test_keys = [k for k, _ in fold['test'] if '_0_' in k ]
        if args.data_set == 'PNA_keys.csv' or args.data_set == 'PNA_keys_201.csv':
            if len(fold['test']) > 0:
                test_keys = [k for k, _ in fold['test'] if '_0_' in k ]
                test_ground_true =[Y for _, Y in fold['test'] if '_0_' in _  ]
            else:
                test_keys = [k for k, _ in fold['val'] if '_0_' in k  ]
                test_ground_true =[Y for _, Y in fold['val'] if '_0_' in _ ]
        elif args.data_set == 'MD_keys.csv' :
            if len(fold['test']) > 0:
                test_keys = [k for k, _ in fold['test'] if '_0_' in k and k[0:4] in validpdb]
                test_ground_true =[Y for _, Y in fold['test'] if '_0_' in _ and _[0:4] in validpdb]
            else:
                test_keys = [k for k, _ in fold['val'] if '_0_' in k and k[0:4] in validpdb]
                test_ground_true =[Y for _, Y in fold['val'] if '_0_' in _ and _[0:4] in validpdb]
        else:
            if len(fold['test']) > 0:
                test_keys = [k for k, _ in fold['test'] if 'model0' in k ]
                test_ground_true =[Y for _, Y in fold['test'] if 'model0' in _ ]
            else:
                test_keys = [k for k, _ in fold['val'] if 'model0' in k ]
                test_ground_true =[Y for _, Y in fold['val'] if 'model0' in _ ]

        train_ground_true =[float(Y) for k, Y in fold['train'] if k[0:4] in train_pdb]
        seen_pdb = set()
        val_ground_true = []
        for k in val_keys:
            for k_fold,Y in fold['train']:
                if k_fold not in seen_pdb and k == k_fold:
                    seen_pdb.add(k_fold)
                    val_ground_true.append(float(Y))
        # val_ground_true = [float(Y) seen_pdb.add(pdb_id) for k in val_keys for k_fold, Y in fold['train'] if k == k_fold and k not in seen_pdb]
        print(len(val_ground_true),len(val_keys))
        # check train_keys and train_ground_true are aligned
        assert len(train_keys) == len(train_ground_true), "train_keys 和 train_ground_true 长度不一致！"
        for key, y in zip(train_keys, train_ground_true):
            assert any(key == k and y == Y for k, Y in fold['train']), f"Key {key} 和 Y {y} 不匹配！"

        # check val_keys and val_ground_true are aligned
        assert len(val_keys) == len(val_ground_true), "val_keys 和 val_ground_true 长度不一致！"
        # for key, y in zip(val_keys, val_ground_true):
        #     assert any(key == k and y == Y for k, Y in fold['train']), f"Key {key} 和 Y {y} 不匹配！"
        # val_ground_true =[Y for _, Y in fold['val'] ]
        # val_ground_true = [float(Y) for k in val_keys for k_fold, Y in fold['train'] if k == k_fold]
        # train_ground_true =[Y for _, Y in fold['train'] if _ in train_keys ]
        # if len(fold['test']) > 0:
        #     test_ground_true =[Y for _, Y in fold['test'] if 'model0' in _ ]
        # else:
        #     test_ground_true =[Y for _, Y in fold['val'] if 'model0' in _ ]
        for key, y in zip(test_keys, test_ground_true):
            assert any(key == k and y == Y for k, Y in fold['test']), f"Key {key} 和 Y {y} 不匹配！ Key {k} 和 Y {Y} 不匹配！ "
        # print(test_ground_true)
        scaler.fit(train_ground_true)
        # train_ground_true_scaler = scaler.transform(train_ground_true)
        # test_ground_true_scaler = scaler.transform(test_ground_true)
        train_ground_true_scaler = train_ground_true
        test_ground_true_scaler = test_ground_true
        # train_ground_true_scaler = safe_negative_log_transform(train_ground_true)
        # test_ground_true_scaler = safe_negative_log_transform(test_ground_true)
        if local_rank == 0:
            print (f'Number of train data: {len(train_keys)}')
            print(f'Number of train value data: {len(train_ground_true)}')
            print (f'Number of val data: {len(val_keys)}')
            print (f'Number of val value data: {len(val_ground_true)}')
            print (f'Number of test data: {len(test_keys)}')
            print (f'Number of test data: {len(test_ground_true)}')
            if args.model == 'M2-PRNet':
                model = EquiScore(args, config)
            elif args.model == 'IPA':
                model = IPAAffinityModel()
            elif args.model == 'GVP':
                model = GVPAffinityModel()

            print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
            # args.device = args.local_rank
            args.device ='cuda' if torch.cuda.is_available() else 'cpu'
            if args.hot_start:
                model ,opt_dict,epoch_start= utils.initialize_model(model, args.device,args,args.save_model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

                optimizer.load_state_dict(opt_dict)

            else:
                
                model = utils.initialize_model(model, args.device,args)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                epoch_start = 0
                if not args.onlytest and i == 0:
                    write_log_head(args,log_path,model,train_keys,val_keys)
                    write_log_head(args,args.log_path_step,model,train_keys,val_keys)
            
            # dataset processing
            train_dataset = ESDataset_m(train_keys, train_ground_true_scaler, graph, allgraph  ,res_level, args, args.data_path,image_processor,args.debug)#keys,args, data_dir,debug
            
            val_dataset = ESDataset_m(val_keys, val_ground_true, graph, allgraph, res_level,args,  args.data_path,image_processor,args.debug)
            test_dataset = ESDataset_m(test_keys, test_ground_true_scaler , graph, allgraph, res_level,args,  args.data_path,image_processor,args.debug) 
            # return
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.ngpu > 1 else None
            val_sampler = SequentialDistributedSampler(val_dataset,args.batch_size) if args.ngpu > 1 else None
            test_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu > 1 else None
        #    use sampler to balance the training data or not 
            if args.sampler:
                train_sampler = PDBBalancedSampler(
                    keys=train_keys,
                    batch_size=args.batch_size,
                    shuffle=True,
                    random_state= args.seed   
                )
                # train_sampler = PDBEpochSampler(train_keys, batch_size=args.batch_size, shuffle=True,random_state=42 )
                train_dataloader = DataLoaderX(train_dataset, args.batch_size, sampler = train_sampler,\
                    shuffle=False, num_workers = args.num_workers, collate_fn=train_dataset.collate,pin_memory=True,prefetch_factor = 4)
            else:
                train_dataloader = DataLoaderX(train_dataset, args.batch_size, sampler = train_sampler,\
                    shuffle=False, num_workers = args.num_workers, collate_fn=train_dataset.collate,pin_memory=True,prefetch_factor = 4)
            val_dataloader = DataLoaderX(val_dataset, args.batch_size, sampler=val_sampler,\
                shuffle=False, num_workers = args.num_workers, collate_fn=val_dataset.collate,pin_memory=True,prefetch_factor = 4)
            test_dataloader = DataLoaderX(test_dataset, args.batch_size, sampler=test_sampler,\
                shuffle=False, num_workers = args.num_workers, collate_fn=test_dataset.collate,pin_memory=True,prefetch_factor = 4) 

            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,pct_start=args.pct_start,\
                steps_per_epoch=len(train_dataloader), epochs=args.epoch,last_epoch = -1 if len(train_dataloader)*epoch_start == 0 else len(train_dataloader)*epoch_start )
            
            #loss function ,in this paper just use cross entropy loss but you can try focal loss too!
            if args.loss_fn == 'bce_loss':
                loss_fn = nn.BCELoss().to(args.device,non_blocking=True)# 
            elif args.loss_fn == 'focal_loss':
                loss_fn = FocalLoss().to(args.device,non_blocking=True)
            elif args.loss_fn == 'cross_entry':
                loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smothing).to(args.device,non_blocking=True)
            elif args.loss_fn == 'mse_loss':
                loss_fn = nn.MSELoss().to(args.device,non_blocking=True)
            elif args.loss_fn == 'poly_loss_ce':
                loss_fn = PolyLoss_CE(epsilon = args.eps).to(args.device,non_blocking=True)
            elif args.loss_fn == 'poly_loss_fl':
                loss_fn = PolyLoss_FL(epsilon=args.eps,gamma = 2.0).to(args.device,non_blocking=True)
            else:
                raise ValueError('not support this loss : %s'%args.loss_fn)
            if args.onlytest:
                best_name = f"workdir/M2-PRNet/2025-11-18-02-14-42/fold_{i}.pt"# 301 - 1200
                checkpoint = torch.load(best_name)
                model.load_state_dict(checkpoint['model'])
                if args.useMultiModel:
                    test_losses,test_true,test_pred,keys,conloss,test_atom_loss,test_res_loss = testAndPrint_aff_m(model,test_dataloader,loss_fn,args,None)
                elif args.model == 'IPA' or args.model == 'GVP':
                    test_losses,test_true,test_pred,keys,conloss = testAndPrint_IPA(model,test_dataloader,loss_fn,args,None)
                else:
                    test_losses,test_true,test_pred,keys,conloss = testAndPrint_aff_m(model,test_dataloader,loss_fn,args,None)
                test_true = np.array(test_true).ravel()
                test_pred = np.array(test_pred).ravel()
                rmse = root_mean_squared_error(test_true, test_pred)
                mae = mean_absolute_error(test_true, test_pred)
                test_r_p = pearsonr(test_true, test_pred)[0]
                test_r_s = spearmanr(test_true, test_pred)[0]
                test_losses = torch.mean(torch.stack(test_losses), dim=0) 
                r2 = r2_score(test_true, test_pred)
                
                print('{}\t rmse {:.7f}\t pcc {:.7f}\t spcc {:.7f}\t mae {:.7f}\n'.format(
                        i, rmse, test_r_p, test_r_s, mae))
                avgpcc += test_r_p 
                avgspcc += test_r_s
                avg_r2 += r2
                avg_rmse += rmse
                avg_mae += mae
                continue
                # return
            # from vgae import VGAE
            # vgae = VGAE(in_feats, hidden_size, latent_size)
            # for epoch in range(100):
            #     for i_batch, (g,Y,key,batch) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
            #         optimizer = torch.optim.Adam(vgae.parameters(), lr=0.01) 
            # return 
            best_loss = 1e8
            best_f1 = -1
            counter = 0
            best_pcc = 0.0
            best_spcc = 0.0
            best_rmse = 1e8
            best_r2 = -1
            best_mae = 1e8
            args.steps = 0
            args.best_loss = 1e8
            args.best_pcc = 0.0
            args.best_spcc = 0.0
            args.best_rmse = 1e8
            args.best_r2 = -1
            args.best_mae = 1e8
            args.test_dataloader = test_dataloader
            args.val_dataloader = val_dataloader
            train_atom_loss,train_res_loss = 0.0,0.0
            for epoch in range(epoch_start,num_epochs):
                st = time.time()
                #collect losses of each iteration
                if args.ngpu > 1:
                    train_sampler.set_epoch(epoch) 
                if args.useMultiModel:
                    model,train_losses,optimizer,scheduler,train_atom_loss,train_res_loss = train_m(model,args,optimizer,loss_fn,train_dataloader,scheduler)
                elif args.model == 'IPA' or args.model == 'GVP':
                    model,train_losses,optimizer,scheduler = train_IPA(model,args,optimizer,loss_fn,train_dataloader,scheduler)
                else:
                    model,train_losses,optimizer,scheduler = train_m(model,args,optimizer,loss_fn,train_dataloader,scheduler)
                if args.ngpu > 1:
                    dist.barrier() 
                val_losses,val_true,val_pred,keys= evaluator_m(model,val_dataloader,loss_fn,args,val_sampler)
                # val_losses = 0.0
                if args.ngpu > 1:
                    dist.barrier() 
                if local_rank == 0:
                    test_losses = 0.0
                    test_atom_loss,test_res_loss =0.0,0.0
                    # val_losses = 0.0
                    train_losses = torch.mean(torch.tensor(train_losses,dtype=torch.float)).data.cpu().numpy()
                    val_losses = torch.mean(torch.tensor(val_losses,dtype=torch.float)).data.cpu().numpy()
                    if epoch%1==0:
                        
                        if args.useMultiModel:
                            test_losses,test_true,test_pred,keys,conloss,test_atom_loss,test_res_loss = testAndPrint_aff_m(model,args.test_dataloader,loss_fn,args,None)
                        elif args.model == 'IPA' or args.model == 'GVP':
                            test_losses,test_true,test_pred,keys,conloss = testAndPrint_IPA(model,args.test_dataloader,loss_fn,args,None)
                        else:
                            test_losses,test_true,test_pred,keys,conloss = testAndPrint_aff_m(model,args.test_dataloader,loss_fn,args,None)
                        
                        test_true = np.array(test_true).ravel()
                        test_pred = np.array(test_pred).ravel()
                        # test_true = scaler.inverse_transform(test_true).astype(np.float32).ravel()
                        # test_pred = scaler.inverse_transform(test_pred).astype(np.float32).ravel()
                        # print(test_pred.shape,test_true.shape)
                        
                        # test_true = safe_inverse_negative_log(test_true)
                        # test_pred = safe_inverse_negative_log(test_pred)
                        # print(test_true,test_pred)
                        test_r_p = pearsonr(test_true, test_pred)[0]
                        test_r_s = spearmanr(test_true, test_pred)[0]
                        rmse = root_mean_squared_error(test_true, test_pred)
                        mae = mean_absolute_error(test_true, test_pred)  
                        r2 = r2_score(test_true, test_pred)
                        test_losses = torch.mean(torch.stack(test_losses), dim=0) 
                        if np.isnan(test_r_p):
                            print('pcc is nan!')
                            print(test_true,test_pred)
                    if args.loss_fn == 'mse_loss':
                        end = time.time()
                        with open(log_path,'a') as f:
                            f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses)  + '\t' + str(end-st)+'\t'+ f'test_Pearson R: {test_r_p:.7f}'+'\t'+f'test_Spearman R: {test_r_s:.7f}'+'\t'+f'test_RMSE: {rmse:.7f}'+'\t'+f'MAE: {mae:.7f}'+'\t'+f'R2: {r2:.7f}'+'\t'+f'conloss:: {conloss:.7f}'
                                    +f'train_atom_loss::{train_atom_loss:.7f}'+'\t' +f'train_res_loss::{train_res_loss:.7f}'+f'test_atom_loss::{test_atom_loss:.7f}'+'\t' +f'test_res_loss::{test_res_loss:.7f}'+'\n')
                            f.close()
                    else:
                        test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(val_true,val_pred)
                        end = time.time()
                        with open(log_path,'a') as f:
                            f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses)\
                            + '\t'+str(test_auroc)+ '\t'+str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'\
                            + str(end-st)+ '\n')
                            f.close()
                    counter +=1 
                    if  val_losses < best_loss:
                        # best_loss = mae - test_r_p
                        best_loss = val_losses
                        
                        best_pcc = test_r_p
                        best_spcc = test_r_s
                        best_r2 = r2
                        best_rmse= rmse
                        best_mae = mae
                        counter = 0
                        save_model(model,optimizer,args,epoch,save_path,cv=i,mode = 'best')

                    if counter > args.patience:
                        save_model(model,optimizer,args,epoch,save_path,cv=i,mode = 'early_stop')
                        print('model early stop !')
                        avgpcc += best_pcc
                        avgspcc += best_spcc
                        avg_r2 += best_r2
                        avg_rmse += best_rmse
                        avg_mae += best_mae
                        break
                    if epoch == num_epochs-1:
                        save_model(model,optimizer,args,epoch,save_path,cv=i,mode = 'end')
                        avgpcc += best_pcc
                        avgspcc += best_spcc
                        avg_r2 += best_r2
                        avg_rmse += best_rmse
                        avg_mae += best_mae
                        print('This is the end of training !')

                if args.ngpu > 1:
                    dist.barrier() 
            if args.ngpu > 1:
                dist.barrier() 
            args.avgpcc +=args.best_pcc
            
            args.avgspcc += args.best_spcc
            args.avgr2 += args.best_r2
            args_avgrmse += args.best_rmse
            args.avgmae += args.best_mae
      
    print('training done!')
    print(f'avg save by step pcc is {args.avgpcc /5.0} spcc is { args.avgspcc/5.0} r2 is {args.avgr2/5.0} rmse is {args_avgrmse/5.0} mae is {args.avgmae/5.0}')
    print(f'avg pcc is {avgpcc/5.0} spcc is {avgspcc/5.0} r2 is {avg_r2/5.0} rmse is {avg_rmse/5.0} mae is {avg_mae/5.0}')
    
if '__main__' == __name__:
    '''distribution training'''
    from torch import distributed as dist
    import torch.multiprocessing as mp
    from utils.dist_utils import *
    from utils.parsing import parse_train_args
    # get args from parsering function
    args = parse_train_args()
    # set gpu to use
    # if args.ngpu>0:
    #     cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=8000, sample=3, nitro_restriction=False, verbose=True)
    #     if cmd[-1] == ',':
    #         os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
    #     else:
    #         os.environ['CUDA_VISIBLE_DEVICES']=cmd
    os.environ["MASTER_ADDR"] = args.MASTER_ADDR
    os.environ["MASTER_PORT"] = args.MASTER_PORT
    run(0,args)
    
    exit()
    
    # from torch.multiprocessing import Process
    # world_size = args.ngpu

    # # use multiprocess to train
    # processes = []
    # for rank in range(world_size):
    #     p = Process(target=run, args=(rank, args))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()


