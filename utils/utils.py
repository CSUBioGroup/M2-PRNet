import numpy as np
import torch
import os
import sys
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from utils.gradcam import FrozenGradCAM
from utils.gradcam_vis import save_cam, save_cam_better, save_cam_with_colorbar,save_cam_gray_overlay,save_cam_paper_style,save_cam_gray_overlay_clean,save_cam_gray_overlay_paper
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from torch import distributed as dist
import os.path
import time
import torch.nn as nn
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from collections import defaultdict
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from sklearn.metrics import accuracy_score,auc,balanced_accuracy_score
from sklearn.metrics import recall_score,precision_score,precision_recall_curve
from sklearn.metrics import confusion_matrix,f1_score
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from model.equiscore import conLoss

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())                            
# from dataset.dataset import ESDataset,DTISampler
from utils.dist_utils import *
N_atom_features = 28
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import dgl
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
def train_contrastive_gan(model, args, optimizer, loss_fn, train_dataloader, scheduler):
    # collect losses of each iteration
    train_losses = [] 
    loss_logits = []
    loss_conts = []
    gan_losses = []
    epoch_start = 0
    discriminator = model.discriminator
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    d_scheduler = torch.optim.lr_scheduler.OneCycleLR(d_optimizer, max_lr=args.max_lr,pct_start=args.pct_start,\
             steps_per_epoch=len(train_dataloader), epochs=args.epoch,last_epoch = -1 if len(train_dataloader)*epoch_start == 0 else len(train_dataloader)*epoch_start )
    model.train()
    discriminator.train()
    # alpha=1.0
    for i_batch, (g_batch, full_g_batch, Y) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        d_optimizer.zero_grad()
        model.zero_grad()
        discriminator.zero_grad()

        g_batch = g_batch.to(args.device, non_blocking=True)
        full_g_batch = full_g_batch.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        Y = Y.unsqueeze(-1)

        # 模型前向传播，得到所有样本的嵌入
        embeddings, logits = model(g_batch, full_g_batch, contrastive=True)

        # 计算logit损失
        loss_logit = loss_fn(logits, Y)
        # PCC损失
        pcc_loss_value = pcc_loss(logits, Y)
        # SPCC损失
        spcc_loss_value = spcc_loss(logits, Y)

        # 拆分输出
        batch_size = args.batch_size
        pos1, pos2 = embeddings[0:batch_size], embeddings[batch_size:2*batch_size]
        neg1, neg2 = embeddings[2*batch_size:3*batch_size], embeddings[3*batch_size:]

        # 计算对比学习损失
        pos_sim = F.cosine_similarity(pos1, pos2)
        neg_sim = F.cosine_similarity(pos1, neg1)
        loss_cont = contrastive_loss(pos_sim, neg_sim)

        # GAN判别器损失
        real_pairs = torch.cat([pos1, pos2], dim=1)
        fake_pairs = torch.cat([pos1, neg1], dim=1)
        real_labels = torch.ones(batch_size, 1).to(args.device)
        fake_labels = torch.zeros(batch_size, 1).to(args.device)

        real_output = discriminator(real_pairs.detach())
        fake_output = discriminator(fake_pairs.detach())

        d_loss_real = F.binary_cross_entropy(real_output, real_labels)
        d_loss_fake = F.binary_cross_entropy(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # 更新判别器
        # d_loss.backward(retain_graph=True)
        # d_optimizer.step()

        # 生成器损失（对比学习损失 + GAN损失）
        g_loss_fake = F.binary_cross_entropy(fake_output, real_labels)
        alpha = 0.0
        loss1 = alpha * loss_cont + (1 - alpha) * loss_logit
        loss = args.mse_weight * loss1 - args.pcc_weight * pcc_loss_value - args.spcc_weight * spcc_loss_value + args.gan_weight * d_loss

        # 反向传播并更新模型参数
        # 反向传播并更新模型参数
        loss.backward()
        optimizer.step()

        # 打印损失
        train_losses.append(loss.item())
        loss_conts.append(loss_cont.item())
        loss_logits.append(loss_logit.item())
        gan_losses.append(d_loss.item())
        if args.lr_decay:
            scheduler.step()
            # d_scheduler.step()
        torch.cuda.empty_cache()
    return model, train_losses, loss_conts, loss_logits, optimizer, scheduler, gan_losses
def get_args_from_json(json_file_path, args_dict):
    """"
    docstring:
        use this function to update the args_dict from a json file if you want to use a json file save parameters 
    input:
        json_file_path: string
            json file path
        args_dict: args dict
            dict

    output:
        args dict
    """

    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict

def initialize_model(model, device, args,load_save_file = False,init_classifer = True):
    """ initialize the model parameters or load the model from a saved file"""
    for param in model.parameters():
        if param.dim() == 1:
            continue
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
            

    if load_save_file:
        state_dict = torch.load(load_save_file,map_location = 'cpu')
        model_dict = state_dict['model']
        model_state_dict = model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if k in model_state_dict}
        model_state_dict.update(model_dict)
        model.load_state_dict(model_state_dict) 
        
        optimizer =state_dict['optimizer']
        epoch = state_dict['epoch']
        print('load save model!')
    if device:
        model = model.to(device)
    elif torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
      
        model = model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                     device_ids=[args.local_rank], 
                                                     output_device=args.local_rank, 
                                                     find_unused_parameters=True, 
                                                     broadcast_buffers=False)
        if load_save_file:
            return model ,optimizer,epoch
        return model
    # model.to(args.local_rank)
    if load_save_file:
        return model ,optimizer,epoch
    return model

def get_logauc(fp, tp, min_fp=0.001, adjusted=False):
    """"
    docstring:
        use this function to calculate logauc 
    input:
        fp: list
            false positive
        tp: list
            true positive

    output: float
        logauc
    """
    
    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if (lam_index != 0):
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    dy = (y[1:] - y[:-1])
    with np.errstate(divide='ignore'):
        intercept = y[1:] - x[1:] * (dy / (x[1:] - x[:-1]))
        intercept[np.isinf(intercept)] = 0.
    norm = np.log10(1. / float(min_fp))
    areas = ((dy / np.log(10.)) + intercept * np.log10(x[1:] / x[:-1])) / norm
    logauc = np.sum(areas)
    if adjusted:
        logauc -= 0.144620062  # random curve logAUC
    return logauc

def get_metrics(train_true,train_pred):
    # lr_decay
    """"
    docstring:
        calculate the metrics for the dataset
    input:
        train_true: list
            label
        train_pred: list
            predicted label

    output: list
        metrics
    """
    try:
        train_pred = np.concatenate(np.array(train_pred,dtype=object), 0).astype(np.float)
        train_true = np.concatenate(np.array(train_true,dtype=object), 0).astype(np.float)
    except:
        pass
    train_pred_label = np.where(train_pred > 0.5,1,0).astype(np.float)

    tn, fp, fn, tp = confusion_matrix(train_true,train_pred_label).ravel()
    train_auroc = roc_auc_score(train_true, train_pred) 
    train_acc = accuracy_score(train_true,train_pred_label)
    train_precision = precision_score(train_true,train_pred_label)
    train_sensitity = tp/(tp + fn)
    train_specifity = tn/(fp+tn)
    ps,rs,_ = precision_recall_curve(train_true,train_pred)
    train_auprc = auc(rs,ps)
    train_f1 = f1_score(train_true,train_pred_label)
    train_balanced_acc = balanced_accuracy_score(train_true,train_pred_label)
    fp,tp,_ = roc_curve(train_true,train_pred)
    train_adjusted_logauroc = get_logauc(fp,tp)
    # BEDROC
    sort_ind = np.argsort(train_pred)[::-1] # Descending order
    BEDROC = CalcBEDROC(train_true[sort_ind].reshape(-1, 1), 0, alpha = 80.5)
    return train_auroc,BEDROC,train_adjusted_logauroc,train_auprc,train_balanced_acc,train_acc,train_precision,train_sensitity,train_specifity,train_f1


def random_split(train_keys, split_ratio=0.9, seed=0, shuffle=True):
    """
    docstring:
        split the dataset into train and validation set by random sampling, this function not useful for new target protein prediction
    """
    
    
    dataset_size = len(train_keys)
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return [train_keys[i] for i in train_idx], [train_keys[i] for i in valid_idx]

def evaluator(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in enumerate(loader):
            
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred = model(g,full_g)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def evaluator_aff(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in enumerate(loader):
            
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred = model(g,full_g)
            mse_loss = loss_fn(pred, Y)
            # print(mse_loss)
            # PCC损失
            pcc_loss_value = pcc_loss(pred, Y)

            # SPCC损失
            spcc_loss_value = spcc_loss(pred, Y)

            # 组合损失 (可以根据需要调整权重)
            loss = args.mse_weight * mse_loss 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def evaluator_aff_equi(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in enumerate(loader):
            batch = g.batch_num_nodes()
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
            
            pred = model(g,batch)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def testAndPrint(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred = model(g,full_g)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def testAndPrint_aff_m(model,loader,loss_fn,args,test_sampler): 
    all_records = [] 
    model.eval() 
    import os
    if not os.path.exists(args.out_cam_dir): 
        os.makedirs(args.out_cam_dir) 
    
    test_losses,test_true,test_pred,keys = [], [],[],[] 
    fused_feats = []
    with torch.no_grad(): 
        if args.image_network: 
        # ======= 初始化 Grad-CAM 三个视图 ======= 
            cam_axial = FrozenGradCAM(model.image_network.axial_backbone, "stages_0") 
            # cam_coronal = GradCAM(model.image_network.coronal_backbone, "stages_3") 
            # cam_sagittal = GradCAM(model.image_network.sagittal_backbone, "stages_3") 
        for i_batch, (g,full_g,Y,key,pro_feat, rna_feat, mol_indicator, chain_indicator, pro_coords,rna_coords, prot_emb, 
                        rna_emb, neighbor_matrix_padded, prot_len, rna_len, mask,prot_whole_emb, rna_whole_emb,front_batch, side_batch, top_batch) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            model.zero_grad() 

            B = front_batch.shape[0]
            # for b in range(B):
            #     base_name = key[b]
            #     save_cam_better(front_batch[b], front_batch[b], f"{args.out_cam_dir}/{base_name}_front_cam.png")
            #     save_cam_better(side_batch[b],  side_batch[b],  f"{args.out_cam_dir}/{base_name}_side_cam.png")
            #     save_cam_better(top_batch[b],   top_batch[b],   f"{args.out_cam_dir}/{base_name}_top_cam.png")
            g = g.to(args.device,non_blocking=True) 
            full_g = full_g.to(args.device,non_blocking=True) 
            pro_feat = pro_feat.to(args.device,non_blocking=True) 
            rna_feat = rna_feat.to(args.device,non_blocking=True) 
            mol_indicator = mol_indicator.to(args.device,non_blocking=True) 
            chain_indicator = chain_indicator.to(args.device,non_blocking=True) 
            pro_coords = pro_coords.to(args.device,non_blocking=True) 
            rna_coords = rna_coords.to(args.device,non_blocking=True) 
            prot_emb = prot_emb.to(args.device,non_blocking=True) 
            rna_emb = rna_emb.to(args.device,non_blocking=True) 
            neighbor_matrix_padded = neighbor_matrix_padded.to(args.device,non_blocking=True) 
            mask = mask.to(args.device,non_blocking=True) 
            # prot_whole_emb = prot_whole_emb.to(args.device,non_blocking=True) 
            # # rna_whole_emb = rna_whole_emb.to(args.device,non_blocking=True) 
            front_batch, side_batch, top_batch = front_batch.to(args.device,non_blocking=True), side_batch.to(args.device,non_blocking=True), top_batch.to(args.device,non_blocking=True) 
            
            Y = Y.to(args.device,non_blocking=True) 
            Y = Y.unsqueeze(-1) 
            if args.useMultiModel: 
                atom_pre, res_pre, pred,score_matrix,conloss,fused_feat, base_att, attn_info = model(g,full_g, rna_feat, rna_emb, rna_coords, pro_feat,prot_emb,pro_coords, mol_indicator, chain_indicator, neighbor_matrix_padded, mask,prot_whole_emb, rna_whole_emb,front_batch, side_batch, top_batch,getScore=True) 
            else: 
                pred,score_matrix,conloss,fused_feat, base_att, attn_info  = model(g,full_g, rna_feat, rna_emb, rna_coords, pro_feat,prot_emb,pro_coords, mol_indicator, chain_indicator, neighbor_matrix_padded, mask,prot_whole_emb, rna_whole_emb ,front_batch, side_batch, top_batch,getScore=True) 
                # print(pred.shape) loss = loss_fn(pred ,Y) 
                atom_loss = loss_fn(atom_pre ,Y) if args.useMultiModel else 0 
                res_loss = loss_fn(res_pre ,Y) if args.useMultiModel else 0 
                # pred = res_pre # loss = loss + 0.3*atom_loss + 0.3*res_loss + args.con_weight*conloss 
            loss = loss_fn(pred ,Y) 
            if args.ngpu > 1: 
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM) 
                loss /= float(dist.get_world_size()) # get all loss value 
                # collect loss, true label and predicted label 
            test_losses.append(loss.data) 
            if args.ngpu > 1: 
                test_true.append(Y.data) 
            else: 
                test_true.append(Y.data) 
            keys.extend(key)
            
            # 收集融合特征
            if fused_feat is not None:
                if isinstance(fused_feat, torch.Tensor):
                    fused_feats.append(fused_feat.detach().cpu().numpy())
                else:
                    fused_feats.append(fused_feat) 
            if pred.shape[1]==2: 
                pred = torch.softmax(pred,dim = -1)[:,1] 
            pred = pred if args.loss_fn == 'auc_loss' else pred 
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data) 
            pred_cpu = pred.detach().cpu().view(-1)
            Y_cpu = Y.detach().cpu().view(-1)
            # print(f'key:{key}  pred:{pred_cpu}  true:{Y_cpu} ')
            if args.image_network and False : 
                B = front_batch.shape[0] 
                heat_front = cam_axial(front_batch) 
                # [1,1,H,W] 
                heat_side = cam_axial(side_batch) 
                heat_top = cam_axial(top_batch) 
                for b in range(B): 
                # # print(b) 
                    base_name = key[b] 
                    save_cam_gray_overlay_paper(front_batch[b], heat_front[b], f"{args.out_cam_dir}/{base_name}_front_cam.png") 
                    save_cam_gray_overlay_paper(side_batch[b], heat_side[b], f"{args.out_cam_dir}/{base_name}_side_cam.png") 
                    save_cam_gray_overlay_paper(top_batch[b], heat_top[b], f"{args.out_cam_dir}/{base_name}_top_cam.png") 
                for b in range(len(key)):
                    all_records.append({
                        "key": key[b][0:4],
                        "pred": float(pred_cpu[b]),
                        "true": float(Y_cpu[b]),
                        "abs_error": abs(float(pred_cpu[b]) - float(Y_cpu[b])),
                        # 如果你想保存特征（可选）
                        # "fused_feat": fused_feat[b].detach().cpu()
                    })
            if args.draw:
                sample_keys = list(key) if isinstance(key, (list, tuple)) else [key]

                src_all, dst_all = g.edges()
                src_all = src_all.detach().cpu()
                dst_all = dst_all.detach().cpu()

                # 原有 score_matrix：通常是 g structural edges 上 bias 后的 score/logit
                mean_scores_all = score_matrix.sum(dim=1).squeeze(-1).detach().cpu().view(-1)

                node_types_all = g.ndata['type'].detach().cpu().numpy()
                node_bases_all = g.ndata['bases'].detach().cpu().numpy()
                node_base_idx_all = g.ndata['base_idx'].detach().cpu().numpy()
                node_atom_types_all = g.ndata['atom_type'].detach().cpu().numpy()
                edge_types_all = g.edata['type'].detach().cpu().view(-1).numpy()

                batch_num_nodes = g.batch_num_nodes().tolist() if hasattr(g, 'batch_num_nodes') else [g.num_nodes()]
                batch_num_edges = g.batch_num_edges().tolist() if hasattr(g, 'batch_num_edges') else [g.num_edges()]

                # -----------------------------
                # Optional tensors from attn_info
                # -----------------------------
                def _to_cpu_flat(name):
                    if 'attn_info' not in globals() or attn_info is None or name not in attn_info:
                        return None
                    x = attn_info[name]
                    try:
                        return x.detach().cpu()
                    except Exception:
                        return None

                g_geo_pre_all = _to_cpu_flat('g_score_geo_pre_bias')
                g_bias_delta_all = _to_cpu_flat('g_edge_bias_delta')
                g_proj_e_all = _to_cpu_flat('g_proj_e_bias')
                g_score_after_bias_all = _to_cpu_flat('g_score_after_edge_bias')
                g_final_softmax_all = _to_cpu_flat('g_score_final_softmax_on_full_g')
                g_e_out_all = _to_cpu_flat('g_e_out_from_geo_score')
                g_length_all = _to_cpu_flat('g_edge_length')

                full_src_all = _to_cpu_flat('full_src')
                full_dst_all = _to_cpu_flat('full_dst')
                full_length_all = _to_cpu_flat('full_edge_length')
                full_geo_pre_all = _to_cpu_flat('full_score_geo_pre_bias')
                full_after_bias_all = _to_cpu_flat('full_score_after_struct_bias')
                full_final_all = _to_cpu_flat('full_score_final_softmax')
                full_edge_type_all = _to_cpu_flat('full_edge_type')

                base_att_tensor = None
                if base_att is not None:
                    import torch as _torch
                    if isinstance(base_att, (list, tuple)):
                        att_stack = _torch.stack(base_att, dim=0)
                        base_att_tensor = att_stack.mean(dim=0).mean(dim=1)
                    else:
                        att_tensor = _torch.as_tensor(base_att)
                        if att_tensor.dim() == 5:
                            base_att_tensor = att_tensor.mean(dim=0).mean(dim=1)
                        elif att_tensor.dim() == 4:
                            base_att_tensor = att_tensor.mean(dim=1)
                        else:
                            base_att_tensor = att_tensor

                node_offset = 0
                edge_offset = 0
                full_edge_offset = 0

                full_batch_num_edges = None
                if full_src_all is not None and hasattr(full_g, 'batch_num_edges'):
                    full_batch_num_edges = full_g.batch_num_edges().tolist()
                elif full_src_all is not None:
                    full_batch_num_edges = [len(full_src_all)]

                for b, sample_key in enumerate(sample_keys):
                    safe_key = str(sample_key).replace('/', '_').replace('\\', '_')
                    n_nodes = int(batch_num_nodes[b]) if b < len(batch_num_nodes) else int(batch_num_nodes[-1])
                    n_edges = int(batch_num_edges[b]) if b < len(batch_num_edges) else int(batch_num_edges[-1])

                    node_slice = slice(node_offset, node_offset + n_nodes)
                    edge_slice = slice(edge_offset, edge_offset + n_edges)

                    node_types = node_types_all[node_slice]
                    node_bases = node_bases_all[node_slice]
                    node_base_idx = node_base_idx_all[node_slice]
                    node_atom_types = node_atom_types_all[node_slice]
                    src = src_all[edge_slice] - node_offset
                    dst = dst_all[edge_slice] - node_offset
                    mean_scores = mean_scores_all[edge_slice]
                    edge_types = edge_types_all[edge_slice]

                    # coords fallback
                    coords = None
                    coord_name = None
                    for cname in ("coors", "coords", "coord", "pos", "xyz", "atom_pos"):
                        if cname in g.ndata:
                            try:
                                coords = g.ndata[cname].detach().cpu().numpy()[node_slice]
                                coord_name = cname
                            except Exception:
                                coords = None
                                coord_name = None
                            break

                    if coords is None:
                        print(f"[EdgeLength] Warning: no coordinate field found in g.ndata for {safe_key}.")
                    else:
                        print(f"[EdgeLength] using g.ndata['{coord_name}'] to compute edge lengths for {safe_key}")

                    def _edge_tensor_scalar(tensor_all, local_i, reducer='mean'):
                        if tensor_all is None:
                            return None
                        try:
                            x = tensor_all[edge_offset + local_i]
                            if x.numel() == 1:
                                return float(x.reshape(-1)[0].item())
                            if reducer == 'sum':
                                return float(x.float().sum().item())
                            if reducer == 'norm':
                                return float(torch.norm(x.float()).item())
                            return float(x.float().mean().item())
                        except Exception:
                            return None

                    edge_info = []
                    for i in range(len(src)):
                        s, d = int(src[i].item()), int(dst[i].item())

                        edge_length = _edge_tensor_scalar(g_length_all, i, reducer='mean')
                        if edge_length is None and coords is not None:
                            try:
                                edge_length = float(np.linalg.norm(coords[s] - coords[d]))
                            except Exception:
                                edge_length = None

                        geo_pre = _edge_tensor_scalar(g_geo_pre_all, i, reducer='mean')
                        bias_delta = _edge_tensor_scalar(g_bias_delta_all, i, reducer='mean')
                        proj_e_bias = _edge_tensor_scalar(g_proj_e_all, i, reducer='mean')
                        after_bias = _edge_tensor_scalar(g_score_after_bias_all, i, reducer='mean')
                        final_softmax = _edge_tensor_scalar(g_final_softmax_all, i, reducer='mean')
                        e_out_mean = _edge_tensor_scalar(g_e_out_all, i, reducer='mean')
                        e_out_norm = _edge_tensor_scalar(g_e_out_all, i, reducer='norm')

                        edge_info.append({
                            "source": s,
                            "dest": d,
                            "weight": float(mean_scores[i].item()),
                            "length": edge_length,
                            "geo_score_pre_bias": geo_pre,
                            "edge_bias_delta": bias_delta,
                            "proj_e_bias_mean": proj_e_bias,
                            "score_after_edge_bias": after_bias,
                            "final_softmax_on_full_g": final_softmax,
                            "e_out_mean": e_out_mean,
                            "e_out_norm": e_out_norm,
                            "src_type": int(node_types[s]),
                            "src_base_idx": int(node_base_idx[s]),
                            "dst_type": int(node_types[d]),
                            "dst_base_idx": int(node_base_idx[d]),
                            "src_base": int(node_bases[s]) if not isinstance(node_bases[s], str) else node_bases[s],
                            "dst_base": int(node_bases[d]) if not isinstance(node_bases[d], str) else node_bases[d],
                            "src_atom": int(node_atom_types[s]),
                            "dst_atom": int(node_atom_types[d]),
                            "edge_type": int(edge_types[i]) if not hasattr(edge_types[i], 'item') else int(edge_types[i].item())
                        })

                    # Main structural-edge CSV: this is the most important one for chemical/edge-feature analysis.
                    structural_cols = [
                        "source", "dest", "edge_type", "length",
                        "weight", "geo_score_pre_bias", "edge_bias_delta", "proj_e_bias_mean",
                        "score_after_edge_bias", "final_softmax_on_full_g",
                        "e_out_mean", "e_out_norm",
                        "src_type", "src_base_idx", "src_base", "src_atom",
                        "dst_type", "dst_base_idx", "dst_base", "dst_atom",
                    ]
                    structural_df = pd.DataFrame(edge_info, columns=structural_cols)
                    if not structural_df.empty:
                        structural_df = structural_df.sort_values("score_after_edge_bias", ascending=False, na_position='last')
                    structural_path = f"data/graph/{safe_key}_structural_edge_weight_components.csv"
                    structural_df.to_csv(structural_path, index=False)
                    print(f"[StructuralEdgeWeights] saved: {structural_path}")

                    # Keep backward-compatible name.
                    edge_path = f"data/graph/{safe_key}_edge_scores.csv"
                    structural_df.to_csv(edge_path, index=False)
                    print(f"[EdgeScores] saved: {edge_path}")

                    # full_g distance graph CSV: pure geometric/full-graph score side.
                    if full_src_all is not None and full_dst_all is not None and full_batch_num_edges is not None:
                        n_full_edges = int(full_batch_num_edges[b]) if b < len(full_batch_num_edges) else int(full_batch_num_edges[-1])
                        full_slice = slice(full_edge_offset, full_edge_offset + n_full_edges)
                        fsrc = full_src_all[full_slice].detach().cpu() - node_offset
                        fdst = full_dst_all[full_slice].detach().cpu() - node_offset

                        structural_pair_set = set((int(s.item()), int(d.item())) for s, d in zip(src, dst))

                        full_rows = []
                        for j in range(len(fsrc)):
                            fs, fd = int(fsrc[j].item()), int(fdst[j].item())
                            def _full_scalar(x_all, reducer='mean'):
                                if x_all is None:
                                    return None
                                try:
                                    x = x_all[full_edge_offset + j]
                                    if x.numel() == 1:
                                        return float(x.reshape(-1)[0].item())
                                    if reducer == 'norm':
                                        return float(torch.norm(x.float()).item())
                                    return float(x.float().mean().item())
                                except Exception:
                                    return None

                            full_et = _full_scalar(full_edge_type_all, reducer='mean')
                            if full_et is not None:
                                try:
                                    full_et = int(full_et)
                                except Exception:
                                    pass

                            full_rows.append({
                                "source": fs,
                                "dest": fd,
                                "length": _full_scalar(full_length_all, reducer='mean'),
                                "geo_score_pre_bias": _full_scalar(full_geo_pre_all, reducer='mean'),
                                "score_after_struct_bias": _full_scalar(full_after_bias_all, reducer='mean'),
                                "final_softmax": _full_scalar(full_final_all, reducer='mean'),
                                "is_structural_edge": (fs, fd) in structural_pair_set,
                                "edge_type": full_et,
                            })

                        full_df = pd.DataFrame(full_rows)
                        full_path = f"data/graph/{safe_key}_full_distance_edge_scores.csv"
                        full_df.to_csv(full_path, index=False)
                        print(f"[FullDistanceEdgeScores] saved: {full_path}")
                        full_edge_offset += n_full_edges

                    # Original aggregation, now based on structural edge score components.
                    pair_stats = defaultdict(lambda: {"sum_weight": 0.0, "count": 0, "max_weight": -1e9})
                    for e in edge_info:
                        # Use score_after_edge_bias first; fallback to weight.
                        ww = e.get("score_after_edge_bias", None)
                        if ww is None or pd.isna(ww):
                            ww = e["weight"]
                        e_weight = float(ww)

                        if e["src_type"] == 0 or e["dst_type"] == 0:
                            continue
                        if e["src_type"] == e["dst_type"]:
                            continue
                        src_key = (e["src_type"], e["src_base"], e["src_base_idx"])
                        dst_key = (e["dst_type"], e["dst_base"], e["dst_base_idx"])
                        if src_key == dst_key:
                            continue
                        pair_key = tuple(sorted([src_key, dst_key], key=lambda x: (x[0], x[2], str(x[1]))))
                        pair_stats[pair_key]["sum_weight"] += e_weight
                        pair_stats[pair_key]["count"] += 1
                        pair_stats[pair_key]["max_weight"] = max(pair_stats[pair_key]["max_weight"], e_weight)

                    pair_rows = []
                    for pair_key, stat in pair_stats.items():
                        a, bb = pair_key
                        pair_rows.append({
                            "src_type": a[0], "src_base": a[1], "src_base_idx": a[2],
                            "dst_type": bb[0], "dst_base": bb[1], "dst_base_idx": bb[2],
                            "sum_weight": stat["sum_weight"],
                            "mean_weight": stat["sum_weight"] / stat["count"] if stat["count"] > 0 else 0.0,
                            "max_weight": stat["max_weight"],
                            "edge_count": stat["count"],
                        })

                    pair_cols = ["src_type", "src_base", "src_base_idx", "dst_type", "dst_base", "dst_base_idx", "sum_weight", "mean_weight", "max_weight", "edge_count"]
                    pair_df = pd.DataFrame(pair_rows, columns=pair_cols)
                    if not pair_df.empty:
                        pair_df = pair_df.sort_values("mean_weight", ascending=False)
                        pair_df = pair_df.drop_duplicates(subset=["src_base_idx"], keep="first")
                    pair_path = f"data/graph/{safe_key}_base_pair_scores.csv"
                    pair_df.to_csv(pair_path, index=False)
                    print(f"[BasePair] saved: {pair_path}")

                    atom_stats = defaultdict(lambda: {"sum_weight": 0.0, "count": 0, "max_weight": -1e9})
                    bond_type_stats = defaultdict(lambda: {"sum_weight": 0.0, "count": 0, "max_weight": -1e9})
                    virtual_stats = {"sum_weight": 0.0, "count": 0, "max_weight": -1e9}
                    distance_stats = {"sum_dist": 0.0, "count": 0, "min_dist": 1e9, "max_dist": 0.0}

                    for e in edge_info:
                        ww = e.get("score_after_edge_bias", None)
                        if ww is None or pd.isna(ww):
                            ww = e["weight"]
                        w = float(ww)
                        s = int(e["source"])
                        d = int(e["dest"])
                        atom_stats[s]["sum_weight"] += w
                        atom_stats[s]["count"] += 1
                        atom_stats[s]["max_weight"] = max(atom_stats[s]["max_weight"], w)
                        atom_stats[d]["sum_weight"] += w
                        atom_stats[d]["count"] += 1
                        atom_stats[d]["max_weight"] = max(atom_stats[d]["max_weight"], w)

                        et = e.get("edge_type", None)
                        try:
                            et_key = int(et)
                        except Exception:
                            et_key = str(et)
                        bond_type_stats[et_key]["sum_weight"] += w
                        bond_type_stats[et_key]["count"] += 1
                        bond_type_stats[et_key]["max_weight"] = max(bond_type_stats[et_key]["max_weight"], w)

                        if int(e["src_type"]) == 0 or int(e["dst_type"]) == 0:
                            virtual_stats["sum_weight"] += w
                            virtual_stats["count"] += 1
                            virtual_stats["max_weight"] = max(virtual_stats["max_weight"], w)

                        if e.get("length", None) is not None:
                            try:
                                dist = float(e["length"])
                                distance_stats["sum_dist"] += dist
                                distance_stats["count"] += 1
                                distance_stats["min_dist"] = min(distance_stats["min_dist"], dist)
                                distance_stats["max_dist"] = max(distance_stats["max_dist"], dist)
                            except Exception:
                                pass

                    atom_rows = []
                    for node_id, st in atom_stats.items():
                        atom_rows.append({
                            "node_id": node_id,
                            "node_type": int(node_types[node_id]),
                            "base_idx": int(node_base_idx[node_id]),
                            "base": int(node_bases[node_id]) if not isinstance(node_bases[node_id], str) else node_bases[node_id],
                            "atom_type": int(node_atom_types[node_id]),
                            "sum_weight": st["sum_weight"],
                            "mean_weight": st["sum_weight"] / st["count"] if st["count"] > 0 else 0.0,
                            "max_weight": st["max_weight"],
                            "edge_count": st["count"],
                        })
                    atom_cols = ["node_id", "node_type", "base_idx", "base", "atom_type", "sum_weight", "mean_weight", "max_weight", "edge_count"]
                    atom_df = pd.DataFrame(atom_rows, columns=atom_cols)
                    if not atom_df.empty:
                        atom_df = atom_df.sort_values("mean_weight", ascending=False)
                    atom_path = f"data/graph/{safe_key}_atom_stats.csv"
                    atom_df.to_csv(atom_path, index=False)
                    print(f"[AtomStats] saved: {atom_path}")

                    bond_rows = []
                    for btype, st in bond_type_stats.items():
                        bond_rows.append({
                            "edge_type": btype,
                            "sum_weight": st["sum_weight"],
                            "mean_weight": st["sum_weight"] / st["count"] if st["count"] > 0 else 0.0,
                            "max_weight": st["max_weight"],
                            "edge_count": st["count"],
                        })
                    bond_cols = ["edge_type", "sum_weight", "mean_weight", "max_weight", "edge_count"]
                    bond_df = pd.DataFrame(bond_rows, columns=bond_cols)
                    if not bond_df.empty:
                        bond_df = bond_df.sort_values("mean_weight", ascending=False)
                    bond_path = f"data/graph/{safe_key}_bond_type_stats.csv"
                    bond_df.to_csv(bond_path, index=False)
                    print(f"[BondStats] saved: {bond_path}")

                    virtual_summary = {
                        "sum_weight": virtual_stats["sum_weight"],
                        "mean_weight": virtual_stats["sum_weight"] / virtual_stats["count"] if virtual_stats["count"] > 0 else 0.0,
                        "max_weight": virtual_stats["max_weight"],
                        "edge_count": virtual_stats["count"],
                    }
                    virtual_path = f"data/graph/{safe_key}_virtual_node_stats.csv"
                    pd.DataFrame([virtual_summary]).to_csv(virtual_path, index=False)
                    print(f"[VirtualStats] saved: {virtual_path}")

                    if distance_stats["count"] > 0:
                        dist_summary = {
                            "avg_dist": distance_stats["sum_dist"] / distance_stats["count"],
                            "min_dist": distance_stats["min_dist"],
                            "max_dist": distance_stats["max_dist"],
                            "pair_count": distance_stats["count"],
                        }
                        dist_path = f"data/graph/{safe_key}_distance_stats.csv"
                        pd.DataFrame([dist_summary]).to_csv(dist_path, index=False)
                        print(f"[DistanceStats] saved: {dist_path}")

                    base_att_rows = []
                    for r in pair_rows:
                        src_idx = int(r["src_base_idx"])
                        dst_idx = int(r["dst_base_idx"])
                        base_att_rows.append({
                            "src_base_idx": src_idx,
                            "dst_base_idx": dst_idx,
                            "src_base": r["src_base"],
                            "dst_base": r["dst_base"],
                            "sum_weight": r["sum_weight"],
                            "mean_weight": r["mean_weight"],
                            "max_weight": r["max_weight"],
                            "edge_count": r["edge_count"],
                            "is_sequential_neighbor": abs(src_idx - dst_idx) == 1,
                            "is_connected_atomic": r["edge_count"] > 0,
                        })
                    base_att_cols = ["src_base_idx", "dst_base_idx", "src_base", "dst_base", "sum_weight", "mean_weight", "max_weight", "edge_count", "is_sequential_neighbor", "is_connected_atomic"]
                    base_att_df = pd.DataFrame(base_att_rows, columns=base_att_cols)
                    if not base_att_df.empty:
                        base_att_df = base_att_df.sort_values("mean_weight", ascending=False)
                    base_att_path = f"data/graph/{safe_key}_base_attention_connectivity.csv"
                    base_att_df.to_csv(base_att_path, index=False)
                    print(f"[BaseAttention] saved: {base_att_path}")

                    try:
                        if base_att_tensor is not None:
                            att_mat = base_att_tensor[b] if hasattr(base_att_tensor, "dim") and base_att_tensor.dim() >= 3 else base_att_tensor
                            try:
                                rna_base_idxs = node_base_idx[node_types == 1]
                                n_rna = int(rna_base_idxs.max()) + 1 if len(rna_base_idxs) > 0 else att_mat.shape[0]
                            except Exception:
                                n_rna = att_mat.shape[0]

                            res_rows = []
                            for i in range(min(n_rna, att_mat.shape[0])):
                                for j in range(min(n_rna, att_mat.shape[1])):
                                    res_rows.append({
                                        "src_base_idx": int(i),
                                        "dst_base_idx": int(j),
                                        "attn_weight": float(att_mat[i, j].item()),
                                        "is_sequential_neighbor": abs(i - j) == 1,
                                    })
                            if res_rows:
                                res_df = pd.DataFrame(res_rows).sort_values("attn_weight", ascending=False)
                                res_path = f"data/graph/{safe_key}_residue_attention_matrix.csv"
                                res_df.to_csv(res_path, index=False)
                                print(f"[ResidueAttention] saved: {res_path}")
                    except Exception as _e:
                        print("Warning: failed to save residue-level attention:", _e)

                    node_offset += n_nodes
                    edge_offset += n_edges

            # if args.draw:
            #     # 节点属性提取
            #     num_nodes = g.num_nodes()
            #     src, dst = g.edges() 
            #     mean_scores = score_matrix.sum(dim=1).squeeze(-1).cpu()  # Shape: (num_edges,)
            #     # print(g.ndata)
            #     node_types = g.ndata['type'].cpu().numpy()  # 节点类型
            #     node_bases = g.ndata['bases'].cpu().numpy()  # 节点碱基类型
            #     node_base_idx = g.ndata['base_idx'].cpu().numpy()  # 碱基索引
            #     node_atom_types = g.ndata['atom_type'].cpu().numpy()  # 原子类型
            #     edge_types = g.edata['type']
            #     # 打印或存储边的详细信息
            #     edge_info = []
            #     for i in range(len(src)):
            #         s, d = src[i].item(), dst[i].item()
            #         edge_info.append({
            #             "source": s,
            #             "dest": d,
            #             "weight": mean_scores[i].item(),
            #             "src_type": node_types[s],
            #             "src_base_idx":node_base_idx[s],
            #             "dst_type": node_types[d],
            #             "dst_base_idx":node_base_idx[d],
            #             "src_base": node_bases[s],
            #             "dst_base": node_bases[d],
            #             "src_atom": node_atom_types[s],
            #             "dst_atom": node_atom_types[d],
            #             "edge_type": edge_types[i]
            #         })
            #     # 可选：存储为文件，供后续分析
            #     import pandas as pd
            #     edge_df = pd.DataFrame(edge_info)
            #     edge_df.to_csv(f"data/graph/{key[0]}.csv", index=False)
            #     import csv
            #     import os

            #     # 定义原子类型和边类型名称
            #     atom_categories = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'other']
            #     edge_type_names = {0: 'edge_type_0', 1: 'edge_type_1'}  # 边类型名称映射

            #     # 初始化统计字典（节点类别 + 边类型）
            #     stats = {category: {'sum': 0.0, 'count': 0} for category in ['virtual'] + atom_categories}
            #     edge_type_stats = {et: {'sum': 0.0, 'count': 0} for et in edge_type_names}  # 边类型统计

            #     for edge in edge_info:
            #         src_type = edge['src_type']
            #         dst_type = edge['dst_type']
            #         weight = edge['weight']
            #         et = edge['edge_type'].item()  # 边类型（0或1）
                    
            #         # 统计节点类别（virtual或原子类型）
            #         if src_type == 0 or dst_type == 0:
            #             stats['virtual']['sum'] += weight
            #             stats['virtual']['count'] += 1
            #         else:
            #             src_atom = atom_categories[edge['src_atom']]
            #             dst_atom = atom_categories[edge['dst_atom']]
            #             stats[src_atom]['sum'] += weight
            #             stats[src_atom]['count'] += 1
            #             stats[dst_atom]['sum'] += weight
            #             stats[dst_atom]['count'] += 1
                    
            #         # 统计边类型（无论是否连接到virtual节点）
            #         edge_type_stats[et]['sum'] += weight
            #         edge_type_stats[et]['count'] += 1

            #     # 计算平均值（节点类别）
            #     averages = {}
            #     for category in ['virtual'] + atom_categories:
            #         total = stats[category]['sum']
            #         count = stats[category]['count']
            #         averages[category] = total / count if count != 0 else 0.0

            #     # 计算平均值（边类型）
            #     for et in edge_type_names:
            #         total = edge_type_stats[et]['sum']
            #         count = edge_type_stats[et]['count']
            #         averages[edge_type_names[et]] = total / count if count != 0 else 0.0

            #     # 确保输出目录存在
            #     output_dir = 'data/graph'
            #     os.makedirs(output_dir, exist_ok=True)

            #     # 生成CSV文件（包含节点类别和边类型）
            #     output_path = os.path.join(output_dir, f'{key[0]}.csv')
            #     ordered_categories = ['virtual'] + atom_categories + list(edge_type_names.values())

            #     with open(output_path, 'w', newline='') as csvfile:
            #         writer = csv.writer(csvfile)
            #         writer.writerow(['category', 'average_weight'])
            #         for cat in ordered_categories:
            #             writer.writerow([cat, averages.get(cat, 0.0)])

            #     print(f"结果已保存至 {output_path}")
            
    # # 按误差从小到大排序
    # all_records = sorted(all_records, key=lambda x: x["abs_error"])

    # top100 = all_records[:30]
    # import pandas as pd

    # df = pd.DataFrame(top100)
    # csv_path = f"/root/private_data/wjk_work/CoPRA/datasets/PRA310/splits/MD_pre30_0.csv"
    # for i in range(8):
    #     if os.path.exists(csv_path):
    #         csv_path = f"/root/private_data/wjk_work/CoPRA/datasets/PRA310/splits/MD_pre30_{i}.csv"
    # df.to_csv(csv_path, index=False)
    # print(f"Saved top-40 samples to {csv_path}")
    # print(f"Top-40 best predictions (by |pred - GT|):")
    # for i, rec in enumerate(top100[:10]):
    #     print(f"{i+1:03d} | {rec['key']} | "
    #         f"pred={rec['pred']:.4f}, GT={rec['true']:.4f}, "
    #         f"|err|={rec['abs_error']:.4f}")
    # top_pdbs = sorted(set(r["key"][:4].upper() for r in top100))
    # csv_path = "/root/private_data/wjk_work/CoPRA/datasets/PRA310/splits/MD.csv"
    # df = pd.read_csv(csv_path)
    # df["PDB"] = df["PDB"].str.upper()

    # df_filtered = df[df["PDB"].isin(top_pdbs)].copy()
    # out_path = "/root/private_data/wjk_work/CoPRA/datasets/PRA310/splits/MD_top100_pdbs.csv"
    # out_path = f"/root/private_data/wjk_work/CoPRA/datasets/PRA310/splits/top30_best_predictions_0.csv"
    # for i in range(8):
    #     if os.path.exists(out_path):
    #         out_path = f"/root/private_data/wjk_work/CoPRA/datasets/PRA310/splits/top30_best_predictions_{i}.csv"
    #         continue
    #     df_filtered.to_csv(out_path, index=False)
    #     break
    # import numpy as np
    # from scipy.stats import pearsonr, spearmanr

    # # 提取 pred / GT
    # preds = np.array([r["pred"] for r in top100], dtype=np.float32)
    # trues = np.array([r["true"] for r in top100], dtype=np.float32)

    # # ===== 回归误差 =====
    # mae = np.mean(np.abs(preds - trues))
    # rmse = np.sqrt(np.mean((preds - trues) ** 2))

    # # ===== 相关性 =====
    # pcc, pcc_pval = pearsonr(preds, trues)
    # spcc, spcc_pval = spearmanr(preds, trues)

    # print("===== Top-100 Best Samples Metrics =====")
    # print(f"MAE   : {mae:.4f}")
    # print(f"RMSE  : {rmse:.4f}")
    # print(f"PCC   : {pcc:.4f}  (p={pcc_pval:.2e})")
    # print(f"SPCC  : {spcc:.4f} (p={spcc_pval:.2e})")
    # gather ngpu result to single tensor
    if args.ngpu > 1:
        test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                        len(test_sampler.dataset)).cpu().numpy()
        test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                        len(test_sampler.dataset)).cpu().numpy()
    
    else:
        test_true = torch.concat(test_true, dim=0).cpu().numpy()
        test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    
    # 合并所有 fused_feats
    if fused_feats and len(fused_feats) > 0:
        try:
            all_fused_feats = np.concatenate(fused_feats, axis=0)
        except:
            all_fused_feats = None
    else:
        all_fused_feats = None
    
    return test_losses,test_true,test_pred,keys,conloss,all_fused_feats
def evaluator_mtrain_m(
    model,
    args,
    optimizer,
    loss_fn,
    train_dataloader,
    scheduler
):
    all_records = [] 
    model.eval() 
    if not os.path.exists(args.out_cam_dir): 
        os.makedirs(args.out_cam_dir) 
   
    with torch.no_grad(): 
        test_losses,test_true,test_pred,keys = [], [],[],[] 
        for i_batch, (g,full_g,Y,key,pro_feat, rna_feat, mol_indicator, chain_indicator, pro_coords,rna_coords, prot_emb, 
                        rna_emb, neighbor_matrix_padded, prot_len, rna_len, mask,prot_whole_emb, rna_whole_emb,front_batch, side_batch, top_batch) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            model.zero_grad() 
            g = g.to(args.device,non_blocking=True) 
            full_g = full_g.to(args.device,non_blocking=True) 
            pro_feat = pro_feat.to(args.device,non_blocking=True) 
            rna_feat = rna_feat.to(args.device,non_blocking=True) 
            mol_indicator = mol_indicator.to(args.device,non_blocking=True) 
            chain_indicator = chain_indicator.to(args.device,non_blocking=True) 
            pro_coords = pro_coords.to(args.device,non_blocking=True) 
            rna_coords = rna_coords.to(args.device,non_blocking=True) 
            prot_emb = prot_emb.to(args.device,non_blocking=True) 
            rna_emb = rna_emb.to(args.device,non_blocking=True) 
            neighbor_matrix_padded = neighbor_matrix_padded.to(args.device,non_blocking=True) 
            mask = mask.to(args.device,non_blocking=True) 
            # prot_whole_emb = prot_whole_emb.to(args.device,non_blocking=True) 
            # # rna_whole_emb = rna_whole_emb.to(args.device,non_blocking=True) 
            front_batch, side_batch, top_batch = front_batch.to(args.device,non_blocking=True), side_batch.to(args.device,non_blocking=True), top_batch.to(args.device,non_blocking=True) 
            # if args.image_network and False: 
            #     B = front_batch.shape[0] 
            #     heat_front = cam_axial(front_batch) 
            #     heat_side = cam_axial(side_batch) 
            #     heat_top = cam_axial(top_batch) 
            #     for b in range(B): 
            #         # print(b) 
            #         base_name = key[b] 
            #         save_cam(front_batch[b], heat_front[b], f"{args.out_cam_dir}/{base_name}_front_cam.png") 
            #         save_cam(side_batch[b], heat_side[b], f"{args.out_cam_dir}/{base_name}_side_cam.png") 
            #         save_cam(top_batch[b], heat_top[b], f"{args.out_cam_dir}/{base_name}_top_cam.png") 
            Y = Y.to(args.device,non_blocking=True) 
            Y = Y.unsqueeze(-1) 
            if args.useMultiModel: 
                atom_pre, res_pre, pred,mean_scores,conloss,fused_feat, base_att = model(g,full_g, rna_feat, rna_emb, rna_coords, pro_feat,prot_emb,pro_coords, mol_indicator, chain_indicator, neighbor_matrix_padded, mask,prot_whole_emb, rna_whole_emb,front_batch, side_batch, top_batch,getScore=True) 
            else: 
                pred,mean_scores,conloss,fused_feat, base_att = model(g,full_g, rna_feat, rna_emb, rna_coords, pro_feat,prot_emb,pro_coords, mol_indicator, chain_indicator, neighbor_matrix_padded, mask,prot_whole_emb, rna_whole_emb ,front_batch, side_batch, top_batch,getScore=True) 
                # print(pred.shape) loss = loss_fn(pred ,Y) 
                atom_loss = loss_fn(atom_pre ,Y) if args.useMultiModel else 0 
                res_loss = loss_fn(res_pre ,Y) if args.useMultiModel else 0 
                # pred = res_pre # loss = loss + 0.3*atom_loss + 0.3*res_loss + args.con_weight*conloss 
                if args.ngpu > 1: 
                    dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM) 
                loss /= float(dist.get_world_size()) # get all loss value 
                # collect loss, true label and predicted label 
                test_losses.append(loss.data) 
                if args.ngpu > 1: 
                    test_true.append(Y.data) 
                else: 
                    test_true.append(Y.data) 
                keys.extend(key) 
                if pred.shape[1]==2: 
                    pred = torch.softmax(pred,dim = -1)[:,1] 
                pred = pred if args.loss_fn == 'auc_loss' else pred 
                test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data) 
    # gather ngpu result to single tensor
    if args.ngpu > 1:
        test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                        len(test_sampler.dataset)).cpu().numpy()
        test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                        len(test_sampler.dataset)).cpu().numpy()
    
    else:
        test_true = torch.concat(test_true, dim=0).cpu().numpy()
        test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def train_m(
    model,
    args,
    optimizer,
    loss_fn,
    train_dataloader,
    scheduler
):
    """
    Train loop for M-DDG model
    """

    # collect losses of each iteration
    train_losses = []

    model.train()

    train_atom_loss, train_res_loss = 0.0, 0.0

    for i_batch, (
        g,
        full_g,
        Y,
        key,
        pro_feat,
        rna_feat,
        mol_indicator,
        chain_indicator,
        pro_coords,
        rna_coords,
        prot_emb,
        rna_emb,
        neighbor_matrix_padded,
        prot_len,
        rna_le,
        mask,
        prot_whole_emb,
        rna_whole_emb,
        front_batch,
        side_batch,
        top_batch
    ) in tqdm.tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader)
    ):
        
        optimizer.zero_grad()
        model.zero_grad()

        # move graph & tensors to device
        g = g.to(args.device, non_blocking=True)
        full_g = full_g.to(args.device, non_blocking=True)

        pro_feat = pro_feat.to(args.device, non_blocking=True)
        rna_feat = rna_feat.to(args.device, non_blocking=True)

        mol_indicator = mol_indicator.to(args.device, non_blocking=True)
        chain_indicator = chain_indicator.to(args.device, non_blocking=True)

        pro_coords = pro_coords.to(args.device, non_blocking=True)
        rna_coords = rna_coords.to(args.device, non_blocking=True)

        prot_emb = prot_emb.to(args.device, non_blocking=True)
        rna_emb = rna_emb.to(args.device, non_blocking=True)

        neighbor_matrix_padded = neighbor_matrix_padded.to(
            args.device,
            non_blocking=True
        )

        mask = mask.to(args.device, non_blocking=True)

        if args.image_network:
            front_batch = front_batch.to(args.device, non_blocking=True)
            side_batch = side_batch.to(args.device, non_blocking=True)
            top_batch = top_batch.to(args.device, non_blocking=True)

        Y = Y.to(args.device, non_blocking=True)
        Y = Y.unsqueeze(-1)

        # Laplacian positional encoding sign flip
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(
                batch_lap_pos_enc.size(1),
                device=args.device
            )
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            g.ndata['lap_pos_enc'] = (
                batch_lap_pos_enc * sign_flip.unsqueeze(0)
            )

        # forward
        if args.useMultiModel:
            atom_pre, res_pre, logits, conloss,fused_feat = model(
                g,
                full_g,
                rna_feat,
                rna_emb,
                rna_coords,
                pro_feat,
                prot_emb,
                pro_coords,
                mol_indicator,
                chain_indicator,
                neighbor_matrix_padded,
                mask,
                prot_whole_emb,
                rna_whole_emb,
                front_batch,
                side_batch,
                top_batch
            )

            loss = loss_fn(logits, Y)

        else:
            logits, conloss, fused_feat = model(
                g,
                full_g,
                rna_feat,
                rna_emb,
                rna_coords,
                pro_feat,
                prot_emb,
                pro_coords,
                mol_indicator,
                chain_indicator,
                neighbor_matrix_padded,
                mask,
                prot_whole_emb,
                rna_whole_emb,
                front_batch,
                side_batch,
                top_batch
            )

            loss = loss_fn(logits, Y) + conloss

        # multi-task loss
        if args.useMultiModel:
            train_atom_loss = loss_fn(atom_pre, Y)
            train_res_loss = loss_fn(res_pre, Y)

            weights = F.softmax(model.loss_weights, dim=0)
            loss = (
                weights[0] * loss
                + weights[1] * train_atom_loss
                + weights[2] * train_res_loss
            )

        train_losses.append(loss.item())

        # gradient accumulation
        loss = loss / args.grad_sum
        loss.backward()

        args.steps += 1
        if args.steps % args.test_step == 0:
            test_losses,test_true,test_pred,keys,conloss,all_fused_feats = testAndPrint_aff_m(model,args.test_dataloader,loss_fn,args,None)
            test_true = np.array(test_true).ravel()
            test_pred = np.array(test_pred).ravel()
            test_r_p = pearsonr(test_true, test_pred)[0]
            test_r_s = spearmanr(test_true, test_pred)[0]
            rmse = root_mean_squared_error(test_true, test_pred)
            mae = mean_absolute_error(test_true, test_pred)  
            r2 = r2_score(test_true, test_pred)
            test_losses = torch.mean(torch.stack(test_losses), dim=0) 
            with open(args.log_path_step,'a') as f:
                f.write(str(args.steps)+ '\t'+str(loss.item())+ '\t'+str(0)+ '\t'+str(test_losses)  + '\t' + str(0)+'\t'+ f'test_Pearson R: {test_r_p:.7f}'+'\t'+f'test_Spearman R: {test_r_s:.7f}'+'\t'+f'test_RMSE: {rmse:.7f}'+'\t'+f'MAE: {mae:.7f}'+'\t'+f'R2: {r2:.7f}'+'\t'+f'conloss:: {conloss:.7f}'
                        +f'train_atom_loss::{train_atom_loss:.7f}'+'\t' +f'train_res_loss::{train_res_loss:.7f}'+f'test_atom_loss::{0:.7f}'+'\t' +f'test_res_loss::{0:.7f}'+'\n')
                f.close()

            if mae - test_r_s < args.best_loss:
                args.best_loss = mae - test_r_s
                args.best_pcc = test_r_p
                args.best_spcc = test_r_s
                args.best_rmse = rmse
                args.best_r2 = r2
                args.best_mae = mae
                save_model(model,optimizer,args,args.steps,args.save_step_path,cv=args.cv,mode = f'step_{args.test_step}')
        if (
            (i_batch + 1) % args.grad_sum == 0
            or i_batch == len(train_dataloader) - 1
        ):
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        # multi-GPU reduction
        if args.ngpu > 1:
            dist.all_reduce(
                loss.data,
                op=torch.distributed.ReduceOp.SUM
            )
            loss /= float(dist.get_world_size())

        loss = loss.data * args.grad_sum

        if args.lr_decay:
            scheduler.step()

    if args.useMultiModel:
        return (
            model,
            train_losses,
            optimizer,
            scheduler,
            train_atom_loss,
            train_res_loss
        )

    return model, train_losses, optimizer, scheduler
def testAndPrint_IPA(model, loader, loss_fn, args, test_sampler=None):
    model.eval()
    with torch.no_grad():
        test_losses, test_true, test_pred, keys = [], [], [], []

        for i_batch, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):
            (
                g,
                full_g,
                Y,
                key,
                pro_feat,
                rna_feat,
                mol_indicator,
                chain_indicator,
                pro_coords,
                rna_coords,
                prot_emb,
                rna_emb,
                neighbor_matrix_padded,
                prot_len,
                rna_len,
                mask,
                prot_whole_emb,
                rna_whole_emb,
                front_batch,
                side_batch,
                top_batch
            ) = batch

            pro_coords = pro_coords.to(args.device, non_blocking=True)
            rna_coords = rna_coords.to(args.device, non_blocking=True)
            prot_emb = prot_emb.to(args.device, non_blocking=True)
            rna_emb = rna_emb.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)

            if neighbor_matrix_padded is not None:
                neighbor_matrix_padded = neighbor_matrix_padded.to(
                    args.device, non_blocking=True
                )

            if torch.is_tensor(prot_len):
                prot_len = prot_len.to(args.device, non_blocking=True)
            else:
                prot_len = torch.as_tensor(prot_len, dtype=torch.long, device=args.device)

            if torch.is_tensor(rna_len):
                rna_len = rna_len.to(args.device, non_blocking=True)
            else:
                rna_len = torch.as_tensor(rna_len, dtype=torch.long, device=args.device)

            Y = Y.to(args.device, non_blocking=True).unsqueeze(-1)

            pred = model(
                pro_coords,
                rna_coords,
                prot_emb,
                rna_emb,
                neighbor_matrix_padded,
                prot_len,
                rna_len,
                mask
            )

            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)

            loss = loss_fn(pred, Y)
            test_losses.append(loss.data)
            test_true.append(Y.data)
            test_pred.append(pred.data)
            keys.extend(key)

        # 默认单卡/本地聚合；多卡且有 sampler 时再做 distributed_concat
        if args.ngpu > 1 and test_sampler is not None:
            test_true = distributed_concat(
                torch.concat(test_true, dim=0), len(test_sampler.dataset)
            ).cpu().numpy()
            test_pred = distributed_concat(
                torch.concat(test_pred, dim=0), len(test_sampler.dataset)
            ).cpu().numpy()
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()

    return test_losses, test_true, test_pred, keys, 0.0
def train_IPA(
    model,
    args,
    optimizer,
    loss_fn,
    train_dataloader,
    scheduler
):

    model.train()

    train_losses = []

    train_atom_loss, train_res_loss = 0.0, 0.0

    for i_batch, batch in tqdm.tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader)
    ):

        (
            g,
            full_g,
            Y,
            key,
            pro_feat,
            rna_feat,
            mol_indicator,
            chain_indicator,
            pro_coords,
            rna_coords,
            prot_emb,
            rna_emb,
            neighbor_matrix_padded,
            prot_len,
            rna_len,
            mask,
            prot_whole_emb,
            rna_whole_emb,
            front_batch,
            side_batch,
            top_batch
        ) = batch

        optimizer.zero_grad()

        # ===== move tensor to device =====

        pro_coords = pro_coords.to(args.device, non_blocking=True)
        rna_coords = rna_coords.to(args.device, non_blocking=True)

        prot_emb = prot_emb.to(args.device, non_blocking=True)
        rna_emb = rna_emb.to(args.device, non_blocking=True)

        mask = mask.to(args.device, non_blocking=True)

        if neighbor_matrix_padded is not None:
            neighbor_matrix_padded = neighbor_matrix_padded.to(
                args.device,
                non_blocking=True
            )

        # prot_len = prot_len.to(args.device)
        # rna_len = rna_len.to(args.device)
        if torch.is_tensor(prot_len):
            prot_len = prot_len.to(args.device, non_blocking=True)
        else:
            prot_len = torch.as_tensor(
                prot_len, dtype=torch.long, device=args.device
            )

        if torch.is_tensor(rna_len):
            rna_len = rna_len.to(args.device, non_blocking=True)
        else:
            rna_len = torch.as_tensor(
                rna_len, dtype=torch.long, device=args.device
            )

        Y = Y.to(args.device).unsqueeze(-1)

        # ===== forward =====

        logits = model(
            pro_coords,
            rna_coords,
            prot_emb,
            rna_emb,
            neighbor_matrix_padded,
            prot_len,
            rna_len,
            mask
        )
        # 统一形状到 [B, 1]
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        # ===== loss =====

        conloss = 0.0
        loss = loss_fn(logits, Y) + conloss

        train_losses.append(loss.item())

        # ===== gradient accumulation =====

        loss = loss / args.grad_sum
        loss.backward()

        if (
            (i_batch + 1) % args.grad_sum == 0
            or i_batch == len(train_dataloader) - 1
        ):
            optimizer.step()
            optimizer.zero_grad()

        # ===== multi GPU =====

        if args.ngpu > 1:
            dist.all_reduce(
                loss,
                op=torch.distributed.ReduceOp.SUM
            )
            loss /= dist.get_world_size()

        # ===== scheduler =====

        if args.lr_decay and (
            (i_batch + 1) % args.grad_sum == 0
        ):
            scheduler.step()

        # ===== evaluation =====

        args.steps += 1

        if args.steps % args.test_step == 0:

            test_losses, test_true, test_pred, keys, conloss = testAndPrint_IPA(
                model,
                args.test_dataloader,
                loss_fn,
                args,
                None
            )

            test_true = np.array(test_true).ravel()
            test_pred = np.array(test_pred).ravel()

            test_r_p = pearsonr(test_true, test_pred)[0]
            test_r_s = spearmanr(test_true, test_pred)[0]

            rmse = root_mean_squared_error(test_true, test_pred)
            mae = mean_absolute_error(test_true, test_pred)
            r2 = r2_score(test_true, test_pred)

            test_losses = torch.mean(torch.stack(test_losses)).item()

            with open(args.log_path_step, "a") as f:

                f.write(
                    f"{args.steps}\t"
                    f"{np.mean(train_losses):.6f}\t"
                    f"{test_losses:.6f}\t"
                    f"Pearson:{test_r_p:.6f}\t"
                    f"Spearman:{test_r_s:.6f}\t"
                    f"RMSE:{rmse:.6f}\t"
                    f"MAE:{mae:.6f}\t"
                    f"R2:{r2:.6f}\t"
                    f"conloss:{conloss:.6f}\n"
                )

            # ===== save best model =====

            if mae - test_r_s < args.best_loss:

                args.best_loss = mae - test_r_s
                args.best_pcc = test_r_p
                args.best_spcc = test_r_s
                args.best_rmse = rmse
                args.best_r2 = r2
                args.best_mae = mae

                save_model(
                    model,
                    optimizer,
                    args,
                    args.steps,
                    args.save_step_path,
                    cv=args.cv,
                    mode=f"step_{args.steps}"
                )

    return model, train_losses, optimizer, scheduler
def testAndPrint_aff(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred = model(g,full_g)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def testAndPrint_aff_equi(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            batch = g.batch_num_nodes()
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            # batch = g.batch_num_nodes()
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
            
            pred = model(g,batch)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def testAndPrint_equi(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,Y,key,batch) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            batch = g.batch_num_nodes()
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            # full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            # 将 batch 转换为适合 radius_graph 的格式
            # print(batch)
            batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
            
            # logits = model(g, batch)
            pred = model(g,batch)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
        results = pd.DataFrame({
            'key': keys,
            'true': test_true.flatten(),
            'pred': test_pred.flatten()
        })
        results.to_csv('test_results.csv', index=False)
        print("Results saved to test_results.csv")
    return test_losses,test_true,test_pred,keys
import copy
import tqdm
def train(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        g = g.to(args.device,non_blocking=True)
        full_g = full_g.to(args.device,non_blocking=True)

        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
       
        logits = model(g,full_g)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        loss = loss_fn(logits, Y)
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def pcc_loss(y_pred, y_true):
    # 计算皮尔逊相关系数损失
    y_pred_mean = torch.mean(y_pred)
    y_true_mean = torch.mean(y_true)
    
    cov = torch.mean((y_pred - y_pred_mean) * (y_true - y_true_mean))
    std_pred = torch.std(y_pred)
    std_true = torch.std(y_true)
    
    pcc = cov / (std_pred * std_true)
    return pcc  

def spcc_loss(y_pred, y_true):
    # 计算斯皮尔曼相关系数损失
    pred_rank = torch.argsort(torch.argsort(y_pred))
    true_rank = torch.argsort(torch.argsort(y_true))
    
    n = y_pred.size(0)
    diff_rank = pred_rank - true_rank
    spcc = 1 - (6 * torch.sum(diff_rank ** 2)) / (n * (n ** 2 - 1))
    return spcc  
def train_aff(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        optimizer.zero_grad()
        model.zero_grad()
        g = g.to(args.device,non_blocking=True)
        full_g = full_g.to(args.device,non_blocking=True)

        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
       
        logits = model(g,full_g)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        mse_loss = loss_fn(logits, Y)
        # PCC损失
        pcc_loss_value = pcc_loss(logits, Y)
        
        # SPCC损失
        spcc_loss_value = spcc_loss(logits, Y)

        # 组合损失 (可以根据需要调整权重)
        loss = args.mse_weight * mse_loss - args.pcc_weight * pcc_loss_value - args.spcc_weight * spcc_loss_value
        
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def contrastive_loss(pos_sim, neg_sim, temperature=0.07):
    # 将相似度缩放并应用softmax
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    
    # 最大化正样本的相似度，最小化负样本的相似度
    pos_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
    
    # 取平均值作为最终损失
    return torch.mean(pos_loss)
def train_contrastive(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    loss_logits = []
    loss_conts = []
    # lossLR=conLoss(args.batch_size).to(args.device)
    model.train()
    # alpha=1.0
    for i_batch, (g_batch,full_g_batch,Y) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        optimizer.zero_grad()
        model.zero_grad()
#         g_pos1 = g_pos1.to(args.device,non_blocking=True)
#         full_g_pos1 = full_g_pos1.to(args.device,non_blocking=True)
#         g_pos2 = g_pos2.to(args.device,non_blocking=True)
#         full_g_pos2 = full_g_pos2.to(args.device,non_blocking=True)
        
#         g_neg1 = g_neg1.to(args.device,non_blocking=True)
#         full_g_neg1 = full_g_neg1.to(args.device,non_blocking=True)
#         g_neg2 = g_neg2.to(args.device,non_blocking=True)
#         full_g_neg2 = full_g_neg2.to(args.device,non_blocking=True)
        # 合并样本
        # g_batch = dgl.batch([g_pos1, g_pos2, g_neg1, g_neg2]).to(args.device, non_blocking=True)
        # full_g_batch = dgl.batch([full_g_pos1, full_g_pos2, full_g_neg1, full_g_neg2]).to(args.device, non_blocking=True)
        g_batch = g_batch.to(args.device, non_blocking=True)
        full_g_batch = full_g_batch.to(args.device, non_blocking=True)
        # 模型前向传播，得到所有样本的嵌入
        embeddings, logits = model(g_batch, full_g_batch,contrastive = True)
        Y = Y.to(args.device,non_blocking=True)
        Y = Y.unsqueeze(-1)
        loss_logit = loss_fn(logits, Y)
        # PCC损失
        pcc_loss_value = pcc_loss(logits, Y)
        
        # SPCC损失
        spcc_loss_value = spcc_loss(logits, Y)

        
        # 拆分输出
        batch_size = args.batch_size
        pos1, pos2 = embeddings[0:batch_size], embeddings[batch_size:2*batch_size]
        neg1, neg2 = embeddings[2*batch_size:3*batch_size], embeddings[3*batch_size:]
        
        # print("flag")
        # pos1,pos2 = model(g_pos1,full_g_pos1),model(g_pos2,full_g_pos2)
        # neg1,neg2 = model(g_neg1,full_g_neg1),model(g_neg2,full_g_neg2)
        
        # print(logits)
        # print(Y.shape)
        pos_sim = F.cosine_similarity(pos1,pos2)
        neg_sim = F.cosine_similarity(pos1,neg1)
        loss_cont = contrastive_loss(pos_sim, neg_sim)
        # 计算对比学习损失
        # loss =  loss_cont + loss_logit
        # if loss_cont<0.1:
        alpha=0.4
        
        loss1 = alpha * loss_cont + (1 - alpha) * loss_logit
        # 组合损失 (可以根据需要调整权重)
        loss = args.mse_weight * loss1 - args.pcc_weight * pcc_loss_value - args.spcc_weight * spcc_loss_value
        # 反向传播并更新模型参数
        loss.backward()
        optimizer.step()

        # 打印损失
        train_losses.append(loss.item())
        loss_conts.append(loss_cont.item())
        loss_logits.append(loss_logit.item())
        if args.lr_decay:
            scheduler.step()
        torch.cuda.empty_cache()
    return model,train_losses,loss_conts,loss_logits,optimizer,scheduler
def train_aff_equi(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        batch = g.batch_num_nodes()
        g = g.to(args.device,non_blocking=True)
        # full_g = full_g.to(args.device,non_blocking=True)
        
        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
       
        batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
        
        logits = model(g, batch)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        loss = loss_fn(logits, Y)
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def train_equi(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,Y,key,batch) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        # continue
        batch = g.batch_num_nodes()
        g = g.to(args.device,non_blocking=True)
        # full_g = full_g.to(args.device,non_blocking=True)
        # batch = g.size(0)
        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
        # print(batch.numel())# 获取批图中的批次信息
        
        
        # 将 batch 转换为适合 radius_graph 的格式
        batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
        
        logits = model(g, batch)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        loss = loss_fn(logits, Y)
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def getToyKey(train_keys):

    """get toy dataset for test"""

    train_keys_toy_d = []
    train_keys_toy_a = []
    
    max_all = 600
    for key in train_keys:
        if '_active_' in key:
            train_keys_toy_a.append(key)
        if '_active_' not in key:
            train_keys_toy_d.append(key)

    if len(train_keys_toy_a) == 0 or len(train_keys_toy_d) == 0:
        return None

    return train_keys_toy_a[:300] + train_keys_toy_d[:(max_all-300)]
def getTestedPro(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            lines = []
            for line in f.readlines():
                if 'actions' in line:
                    lines.append(line)
        lines = [line.split('\t')[0] for line in lines]
        return lines
    else:
        return []
def getEF(model,args,test_path,save_path,debug,batch_size,loss_fn,rates = 0.01,flag = '',prot_split_flag = '_'):
        """calculate EF of test dataset, since dataset have 102/81 proteins ,so we need to calculate EF of each protein one by one!"""
        save_file = save_path + '/EF_test' + flag
        tested_pros = getTestedPro(save_file)
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split(prot_split_flag)
 
            if '_active' in key:
                pros[key_split[0]].insert(0,os.path.join(test_path ,key))
            else:
                ''' all positive label sample will be place in head of list'''
                pros[key_split[0]].append(os.path.join(test_path ,key))


        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :
                if pro in tested_pros:
                    if args.ngpu >= 1:
                        dist.barrier()
                    print('this pro :  %s  is tested'%pro)
                    continue
                test_keys_pro = pros[pro]
                if len(test_keys_pro) == 0:
                    if args.ngpu >= 1:
                        dist.barrier()
                    continue
                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)

                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu >= 1:
                    dist.barrier()
                if args.local_rank == 0:
                    test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    for key in test_keys_pro:
                        key_split = key.split('_')
                        if '_active' in key:
                            Y_sum += 1
                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)

                    EF = []
                    hits_list = []
                    for rate in rates:
                        ''' cal different rates of EF'''
                        find_limit = int(len(test_keys_pro)*rate)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)

                    
                    EF_str = '['
                    hits_str = '['
                    for ef,hits in zip(EF,hits_list):
                        EF_str += '%.3f'%ef+'\t'
                        hits_str += ' %d '%hits
                    EF_str += ']'
                    hits_str += ']'
                    end = time.time()
                    with open(save_file,'a') as f:
                        f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\t'+'loss:' + str(test_losses)+'\n'\
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t' + str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu >= 1:
                    dist.barrier()
                continue
            if args.ngpu >= 1:
                dist.barrier()
        if args.local_rank == 0:
            EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
            EFs_str = '['
            for ef in EFs:
                EFs_str += str(ef)+'\t'
            EFs_str += ']'
            args_dict = vars(args)
            with open(save_file,'a') as f:
                    f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                    for item in args_dict.keys():
                        f.write(item + ' : '+str(args_dict[item]) + '\n')
                    f.close()
        if args.ngpu >= 1:
            # Keeping processes in sync
            dist.barrier()
def getNumPose(test_keys,nums = 5):

    """get the first nums pose for each ligand to prediction"""

    ligands = defaultdict(list)
    for key in test_keys:
        key_split = key.split('_')
        ligand_name = '_'.join(key_split[-2].split('-')[:-1])
        ligands[ligand_name].append(key)
    result = []
    for ligand_name in ligands.keys():
        ligands[ligand_name].sort(key = lambda x : int(x.split('_')[-2].split('-')[-1]),reverse=False)
        result += ligands[ligand_name][:nums]
    return result
def getIdxPose(test_keys,idx = 0):
    """"get the idx pose for each ligand to prediction"""
    ligands = defaultdict(list)


    for key in test_keys:
        key_split = key.split('_')
        ligand_name = '_'.join(key_split[-2].split('-')[:-1])
        ligands[ligand_name].append(key)
    result = []
    for ligand_name in ligands.keys():

        ligands[ligand_name].sort(key = lambda x : int(x.split('_')[-2].split('-')[-1]),reverse=False)
        if idx < len(ligands[ligand_name]):
            result.append(ligands[ligand_name][idx]) 
        else:
            result.append(ligands[ligand_name][-1]) 
    return result
def getEFMultiPose(model,args,test_path,save_path,debug,batch_size,loss_fn,rates = 0.01,flag = '',pose_num = 5,idx_style = False):
        """calulate EF for multi pose complex"""
        save_file = save_path + '/EF_test_multi_pose' + '_{}_'.format(pose_num) + flag
        test_keys = os.listdir(test_path)
        # for multi pose complex, get pose_num poses to cal EF
        tested_pros = getTestedPro(save_file)
        if idx_style:
            test_keys = getIdxPose(test_keys,idx = pose_num)
        else:
            test_keys = getNumPose(test_keys,nums = pose_num) 
 
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            pros[key_split[0]].append(os.path.join(test_path , key))
        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :
                if pro in tested_pros:
                    if args.ngpu >= 1:
                        dist.barrier()
                    print('this pro :  %s  is tested'%pro)
                    continue
                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    if args.ngpu >= 1:
                        dist.barrier()
                    continue
                print('protein keys num ',len(test_keys_pro))

                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)
                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu >= 1:
                    dist.barrier()
                if args.local_rank == 0:
 
                    test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    # multi pose 
                    # get max logits for every ligand
                    key_logits = defaultdict(list)
                    for pred,key in zip(test_pred,test_keys_pro):
                        new_key = '_'.join(key.split('/')[-1].split('_')[:-2] + key.split('/')[-1].split('_')[-2].split('-')[:-1])
                        key_logits[new_key].append(pred)
                    new_keys = list(key_logits.keys())
                    max_pose_logits = [max(logits) for logits in  list(key_logits.values())]

                    test_keys_pro = []
                    test_pred = []
                    for key,logit in zip(new_keys,max_pose_logits):
                        key_split = key.split('_') 
                        if 'actives' in key_split:
                            test_keys_pro.insert(0,key)
                            test_pred.insert(0,logit)
                            Y_sum += 1
                        else:
                            ''' all positive label sample will be place in head of list'''
                            test_keys_pro.append(key)
                            test_pred.append(logit)

                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)
                    
                    EF = []
                    hits_list = []
                    for rate in rates:
                        find_limit = int(len(test_keys_pro)*rate)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)
                    
                    EF_str = '['
                    hits_str = '['
                    for ef,hits in zip(EF,hits_list):
                        EF_str += '%.3f'%ef+'\t'
                        hits_str += ' %d '%hits
                    EF_str += ']'
                    hits_str += ']'
                    end = time.time()
                    with open(save_file,'a') as f:
                        f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\t'+'loss:' + str(test_losses)+'\n'\
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t' + str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu >= 1:
                    dist.barrier()
                continue
            if args.ngpu >= 1:
                dist.barrier()
        if args.local_rank == 0:
            EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
            EFs_str = '['
            for ef in EFs:
                EFs_str += str(ef)+'\t'
            EFs_str += ']'
            args_dict = vars(args)
            with open(save_file,'a') as f:
                    f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                    for item in args_dict.keys():
                        f.write(item + ' : '+str(args_dict[item]) + '\n')
                    f.close()
        if args.ngpu >= 1:
            dist.barrier()
def getEF_from_MSE(model,args,test_path,save_path,device,debug,batch_size,A2_limit,loss_fn,rates = 0.01):
        """cal EF for regression model if you want to training a regression model, you can use this function to cal EF"""
        
        save_file = save_path + '/EF_test'
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            if 'active' in key_split:
                pros[key_split[0]].insert(0,key)
            else:
                pros[key_split[0]].append(key)

        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :

                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    continue
                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                test_dataloader = DataLoader(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = args.num_workers, collate_fn=test_dataset.collate)
                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args)
                test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                test_losses = np.mean(np.array(test_losses))
                # print(test_losses)
                Y_sum = 0
                for key in test_keys_pro:
                    key_split = key.split('_')
                    if 'active' in key_split:
                        Y_sum += 1
                actions = int(Y_sum)
                action_rate = actions/len(test_keys_pro)
                test_pred = np.concatenate(np.array(test_pred), 0)
                EF = []
                hits_list = []
                for rate in rates:
                    find_limit = int(len(test_keys_pro)*rate)
                    _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                    hits = torch.sum(indices[:find_limit] < actions)
                    EF.append((hits/find_limit)/action_rate)
                    hits_list.append(hits)
                
                EF_str = '['
                hits_str = '['
                for ef,hits in zip(EF,hits_list):
                    EF_str += '%.3f'%ef+'\t'
                    hits_str += ' %d '%hits
                EF_str += ']'
                hits_str += ']'
                end = time.time()
                with open(save_file,'a') as f:
                    f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\n'\
                        +'EF:'+rate_str)
                    f.write( EF_str)
                    f.close()
                EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                continue
        EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
        EFs_str = '['
        for ef in EFs:
            EFs_str += str(ef)+'\t'
        EFs_str += ']'
        args_dict = vars(args)
        with open(save_file,'a') as f:
                f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                for item in args_dict.keys():
                    f.write(item + ' : '+str(args_dict[item]) + '\n')
                f.close()
from collections import defaultdict
import numpy as np
import pickle

def get_train_val_keys(keys):
    train_keys = keys
    pro_dict = defaultdict(list)
    for key in train_keys:
        pro = key.split('_')[0]
        pro_dict[pro].append(key)
    pro_list = list(pro_dict.keys())
    indices = np.arange(len(pro_list))
    np.random.shuffle(indices)
    train_num = int(len(indices)*0.8)
    count = 0
    train_list = []
    val_list = []
    for i in indices:
        count +=1
        if count < train_num:
            train_list += pro_dict[pro_list[i]]
        else:
            val_list +=  pro_dict[pro_list[i]]
    return train_list,val_list
def get_dataloader(args,train_keys,val_keys,val_shuffle=False):
    """"
    docstring:
        get dataloader for train and validation
    input:
        train_keys: list of train keys
            train file paths
        val_keys: list of validation keys
            validation file paths

    output: dataloader for train and validation
        (train_dataloader,val_dataloader)
    """
    train_dataset = ESDataset(train_keys,args, args.data_path,args.debug)
    val_dataset = ESDataset(val_keys,args, args.data_path,args.debug)
   
    if args.sampler:

        num_train_chembl = len([0 for k in train_keys if '_active' in k])
        num_train_decoy = len([0 for k in train_keys if '_active' not in k])
        train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
        train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=False,num_workers = args.num_workers, collate_fn=train_dataset.collate,\
            sampler = train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=True, num_workers = args.num_workers, collate_fn=train_dataset.collate)
    val_dataloader = DataLoader(val_dataset, args.batch_size, \
        shuffle=val_shuffle, num_workers = args.num_workers, collate_fn=val_dataset.collate)
    return train_dataloader,val_dataloader

def write_log_head(args,log_path,model,train_keys,val_keys):
    """a function to write the head of log file at the beginning of training"""
    args_dict = vars(args)
    with open(log_path,'w')as f:
        f.write(f'Number of train data: {len(train_keys)}' +'\n'+ f'Number of val data: {len(val_keys)}' + '\n')
        f.write(f'number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}' +'\n')
        for item in args_dict.keys():
            f.write(item + ' : '+str(args_dict[item]) + '\n')
        f.write('epoch'+'\t'+'train_loss'+'\t'+'val_loss'+'\t'+'test_loss' #'\t'+'train_auroc'+ '\t'+'train_adjust_logauroc'+ '\t'+'train_auprc'+ '\t'+'train_balanced_acc'+ '\t'+'train_acc'+ '\t'+'train_precision'+ '\t'+'train_sensitity'+ '\t'+'train_specifity'+ '\t'+'train_f1'+ '\t'\
        + '\t' + 'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
        f.close()
def save_model(model,optimizer,args,epoch,save_path,cv,mode = 'best'):
    """a function to save model"""
    best_name = save_path + f'/save_{mode}_model_{cv}'+'.pt'
    if args.debug:
        best_name = save_path + '/save_{}_model_debug'.format(mode)+'.pt'

    torch.save({'model':model.module.state_dict() if isinstance(model,nn.parallel.DistributedDataParallel) else model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch}, best_name)

def shuffle_train_keys(train_keys):
    """shuffle train keys by protein"""
    sample_dict = defaultdict(list)
    for i in train_keys:
        key = i.split('/')[-1].split('_')[0]
        sample_dict[key].append(i)
    keys = list(sample_dict.keys())
    np.random.shuffle(keys)
    new_keys = []
    batch_sizes = []

    for i,key in enumerate(keys):
        temp = sample_dict[key]
        np.random.shuffle(temp)
        new_keys += temp
        batch_sizes.append(len(temp))
    return new_keys,batch_sizes
