import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import WeightAndSum
import dgl.function as fn
import e3nn
from e3nn import o3

from model.fusion import ImprovedFusionModel
from model.graph_transformer import RNAProteinGraphTransformer
from model.image_network import MultiViewConvNeXt, TrainableImageNetwork

# from model.graph_transformer import RNAProteinGraphTransformer
"""
with edge features
"""
import matplotlib.pyplot as plt
from model.equiscore_layer import EquiScoreLayer
from utils.equiscore_utils import MLPReadout
import torch.nn.init as init
class conLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.05, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)
 
        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)  # 在第一层后添加Dropout
        x = F.relu(self.fc2(x))
        x = self.drop(x)  # 在第二层后添加Dropout
        x = torch.sigmoid(self.fc3(x))
        return x

class CrossAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)
        self.out = nn.Linear(out_features, out_features)

    def forward(self, ligand, pocket):
        # 计算Query, Key, Value
        Q = self.query(ligand)
        K = self.key(pocket)
        V = self.value(pocket)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算输出特征
        attention_output = torch.matmul(attention_weights, V)
        output = self.out(attention_output)

        return output
class EquiScore(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        if args.data_set == 'PNA_keys.csv' or args.data_set == 'PNA_keys_201.csv' or args.data_set == 'MD_keys.csv' or args.data_set == 'data/case/case.csv' or args.data_set == 'PNA_keys_updated.csv'  or args.data_set == 'MD70.csv' or args.data_set == 'data/vegf/vegf_case.csv':
            self.maxlen = 1529
        else:
            self.maxlen = 2200
        self.res_level = RNAProteinGraphTransformer(config)
        self.fusion_network = ImprovedFusionModel(args)
        # self.image_network = TrainableImageNetwork(hidden_dim = self.args.n_out_feature * 3)
        self.image_network = MultiViewConvNeXt(args)
        self.align_layer = nn.Linear(self.maxlen, self.args.n_out_feature)
        atom_dim = 16*12 if self.args.FP else 10*6
        self.atom_encoder = nn.Embedding(atom_dim  + 1, self.args.n_out_feature, padding_idx=0)
        self.edge_encoder = nn.Embedding( 36* 5 + 1, self.args.edge_dim, padding_idx=0) if args.edge_bias is True else nn.Identity()
        self.rel_pos_encoder = nn.Embedding(512, self.args.edge_dim, padding_idx=0) if args.rel_pos_bias is True else nn.Identity()#rel_pos
        self.in_degree_encoder = nn.Embedding(10, self.args.n_out_feature, padding_idx=0) if args.in_degree_bias is True else nn.Identity()
        # self.irreps_sh='1x0e+1x1e+1x2e'
        # self.irreps_edge_attr = o3.Irreps(irreps_sh)
        self.loss_weights = nn.Parameter(torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32))
        if args.rel_pos_bias:
            self.linear_rel_pos =  nn.Linear(self.args.edge_dim, self.args.head_size) 
        if self.args.lap_pos_enc:
            self.embedding_lap_pos_enc = nn.Linear(self.args.pos_enc_dim, self.args.n_out_feature)
        self.layers = nn.ModuleList([ EquiScoreLayer(self.args) \
                for _ in range(self.args.n_graph_layer) ]) 
        # for layer in self.layers[:2]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        if args.useMultiModel:
            self.MLP_layer_r = MLPReadout(self.args)   # 1 out dim if regression problem   
            self.MLP_layer_a = MLPReadout(self.args)  
        # self.weight_and_sum_kl = WeightAndSum(self.args.n_out_feature)     
        self.weight_and_sum = WeightAndSum(self.args.n_out_feature)   
        self.discriminator = Discriminator(input_dim=2 * self.args.n_out_feature) 
        # self.ddg_prehead = nn.Sequential(
        #     nn.Linear(args.n_out_feature * 2, args.n_out_feature),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(args.n_out_feature, args.n_out_feature // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(args.n_out_feature // 2, 1)
        # )
        self._initialize_weights()
        # self.weight_and_sum2 = WeightAndSum(self.args.n_out_feature)     
        # self.cross_attention = CrossAttention(in_features=args.n_out_feature, out_features=args.n_out_feature)
    def _initialize_weights(self):
        # 遍历模型的所有子模块
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Xavier 初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为 0
            # elif isinstance(m, nn.Conv2d):
            #     init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 初始化
            #     if m.bias is not None:
            #         init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        init.constant_(param.data, 0)
    def getAtt(self,g, full_g):
        
        h = g.ndata['x']

        h = self.atom_encoder(h.long()).mean(-2)

        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        for conv in self.layers:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        # only ligand atom's features are used to task layer 
        # h = h * g.ndata['V']
        
        hg = self.weight_and_sum(g,h)
        hg = self.MLP_layer(hg)
        
        return h,g,full_g,hg
    def getAttFirstLayer(self,g,full_g):
        """
        A tool function to get the attention of the first layer
        """
        h = g.ndata['x']
        

        h = self.atom_encoder(h.long()).mean(-2)

        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)

        for conv in [self.layers[0]]:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        # only ligand atom's features are used to task layer
        # h = h * g.ndata['V']
        hg = self.weight_and_sum(g,h)
        hg = self.MLP_layer(hg)
        
        return h,g,full_g,hg
    
    def forward(self, g, full_g, rna_feat, rna_emb, rna_coords, pro_feat,prot_emb,pro_coords, mol_indicator, chain_indicator,
                    neighbor_matrix_padded, mask, prot_whole_emb, rna_whole_emb,front_batch, side_batch, top_batch, contrastive=True, getScore = False):
        """
        Parameters
        ----------
        g : dgl.DGLGraph 
            convalent and IFP based graph 

        full_g :dgl.DGLGraph
            geometric based graph

		Returns
		-------
            probability of binding

        """
        res_feat = self.res_level(rna_feat,rna_emb, rna_coords, pro_feat,prot_emb,pro_coords, mol_indicator, chain_indicator,
                    neighbor_matrix_padded, mask)
        h = g.ndata['x']
        h = self.atom_encoder(h.long()).mean(-2)
        g.apply_edges(fn.u_sub_v('coors', 'coors', 'detla_coors'))
        h_vector = g.edata['detla_coors'].float()
        
        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        # if self.args.rel_pos_bias:
        #     if self.equi:
        #         edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,x=h_vector, normalize=True, normalization='component')
        #         print(edge_sh.shape)
        #     else:
        #         e_bias = self.linear_rel_pos(h_vector)
        #     # print(h.shape)
        #     # print(e.shape)
        #     # print(e_bias.shape)
        #     e = e + e_bias
        i = 0
        for conv in self.layers:
            
            h, e, score_matrix, attn_info  = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
            # if contrastive and i == 1:  # 只在前两层使用对比学习
            #     # h_contrastive = h.clone()  # 保存前两层的嵌入用于对比学习
            #     h_contrastive = self.weight_and_sum_kl(g,h)
            #     # print(h_contrastive.shape)
            #     # print(h.shape)
            #     return h_contrastive
            # i+=1
        
        if self.args.crossatt:
            ligand = h * g.ndata['V']
            pocket = h * (1 - g.ndata['V'])
            # 计算ligand和pocket的表征
            ligand_repr = self.weight_and_sum(g, ligand)
            pocket_repr = self.weight_and_sum2(g, pocket)

            # 计算cross-attention
            cross_attn_output = self.cross_attention(pocket_repr, ligand_repr)

            # 计算亲和力值
            affinity = self.MLP_layer(cross_attn_output)
            return affinity
        if self.args.ligandonly:
            h = h * g.ndata['V']
        hg = self.weight_and_sum(g,h)
        # print(hg.shape)
        # print(res_feat['output'].shape)
        # hg = torch.cat([hg,self.align_layer(res_feat['output'].squeeze(-1))],dim=-1)
        res_feat = self.align_layer(res_feat['output'].squeeze(-1))
        # llm_seq_feat = torch.cat([prot_whole_emb, rna_whole_emb], dim=1)
        if self.args.image_network:
            image_feat = self.image_network(front_batch, side_batch, top_batch)
            # print("Image feature shape:", image_feat.shape)
        else:
            image_feat = None
            # llm_seq_feat = torch.cat([llm_seq_feat, image_feat], dim=1)
        if self.args.useMultiModel:
            atom_pre = self.MLP_layer_a(hg)
            res_pre = self.MLP_layer_r(res_feat)
        contrastive_loss = 0.0
        if self.args.nores_feat:
            res_feat = res_feat * torch.zeros_like(res_feat)
        if self.args.noatom_feat:
            hg = hg * torch.zeros_like(hg)
        if self.args.contrastive:
            final_output, contrastive_loss, fused_feat = self.fusion_network(hg, res_feat, image_feat=image_feat, contrastive=self.args.contrastive)
            if getScore:
                base_att = res_feat.get('attention_weights', None) if isinstance(res_feat, dict) else None
                if self.args.useMultiModel:
                    return atom_pre, res_pre, final_output, score_matrix, contrastive_loss, fused_feat, base_att, attn_info 
                else:
                    return final_output, score_matrix, contrastive_loss, fused_feat, base_att, attn_info 
            else: 
                if self.args.useMultiModel:
                    return atom_pre, res_pre, final_output, contrastive_loss, fused_feat
                else:
                    return final_output, contrastive_loss, fused_feat
        else:
            
            final_output, fused_feat = self.fusion_network(hg, res_feat,image_feat=image_feat, contrastive=self.args.contrastive)
            if getScore:
                base_att = res_feat.get('attention_weights', None) if isinstance(res_feat, dict) else None
                return final_output, score_matrix, contrastive_loss, fused_feat, base_att, attn_info 
            else:
                return final_output, contrastive_loss, fused_feat
        if contrastive:
            return hg,self.MLP_layer(hg)
        if getScore:
            return self.MLP_layer(hg),score_matrix,attn_info 
        else: 
            return self.MLP_layer(hg)
