import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# from PNAGraph.utils.equiscore_utils import CoorsNorm
class CoorsNorm(nn.Module):
    """
    Norm the coors

    """
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale
class GaussianDistanceEncoding(nn.Module):
    """高斯距离编码，将距离映射为衰减权重"""
    def __init__(self, num_kernels=32, sigma_min=0.1, sigma_max=10.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # 创建一组高斯核的标准差
        sigmas = torch.linspace(sigma_min, sigma_max, num_kernels)
        self.register_buffer('sigmas', sigmas)
        
    def forward(self, distances):
        """
        Args:
            distances: [batch_size, num_nodes, num_nodes] 或 [num_nodes, num_nodes]
        Returns:
            gaussian_weights: [..., num_kernels] 高斯权重
        """
        # 扩展维度以便广播计算
        distances = distances.unsqueeze(-1)  # [..., num_nodes, num_nodes, 1]
        sigmas = self.sigmas.view(1, 1, 1, -1)  # [1, 1, 1, num_kernels]
        
        # 计算高斯权重
        gaussian_weights = torch.exp(-0.5 * (distances / sigmas) ** 2)
        
        return gaussian_weights

class MultiModalFeatureProjection(nn.Module):
    """多模态特征投影，处理不同维度的RNA和蛋白质特征"""
    def __init__(self, rna_biochem_dim, rna_seq_dim, protein_biochem_dim, protein_seq_dim, hidden_dim):
        super().__init__()
        
        # RNA特征投影
        self.rna_biochem_proj = nn.Linear(rna_biochem_dim, hidden_dim // 2)
        self.rna_seq_proj = nn.Linear(rna_seq_dim, hidden_dim // 2)
        self.rna_layer_norm = nn.LayerNorm(hidden_dim)
        
        # 蛋白质特征投影
        self.protein_biochem_proj = nn.Linear(protein_biochem_dim, hidden_dim // 2)
        self.protein_seq_proj = nn.Linear(protein_seq_dim, hidden_dim // 2)
        self.protein_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, rna_biochem_feats, rna_seq_feats, protein_biochem_feats, protein_seq_feats):
        """
        Args:
            rna_biochem_feats: [num_rna_nodes, rna_biochem_dim]
            rna_seq_feats: [num_rna_nodes, rna_seq_dim]
            protein_biochem_feats: [num_protein_nodes, protein_biochem_dim]
            protein_seq_feats: [num_protein_nodes, protein_seq_dim]
        """
        # 投影RNA特征
        rna_biochem = self.rna_biochem_proj(rna_biochem_feats)
        rna_seq = self.rna_seq_proj(rna_seq_feats)
        rna_feats = torch.cat([rna_biochem, rna_seq], dim=-1)
        rna_feats = self.rna_layer_norm(rna_feats)
        
        # 投影蛋白质特征
        protein_biochem = self.protein_biochem_proj(protein_biochem_feats)
        protein_seq = self.protein_seq_proj(protein_seq_feats)
        protein_feats = torch.cat([protein_biochem, protein_seq], dim=-1)
        protein_feats = self.protein_layer_norm(protein_feats)
        
        return rna_feats, protein_feats

class DistanceAwareAttention(nn.Module):
    """距离感知的多头注意力机制"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 距离权重投影
        self.distance_proj = nn.Linear(32, num_heads)  # 32是高斯核的数量
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, distance_weights, mask=None):
        """
        Args:
            query, key, value: [num_nodes, hidden_dim]
            distance_weights: [num_nodes, num_nodes, num_kernels]
            mask: [num_nodes, num_nodes] 可选注意力掩码
        """
        batch_size, num_nodes, _ = query.size()
        
        # 投影到查询、键、值
        Q = self.q_proj(query).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 添加距离偏置
        distance_bias = self.distance_proj(distance_weights)  # [batch,num_nodes, num_nodes, num_heads]
        # print(distance_bias.shape)
        distance_bias = distance_bias.permute(0, 3, 1, 2)  # [batch,, num_heads, num_nodes, num_nodes]
        
        attn_scores = attn_scores + distance_bias
        # print(mask.shape, attn_scores.shape)
        # 应用注意力掩码（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) 
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, self.hidden_dim
        )
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class GraphTransformerLayer(nn.Module):
    """图Transformer层"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.self_attention = DistanceAwareAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, distance_weights, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.self_attention(x, x, x, distance_weights, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class RNAProteinGraphTransformer(nn.Module):
    """RNA-蛋白质图Transformer"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.dropout_rate = config['dropout']
        self.coors_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(32, self.num_heads))
        # 特征投影
        self.feature_projection = MultiModalFeatureProjection(
            rna_biochem_dim=config['rna_biochem_dim'],
            rna_seq_dim=config['rna_seq_dim'],
            protein_biochem_dim=config['protein_biochem_dim'],
            protein_seq_dim=config['protein_seq_dim'],
            hidden_dim=config['hidden_dim']
        )
        self.dis_update = nn.Linear(32,32)
        self.coor_norm = CoorsNorm()
        # 高斯距离编码
        self.distance_encoding = GaussianDistanceEncoding(
            num_kernels=config['num_gaussian_kernels'],
            sigma_min=config['sigma_min'],
            sigma_max=config['sigma_max']
        )
        # self.distance_encoding = nn.Linear(1370+159,1370+159)
        # 节点类型嵌入
        self.rna_type_embedding = nn.Embedding(1, config['hidden_dim'])
        self.protein_type_embedding = nn.Embedding(1, config['hidden_dim'])
        self.mol_indicator_embedding = nn.Embedding(2, config['hidden_dim'])
        self.chain_indicator_embedding = nn.Embedding(6, config['hidden_dim'])  #
        self.coord_embedding = nn.Linear(3, config['hidden_dim'])
        # Transformer层
        self.layers = nn.ModuleList([
            GraphTransformerLayer(config['hidden_dim'], config['num_heads'], config['dropout'])
            for _ in range(config['num_layers'])
        ])
        
        # 输出头（根据任务需要调整）
        self.output_head = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, config['output_dim'])
        )
        self.norm1 = nn.LayerNorm(config['hidden_dim'])  # 拼接后归一化
        self.norm2 = nn.LayerNorm(config['hidden_dim'])  # 所有嵌入相加后归一化
        self.dropout = nn.Dropout(config['dropout'])
    def _build_node_valid_mask(self, rna_feat, pro_feat, mask=None):
        # 非零向量视为有效节点
        rna_valid = (rna_feat.abs().sum(dim=-1) > 0)   # [B, R]
        pro_valid = (pro_feat.abs().sum(dim=-1) > 0)   # [B, P]
        node_valid = torch.cat([rna_valid, pro_valid], dim=1)  # [B, R+P]

        if mask is not None:
            node_valid = node_valid & mask.bool()
        return node_valid, rna_valid, pro_valid    
    def forward(self, rna_biochem_feats, rna_seq_feats, rna_coords, protein_biochem_feats, 
                protein_seq_feats, protein_coords, mol_indicator, chain_indicator, adj_matrix,mask=None):
        """
        Args:
            rna_biochem_feats: [batch_size, num_rna_nodes, rna_biochem_dim]
            rna_seq_feats: [batch_size, num_rna_nodes, rna_seq_dim]
            protein_biochem_feats: [batch_size, num_protein_nodes, protein_biochem_dim]
            protein_seq_feats: [batch_size, num_protein_nodes, protein_seq_dim]
            adj_matrix: [batch_size, num_nodes, num_nodes] 距离矩阵
        """
        batch_size = rna_biochem_feats.size(0)
        num_rna_nodes = rna_biochem_feats.size(1)
        num_protein_nodes = protein_biochem_feats.size(1)
        num_nodes = num_rna_nodes + num_protein_nodes
        # print(rna_biochem_feats.shape, rna_seq_feats.shape, protein_biochem_feats.shape, protein_seq_feats.shape , mask.shape)
        # print(mol_indicator.shape, chain_indicator.shape, rna_coords.shape, protein_coords.shape, adj_matrix.shape)
        # 1. 特征投影
        rna_feats, protein_feats = self.feature_projection(
            rna_biochem_feats, rna_seq_feats, protein_biochem_feats, protein_seq_feats
        )
        coords = torch.cat([rna_coords, protein_coords], dim=1) 
        coords = self.coor_norm(coords)
        # protein_coords = self.coor_norm((protein_coords)

        # diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        # distance_matrix = self.coors_mlp(torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8))
        # adj_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        # 3. 拼接所有节点特征
        all_feats = torch.cat([rna_feats, protein_feats], dim=1)  # [batch_size, num_nodes, hidden_dim]
        all_feats = self.norm1(all_feats)
        # print(self.mol_indicator_embedding(mol_indicator).shape)
        mol_indicator_indices = mol_indicator.argmax(dim=-1)
        all_feats += self.mol_indicator_embedding(mol_indicator_indices)
        chain_indicator_indices = chain_indicator.argmax(dim=-1)
        all_feats += self.chain_indicator_embedding(chain_indicator_indices)
        all_feats += self.coord_embedding(coords)
        all_feats = self.norm2(all_feats)
        if mask is not None:
            all_feats = all_feats * mask.unsqueeze(-1)

        # 4. 距离编码与注意力掩码
        # 注意：在输入的 adj_matrix 中，0 既表示 "无连接"，也用于对角线的自连接（distance=0）。
        # 需要确保自连接不会被当作无连接而 mask 掉，同时在做高斯距离编码时不要把 "无连接(0)" 和 "self(0)" 混淆。
        # 创建注意力掩码：有连接为1，无连接为0，但强制保留对角线（self-connection）为1
        attention_mask = (adj_matrix != 0).float()  # 有连接的为1，无连接为0
        diag = torch.eye(num_nodes, device=adj_matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = torch.clamp(attention_mask + diag, 0, 1)  # 确保对角线为1

        # 为距离编码准备一个不会把 "无连接(0)" 当作 "self(0)" 的副本：
        # 将无连接的位置设为一个很大的值（例如 1e6），使得高斯编码在这些位置接近0；
        # 而对角线仍保持0，以保留 self 信息。
        dist_for_encoding = adj_matrix.clone()
        dist_for_encoding = dist_for_encoding.masked_fill(attention_mask == 0, 1e6)

        distance_weights = self.distance_encoding(dist_for_encoding)  # [batch_size, num_nodes, num_nodes, num_kernels]

        # 6. 通过Transformer层
        hidden_states = all_feats
        all_attention_weights = []

        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, distance_weights, attention_mask)
            avg_attn_weights = attn_weights.mean(dim=1)
            avg_attn_weights_expanded = avg_attn_weights.unsqueeze(-1) 
            distance_weights = distance_weights * avg_attn_weights_expanded  # 加权
            distance_weights = self.dis_update(distance_weights)  # 通过线性层更新
            hidden_states = self.dropout(hidden_states)
            all_attention_weights.append(attn_weights)

        # 7. 输出
        # print(hidden_states.shape)
        output = self.output_head(hidden_states)
        # print(output.shape)
        return {
            'node_embeddings': hidden_states,
            'output': output,
            'attention_weights': all_attention_weights
        }

# 配置示例
config = {
    'rna_biochem_dim': 64,      # RNA生化特征维度
    'rna_seq_dim': 512,         # RNA序列特征维度
    'protein_biochem_dim': 128, # 蛋白质生化特征维度
    'protein_seq_dim': 1024,    # 蛋白质序列特征维度
    'hidden_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.1,
    'num_gaussian_kernels': 32,
    'sigma_min': 0.1,
    'sigma_max': 10.0,
    'output_dim': 1  # 根据你的任务调整
}

# 创建模型
model = RNAProteinGraphTransformer(config)

# 打印模型参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型参数量: {count_parameters(model):,}")

# 示例使用
def example_usage():
    batch_size = 2
    num_rna_nodes = 10
    num_protein_nodes = 5
    
    # 创建示例输入
    rna_biochem = torch.randn(batch_size, num_rna_nodes, config['rna_biochem_dim'])
    rna_seq = torch.randn(batch_size, num_rna_nodes, config['rna_seq_dim'])
    protein_biochem = torch.randn(batch_size, num_protein_nodes, config['protein_biochem_dim'])
    protein_seq = torch.randn(batch_size, num_protein_nodes, config['protein_seq_dim'])
    
    # 创建示例邻接矩阵（距离矩阵）
    num_nodes = num_rna_nodes + num_protein_nodes
    adj_matrix = torch.rand(batch_size, num_nodes, num_nodes) * 10  # 距离范围0-10
    # 设置对角线为0，无连接的位置为0
    for i in range(batch_size):
        adj_matrix[i] = adj_matrix[i].triu(1) + adj_matrix[i].triu(1).transpose(-1, -2)
    
    # 前向传播
    outputs = model(rna_biochem, rna_seq, protein_biochem, protein_seq, adj_matrix)
    
    print(f"输入形状: RNA生化 {rna_biochem.shape}, RNA序列 {rna_seq.shape}")
    print(f"         Protein生化 {protein_biochem.shape}, Protein序列 {protein_seq.shape}")
    print(f"输出形状: {outputs['output'].shape}")
    print(f"节点嵌入形状: {outputs['node_embeddings'].shape}")

if __name__ == "__main__":
    example_usage()