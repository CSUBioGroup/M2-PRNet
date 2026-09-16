import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_feat, key_feat, value_feat):
        """
        query_feat: 来自一个图的特征 [batch_size, seq_len, hidden_dim]
        key_feat, value_feat: 来自另一个图的特征 [batch_size, seq_len, hidden_dim]
        """
        attn_output, attn_weights = self.cross_attention(
            query=query_feat,
            key=key_feat,
            value=value_feat
        )
        # 残差连接 + 层归一化
        output = self.layer_norm(query_feat + self.dropout(attn_output))
        return output, attn_weights
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, feat1, feat2):
        """
        feat1, feat2: [batch_size, hidden_dim] 或 [batch_size, seq_len, hidden_dim]
        """
        concatenated = torch.cat([feat1, feat2], dim=-1)
        
        # 计算门控权重
        gate_weights = self.gate(concatenated)
        
        # 变换融合特征
        transformed = self.transform(concatenated)
        
        # 门控融合
        fused_feat = gate_weights * feat1 + (1 - gate_weights) * feat2 + transformed
        
        return self.layer_norm(fused_feat)
class TripleGatedFusion(nn.Module):
    """三路门控融合"""
    def __init__(self, dim):
        super().__init__()
        self.gate_atom = nn.Linear(dim, dim)
        self.gate_residue = nn.Linear(dim, dim)
        self.gate_llm = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, atom_feat, residue_feat, llm_feat):
        gate_weights = self.sigmoid(
            self.gate_atom(atom_feat) + 
            self.gate_residue(residue_feat) + 
            self.gate_llm(llm_feat)
        )
        
        gated_atom = gate_weights * atom_feat
        gated_residue = gate_weights * residue_feat  
        gated_llm = gate_weights * llm_feat
        
        # 加权融合
        fused = gated_atom + gated_residue + gated_llm
        return fused
class HierarchicalFusion(nn.Module):
    def __init__(self, atom_dim, residue_dim, fusion_dim, output_dim, image_dim=None, llm_seq_dim=None):
        super().__init__()
        
        # 不同层次的投影
        self.atom_proj = nn.Linear(atom_dim, fusion_dim)
        self.residue_proj = nn.Linear(residue_dim, fusion_dim)
        if llm_seq_dim is not None:
            self.llm_seq_proj = nn.Linear(llm_seq_dim, fusion_dim)  # 新增LLM投影
            # 多层次融合 - 扩展为三路
            self.cross_attention_atom = CrossAttentionFusion(fusion_dim, num_heads=8)
            self.cross_attention_residue = CrossAttentionFusion(fusion_dim, num_heads=8)
            self.cross_attention_llm = CrossAttentionFusion(fusion_dim, num_heads=8)
            # 三路门控融合
            self.gated_fusion = TripleGatedFusion(fusion_dim)
        else:
            self.gated_fusion = GatedFusion(fusion_dim)
            # self.gated_fusion2 = GatedFusion(fusion_dim)
            # 多层次融合
            self.cross_attention_rq = CrossAttentionFusion(fusion_dim, num_heads=8)
            self.cross_attention2_pq = CrossAttentionFusion(fusion_dim, num_heads=8)
            self.cross_attention_image = CrossAttentionFusion(fusion_dim*2, num_heads=8)

       
        # 注意力融合机制
        self.cross_attention_image = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        # 特征投影层（用于维度对齐）
        self.feature_projection = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 输出预测
        self.output_network = nn.Sequential(
            nn.Linear(fusion_dim , fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
    def _gated_fusion(self, graph_features, image_features, attended_features):
        """改进后的门控融合方案"""
        # 拼接基础特征 [batch, hidden * 2]
        base_features = torch.cat([graph_features, image_features], dim=1)
        
        # 计算门控权重 [batch, hidden]
        gate_weights = self.fusion_gate(base_features)
        gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-6)  # 归一化
        
        # 特征投影 [batch, hidden]
        base_projected = self.feature_projection(base_features)
        
        # 门控融合 + 残差连接
        final_features = gate_weights * attended_features + (1 - gate_weights) * base_projected
        final_features = final_features + attended_features  # 残差连接
        
        return final_features
    
    def _weighted_fusion(self, graph_features, image_features, attended_features):
        """加权融合方案（推荐）"""
        # 方案1: 直接使用注意力特征作为主要特征，用原始特征增强
        # [batch, hidden * 3]
        all_features = torch.cat([graph_features, image_features, attended_features], dim=1)
        
        # 自适应加权 [batch, hidden]
        final_features = self.feature_projection(all_features) + attended_features  # 残差连接
        return final_features
    
    def _residual_fusion(self, graph_features, image_features, attended_features):
        """残差融合方案"""
        # 将图特征和图像特征投影到同一空间
        graph_proj = self.feature_projection(
            torch.cat([graph_features, torch.zeros_like(graph_features)], dim=1)
        )  # [batch, hidden]
        image_proj = self.feature_projection(
            torch.cat([torch.zeros_like(image_features), image_features], dim=1)
        )  # [batch, hidden]
        
        # 残差融合
        final_features = attended_features + graph_proj + image_proj
        return final_features
    def forward(self, atom_feat, residue_feat, image_feat=None, llm_seq_feat=None):
        # 投影到统一维度
        atom_proj = self.atom_proj(atom_feat)
        residue_proj = self.residue_proj(residue_feat)
        if hasattr(self, 'llm_seq_proj'):
            llm_seq_proj = self.llm_seq_proj(llm_seq_feat)
        
            # 三路交叉注意力融合
            atom_enhanced, _ = self.cross_attention_atom(
                atom_proj.unsqueeze(1), 
                torch.cat([residue_proj.unsqueeze(1), llm_seq_proj.unsqueeze(1)], dim=1),
                torch.cat([residue_proj.unsqueeze(1), llm_seq_proj.unsqueeze(1)], dim=1)
            )
            
            residue_enhanced, _ = self.cross_attention_residue(
                residue_proj.unsqueeze(1),
                torch.cat([atom_proj.unsqueeze(1), llm_seq_proj.unsqueeze(1)], dim=1),
                torch.cat([atom_proj.unsqueeze(1), llm_seq_proj.unsqueeze(1)], dim=1)
            )
            
            llm_enhanced, _ = self.cross_attention_llm(
                llm_seq_proj.unsqueeze(1),
                torch.cat([atom_proj.unsqueeze(1), residue_proj.unsqueeze(1)], dim=1),
                torch.cat([atom_proj.unsqueeze(1), residue_proj.unsqueeze(1)], dim=1)
            )
            
            # 三路门控融合
            fused_feat = self.gated_fusion(
                atom_enhanced.squeeze(1), 
                residue_enhanced.squeeze(1), 
                llm_enhanced.squeeze(1)
            )
        else:
            # print(atom_proj.shape,residue_proj.shape)
            # 交叉注意力融合
            atom_enhanced, _ = self.cross_attention_rq(atom_proj, residue_proj, residue_proj)
            residue_enhanced, _ = self.cross_attention2_pq(residue_proj, atom_proj, atom_proj)
            
            # 门控融合
            fused_feat = self.gated_fusion(atom_enhanced, residue_enhanced)
            
            # 池化得到图级表征
            if len(fused_feat.shape) == 3:  # [batch, seq, dim]
                fused_feat = fused_feat.mean(dim=1)  # 平均池化
            else:
                fused_feat = fused_feat
            if  image_feat is not None:
                # 3. 跨模态注意力融合
                fused_feat = fused_feat.unsqueeze(1)  # [batch, 1, hidden]
                image_feat = image_feat.unsqueeze(1)  # [batch, 1, hidden]
                multimodal_features = torch.cat([fused_feat, image_feat], dim=1)
                attended_features, _ = self.cross_attention_image(
                    multimodal_features, multimodal_features, multimodal_features
                )  # [batch, 2, hidden]
                
                # # 池化得到最终特征
                attended_features = attended_features.mean(dim=1)  # [batch, hidden]
                fused_feat = self._gated_fusion(fused_feat.squeeze(1), image_feat.squeeze(1), attended_features)
                # # 4. 残差连接 + 融合
                # combined_features = torch.cat([fused_feat, image_feat], dim=1)  # [batch, hidden*2]
                # # combined_features
                # print("Combined feature shape:", combined_features.shape)
                # print("Fused feature shape:", fused_features.shape)
                # # 与注意力特征融合
                # fused_feat = combined_features + fused_features
            # 最终预测
            # print(graph_feat.shape)
        output = self.output_network(fused_feat)
        # fused_feat = atom_feat
        # fused_feat = residue_feat
        # fused_feat = image_feat
        return output,fused_feat
class ImprovedFusionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 层次化融合模块
        self.fusion_network = HierarchicalFusion(
            atom_dim=args.n_out_feature,  # 原子图表征维度
            residue_dim=args.n_out_feature,  # 碱基图表征维度  
            fusion_dim=args.n_out_feature,  # 融合维度
            image_dim = args.n_out_feature,
            # llm_seq_dim= args.n_out_feature,
            output_dim=1  # 输出维度
        )
        self.llm_seq_proj = nn.Linear(args.llm_seq_dim * 2, args.n_out_feature)  # LLM序列特征投影
        # 对齐层（如果需要）
        # self.align_layer = nn.Linear(args.residue_output_dim, args.fusion_dim)
        self.feat1_proj = nn.Linear(args.n_out_feature, args.n_out_feature)  # 原子级特征投影
        self.feat2_proj = nn.Linear(args.n_out_feature, args.n_out_feature)  # 碱基级特征投影
        self.image_proj = nn.Linear(args.n_out_feature , args.n_out_feature)  # 图像特征投影
        # self.mlp = nn.Sequential(
        #     nn.Linear(args.n_out_feature * 3, args.n_out_feature),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(args.n_out_feature, args.n_out_feature // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(args.n_out_feature // 2, 1),
        # )
    def forward(self, atom_graph_feat, residue_graph_feat,image_feat = None, llm_seq_feat = None, contrastive=False):
        """
        atom_graph_feat: 原子图的图表征 [batch_size, atom_dim]
        residue_graph_feat: 碱基图的节点表征 [batch_size, seq_len, residue_dim]
        """
        # 如果碱基图输出是节点级，先池化成图级
        if len(residue_graph_feat.shape) == 3:
            residue_graph_feat = residue_graph_feat.mean(dim=1)  # 平均池化
        if self.args.image_network and image_feat is not None:
            image_proj = self.image_proj(image_feat)
        else:
            image_proj = None
            # 融合图像特征
            # residue_graph_feat = residue_graph_feat + image_proj
        # 投影LLM序列特征
        # 三路融合
        if llm_seq_feat is not None:
            llm_seq_projected = self.llm_seq_proj(llm_seq_feat)
            fused_output, fused_feat = self.fusion_network(atom_graph_feat, residue_graph_feat, llm_seq_projected)
        else:
            # 层次化融合
            fused_output, fused_feat = self.fusion_network(atom_graph_feat, residue_graph_feat, image_proj)
        
        # 对齐维度（如果需要）
        # residue_aligned = self.align_layer(residue_graph_feat)
        # fused_feat = torch.cat([atom_graph_feat, residue_graph_feat, image_proj], dim=-1)  # [B, 3D]
        # fused_output = self.mlp(fused_feat)  # [B, 1]
        # fused_feat = None
        
        # 对比学习（如果启用）
        if contrastive:
            # 添加对比学习损失
            # contrastive_loss = self.contrastive_loss(atom_graph_feat, residue_graph_feat, image_proj)
            contrastive_loss = self.contrastive_loss(atom_graph_feat, residue_graph_feat)
            return fused_output, contrastive_loss, fused_feat
        
        return fused_output,fused_feat
    def multi_scale_contrastive_loss(self, atom_feats, residue_feats, temperature=0.5):
        """多尺度对比学习"""
        # 分别处理不同尺度特征
        atom_feats = F.normalize(self.feat1_proj(atom_feats), dim=1)
        residue_feats = F.normalize(self.feat2_proj(residue_feats), dim=1)
        
        # 计算跨尺度相似度
        cross_similarity = torch.matmul(atom_feats, residue_feats.T) / temperature
        
        # 计算同尺度相似度（作为正则化）
        atom_similarity = torch.matmul(atom_feats, atom_feats.T) / temperature
        residue_similarity = torch.matmul(residue_feats, residue_feats.T) / temperature
        
        batch_size = atom_feats.size(0)
        labels = torch.arange(batch_size).to(atom_feats.device)
        
        # 组合损失
        cross_loss = F.cross_entropy(cross_similarity, labels)
        atom_loss = F.cross_entropy(atom_similarity, labels)
        residue_loss = F.cross_entropy(residue_similarity, labels)
        
        return cross_loss + 0.3 * (atom_loss + residue_loss)
    def contrastive_loss(self, feat1, feat2, feat3 = None, temperature=0.5):
        """对比学习损失，增强两个图表征的一致性"""
        # 归一化特征
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        if feat3 is not None:
            feat3 = F.normalize(feat3, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(feat1, feat2.T) / temperature
        
        if feat3 is not None:
            similarity += torch.matmul(feat1, feat3.T) / temperature
            similarity += torch.matmul(feat2, feat3.T) / temperature
        # 对比损失
        labels = torch.arange(feat1.size(0)).to(feat1.device)
        loss = F.cross_entropy(similarity, labels)
        #仅仅启用余弦相似度
        # contrastive_loss = 1 - F.cosine_similarity(feat1, feat2).mean()
        
        return loss