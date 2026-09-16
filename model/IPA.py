import torch
import torch.nn as nn

from openfold.model.structure_module import InvariantPointAttention
from openfold.utils.rigid_utils import Rigid ,Rotation


class IPAAffinityModel(nn.Module):

    def __init__(
        self,
        node_dim=1280,
        ipa_dim=256,   # 新增：IPA内部维度
        pair_dim=128,
        ipa_depth=3,
        heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.pair_embedding = nn.Sequential(
            nn.Linear(1, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, pair_dim)
        )
        # 新增：输入降维
        self.drop = nn.Dropout(dropout)
        self.input_proj = nn.Linear(node_dim, ipa_dim)
        self.ipa_blocks = nn.ModuleList([
            InvariantPointAttention(
                c_s=ipa_dim,      # 改为降维后通道
                c_z=pair_dim,
                c_hidden=16,
                no_heads=heads,
                no_qk_points=4,
                no_v_points=8,
                
            )
            for _ in range(ipa_depth)
        ])

        self.layer_norm = nn.LayerNorm(ipa_dim)

        self.affinity_head = nn.Sequential(
            nn.Linear(ipa_dim, ipa_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ipa_dim, 1)
        )

    def build_pair_feature(self, coords):
        """
        构建 pair feature (distance embedding)
        """
        dist = torch.cdist(coords, coords)  # (B,N,N)
        dist = dist.unsqueeze(-1)

        pair_feat = self.pair_embedding(dist)

        return pair_feat

    def forward(
        self,
        pro_coords,
        rna_coords,
        prot_emb,
        rna_emb,
        neighbor_matrix_padded,
        prot_len,
        rna_len,
        mask
    ):

        # =========================
        # 1 拼接 protein + RNA
        # =========================

        coords = torch.cat([pro_coords, rna_coords], dim=1)
        single = torch.cat([prot_emb, rna_emb], dim=1)
        
        B, N, _ = coords.shape

        # =========================
        # 2 pair feature
        # =========================

        pair = self.build_pair_feature(coords)

        if neighbor_matrix_padded is not None:
            pair = pair * neighbor_matrix_padded.unsqueeze(-1)

        # =========================
        # 3 rigid frame
        # =========================

        # rigids = Rigid.from_translation(coords)
        eye = torch.eye(3, dtype=coords.dtype, device=coords.device)
        rot_mats = eye.view(*([1] * (coords.ndim - 1)), 3, 3).expand(
            *coords.shape[:-1], 3, 3
        ).contiguous()
        rigids = Rigid(Rotation(rot_mats=rot_mats), coords)

        # =========================
        # 4 IPA blocks
        # =========================
        single = self.input_proj(single)
        single = self.drop(single)
        s = single
        # print("Initial s shape:", s.shape)
        # print("Initial pair shape:", pair.shape)
        # print("Initial rigids shape:", rigids.translation.shape)
        # print("Initial mask shape:", mask.shape)
        
        for ipa in self.ipa_blocks:

            s = s + ipa(
                s,
                pair,
                rigids,
                mask
            )

        s = self.layer_norm(s)

        # =========================
        # 5 interface pooling
        # =========================

        # rna_rep = s[:, prot_len:, :]
        rna_rep = s

        pooled = rna_rep.mean(dim=1)

        # =========================
        # 6 affinity prediction
        # =========================

        affinity = self.affinity_head(pooled)

        return affinity.squeeze(-1)