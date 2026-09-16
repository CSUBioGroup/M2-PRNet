import torch
import torch.nn as nn
import torch.nn.functional as F


class GVP(nn.Module):
    def __init__(self, s_in, v_in, s_out, v_out):
        super().__init__()
        self.s_proj = nn.Linear(s_in + v_out, s_out)
        self.v_gate = nn.Linear(s_out, v_out)
        self.W_v = nn.Parameter(torch.randn(v_in, v_out) * 0.02)

    def forward(self, s, v):
        # s: [B, N, s_in], v: [B, N, v_in, 3]
        v_lin = torch.einsum("bnvc,vu->bnuc", v, self.W_v)            # [B,N,v_out,3]
        v_norm = torch.norm(v_lin, dim=-1)                             # [B,N,v_out]
        s_out = self.s_proj(torch.cat([s, v_norm], dim=-1))            # [B,N,s_out]
        gate = torch.sigmoid(self.v_gate(s_out)).unsqueeze(-1)         # [B,N,v_out,1]
        v_out = v_lin * gate
        return s_out, v_out


class GVPBlock(nn.Module):
    def __init__(self, s_dim=128, v_dim=16, dropout=0.1):
        super().__init__()
        self.s2v = nn.Linear(s_dim, v_dim)
        self.msg_s = nn.Sequential(
            nn.Linear(s_dim * 2 + 1, s_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(s_dim, s_dim),
        )
        self.gvp = GVP(s_dim, v_dim, s_dim, v_dim)
        self.norm = nn.LayerNorm(s_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, s, v, coords, node_mask=None, adj=None):
        # coords: [B,N,3]
        rel = coords[:, :, None, :] - coords[:, None, :, :]            # [B,N,N,3]
        dist = torch.norm(rel, dim=-1).clamp_min(1e-6)                 # [B,N,N]
        direction = rel / dist.unsqueeze(-1)

        logits = -dist
        if adj is not None:
            logits = logits.masked_fill(adj <= 0, -1e4)

        if node_mask is not None:
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            logits = logits.masked_fill(~pair_mask, -1e4)

        attn = torch.softmax(logits, dim=-1)                           # [B,N,N]
        neigh_s = torch.einsum("bij,bjd->bid", attn, s)                # [B,N,s_dim]
        weighted_dist = (attn * dist).sum(dim=-1, keepdim=True)        # [B,N,1]

        s_for_v = self.s2v(s)                                          # [B,N,v_dim]
        neigh_v = torch.einsum("bij,bjv,bijc->bivc", attn, s_for_v, direction)

        m_s = self.msg_s(torch.cat([s, neigh_s, weighted_dist], dim=-1))
        ds, dv = self.gvp(m_s, neigh_v)

        s = self.norm(s + self.drop(ds))
        v = v + self.drop(dv)
        return s, v


class GVPAffinityModel(nn.Module):
    def __init__(self, node_dim=1280, s_dim=128, v_dim=16, depth=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, s_dim)
        self.blocks = nn.ModuleList([GVPBlock(s_dim, v_dim, dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(s_dim, 1),
        )

    def forward(
        self,
        pro_coords,
        rna_coords,
        prot_emb,
        rna_emb,
        neighbor_matrix_padded,
        prot_len,
        rna_len,
        mask,
    ):
        coords = torch.cat([pro_coords, rna_coords], dim=1)            # [B,N,3]
        s = torch.cat([prot_emb, rna_emb], dim=1)                      # [B,N,node_dim]
        s = self.input_proj(s)

        B, N, _ = coords.shape
        v = torch.zeros(B, N, 16, 3, device=coords.device, dtype=coords.dtype)

        node_mask = mask.bool() if mask is not None else None
        adj = neighbor_matrix_padded if neighbor_matrix_padded is not None else None

        for blk in self.blocks:
            s, v = blk(s, v, coords, node_mask=node_mask, adj=adj)

        if node_mask is not None:
            denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (s * node_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = s.mean(dim=1)

        return self.head(pooled)  # [B,1]