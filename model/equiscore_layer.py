import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import numpy as np
import os
# os.path.append('../utils')
from utils.equiscore_utils import *

class ForceEdgeWeight(nn.Module):
    def __init__(self):
        super(ForceEdgeWeight, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def forward(self, distances):
        r = distances
        force = self.a / r + self.b / r**6 + self.c / r**12
        return force

"""
    Multi Attention Head
"""
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(ffn_size, hidden_size)
    def forward(self, x):
        x = self.ffn_dropout(self.layer1(x))
        x = self.gelu(x)
        x = self.layer2(x)
        return x
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,edge_dim,dropout_rate = 0.2):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=True)
        self.proj_e = nn.Linear(edge_dim, num_heads, bias=True)
        self.attn_proj = nn.Linear(num_heads,edge_dim)
        self.output_layer = nn.Linear(self.out_dim * num_heads, self.out_dim * num_heads)
        self.output_layer_edge = nn.Linear(edge_dim, edge_dim)
        self.coor_norm = CoorsNorm()
        self.coors_mlp = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(edge_dim, num_heads))
        self.force_edge_weight = ForceEdgeWeight()
    def propagate_attention(self, g, full_g, return_attn: bool = False):
        dis_force = False
        attn_info = {} if return_attn else None

        # 1) full_g: geometric distance graph attention logits
        full_g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        full_g.apply_edges(fn.u_sub_v('coors', 'coors', 'detla_coors'))
        full_g.apply_edges(square('detla_coors', 'rel_pos_3d'))

        # raw edge length on full_g
        full_g.apply_edges(lambda edges: {
            'edge_length': torch.sqrt(edges.data['rel_pos_3d'].sum(-1) + 1e-12)
        })

        if dis_force:
            full_g.edata['force'] = self.force_edge_weight(full_g.edata['edge_length'].float())

        full_g.edata['rel_pos_3d'] = self.coors_mlp(full_g.edata['rel_pos_3d'].float())
        full_g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        if dis_force:
            full_g.edata['score'] = (
                full_g.edata['score'].sum(-1, keepdim=True)
                * full_g.edata['force'].view(-1, 1, 1)
            )

        full_g.apply_edges(guss_decoy('score', 'rel_pos_3d'))

        if return_attn:
            attn_info['full_src'], attn_info['full_dst'] = full_g.edges()
            attn_info['full_edge_length'] = full_g.edata['edge_length'].detach().clone()
            attn_info['full_score_geo_pre_bias'] = full_g.edata['score'].detach().clone()
            if 'type' in full_g.edata:
                attn_info['full_edge_type'] = full_g.edata['type'].detach().clone()

        # 2) read full_g geometric score on structural g edges
        src, dst = g.edges()
        g.edata['score'] = full_g.edge_subgraph(
            full_g.edge_ids(src, dst),
            relabel_nodes=False
        ).edata['score']

        if return_attn:
            attn_info['g_src'] = src.detach().clone()
            attn_info['g_dst'] = dst.detach().clone()
            attn_info['g_score_geo_pre_bias'] = g.edata['score'].detach().clone()

            if 'type' in g.edata:
                attn_info['g_edge_type'] = g.edata['type'].detach().clone()

            if 'proj_e' in g.edata:
                attn_info['g_proj_e_bias'] = g.edata['proj_e'].detach().clone()

            # g structural edge length, using full_g/g shared coors
            if 'coors' in g.ndata:
                g.apply_edges(fn.u_sub_v('coors', 'coors', 'edge_delta_coors_for_export'))
                g.apply_edges(lambda edges: {
                    'edge_length_for_export': torch.sqrt(
                        (edges.data['edge_delta_coors_for_export'] ** 2).sum(-1) + 1e-12
                    )
                })
                attn_info['g_edge_length'] = g.edata['edge_length_for_export'].detach().clone()

        # 3) edge feature update from geometric score
        g.edata['e_out'] = self.attn_proj(
            g.edata['score'].view(-1, self.num_heads).contiguous()
        )

        if return_attn:
            attn_info['g_e_out_from_geo_score'] = g.edata['e_out'].detach().clone()

        # 4) add edge feature bias; this is the component most directly related to chemical/edge features
        g.apply_edges(edge_bias('score', 'proj_e'))

        if return_attn:
            attn_info['g_score_after_edge_bias'] = g.edata['score'].detach().clone()
            try:
                attn_info['g_edge_bias_delta'] = (
                    attn_info['g_score_after_edge_bias']
                    - attn_info['g_score_geo_pre_bias']
                ).detach().clone()
            except Exception:
                pass

        # write biased structural-edge scores back to full_g
        full_g.apply_edges(func=partUpdataScore('score', 'score', g), edges=g.edges())

        if return_attn:
            attn_info['full_score_after_struct_bias'] = full_g.edata['score'].detach().clone()

        # 5) final softmax on full_g for node update
        eids = full_g.edges()
        full_g.edata['score'] = edge_softmax(
            graph=full_g,
            logits=full_g.edata['score'].clamp(-5, 5)
        )

        if return_attn:
            attn_info['full_score_final_softmax'] = full_g.edata['score'].detach().clone()
            try:
                attn_info['g_score_final_softmax_on_full_g'] = full_g.edge_subgraph(
                    full_g.edge_ids(src, dst),
                    relabel_nodes=False
                ).edata['score'].detach().clone()
            except Exception:
                pass

        # 6) coordinate and feature update, same as original
        full_g.apply_edges(edge_mul_score('detla_coors', 'score'))
        full_g.send_and_recv(
            eids,
            dgl.function.copy_e('detla_coors', 'detla_coors'),
            fn.sum('detla_coors', 'coors_add')
        )
        full_g.ndata['coors'] += full_g.ndata['coors_add']

        full_g.edata['score'] = self.attn_dropout(full_g.edata['score'])
        full_g.send_and_recv(
            eids,
            fn.u_mul_e('V_h', 'score', 'V_h'),
            fn.sum('V_h', 'wV')
        )

        if return_attn:
            return attn_info


    def forward(self, g, full_g, h, e, return_attn: bool = True):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        full_g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        full_g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        full_g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, 1)

        full_g.ndata['coors'] = self.coor_norm(full_g.ndata['coors'])

        # Let g compute structural edge length using the same normalized coordinates.
        if 'coors' in full_g.ndata:
            g.ndata['coors'] = full_g.ndata['coors']

        attn_info = self.propagate_attention(g, full_g, return_attn=return_attn)

        e_out = self.output_layer_edge(g.edata['e_out'] + e)
        h_out = full_g.ndata['wV']
        h_out = self.output_layer(h_out.view(-1, self.out_dim * self.num_heads))

        # This remains compatible with the old draw code.
        # It is g structural-edge score after edge_bias, not the pure full_g distance score.
        score_matrix = g.edata['score']

        if return_attn:
            return h_out, e_out, score_matrix, attn_info
        return h_out, e_out, score_matrix
    # def propagate_attention(self, g,full_g):
    #     """
    #     attention propagation with proir informations
    #     Parameters
    #     ----------
    #     g : dgl.DGLGraph 
    #         convalent and IFP based graph 

    #     full_g :dgl.DGLGraph
    #         geometric based graph
	
	# 	Returns
	# 	-------
        
    #     """
    #     dis_force = False
    #     ############### geometric distance based graph attention module ################################
    #     full_g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        
    #     ################################## transform coors as rel distance to decay attention score ####
    #     full_g.apply_edges(fn.u_sub_v('coors', 'coors', 'detla_coors')) 
    #     full_g.apply_edges(square('detla_coors', 'rel_pos_3d'))
    #     # print(full_g.edata['rel_pos_3d'][0])
    #     if dis_force:
    #         # Compute actual distances
    #         full_g.apply_edges(lambda edges: {'distance': torch.sqrt(edges.data['rel_pos_3d'].sum(-1))})
    #         # Compute force-based edge weight
    #         # print(full_g.edata['rel_pos_3d'].shape)
    #         # print(full_g.edata['rel_pos_3d'][1])
    #         # print(full_g.edata['distance'][1])
    #         full_g.edata['force'] = self.force_edge_weight(full_g.edata['distance'].float())
    #     full_g.edata['rel_pos_3d'] = self.coors_mlp(full_g.edata['rel_pos_3d'].float())
    #     # scaling
    #     full_g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
    #     ########################################
    #     # distance gate 
    #     if dis_force:
    #         # print("Shape of score:", full_g.edata['score'].shape)
    #         # print("Shape of force:", full_g.edata['force'].shape)
    #         # Distance gate using the force-based edge weight
    #         # full_g.apply_edges({'score': full_g.edata['score'] * full_g.edata['force'].view(-1, 1, 1)})
    #         # print(full_g.edata['score'][0])
    #         # print("force",full_g.edata['force'])
    #         full_g.edata['score'] = full_g.edata['score'].sum(-1, keepdim=True) * full_g.edata['force'].view(-1, 1, 1)
    #         # print(full_g.edata['score'][0])
    #         # print("Shape of score:", full_g.edata['score'].shape)
    #     # else:
    #         # distance gate 
    #     full_g.apply_edges(guss_decoy('score','rel_pos_3d'))
    #     # print("Shape of score:", full_g.edata['score'][0])
    #     # full_g.edata['score'] = full_g.edata['score'].sum(-1, keepdim=True) # only be used to ablation study
    #     ##########################################
    #     # read score on structual based edges
    #     # sent attn score to structual based edges
    #     src,dst = g.edges() # get structual based edges
    #     # Create a tensor of edges (src, dst) pairs
    #     # edges = torch.stack([src, dst], dim=1)

    #     # # To check for multiple edges, you can sort the pairs (to treat (i, j) and (j, i) as the same edge)
    #     # edges_sorted = torch.sort(edges, dim=1)[0]  # Sort each pair

    #     # # Find duplicate edges by checking for repeated (src, dst) pairs
    #     # unique_edges, counts = torch.unique(edges_sorted, dim=0, return_counts=True)

    #     # # Print out the duplicate edges (edges with count > 1)
    #     # for edge, count in zip(unique_edges, counts):
    #     #     if count > 1:
    #     #         print(f"Duplicate edge found: {edge} appears {count.item()} times")


    #     g.edata['score'] = full_g.edge_subgraph(full_g.edge_ids(src,dst),relabel_nodes=False).edata['score']
    #     # project score to edge features to update features
    #     g.edata['e_out'] = self.attn_proj(g.edata['score'].view(-1,self.num_heads).contiguous()) # score to edge features
    #     ############### structual edges(covalent bond based edges) bias################################
    #     # Compute attention score bias
    #     g.apply_edges(edge_bias('score', 'proj_e'))  # add edge bias 
    #     # add edge_bias and update on geometric distanced based edges
    #     full_g.apply_edges(func=partUpdataScore('score','score',g),edges=g.edges()) # 
    #     # Copy edge features as e_out to be passed to FFN_e
    #     ###################################################
    #     # softmax
    #     # for softmax numerical stability
    #     eids = full_g.edges()
    #     ################################
    #     full_g.edata['score'] = edge_softmax(graph = full_g,logits = full_g.edata['score'].clamp(-5,5))
    #     ############## score as coors update factor and update vector features ##############
    #     full_g.apply_edges(edge_mul_score('detla_coors', 'score'))# accumlate detla_coors 
    #     full_g.send_and_recv(eids, dgl.function.copy_e('detla_coors','detla_coors'), fn.sum('detla_coors', 'coors_add'))
    #     # to update vector features
    #     full_g.ndata['coors'] += full_g.ndata['coors_add'] 
    #     #################################################################
    #     #########################################################

        
    #     # Print source, destination, and score matrix
    #     # print("Source nodes, Destination nodes, and Score matrix:")
    #     # for s, d, score in zip(src.tolist(), dst.tolist(), score_matrix.tolist()):
    #     #     print(f"Edge from node {s} to node {d} has score: {score}")

    #     # attention dropout for control overfitting
    #     full_g.edata['score'] = self.attn_dropout(full_g.edata['score'])
    #     #feature update
    #     # full_g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        
    #     #dgl2
    #     full_g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))



    # def forward(self, g, full_g,h, e):
    #     Q_h = self.Q(h)
    #     K_h = self.K(h)
    #     V_h = self.V(h)
    #     proj_e = self.proj_e(e)

    #     # get projections for multi-head attention
    #     full_g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
    #     full_g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
    #     full_g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
    #     g.edata['proj_e'] = proj_e.view(-1, self.num_heads, 1)
    #     ########################## norm coors for EquiScore ############### 
    #     full_g.ndata['coors'] = self.coor_norm(full_g.ndata['coors'])

    #     self.propagate_attention(g,full_g)
    #     e_out = self.output_layer_edge(g.edata['e_out'] + e)
    #     h_out = full_g.ndata['wV'] 
    #     h_out = self.output_layer(h_out.view(-1, self.out_dim * self.num_heads))
    #     score_matrix = g.edata['score']
        
    #     return h_out, e_out, score_matrix
    
class EquiScoreLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = MultiHeadAttentionLayer(self.args.n_out_feature, self.args.n_out_feature//self.args.head_size, self.args.head_size,self.args.edge_dim,self.args.attention_dropout_rate)
        self.self_ffn_dropout = nn.Dropout(self.args.dropout_rate)
        self.self_ffn_dropout_2 = nn.Dropout(self.args.dropout_rate)
        self.ffn_dropout_edge = nn.Dropout(self.args.dropout_rate)
        self.ffn_dropout_edge_2 = nn.Dropout(self.args.dropout_rate)
        self.layer_norm1_h = GraphNorm(hidden_dim = self.args.n_out_feature)
        self.layer_norm1_e = nn.LayerNorm(self.args.edge_dim)
        # FFN for h
        self.FFN_h_layer = FeedForwardNetwork(self.args.n_out_feature, self.args.ffn_size, self.args.dropout_rate)

        self.FFN_e_layer = FeedForwardNetwork(self.args.edge_dim, self.args.ffn_size, self.args.dropout_rate)
 
        self.layer_norm2_h = GraphNorm(hidden_dim = self.args.n_out_feature)
        self.layer_norm2_e = nn.LayerNorm(self.args.edge_dim)
            
    def forward(self, g, full_g,x, e):
        """
        update the node embedding and edge embedding
        Parameters
        ----------
        g : dgl.DGLGraph 
            convalent and IFP based graph 

        full_g :dgl.DGLGraph
            geometric based graph
        x : torch.Tensor
            nodes embeddings
        e : torch.Tensor
            edges embeddings
	
		Returns
		-------
        x : torch.Tensor
            updated nodes embeddings
        e : torch.Tensor
            updated edges embeddings
        
        """

        y = self.layer_norm1_h(g,x)
        e_norm = self.layer_norm1_e(e)
        y, e_norm, score_matrix,att_info  = self.attention(g,full_g, y, e_norm)
        e_norm = self.ffn_dropout_edge(e_norm)
        e = e + e_norm
        e_norm = self.layer_norm2_e(e)
        e_norm = self.FFN_e_layer(e_norm)
        e_norm = self.ffn_dropout_edge_2(e_norm)
        e = e + e_norm
        # x layer module
        y = self.self_ffn_dropout(y)
        x = x + y
        y =  self.layer_norm2_h(g,x)
        y =  self.FFN_h_layer(y)
        y = self.self_ffn_dropout_2(y)
        x = x + y
        return x, e, score_matrix, att_info
        
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={})'.format(self.__class__.__name__,
                                             self.args.n_out_feature,
                                             self.args.n_out_feature, self.head_size)