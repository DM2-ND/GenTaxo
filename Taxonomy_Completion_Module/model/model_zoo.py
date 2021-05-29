import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax as dgl_edge_softmax
from dgl.nn.pytorch.glob import SumPooling, MaxPooling
import math


def BinarySelector(x, a=1000):
    return torch.sigmoid(a*x)


""" 
Graph Propagation Modules: GCN, GAT, PGCN, PGAT
"""
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_alpha)
        self.softmax = dgl_edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
            else:
                self.res_fc = None

    def forward(self, g, feature):
        # prepare
        h = self.feat_drop(feature)  # NxD
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1) # N x H x 1
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1) # N x H x 1
        g.ndata['ft'] = ft
        g.ndata['a1'] = a1
        g.ndata['a2'] = a2
        # 1. compute edge attention
        g.apply_edges(self.edge_attention)
        # 2. compute softmax
        self.edge_softmax(g)
        # 3. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        ret = g.ndata['ft']
        # 4. residual
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                resval = torch.unsqueeze(h, 1)  # Nx1xD'
            ret = resval + ret
        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a' : a}

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation, in_dropout=0.1, hidden_dropout=0.1, output_dropout=0.0):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_dim, hidden_dim, activation, in_dropout))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, activation, hidden_dropout))
        # output layer
        self.layers.append(GCNLayer(hidden_dim, out_dim, None, output_dropout))

    def forward(self, g, features):
        h = features
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(h.device)
        g.ndata['norm'] = norm.unsqueeze(1)
        for layer in self.layers:
            h = layer(g, h)
        return h

class PGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pos_dim, num_layers, activation, in_dropout=0.1, hidden_dropout=0.1, output_dropout=0.0, position_vocab_size=3):
        super(PGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.prop_position_embeddings = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_dim+pos_dim, hidden_dim, activation, in_dropout))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim+pos_dim, hidden_dim, activation, hidden_dropout))
            self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # output layer
        self.layers.append(GCNLayer(hidden_dim+pos_dim, out_dim, None, output_dropout))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))

    def forward(self, g, features):
        h = features
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(h.device)
        g.ndata['norm'] = norm.unsqueeze(1)

        positions = g.ndata.pop('pos').to(h.device)
        for idx, layer in enumerate(self.layers):
            p = self.prop_position_embeddings[idx](positions)
            h = layer(g, torch.cat((h, p), 1))
        return h

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads, activation, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input layer, no residual
        self.gat_layers.append(GATLayer(in_dim, hidden_dim, heads[0], feat_drop, attn_drop, leaky_relu_alpha, False))
        # hidden layers, due to multi-head, the in_dim = hidden_dim * num_heads
        for l in range(1, num_layers):
            self.gat_layers.append(GATLayer(hidden_dim * heads[l-1], hidden_dim, heads[l], feat_drop, attn_drop, leaky_relu_alpha, residual))
        # output layer
        self.gat_layers.append(GATLayer(hidden_dim * heads[-2], out_dim, heads[-1], feat_drop, attn_drop, leaky_relu_alpha, residual))

    def forward(self, g, features):
        h = features
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.activation(h)
        # output projection
        h = self.gat_layers[-1](g, h).mean(1)
        return h

class PGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pos_dim, num_layers, heads, activation, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False, position_vocab_size=3):
        super(PGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.prop_position_embeddings = nn.ModuleList()
        self.activation = activation
        # input layer, no residual
        self.gat_layers.append(GATLayer(in_dim+pos_dim, hidden_dim, heads[0], feat_drop, attn_drop, leaky_relu_alpha, False))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # hidden layers, due to multi-head, the in_dim = hidden_dim * num_heads
        for l in range(1, num_layers):
            self.gat_layers.append(GATLayer(hidden_dim * heads[l-1] + pos_dim, hidden_dim, heads[l], feat_drop, attn_drop, leaky_relu_alpha, residual))
            self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))
        # output layer
        self.gat_layers.append(GATLayer(hidden_dim * heads[-2] + pos_dim, out_dim, heads[-1], feat_drop, attn_drop, leaky_relu_alpha, residual))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))

    def forward(self, g, features):
        h = features
        positions = g.ndata.pop('pos').to(h.device)
        for l in range(self.num_layers):
            p = self.prop_position_embeddings[l](positions)
            h = self.gat_layers[l](g, torch.cat((h, p), 1)).flatten(1)
            h = self.activation(h)
        # output projection
        p = self.prop_position_embeddings[-1](positions)
        h = self.gat_layers[-1](g, torch.cat((h, p), 1)).mean(1)
        return h


""" 
Graph Readout Modules: MR, WMR, CR, [SumPooling, MaxPooling]
TODO: try GlobalAttentionPooling
"""
class MeanReadout(nn.Module):
    def __init__(self):
        super(MeanReadout, self).__init__()
    
    def forward(self, g, pos=None):
        return dgl.mean_nodes(g, 'h')
        
class WeightedMeanReadout(nn.Module):
    def __init__(self, position_vocab_size=3):
        super(WeightedMeanReadout, self).__init__()
        self.position_weights = nn.Embedding(position_vocab_size, 1)
        self.nonlinear = F.softplus
    
    def forward(self, g, pos):
        g.ndata['a'] = self.nonlinear(self.position_weights(pos))
        return dgl.mean_nodes(g, 'h', 'a')

class ConcatReadout(nn.Module):
    def __init__(self):
        super(ConcatReadout, self).__init__()
    
    def forward(self, g, pos):
        normalizer = torch.tensor(g.batch_num_nodes).unsqueeze_(1).float().to(pos.device)

        g.ndata['a_gp'] = (pos == 0).float()
        gp_embed = dgl.sum_nodes(g, 'h', 'a_gp') / normalizer
        g.ndata['a_p'] = (pos == 1).float()
        p_embed = dgl.mean_nodes(g, 'h', 'a_p')
        g.ndata['a_sib'] = (pos == 2).float()
        sib_embed = dgl.sum_nodes(g, 'h', 'a_sib') / normalizer
        
        return torch.cat((gp_embed, p_embed, sib_embed), 1)


class ConcatEdgeReadout(nn.Module):
    def __init__(self):
        super(ConcatEdgeReadout, self).__init__()

    def forward(self, g, pos):
        normalizer = torch.tensor(g.batch_num_nodes).unsqueeze_(1).float().to(pos.device)

        g.ndata['a_pu'] = (pos == 0).float()
        pu_embed = dgl.sum_nodes(g, 'h', 'a_pu') / normalizer
        g.ndata['a_u'] = (pos == 1).float()
        u_embed = dgl.mean_nodes(g, 'h', 'a_u')
        g.ndata['a_cu'] = (pos == 2).float()
        cu_embed = dgl.sum_nodes(g, 'h', 'a_cu') / normalizer

        g.ndata['a_pseudo'] = (pos == 3).float()
        pseudo_embed = dgl.sum_nodes(g, 'h', 'a_pseudo') / normalizer

        g.ndata['a_pv'] = (pos == 4).float()
        pv_embed = dgl.sum_nodes(g, 'h', 'a_pv') / normalizer
        g.ndata['a_v'] = (pos == 5).float()
        v_embed = dgl.mean_nodes(g, 'h', 'a_v')
        g.ndata['a_cv'] = (pos == 6).float()
        cv_embed = dgl.sum_nodes(g, 'h', 'a_cv') / normalizer

        return torch.cat((pu_embed, u_embed, cu_embed, pseudo_embed, pv_embed, v_embed, cv_embed), 1)

class SumReadout(nn.Module):
    def __init__(self):
        super(SumReadout, self).__init__()
        self.sum_pooler = SumPooling()
    
    def forward(self, g):
        feat = g.ndata['h']
        return self.sum_pooler(g, feat)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()
        self.max_pooler = MaxPooling()
    
    def forward(self, g):
        feat = g.ndata['h']
        return self.max_pooler(g, feat)

""" 
Matching Modules: MLP, LBM, [NTN]
"""


class DST(nn.Module):
    def __init__(self, l_dim, r_dim, non_linear=F.tanh):
        super(DST, self).__init__()
        self.ffn1 = nn.Linear(l_dim*2, r_dim, bias=False)
        self.ffn2 = nn.Linear(r_dim, r_dim, bias=False)
        self.u_R = nn.Linear(r_dim, 1, bias=False)
        self.f = non_linear

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        e = torch.cat((e1, e2), -1)
        return torch.mean(torch.abs(self.ffn1(e) - self.ffn2(q)), dim=-1, keepdim=True)
        # return self.u_R(self.f(torch.abs(self.ffn1(e) - self.ffn2(q))))


class SLP(nn.Module):
    def __init__(self, l_dim, r_dim, hidden_dim, non_linear=F.tanh):
        super(SLP, self).__init__()
        self.u_R = nn.Linear(hidden_dim, 1, bias=False)
        self.f = non_linear
        self.ffn = nn.Linear(l_dim*2+r_dim, hidden_dim, bias=False)

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.u_R(self.f(self.ffn(torch.cat((e1, e2, q), 1))))


class MLP(nn.Module):
    def __init__(self, l_dim, r_dim, hidden_dim, k, non_linear=F.tanh):
        super(MLP, self).__init__()
        activation = nn.ReLU()  
        self.ffn = nn.Sequential(
            nn.Linear(l_dim*2+r_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, k, bias=False)
        )
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.u_R(self.f(self.ffn(torch.cat((e1, e2, q), 1))))


class RawMLP(nn.Module):
    def __init__(self, l_dim, r_dim, hidden_dim, k, non_linear=F.tanh):
        super(RawMLP, self).__init__()
        activation = nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(l_dim+r_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, k, bias=False)
        )
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear

    def forward(self, e, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.u_R(self.f(self.ffn(torch.cat((e, q), 1))))


class BIM(nn.Module):
    def __init__(self, l_dim, r_dim):
        super(BIM, self).__init__()
        self.W = nn.Bilinear(l_dim*2, r_dim, 1, bias=False)
        
    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        e = torch.cat((e1, e2), -1)
        return self.W(e, q)


class RawBIM(nn.Module):
    def __init__(self, l_dim, r_dim):
        super(RawBIM, self).__init__()
        self.W = nn.Bilinear(l_dim, r_dim, 1, bias=False)

    def forward(self, e, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.W(e, q)


class LBM(nn.Module):
    def __init__(self, l_dim, r_dim):
        super(LBM, self).__init__()
        self.W = nn.Bilinear(l_dim*2, r_dim, 1, bias=False)
        
    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        e = torch.cat((e1, e2), -1)
        return torch.exp(self.W(e, q))


class Arborist(nn.Module):
    def __init__(self, l_dim, r_dim, k=5):
        super(Arborist, self).__init__()
        self.u = nn.Linear(l_dim * 2, k, bias=False)
        self.W = nn.Bilinear(l_dim * 2, r_dim, k, bias=False)

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        e = torch.cat((e1, e2), -1)
        u = self.u(e)
        w = self.W(e, q)
        return torch.sum(u*w, dim=-1, keepdim=True)


class RawArborist(nn.Module):
    def __init__(self, l_dim, r_dim, k=5):
        super(RawArborist, self).__init__()
        self.u = nn.Linear(l_dim, k, bias=False)
        self.W = nn.Bilinear(l_dim, r_dim, k, bias=False)

    def forward(self, e, q):
        u = self.u(e)
        w = self.W(e, q)
        return torch.sum(u*w, dim=-1, keepdim=True)


class NTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=F.tanh):
        super(NTN, self).__init__()
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear
        self.W = nn.Bilinear(l_dim*2, r_dim, k, bias=False)
        self.V = nn.Linear(l_dim*2+r_dim, k, bias=False)
        
    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        e = torch.cat((e1, e2), -1)
        return self.u_R(self.f(self.W(e, q) + self.V(torch.cat((e, q), 1))))


class RawNTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=torch.tanh):
        super(RawNTN, self).__init__()
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear
        self.W = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.V = nn.Linear(l_dim + r_dim, k, bias=False)

    def forward(self, e, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.u_R(self.f(self.W(e, q) + self.V(torch.cat((e, q), 1))))


class TriNTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=F.tanh):
        super(TriNTN, self).__init__()
        self.match_l = RawNTN(l_dim, r_dim, k, non_linear)
        self.match_r = RawNTN(l_dim, r_dim, k, non_linear)
        self.match = RawNTN(l_dim*2, r_dim, k, non_linear)

    def forward(self, e1, e2, q):
        l_scores = self.match_l(e1, q)
        r_scores = self.match_r(e2, q)
        scores = self.match(torch.cat((e1, e2), -1), q) + l_scores + r_scores
        return scores


class CNTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=F.tanh):
        super(CNTN, self).__init__()
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear
        self.W = nn.Bilinear(l_dim * 2, r_dim, k, bias=True)
        self.V = nn.Linear(l_dim * 2 + r_dim, k, bias=False)
        self.control = nn.Sequential(nn.Linear(l_dim*2+r_dim, l_dim*2, bias=False), nn.Sigmoid())

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        # ec1 = e1 * self.control_l(torch.cat((e1, q), -1))
        # ec2 = e2 * self.control_r(torch.cat((e2, q), -1))
        # ec = torch.cat((ec1, ec2), -1)

        e = torch.cat((e1, e2), -1)
        c = self.control(torch.cat((e, q), -1))
        ec = e * c
        return self.u_R(self.f(self.W(ec, q) + self.V(torch.cat((ec, q), 1))))


class RawCNTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=torch.tanh):
        super(RawCNTN, self).__init__()
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear
        self.W = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.V = nn.Linear(l_dim + r_dim, k, bias=False)
        self.control = nn.Sequential(nn.Linear(l_dim + r_dim, l_dim, bias=False), nn.Sigmoid())

    def forward(self, e, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        c = self.control(torch.cat((e, q), -1))
        ec = e * c
        return self.u_R(self.f(self.W(ec, q) + self.V(torch.cat((ec, q), 1))))


class TriCNTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=F.tanh):
        super(TriCNTN, self).__init__()
        self.match_l = RawCNTN(l_dim, r_dim, k, non_linear)
        self.match_r = RawCNTN(l_dim, r_dim, k, non_linear)
        self.match = RawCNTN(l_dim*2, r_dim, k, non_linear)

    def forward(self, e1, e2, q):
        l_scores = self.match_l(e1, q)
        r_scores = self.match_r(e2, q)
        scores = self.match(torch.cat((e1, e2), -1), q) + l_scores + r_scores
        return scores


class TMN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=nn.LeakyReLU(0.2)):
    # def __init__(self, l_dim, r_dim, k=5, non_linear=nn.Tanh()):
        super(TMN, self).__init__()

        self.u = nn.Linear(k*3, 1, bias=False)
        self.u_l = nn.Linear(k, 1, bias=False)
        self.u_r = nn.Linear(k, 1, bias=False)
        self.u_e = nn.Linear(k, 1, bias=False)
        self.f = non_linear # if GNN/LSTM encoders are used, tanh should not, because they are not compatible
        self.W_l = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W_r = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W = nn.Bilinear(l_dim*2, r_dim, k, bias=True)
        self.V_l = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V_r = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V = nn.Linear(l_dim*2 + r_dim, k, bias=False)

        self.control = nn.Sequential(nn.Linear(l_dim*2+r_dim, l_dim*2, bias=False), nn.Sigmoid())
        self.control_l = nn.Sequential(nn.Linear(l_dim+r_dim, l_dim, bias=False), nn.Sigmoid())
        self.control_r = nn.Sequential(nn.Linear(l_dim+r_dim, l_dim, bias=False), nn.Sigmoid())

    def forward(self, e1, e2, q):
        ec1 = e1 * self.control_l(torch.cat((e1, q), -1))
        ec2 = e2 * self.control_r(torch.cat((e2, q), -1))
        e = torch.cat((e1, e2), 1)
        ec = e * self.control(torch.cat((e, q), -1))
        l = self.W_l(ec1, q) + self.V_l(torch.cat((ec1, q), 1))
        r = self.W_r(ec2, q) + self.V_r(torch.cat((ec2, q), 1))
        e = self.W(ec, q) + self.V(torch.cat((ec, q), 1))
        l_scores = self.u_l(self.f(l))
        r_scores = self.u_r(self.f(r))
        e_scores = self.u_e(self.f(e))
        scores = self.u(self.f(torch.cat((e.detach(), l.detach(), r.detach()), -1)))
        if self.training:
            return scores, l_scores, r_scores, e_scores
        else:
            return scores


class AbstractMultiViewTMN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=torch.tanh):
        super(AbstractMultiViewTMN, self).__init__()

        self.u_l = nn.Linear(k, 1, bias=False)
        self.u_r = nn.Linear(k, 1, bias=False)
        self.u_t = nn.Linear(k, 1, bias=False)
        self.u_e = nn.Linear(k, 1, bias=False)
        self.f = non_linear
        self.W_l = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W_r = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W_t = nn.Bilinear(l_dim, l_dim, k, bias=True)
        self.W = nn.Bilinear(l_dim*2, r_dim, k, bias=True)
        self.V_l = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V_t = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V_r = nn.Linear(l_dim*2, k, bias=False)
        self.V = nn.Linear(l_dim*2 + r_dim, k, bias=False)

        self.control = nn.Sequential(nn.Linear(l_dim*2+r_dim, l_dim*2, bias=False), nn.Sigmoid())
        self.control_l = nn.Sequential(nn.Linear(l_dim+r_dim, l_dim, bias=False), nn.Sigmoid())
        self.control_r = nn.Sequential(nn.Linear(l_dim+r_dim, l_dim, bias=False), nn.Sigmoid())

    def forward(self, e1, e2, q, partial):
        ec1 = e1 * self.control_l(torch.cat((e1, q), -1))
        ec2 = e2 * self.control_r(torch.cat((e2, q), -1))
        e = torch.cat((e1, e2), 1)
        ec = e * self.control(torch.cat((e, q), -1))
        l = self.W_l(ec1, q) + self.V_l(torch.cat((ec1, q), 1))
        r = self.W_r(ec2, q) + self.V_r(torch.cat((ec2, q), 1))
        t = self.W_t(ec1, ec2) + self.V_t(torch.cat((ec1, ec2), 1))
        e = self.W(ec, q) + self.V(torch.cat((ec, q), 1))
        l_scores = self.u_l(self.f(l))
        r_scores = self.u_r(self.f(r))
        t_scores = self.u_t(self.f(t))
        e_scores = self.u_e(self.f(e))
        if partial:
            return (l_scores, r_scores, t_scores, e_scores)
        else:
            return torch.cat((l.detach(), r.detach(), t.detach(), e.detach()), -1)


class PairViewTMN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=torch.tanh):
        super(PairViewTMN, self).__init__()

        self.f = non_linear
        self.u = nn.Sequential(
            nn.Linear(k*4*2, k*4),
            nn.LeakyReLU(0.2),
            nn.Linear(k*4, 1),
        )
        # self.u = nn.Linear(k*4*2, 1, bias=False)

        self.singleview_match1 = AbstractMultiViewTMN(l_dim, r_dim, k, non_linear)
        self.singleview_match2 = AbstractMultiViewTMN(l_dim, r_dim, k, non_linear)

    def forward(self, e1, e2, q, partial=False):
        e11, e12 = e1
        e21, e22 = e2
        if partial and self.training:
            (l_scores1, r_scores1, t_scores1, e_scores1) = self.singleview_match1(e11, e21, q, True)
            (l_scores2, r_scores2, t_scores2, e_scores2) = self.singleview_match2(e12, e22, q, True)
            return torch.cat((l_scores1, l_scores2), -1), \
                   torch.cat((r_scores1, r_scores2), -1), \
                   torch.cat((t_scores1, t_scores2), -1), \
                   torch.cat((e_scores1, e_scores2), -1)
        else:
            f1 = self.singleview_match1(e11, e21, q, False)
            f2 = self.singleview_match2(e12, e22, q, False)
            scores = self.u(self.f(torch.cat((f1, f2), -1)))
            return scores


class MultiViewTMN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=torch.tanh):
        super(MultiViewTMN, self).__init__()

        self.f = non_linear
        # self.u = nn.Linear(k*4*3, 1, bias=False)
        self.u = nn.Sequential(
            nn.Linear(k * 4 * 3, k * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(k * 4, 1),
        )

        self.singleview_match1 = AbstractMultiViewTMN(l_dim, r_dim, k, non_linear)
        self.singleview_match2 = AbstractMultiViewTMN(l_dim, r_dim, k, non_linear)
        self.singleview_match3 = AbstractMultiViewTMN(l_dim, r_dim, k, non_linear)

    def forward(self, e1, e2, q, partial=False):
        e11, e12, e13 = e1
        e21, e22, e23 = e2
        if partial and self.training:
            (l_scores1, r_scores1, t_scores1, e_scores1) = self.singleview_match1(e11, e21, q, True)
            (l_scores2, r_scores2, t_scores2, e_scores2) = self.singleview_match2(e12, e22, q, True)
            (l_scores3, r_scores3, t_scores3, e_scores3) = self.singleview_match3(e13, e23, q, True)
            return torch.cat((l_scores1, l_scores2, l_scores3), -1), \
                   torch.cat((r_scores1, r_scores2, r_scores3), -1), \
                   torch.cat((t_scores1, t_scores2, t_scores3), -1), \
                   torch.cat((e_scores1, e_scores2, e_scores3), -1)
        else:
            f1 = self.singleview_match1(e11, e21, q, False)
            f2 = self.singleview_match2(e12, e22, q, False)
            f3 = self.singleview_match3(e13, e23, q, False)
            scores = self.u(self.f(torch.cat((f1, f2, f3), -1)))
            return scores