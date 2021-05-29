import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from itertools import combinations
from base import BaseModel
from .model_zoo import *


class AbstractPathModel(nn.Module):
    def __init__(self):
        super(AbstractPathModel, self).__init__()

    def init(self, **options):
        self.hidden_size = options['out_dim']
        in_dim = options['in_dim']
        out_dim = options['out_dim']
        self.p_lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size, batch_first=True)
        self.c_lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size, batch_first=True)
        self.p_control = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())
        self.c_control = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU())

    def init_hidden(self, batch_size, device):
        hidden = (torch.randn(1, batch_size, self.hidden_size).to(device), torch.randn(1, batch_size, self.hidden_size).to(device))
        return hidden

    def encode_parent_path(self, p, lens):
        batch_size, seq_len = p.size()
        hidden = self.init_hidden(batch_size, self.device)
        p = self.embedding(p)
        c = self.p_control(p[:, 0, :]).view(batch_size, 1, -1)
        X = torch.nn.utils.rnn.pack_padded_sequence(p, lens, batch_first=True, enforce_sorted=False)
        X, hidden = self.p_lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = (c*X).max(dim=1)[0]
        return X

    def encode_child_path(self, p, lens):
        batch_size, seq_len = p.size()
        hidden = self.init_hidden(batch_size, self.device)
        p = self.embedding(p)
        c = self.c_control(p[:, 0, :]).view(batch_size, 1, -1)
        X = torch.nn.utils.rnn.pack_padded_sequence(p, lens, batch_first=True, enforce_sorted=False)
        X, hidden = self.c_lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = (c*X).max(dim=1)[0]
        return X

    def forward_path_encoders(self, pu, pv, lens):
        pu = pu.to(self.device)
        pv = pv.to(self.device)
        lens = lens.to(self.device)
        hpu = self.encode_parent_path(pu, lens[:, 0])
        hpv = self.encode_child_path(pv, lens[:, 1])
        return hpu, hpv


class AbstractGraphModel(nn.Module):
    def __init__(self):
        super(AbstractGraphModel, self).__init__()

    def init(self, **options):
        propagation_method = options['propagation_method']
        readout_method = options['readout_method']
        options = options
        if propagation_method == "GCN":
            self.parent_graph_propagate = GCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                activation=F.leaky_relu, in_dropout=options["feat_drop"], hidden_dropout=options["hidden_drop"],
                output_dropout=options["out_drop"])
            self.child_graph_propagate = GCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                activation=F.leaky_relu, in_dropout=options["feat_drop"], hidden_dropout=options["hidden_drop"],
                output_dropout=options["out_drop"])
        elif propagation_method == "PGCN":
            self.parent_graph_propagate = PGCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], activation=F.leaky_relu, in_dropout=options["feat_drop"],
                hidden_dropout=options["hidden_drop"], output_dropout=options["out_drop"])
            self.child_graph_propagate = PGCN(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], activation=F.leaky_relu, in_dropout=options["feat_drop"],
                hidden_dropout=options["hidden_drop"], output_dropout=options["out_drop"])
        elif propagation_method == "GAT":
            self.parent_graph_propagate = GAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                heads=options["heads"], activation=F.leaky_relu, feat_drop=options["feat_drop"],
                attn_drop=options["attn_drop"])
            self.child_graph_propagate = GAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], num_layers=options["num_layers"],
                heads=options["heads"], activation=F.leaky_relu, feat_drop=options["feat_drop"],
                attn_drop=options["attn_drop"])
        elif propagation_method == "PGAT":
            self.parent_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
            self.child_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
        else:
            assert f"Unacceptable Graph Propagation Method: {self.propagation_method}"

        if readout_method == "MR":
            self.p_readout = MeanReadout()
            self.c_readout = MeanReadout()
        elif readout_method == "WMR":
            self.p_readout = WeightedMeanReadout()
            self.c_readout = WeightedMeanReadout()
        else:
            assert f"Unacceptable Readout Method: {self.readout_method}"

    def encode_parent_graph(self, g):
        h = self.embedding(g.ndata['_id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.parent_graph_propagate(g, h)
        h = self.p_readout(g, pos)
        return h

    def encode_child_graph(self, g):
        h = self.embedding(g.ndata['_id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.child_graph_propagate(g, h)
        h = self.c_readout(g, pos)
        return h

    def forward_graph_encoders(self, gu, gv):
        hgu = self.encode_parent_graph(gu)
        hgv = self.encode_child_graph(gv)
        return hgu, hgv


class MatchModel(BaseModel, AbstractPathModel, AbstractGraphModel):
    def __init__(self, mode, **options):
        super(MatchModel, self).__init__()
        self.options = options
        self.mode = mode

        l_dim = 0
        if 'r' in self.mode:
            l_dim += options["in_dim"]
        if 'g' in self.mode:
            l_dim += options["out_dim"]
            AbstractGraphModel.init(self, **options)
        if 'p' in self.mode:
            l_dim += options["out_dim"]
            AbstractPathModel.init(self, **options)
        self.l_dim = l_dim
        self.r_dim = options["in_dim"]

        if options['matching_method'] == "MLP":
            self.match = MLP(self.l_dim, self.r_dim, 100, options["k"])
        elif options['matching_method'] == "SLP":
            self.match = SLP(self.l_dim, self.r_dim, 100)
        elif options['matching_method'] == "DST":
            self.match = DST(self.l_dim, self.r_dim)
        elif options['matching_method'] == "LBM":
            self.match = LBM(self.l_dim, self.r_dim)
        elif options['matching_method'] == "BIM":
            self.match = BIM(self.l_dim, self.r_dim)
        elif options['matching_method'] == "Arborist":
            self.match = Arborist(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "NTN":
            self.match = NTN(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "CNTN":
            self.match = CNTN(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "TMN":
            self.match = TMN(self.l_dim, self.r_dim, options["k"])
        else:
            assert f"Unacceptable Matching Method: {options['matching_method']}"

    def forward_encoders(self, u=None, v=None, gu=None, gv=None, pu=None, pv=None, lens=None):
        ur, vr = [], []
        if 'r' in self.mode:
            hu = self.embedding(u.to(self.device))
            hv = self.embedding(v.to(self.device))
            ur.append(hu)
            vr.append(hv)
        if 'g' in self.mode:
            gu = dgl.batch(gu)
            gv = dgl.batch(gv)
            hgu, hgv = self.forward_graph_encoders(gu, gv)
            ur.append(hgu)
            vr.append(hgv)
        if 'p' in self.mode:
            hpu, hpv = self.forward_path_encoders(pu, pv, lens)
            ur.append(hpu)
            vr.append(hpv)
        ur = torch.cat(ur, -1)
        vr = torch.cat(vr, -1)
        return ur, vr

    def forward(self, q, *inputs):
        qf = self.embedding(q.to(self.device))
        ur, vr = self.forward_encoders(*inputs)
        scores = self.match(ur, vr, qf)
        return scores


class ExpanMatchModel(BaseModel, AbstractPathModel, AbstractGraphModel):
    def __init__(self, mode, **options):
        super(ExpanMatchModel, self).__init__()
        self.options = options
        self.mode = mode

        l_dim = 0
        if 'r' in self.mode:
            l_dim += options["in_dim"]
        if 'g' in self.mode:
            l_dim += options["out_dim"]
            AbstractGraphModel.init(self, **options)
        if 'p' in self.mode:
            l_dim += options["out_dim"]
            AbstractPathModel.init(self, **options)
        self.l_dim = l_dim
        self.r_dim = options["in_dim"]

        if options['matching_method'] == "NTN":
            self.match = RawNTN(self.l_dim, self.r_dim, options["k"])
        if options['matching_method'] == "BIM":
            self.match = RawBIM(self.l_dim, self.r_dim)
        if options['matching_method'] == "MLP":
            self.match = RawMLP(self.l_dim, self.r_dim, 100, options["k"])
        elif options['matching_method'] == "ARB":
            self.match = RawArborist(self.l_dim, self.r_dim, options["k"])
        else:
            assert f"Unacceptable Matching Method: {options['matching_method']}"

    def forward_encoders(self, u=None, gu=None, pu=None, lens=None):
        ur = []
        if 'r' in self.mode:
            hu = self.embedding(u.to(self.device))
            ur.append(hu)
        if 'g' in self.mode:
            gu = dgl.batch(gu)
            hgu = self.encode_parent_graph(gu)
            ur.append(hgu)
        if 'p' in self.mode:
            pu = pu.to(self.device)
            lens = lens.to(self.device)
            hpu = self.encode_parent_path(pu, lens)
            ur.append(hpu)
        ur = torch.cat(ur, -1)
        return ur

    def forward(self, q, us, graphs, paths, lens):
        qf = self.embedding(q.to(self.device))
        ur = self.forward_encoders(us, graphs, paths, lens)
        scores = self.match(ur, qf)
        return scores
