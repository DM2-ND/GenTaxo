import torch
import torch.nn as nn 
import torch.nn.functional as F
from itertools import product
import numpy as np
import more_itertools as mit
import re


EPS = 1e-9


def nll_loss(output, target):
    return F.nll_loss(output, target) 


def square_exp_loss(output, target, beta=1.0):
    """
    output: a (batch_size, 1) tensor, value should be positive
    target: a (batch_size, ) tensor of dtype int
    beta: a float weight of negative samples
    """
    loss = (output[target==1]**2).sum() + beta * torch.exp(-1.0*output[target==0]).sum()
    return loss


def bce_loss(output, target, reduction="mean"):
    """
    output: a (batch_size, 1) tensor
    target: a (batch_size, ) tensor of dtype int
    
    Note: here we revert the `target` because `output` is the "energy score" and thus smaller value indicates it is more likely to be a true position 
    """
    loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float(), reduction=reduction)
    return loss


def weighted_bce_loss(output, target, weight):
    loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float(), reduction="none")*weight
    return loss.sum() / weight.sum()


def cross_entropy_loss(output, target, beta=1.0):
    loss = F.cross_entropy(output, target.long(), reduction="mean")
    return loss


def kl_div_loss(output, target):
    loss = F.kl_div(output.log_softmax(1), target, reduction="batchmean")
    return loss


def margin_rank_loss(output, target, sample_size=32, margin=1.0):
    label = target.cpu().numpy()
    pos_indices = []
    neg_indices = []
    for cnt, sublabel in enumerate(mit.sliced(label, sample_size)):
        pos, neg = [], []
        for i, l in enumerate(sublabel):
            i += cnt * sample_size
            if l:
                pos.append(i)
            else:
                neg.append(i)
        len_p = len(pos)
        len_n = len(neg)
        pos_indices.extend([i for i in pos for _ in range(len_n)])
        neg_indices.extend(neg * len_p)

    y = -1 * torch.ones(output[pos_indices,:].shape[0]).to(target.device)
    loss = F.margin_ranking_loss(output[pos_indices,:], output[neg_indices,:], y, margin=margin, reduction="mean")
    return loss


def info_nce_loss(output, target):
    """
    output: a (batch_size, 1+negative_size) tensor
    target: a (batch_size, ) tensor of dtype long, all zeros
    """
    return F.cross_entropy(output, target, reduction="mean")


class DistMarginLoss:
    def __init__(self, spdist):
        self.spdist = torch.FloatTensor(spdist)  # vocab_size x vocab_size
        self.spdist /= self.spdist.max()

    def loss(self, output, target, nodes):
        label = target.cpu().numpy()
        sep_01 = np.array([0, 1], dtype=label.dtype)
        sep_10 = np.array([1, 0], dtype=label.dtype)

        # fast way to find subarray indices in a large array, c.f. https://stackoverflow.com/questions/14890216/return-the-indexes-of-a-sub-array-in-an-array
        sep10_indices = [(m.start() // label.itemsize)+1 for m in re.finditer(sep_10.tostring(), label.tostring())]
        end_indices = [(m.start() // label.itemsize)+1 for m in re.finditer(sep_01.tostring(), label.tostring())]
        end_indices.append(len(label))
        start_indices = [0] + end_indices[:-1]

        pair_indices = []
        for start, middle, end in zip(start_indices, sep10_indices, end_indices):
            pair_indices.extend(list(product(range(start, middle), range(middle, end))))
        positive_indices, negative_indices = zip(*pair_indices)
        positive_indices = list(positive_indices)
        negative_indices = list(negative_indices)
        positive_node_ids = [nodes[i] for i in positive_indices]
        negative_node_ids = [nodes[i] for i in negative_indices]
        margins = self.spdist[positive_node_ids, negative_node_ids].to(target.device)
        output = output.view(-1)

        # y = -1 * torch.ones(output[positive_indices,:].shape[0]).to(target.device)
        # loss = F.margin_ranking_loss(output[positive_indices,:], output[negative_indices,:], y, margin=margin, reduction="sum")
        loss = (-output[positive_indices].sigmoid().clamp(min=EPS) + output[negative_indices].sigmoid().clamp(min=EPS) + margins.clamp(min=EPS)).clamp(min=0)
        return loss.mean()
