import argparse
from collections import deque
import torch
import torch.nn.functional as F
import model.metric as module_metric
from data_loader.dataset import MAGDataset
from functools import partial
from itertools import chain
from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np
from itertools import chain
from networkx.algorithms import descendants


def rearrange(energy_scores, candidate_position_idx, true_position_idx):
    tmp = np.array([[x==y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels


def find_insert_posistion(node_ids, core_subgraph, holdout_graph, pseudo_leaf_node):
    node2pos = {}
    subgraph = core_subgraph
    for node in node_ids:
        parents = set()
        children = set()
        ps = deque(holdout_graph.predecessors(node))
        cs = deque(holdout_graph.successors(node))
        while ps:
            p = ps.popleft()
            if p in subgraph:
                parents.add(p)
            else:
                ps += list(holdout_graph.predecessors(p))
        while cs:
            c = cs.popleft()
            if c in subgraph:
                children.add(c)
            else:
                cs += list(holdout_graph.successors(c))
        if not children:
            children.add(pseudo_leaf_node)
        position = [(p, c) for p in parents for c in children]
        node2pos[node] = position
    return node2pos


def get_holdout_subgraph(node_ids, full_graph):
    node_to_remove = [n for n in full_graph.nodes if n not in node_ids]
    subgraph = full_graph.subgraph(node_ids).copy()
    for node in node_to_remove:
        parents = set()
        children = set()
        ps = deque(full_graph.predecessors(node))
        cs = deque(full_graph.successors(node))
        while ps:
            p = ps.popleft()
            if p in subgraph:
                parents.add(p)
            else:
                ps += list(full_graph.predecessors(p))
        while cs:
            c = cs.popleft()
            if c in subgraph:
                children.add(c)
            else:
                cs += list(full_graph.successors(c))
        for p in parents:
            for c in children:
                subgraph.add_edge(p, c)
    # remove jump edges
    node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
    for node in subgraph.nodes():
        if subgraph.out_degree(node) > 1:
            successors1 = set(subgraph.successors(node))
            successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
            checkset = successors1.intersection(successors2)
            if checkset:
                for s in checkset:
                    subgraph.remove_edge(node, s)
    return subgraph


def distances(query_node, edges, kv, pseudo_leaf_node):
    node2dist = {}
    dists = []
    for u, v in edges:
        if u in node2dist:
            du = node2dist[u]
        else:
            du = kv.distance(query_node, str(u))
            node2dist[u] = du
        if v == pseudo_leaf_node:
            dists.append(du)
        else:
            if v in node2dist:
                dv = node2dist[v]
            else:
                dv = kv.distance(query_node, str(v))
                node2dist[v] = dv
            dists.append((du + dv) / 2)
    return dists


def calculate_depth(graph):
    def depth(node):
        l = graph.successors(node)
        d = 1
        while True:
            next_level = set(chain.from_iterable([graph.successors(n) for n in l]))
            if next_level:
                l = next_level
                d += 1
            else:
                return d
    roots = [n for n in graph if graph.in_degree(n)==0]
    d = max([depth(r) for r in roots]) + 1
    return d


def main(args, metrics):
    graph_dataset = MAGDataset(name="", path=args.data_path, raw=False)
    metrics = [getattr(module_metric, met) for met in metrics]
    pre_metric = partial(module_metric.obtain_ranks, mode=0)

    full_graph = graph_dataset.g_full.to_networkx()
    core_subgraph = get_holdout_subgraph(graph_dataset.train_node_ids, full_graph)
    pseudo_leaf_node = -1
    for node in list(core_subgraph.nodes()):
        core_subgraph.add_edge(node, pseudo_leaf_node)
    node2descendants = {n: set(descendants(core_subgraph, n)) for n in core_subgraph.nodes}
    candidate_positions = list(set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()])))

    holdout_subgraph = get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids, full_graph)
    node2pos = find_insert_posistion(graph_dataset.test_node_ids, core_subgraph, holdout_subgraph, pseudo_leaf_node)

    node_features = graph_dataset.g_full.ndata['x']
    node_features = F.normalize(node_features, p=2, dim=1)
    kv = KeyedVectors(vector_size=node_features.shape[1])
    kv.add([str(i) for i in range(len(node_features))], node_features.numpy())

    all_ranks = []
    for node in tqdm(graph_dataset.test_node_ids):
        dists = distances(str(node), candidate_positions, kv, pseudo_leaf_node)
        scores, labels = rearrange(torch.Tensor(dists), candidate_positions, node2pos[node])
        all_ranks.extend(pre_metric(scores, labels))
    total_metrics = [metric(all_ranks) for metric in metrics]

    for i, mtr in enumerate(metrics):
        print('    {:15s}: {}'.format(mtr.__name__, total_metrics[i]))

    return


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data_path', default=None, type=str)
    args = args.parse_args()
    metrics = ["macro_mr", "micro_mr", "hit_at_1", "hit_at_5", "hit_at_10", "precision_at_1", "precision_at_5", "precision_at_10", "mrr_scaled_10"]
    print(args.data_path)
    main(args, metrics)
