import networkx as nx
from networkx.algorithms import descendants, ancestors
import dgl
from gensim.models import KeyedVectors
import numpy as np 
import torch 
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import pickle
import time
from tqdm import tqdm
import random
import copy
from itertools import chain, product, combinations
import os
import multiprocessing as mp
from functools import partial
from collections import defaultdict, deque
import more_itertools as mit


MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000


def add_edge_for_dgl(g, n1, n2):
    """
    https://github.com/dmlc/dgl/issues/1476 there is a bug in dgl add edges, so we need a wrapper
    """
    if not ((isinstance(n1, list) and len(n1) == 0) or (isinstance(n2, list) and len(n2) == 0)):
        g.add_edges(n1, n2)


def single_source_shortest_path_length(source,G,cutoff=None):
    """Compute the shortest path lengths from source to all reachable nodes.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    Returns
    -------
    lengths : dictionary
        Dictionary of shortest path lengths keyed by target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length=nx.single_source_shortest_path_length(G,0)
    >>> length[4]
    4
    >>> print(length)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    See Also
    --------
    shortest_path_length
    """
    seen={}                  # level (number of hops) when seen in BFS
    level=0                  # the current level
    nextlevel={source:1}  # dict of nodes to check at next level
    while nextlevel:
        thislevel=nextlevel  # advance to next level
        nextlevel={}         # and start a new list (fringe)
        for v in thislevel:
            if v not in seen:
                seen[v]=level # set the level of vertex v
                nextlevel.update(G[v]) # add neighbors of v
        if (cutoff is not None and cutoff <= level):  break
        level=level+1
    return (source, seen)  # return all path lengths as dictionary


def parallel_all_pairs_shortest_path_length(g, node_ids, num_workers=20):
    # TODO This can be trivially parallelized.
    res = {}
    pool = mp.Pool(processes=num_workers)
    p = partial(single_source_shortest_path_length, G=g, cutoff=None)
    result_list = pool.map(p, node_ids)
    for result in result_list:
        res[result[0]] = result[1]
    pool.close()
    pool.join()
    return res


class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0, c_count=0, create_date="None"):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date
        
    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)
        
    def __lt__(self, another_taxon):
        if self.level < another_taxon.level:
            return True
        else:
            return self.rank < another_taxon.rank


class MAGDataset(object):
    def __init__(self, name, path, embed_suffix="", raw=True, existing_partition=False, partition_pattern='leaf', shortest_path=False):
        """ Raw dataset class for MAG dataset

        Parameters
        ----------
        name : str
            taxonomy name
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed_suffix : str
            suffix of embedding file name, by default ""
        raw : bool, optional
            load raw dataset from txt (True) files or load pickled dataset (False), by default True
        existing_partition : bool, optional
            whether to use the existing the train/validation/test partitions or randomly sample new ones, by default False
        """
        self.name = name  # taxonomy name
        self.embed_suffix = embed_suffix
        self.existing_partition = existing_partition
        self.partition_pattern = partition_pattern
        self.g_full = dgl.DGLGraph()  # full graph, including masked train/validation node indices
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids
        self.test_node_ids = []  # a list of test node_ids
        self.shortest_path = shortest_path

        if raw:
            self._load_dataset_raw(path)
        else:
            self._load_dataset_pickled(path)

    def _load_dataset_pickled(self, pickle_path):
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)

        self.name = data["name"]
        self.g_full = data["g_full"]
        self.vocab = data["vocab"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]
        if self.shortest_path:
            self.shortest_path = data['shortest_path']

    def _load_dataset_raw(self, dir_path):
        """ Load data from three seperated files, generate train/validation/test partitions, and save to binary pickled dataset.
        Please refer to the README.md file for details.


        Parameters
        ----------
        dir_path : str
            The path to a directory containing three input files.
        """
        node_file_name = os.path.join(dir_path, f"{self.name}.terms")
        edge_file_name = os.path.join(dir_path, f"{self.name}.taxo")
        if self.embed_suffix == "":
            embedding_file_name = os.path.join(dir_path, f"{self.name}.terms.embed")
            output_pickle_file_name = os.path.join(dir_path, f"{self.name}.pickle.bin")
        else:
            embedding_file_name = os.path.join(dir_path, f"{self.name}.terms.{self.embed_suffix}.embed")
            output_pickle_file_name = os.path.join(dir_path, f"{self.name}.{self.embed_suffix}.pickle.bin")
        if self.existing_partition:
            train_node_file_name = os.path.join(dir_path, f"{self.name}.terms.train")
            validation_node_file_name = os.path.join(dir_path, f"{self.name}.terms.validation")
            test_file_name = os.path.join(dir_path, f"{self.name}.terms.test")

        tx_id2taxon = {}
        taxonomy = nx.DiGraph()

        # load nodes
        with open(node_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading terms"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    taxon = Taxon(tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                    tx_id2taxon[segs[0]] = taxon
                    taxonomy.add_node(taxon)

        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    taxonomy.add_edge(parent_taxon, child_taxon)

        # load embedding features
        print("Loading embedding ...")
        embeddings = KeyedVectors.load_word2vec_format(embedding_file_name)
        print(f"Finish loading embedding of size {embeddings.vectors.shape}")

        # load train/validation/test partition files if needed
        if self.existing_partition:
            print("Loading existing train/validation/test partitions")
            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)

        # generate vocab, tx_id is the old taxon_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
        tx_id2node_id = {node.tx_id:idx for idx, node in enumerate(taxonomy.nodes()) }
        node_id2tx_id = {v:k for k, v in tx_id2node_id.items()}
        self.vocab = [tx_id2taxon[node_id2tx_id[node_id]].norm_name + "@@@" + str(node_id) for node_id in node_id2tx_id]

        # generate dgl.DGLGraph()
        edges = []
        for edge in taxonomy.edges():
            parent_node_id = tx_id2node_id[edge[0].tx_id]
            child_node_id = tx_id2node_id[edge[1].tx_id]
            edges.append([parent_node_id, child_node_id])

        node_features = np.zeros(embeddings.vectors.shape)
        for node_id, tx_id in node_id2tx_id.items():
            node_features[node_id, :] = embeddings[tx_id]
        node_features = torch.FloatTensor(node_features)

        self.g_full.add_nodes(len(node_id2tx_id), {'x': node_features})
        self.g_full.add_edges([e[0] for e in edges], [e[1] for e in edges])

        # generate validation/test node_indices using either existing partitions or randomly sampled partition
        if self.existing_partition:
            self.train_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_train_node_list]
            self.validation_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_validation_node_list]
            self.test_node_ids = [tx_id2node_id[tx_id] for tx_id in raw_test_node_list]
        else:
            print("Partition graph ...")
            if self.partition_pattern == 'leaf':
                leaf_node_ids = []
                for node in taxonomy.nodes():
                    if taxonomy.out_degree(node) == 0:
                        leaf_node_ids.append(tx_id2node_id[node.tx_id])

                random.seed(47)
                random.shuffle(leaf_node_ids)
                validation_size = min(int(len(leaf_node_ids) * 0.1), MAX_VALIDATION_SIZE)
                test_size = min(int(len(leaf_node_ids) * 0.1), MAX_TEST_SIZE)
                self.validation_node_ids = leaf_node_ids[:validation_size]
                self.test_node_ids = leaf_node_ids[validation_size:(validation_size+test_size)]
                self.train_node_ids = [node_id for node_id in node_id2tx_id if node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
            elif self.partition_pattern == 'internal':
                root_node = [node for node in taxonomy.nodes() if taxonomy.in_degree(node) == 0]
                sampled_node_ids = [tx_id2node_id[node.tx_id] for node in taxonomy.nodes() if node not in root_node]
                random.seed(47)
                random.shuffle(sampled_node_ids)

                validation_size = min(int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
                test_size = min(int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)
                self.validation_node_ids = sampled_node_ids[:validation_size]
                self.test_node_ids = sampled_node_ids[validation_size:(validation_size+test_size)]
                self.train_node_ids = [node_id for node_id in node_id2tx_id if node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
            else:
                raise ValueError('Unknown partition method!')
            print("Finish partition graph ...")

        # Compute shortest path distances
        if self.shortest_path:
            dag = self._get_holdout_subgraph(self.train_node_ids).to_undirected()
            numnodes = len(node_id2tx_id)
            spdists = -1 * (np.ones((numnodes, numnodes), dtype=np.float))
            res = parallel_all_pairs_shortest_path_length(dag, self.train_node_ids)
            for u, dists in res.items():
                for v, dist in dists.items():
                    spdists[u][v] = int(dist)

            spdists[spdists == -1] = int(spdists.max())
            self.shortest_path = spdists

        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, 'wb') as fout:
            # Pickle the 'data' dictionary using the highest protocol available.
            data = {
                "name": self.name,
                "g_full": self.g_full,
                "vocab": self.vocab,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
                "shortest_path": self.shortest_path
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list

    def _get_holdout_subgraph(self, node_ids):
        full_graph = self.g_full.to_networkx()
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
            for p, c in product(parents, children):
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


class RawDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        self.test_topk = test_topk

        self.node_features = graph_dataset.g_full.ndata['x']
        full_graph = graph_dataset.g_full.to_networkx()
        train_node_ids = graph_dataset.train_node_ids
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        if len(roots) > 1:
            self.root = len(full_graph.nodes)
            for r in roots:
                full_graph.add_edge(self.root, r)
            root_vector = torch.mean(self.node_features[roots], dim=0, keepdim=True)
            self.node_features = torch.cat((self.node_features, root_vector), 0)
            self.vocab = graph_dataset.vocab + ['root', 'leaf']
            train_node_ids.append(self.root)
        else:
            self.root = roots[0]
            self.vocab = graph_dataset.vocab + ['leaf']
        self.full_graph = full_graph

        if mode == 'train':
            # add pseudo leaf node to core graph
            self.core_subgraph = self._get_holdout_subgraph(train_node_ids)
            self.pseudo_leaf_node = len(full_graph.nodes)
            for node in list(self.core_subgraph.nodes()):
                self.core_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]
            # for pseudo leaf node
            leaf_vector = torch.zeros((1, self.node_features.size(1))) # zero vector works best
            self.node_features = torch.cat((self.node_features, leaf_vector), 0)
            if self.normalize_embed:
                self.node_features = F.normalize(self.node_features, p=2, dim=1)

            # add interested node list and subgraph
            # remove supersource nodes (i.e., nodes without in-degree 0)
            interested_node_set = set(train_node_ids) - set([self.root])
            self.node_list = list(interested_node_set)

            # build node2pos, node2nbs, node2edge
            self.node2pos, self.node2edge = {}, {}
            self.node2parents, self.node2children, self.node2nbs = {}, {}, {self.pseudo_leaf_node:[]}
            for node in interested_node_set:
                parents = set(self.core_subgraph.predecessors(node))
                children = set(self.core_subgraph.successors(node))
                if len(children) > 1:
                    children = [i for i in children if i != self.pseudo_leaf_node]
                node_pos_edges = [(pre, suc) for pre in parents for suc in children if pre!=suc]
                self.node2edge[node] = set(self.core_subgraph.in_edges(node)).union(set(self.core_subgraph.out_edges(node)))
                self.node2pos[node] = node_pos_edges
                self.node2parents[node] = parents
                self.node2children[node] = children
                self.node2nbs[node] = parents.union(children)
            self.node2nbs[self.root] = set([n for n in self.core_subgraph.successors(self.root) if n != self.pseudo_leaf_node])

            self.valid_node_list = graph_dataset.validation_node_ids
            holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids)
            self.valid_node2pos = self._find_insert_posistion(graph_dataset.validation_node_ids, holdout_subgraph)

            self.test_node_list = graph_dataset.test_node_ids
            holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids)
            self.test_node2pos = self._find_insert_posistion(graph_dataset.test_node_ids, holdout_subgraph)

            # used for sampling negative positions during train/validation stage
            self.pointer = 0
            self.all_edges = list(self._get_candidate_positions(self.core_subgraph))
            self.edge2dist = {(u, v): nx.shortest_path_length(self.core_subgraph, u, v) for (u, v) in self.all_edges}
            random.shuffle(self.all_edges)
        elif mode == 'test':
            # add pseudo leaf node to core graph
            self.core_subgraph = self.full_graph
            self.pseudo_leaf_node = len(full_graph.nodes)
            self.node_list = list(self.core_subgraph.nodes())
            for node in self.node_list:
                self.core_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if
                               self.core_subgraph.out_degree(node) == 1]
            # for pseudo leaf node
            leaf_vector = torch.zeros((1, self.node_features.size(1)))  # zero vector works best
            self.node_features = torch.cat((self.node_features, leaf_vector), 0)
            if self.normalize_embed:
                self.node_features = F.normalize(self.node_features, p=2, dim=1)

            # used for sampling negative positions during train/validation stage
            self.all_edges = list(self._get_candidate_positions(self.core_subgraph))

        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    def __str__(self):
        return f"{self.__class__.__name__} mode:{self.mode}"

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                res.append([u, v, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            res.append([u, v, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            res.append([u, v, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def _get_holdout_subgraph(self, node_ids):
        node_to_remove = [n for n in self.full_graph.nodes if n not in node_ids]
        subgraph = self.full_graph.subgraph(node_ids).copy()
        for node in node_to_remove:
            parents = set()
            children = set()
            ps = deque(self.full_graph.predecessors(node))
            cs = deque(self.full_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.full_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.full_graph.successors(c))
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

    def _get_candidate_positions(self, graph):
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates

    def _find_insert_posistion(self, node_ids, holdout_graph, ignore=[]):
        node2pos = {}
        subgraph = self.core_subgraph
        for node in node_ids:
            if node in ignore:
                continue
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
                children.add(self.pseudo_leaf_node)
            position = [(p, c) for p in parents for c in children if p!=c]
            node2pos[node] = position
        return node2pos

    def _get_negative_anchors(self, query_node, negative_size):
        if self.sampling_mode == 0:
            return self._get_at_most_k_negatives(query_node, negative_size)
        elif self.sampling_mode == 1:
            return self._get_exactly_k_negatives(query_node, negative_size)

    def _get_at_most_k_negatives(self, query_node, negative_size):
        """ Generate AT MOST negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        while True:
            negatives = [ele for ele in self.all_edges[self.pointer: self.pointer + negative_size] if
                         ele not in self.node2pos[query_node] and ele not in self.node2edge[query_node]]
            if len(negatives) > 0:
                break
            self.pointer += negative_size
            if self.pointer >= len(self.all_edges):
                self.pointer = 0

        return negatives

    def _get_exactly_k_negatives(self, query_node, negative_size, ignore=[]):
        """ Generate EXACTLY negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in self.all_edges[self.pointer: self.pointer + n_lack] if
                                  ele not in self.node2pos[query_node] and ele not in self.node2edge[query_node] and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_edges):
                self.pointer = 0
                random.shuffle(self.all_edges)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives


class GraphDataset(RawDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        super(GraphDataset, self).__init__(graph_dataset, mode, sampling_mode, negative_size, max_pos_size,
                                           expand_factor, cache_refresh_time, normalize_embed, test_topk)

        # used for caching local subgraphs
        self.cache = {}  # if g = self.cache[anchor_node], then g is the egonet centered on the anchor_node
        self.cache_counter = {}  # if n = self.cache[anchor_node], then n is the number of times you used this cache

        lg = dgl.DGLGraph()
        lg.add_nodes(1, {"_id": torch.tensor([self.pseudo_leaf_node]), "pos": torch.tensor([1])})
        lg.add_edges(lg.nodes(), lg.nodes())
        self.cache[self.pseudo_leaf_node] = lg

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
                res.append([u, v, u_egonet, v_egonet, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            res.append([u, v, u_egonet, v_egonet, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            res.append([u, v, u_egonet, v_egonet, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def _check_cache_flag(self, node):
        return (node in self.cache) and (self.cache_counter[node] < self.cache_refresh_time)

    def _get_subgraph_and_node_pair(self, query_node, anchor_node_u, anchor_node_v):
        """ Generate anchor_egonet and obtain query_node feature

        instance_mode: 0 means negative example, 1 means positive example
        """

        # [IMPORTANT]
        # if anchor_node_u == self.pseudo_leaf_node:
        #     return self.cache[anchor_node_u]

        if anchor_node_u == self.pseudo_leaf_node:
            g_u = self.cache[anchor_node_u]
        else:
            u_cache_flag = self._check_cache_flag(anchor_node_u)
            u_flag = ((query_node < 0) or (anchor_node_u not in self.node2nbs[query_node])) and (anchor_node_u not in self.node2nbs[anchor_node_v])
            if u_flag and u_cache_flag:
                g_u = self.cache[anchor_node_u]
                self.cache_counter[anchor_node_u] += 1
            else:
                g_u = self._get_subgraph(query_node, anchor_node_u, anchor_node_v, u_flag)
                if u_flag:  # save to cache
                    self.cache[anchor_node_u] = g_u
                    self.cache_counter[anchor_node_u] = 0

        if anchor_node_v == self.pseudo_leaf_node:
            g_v = self.cache[anchor_node_v]
        else:
            v_cache_flag = self._check_cache_flag(anchor_node_v)
            v_flag = ((query_node < 0) or (anchor_node_v not in self.node2nbs[query_node])) and (anchor_node_v not in self.node2nbs[anchor_node_u])
            if v_flag and v_cache_flag:
                g_v = self.cache[anchor_node_v]
                self.cache_counter[anchor_node_v] += 1
            else:
                g_v = self._get_subgraph(query_node, anchor_node_v, anchor_node_u, v_flag)
                if v_flag:  # save to cache
                    self.cache[anchor_node_v] = g_v
                    self.cache_counter[anchor_node_v] = 0

        return g_u, g_v

    def _get_subgraph(self, query_node, anchor_node, other_anchor_node, instance_mode):
        if instance_mode:  # do not need to worry about query_node appears to be the child of anchor_node
            # parents of anchor node
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor)]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)

                # # anchor node itself
                # parent_node_idx = 0
                # nodes = [anchor_node]
                # nodes_pos = [1]

            else:
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node)]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if n != self.pseudo_leaf_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor) if
                                n != self.pseudo_leaf_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))
        else:  # remove query_node from the children set of anchor_node
            # TODO maybe include query node's neighbor
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor) if n!=query_node and n!=other_anchor_node]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # parent_node_idx = 0
                # nodes = [anchor_node]
                # nodes_pos = [1]
            # parents of anchor node
            else:
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node) if n != query_node and n!=other_anchor_node]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if
                                n != self.pseudo_leaf_node and n != query_node and n!=other_anchor_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor) if
                                n != self.pseudo_leaf_node and n != query_node and n!=other_anchor_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))

        # create dgl graph with features
        g = dgl.DGLGraph()
        g.add_nodes(len(nodes), {"_id": torch.tensor(nodes), "pos": torch.tensor(nodes_pos)})
        add_edge_for_dgl(g, list(range(parent_node_idx)), parent_node_idx)
        add_edge_for_dgl(g, parent_node_idx, list(range(parent_node_idx + 1, len(nodes))))

        # add self-cycle
        g.add_edges(g.nodes(), g.nodes())

        return g


class PathDataset(RawDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        super(PathDataset, self).__init__(graph_dataset, mode, sampling_mode, negative_size, max_pos_size,
                                           expand_factor, cache_refresh_time, normalize_embed, test_topk)
        self.node2root_path = self._get_path_to_root()
        self.node2leaf_path = self._get_path_to_leaf()

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
                res.append([u, v, u_path, v_path, lens, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            res.append([u, v, u_path, v_path, lens, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            res.append([u, v, u_path, v_path, lens, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def _get_path_to_root(self):
        node2root_path = {n:[] for n in self.node_list}
        q = deque([self.root])
        node2root_path[self.root] = [[self.root]]
        visit = []
        while q:
            i = q.popleft()
            if i in visit:
                continue
            else:
                visit.append(i)
            children = self.core_subgraph.successors(i)
            for c in children:
                if c == self.pseudo_leaf_node:
                    continue
                if c not in q:
                    q.append(c)
                for path in node2root_path[i]:
                    node2root_path[c].append([c]+path)
        return node2root_path

    def _get_path_to_leaf(self):
        leafs = [n for n in self.core_subgraph.nodes if self.core_subgraph.out_degree(n)==1]
        node2leaf_path = {n:[] for n in self.node_list}
        q = deque(leafs)
        for n in leafs:
            node2leaf_path[n] = [[n, self.pseudo_leaf_node]]
        visit = []
        while q:
            i = q.popleft()
            if i in visit:
                continue
            else:
                visit.append(i)
            parents = self.core_subgraph.predecessors(i)
            for p in parents:
                if p == self.root:
                    continue
                if p not in q:
                    q.append(p)
                for path in node2leaf_path[i]:
                    node2leaf_path[p].append([p]+path)
        return node2leaf_path

    def _get_edge_node_path(self, query_node, edge):
        pu = random.choice(self.node2root_path[edge[0]])
        pu = [n for n in pu if n!=query_node]
        if edge[1] == self.pseudo_leaf_node:
            pv = [self.pseudo_leaf_node]
        else:
            pv = random.choice(self.node2leaf_path[edge[1]])
            pv = [n for n in pv if n!=query_node]
        len_pu = len(pu)
        len_pv = len(pv)
        return pu, pv, (len_pu, len_pv)

    def _get_batch_edge_node_path(self, edges):
        bpu, bpv, lens = zip(*[self._get_edge_node_path(None, edge) for edge in edges])
        lens = torch.tensor(lens)
        max_u, max_v = lens.max(dim=0)[0]
        bpu = [p+[self.pseudo_leaf_node]*(max_u-len(p)) for p in bpu]
        bpv = [p+[self.pseudo_leaf_node]*(max_v-len(p)) for p in bpv]
        return torch.tensor(bpu), torch.tensor(bpv), lens


class GraphPathDataset(GraphDataset, PathDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        super(GraphPathDataset, self).__init__(graph_dataset, mode, sampling_mode, negative_size, max_pos_size,
                                          expand_factor, cache_refresh_time, normalize_embed, test_topk)

    def __getitem__(self, idx):
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
                u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
                res.append([u, v, u_egonet, v_egonet, u_path, v_path, lens, query_node, (1, 1, 1, 1)])
        elif self.sampling_mode > 0:
            u, v = random.choice(self.node2pos[query_node])
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            res.append([u, v, u_egonet, v_egonet, u_path, v_path, lens, query_node, (1, 1, 1, 1)])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u, v in negative_anchors:
            u_egonet, v_egonet = self._get_subgraph_and_node_pair(query_node, u, v)
            u_path, v_path, lens = self._get_edge_node_path(query_node, (u, v))
            u_flag = int(u in self.node2parents[query_node])
            v_flag = int(v in self.node2children[query_node])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            res.append([u, v, u_egonet, v_egonet, u_path, v_path, lens, query_node, (0, u_flag, v_flag, e_flag)])

        return tuple(res)


class ExpanDataset(GraphPathDataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100, expand_factor=64, cache_refresh_time=128, normalize_embed=False, test_topk=-1):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        self.test_topk = test_topk

        self.node_features = graph_dataset.g_full.ndata['x']
        full_graph = graph_dataset.g_full.to_networkx()
        train_node_ids = graph_dataset.train_node_ids
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        if len(roots) > 1:
            self.root = len(full_graph.nodes)
            for r in roots:
                full_graph.add_edge(self.root, r)
            root_vector = torch.mean(self.node_features[roots], dim=0, keepdim=True)
            self.node_features = torch.cat((self.node_features, root_vector), 0)
            self.vocab = graph_dataset.vocab + ['root', 'leaf']
            train_node_ids.append(self.root)
        else:
            self.root = roots[0]
            self.vocab = graph_dataset.vocab + ['leaf']
        self.full_graph = full_graph

        # add pseudo leaf node to core graph
        self.core_subgraph = self._get_holdout_subgraph(train_node_ids)
        self.pseudo_leaf_node = len(full_graph.nodes)
        for node in list(self.core_subgraph.nodes()):
            self.core_subgraph.add_edge(node, self.pseudo_leaf_node)
        self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]
        # for pseudo leaf node
        leaf_vector = torch.zeros((1, self.node_features.size(1)))  # zero vector works best
        self.node_features = torch.cat((self.node_features, leaf_vector), 0)
        if self.normalize_embed:
            self.node_features = F.normalize(self.node_features, p=2, dim=1)

        # add interested node list and subgraph
        # remove supersource nodes (i.e., nodes without in-degree 0)
        interested_node_set = set(train_node_ids) - set([self.root])
        self.node_list = list(interested_node_set)

        # build node2pos, node2nbs, node2edge
        self.node2pos = {}
        self.node2parents, self.node2children, self.node2nbs = {}, {}, {self.pseudo_leaf_node: []}
        for node in interested_node_set:
            parents = set(self.core_subgraph.predecessors(node))
            children = set(self.core_subgraph.successors(node))
            if len(children) > 1:
                children = [i for i in children if i != self.pseudo_leaf_node]
            self.node2pos[node] = list(parents)
            self.node2parents[node] = parents
            self.node2children[node] = children
            self.node2nbs[node] = parents.union(children)

        holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids)
        valid_node2pos = self._find_insert_posistion(graph_dataset.validation_node_ids, holdout_subgraph)
        self.valid_node2pos = {node: set([p for (p, c) in pos_l if c == self.pseudo_leaf_node]) for node, pos_l in valid_node2pos.items()}
        self.valid_node2parents = {node: set([p for (p, c) in pos_l]) for node, pos_l in valid_node2pos.items()}
        self.valid_node_list = [node for node, pos in self.valid_node2pos.items() if len(pos)]

        holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids)
        test_node2pos = self._find_insert_posistion(graph_dataset.test_node_ids, holdout_subgraph)
        self.test_node2pos = {node: set([p for (p, c) in pos_l if c == self.pseudo_leaf_node]) for node, pos_l in test_node2pos.items()}
        self.test_node2parent = {node: set([p for (p, c) in pos_l]) for node, pos_l in test_node2pos.items()}
        self.test_node_list = [node for node, pos in self.test_node2pos.items() if len(pos)]

        # used for sampling negative positions during train/validation stage
        self.pointer = 0
        self.all_nodes = list(self.core_subgraph.nodes())
        random.shuffle(self.all_nodes)

        # used for caching local subgraphs
        self.cache = {}  # if g = self.cache[anchor_node], then g is the egonet centered on the anchor_node
        self.cache_counter = {}  # if n = self.cache[anchor_node], then n is the number of times you used this cache

        self.node2root_path = self._get_path_to_root()

        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        One data instance is a list of (anchor_egonet, query_node_feature, label) triplets.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        query_node = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            pos_positions = self.node2pos[query_node]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u in pos_positions:
                u_egonet = self._get_subgraph_and_node_pair(query_node, u)
                u_path, lens = self._get_edge_node_path(query_node, u)
                res.append([u, u_egonet, u_path, lens, query_node, 1])
        elif self.sampling_mode > 0:
            u = random.choice(self.node2pos[query_node])
            u_egonet = self._get_subgraph_and_node_pair(query_node, u)
            u_path, lens = self._get_edge_node_path(query_node, u)
            res.append([u, u_egonet, u_path, lens, query_node, 1])

        # select negative parents
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_anchors = self._get_negative_anchors(query_node, negative_size)

        # generate negative triplets
        for u in negative_anchors:
            u_egonet = self._get_subgraph_and_node_pair(query_node, u)
            u_path, lens = self._get_edge_node_path(query_node, u)
            res.append([u, u_egonet, u_path, lens, query_node, 0])

        return tuple(res)

    def _get_negative_anchors(self, query_node, negative_size):
        if self.sampling_mode == 0:
            return self._get_at_most_k_negatives(query_node, negative_size)
        elif self.sampling_mode == 1:
            return self._get_exactly_k_negatives(query_node, negative_size)

    def _get_at_most_k_negatives(self, query_node, negative_size):
        """ Generate AT MOST negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_nodes)

        while True:
            negatives = [ele for ele in self.all_nodes[self.pointer: self.pointer + negative_size] if
                         ele not in self.node2pos[query_node]]
            if len(negatives) > 0:
                break
            self.pointer += negative_size
            if self.pointer >= len(self.all_nodes):
                self.pointer = 0

        return negatives

    def _get_exactly_k_negatives(self, query_node, negative_size, ignore=[]):
        """ Generate EXACTLY negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_nodes)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in self.all_nodes[self.pointer: self.pointer + n_lack] if ele not in self.node2pos[query_node] and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_nodes):
                self.pointer = 0
                random.shuffle(self.all_nodes)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives

    def _get_subgraph_and_node_pair(self, query_node, anchor_node_u):
        """ Generate anchor_egonet and obtain query_node feature

        instance_mode: 0 means negative example, 1 means positive example
        """

        # [IMPORTANT]
        cache_flag = self._check_cache_flag(anchor_node_u)
        flag = (query_node < 0) or (anchor_node_u not in self.node2nbs[query_node])
        if flag and cache_flag:
            g_u = self.cache[anchor_node_u]
            # self.cache_counter[anchor_node_u] += 1
        else:
            g_u = self._get_subgraph(query_node, anchor_node_u, flag)
            if flag:  # save to cache
                self.cache[anchor_node_u] = g_u
                self.cache_counter[anchor_node_u] = 0

        return g_u

    def _get_subgraph(self, query_node, anchor_node, instance_mode):
        if instance_mode:  # do not need to worry about query_node appears to be the child of anchor_node
            # parents of anchor node
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor)]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)

                # # anchor node itself
                # parent_node_idx = 0
                # nodes = [anchor_node]
                # nodes_pos = [1]

            else:
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node)]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if n != self.pseudo_leaf_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor)
                                if
                                n != self.pseudo_leaf_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))
        else:  # remove query_node from the children set of anchor_node
            # TODO maybe include query node's neighbor
            if anchor_node == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k=self.expand_factor) if n != query_node]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # parent_node_idx = 0
                # nodes = [anchor_node]
                # nodes_pos = [1]
            # parents of anchor node
            else:
                nodes = [n for n in self.core_subgraph.predecessors(anchor_node) if n != query_node]
                nodes_pos = [0] * len(nodes)
                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor_node)
                nodes_pos.append(1)
                # siblings of query node (i.e., children of anchor node)
                if self.core_subgraph.out_degree(anchor_node) <= self.expand_factor:
                    siblings = [n for n in self.core_subgraph.successors(anchor_node) if
                                n != self.pseudo_leaf_node and n != query_node]
                else:
                    siblings = [n for n in
                                random.choices(list(self.core_subgraph.successors(anchor_node)), k=self.expand_factor)
                                if
                                n != self.pseudo_leaf_node and n != query_node]
                nodes.extend(siblings)
                nodes_pos.extend([2] * len(siblings))

        # create dgl graph with features
        g = dgl.DGLGraph()
        g.add_nodes(len(nodes), {"_id": torch.tensor(nodes), "pos": torch.tensor(nodes_pos)})
        add_edge_for_dgl(g, list(range(parent_node_idx)), parent_node_idx)
        add_edge_for_dgl(g, parent_node_idx, list(range(parent_node_idx + 1, len(nodes))))

        # add self-cycle
        g.add_edges(g.nodes(), g.nodes())

        return g

    def _get_edge_node_path(self, query_node, parent):
        if parent == self.pseudo_leaf_node:
            pu = [self.pseudo_leaf_node]
        else:
            pu = random.choice(self.node2root_path[parent])
            pu = [n for n in pu if n!=query_node]
        len_pu = len(pu)
        return pu, len_pu

    def _get_batch_edge_node_path(self, edges):
        bpu, lens = zip(*[self._get_edge_node_path(None, edge) for edge in edges])
        lens = torch.tensor(lens)
        max_u = lens.max(dim=0)[0]
        bpu = [p+[self.pseudo_leaf_node]*(max_u-len(p)) for p in bpu]
        return torch.tensor(bpu), lens