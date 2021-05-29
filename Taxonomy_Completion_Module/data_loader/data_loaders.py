from torch.utils.data import DataLoader
from .dataset import *
import dgl
import torch
from itertools import chain 

BATCH_GRAPH_NODE_LIMIT = 100000


class UnifiedDataLoader(DataLoader):
    def __init__(self, mode, data_path, sampling_mode=1, batch_size=10, negative_size=20, max_pos_size=100,
                 expand_factor=50, shuffle=True, num_workers=8, cache_refresh_time=64, normalize_embed=False,
                 test_topk=-1, test=0):
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.shuffle = shuffle
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed
        test_flag = 'test' if test else 'train'

        raw_graph_dataset = MAGDataset(name="", path=data_path, raw=False)
        if 'g' in mode and 'p' in mode:
            msk_graph_dataset = GraphPathDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode,
                                                 negative_size=negative_size, max_pos_size=max_pos_size,
                                                 expand_factor=expand_factor, cache_refresh_time=cache_refresh_time,
                                                 normalize_embed=normalize_embed, test_topk=test_topk)
        elif 'g' in mode:
            msk_graph_dataset = GraphDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode,
                                             negative_size=negative_size,
                                             max_pos_size=max_pos_size, expand_factor=expand_factor,
                                             cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                             test_topk=test_topk)
        elif 'p' in mode:
            msk_graph_dataset = PathDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode, negative_size=negative_size,
                                            max_pos_size=max_pos_size, expand_factor=expand_factor,
                                            cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                            test_topk=test_topk)
        else:
            msk_graph_dataset = RawDataset(raw_graph_dataset, mode=test_flag,sampling_mode=sampling_mode, negative_size=negative_size,
                                           max_pos_size=max_pos_size, expand_factor=expand_factor,
                                           cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                           test_topk=test_topk)
        self.dataset = msk_graph_dataset
        self.num_workers = num_workers
        super(UnifiedDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                collate_fn=self.collate_fn, num_workers=self.num_workers,
                                                pin_memory=True)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader

    def collate_fn(self, samples):
        if 'g' in self.mode and 'p' in self.mode:
            us, vs, graphs_u, graphs_v, paths_u, paths_v, lens, queries, labels = map(list, zip(*chain(*samples)))
            lens = torch.tensor(lens)
            max_u, max_v = lens.max(dim=0)[0]
            paths_u = [p + [self.dataset.pseudo_leaf_node] * (max_u - len(p)) for p in paths_u]
            paths_v = [p + [self.dataset.pseudo_leaf_node] * (max_v - len(p)) for p in paths_v]
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), \
                   graphs_u, graphs_v, torch.tensor(paths_u), torch.tensor(paths_v), lens
        elif 'g' in self.mode:
            us, vs, graphs_u, graphs_v, queries, labels = map(list, zip(*chain(*samples)))
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), graphs_u, graphs_v, \
                   None, None, None
        elif 'p' in self.mode:
            us, vs, paths_u, paths_v, lens, queries, labels = map(list, zip(*chain(*samples)))
            lens = torch.tensor(lens)
            max_u, max_v = lens.max(dim=0)[0]
            paths_u = [p + [self.dataset.pseudo_leaf_node] * (max_u - len(p)) for p in paths_u]
            paths_v = [p + [self.dataset.pseudo_leaf_node] * (max_v - len(p)) for p in paths_v]
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), None, None, \
                   torch.tensor(paths_u), torch.tensor(paths_v), lens
        else:
            us, vs, queries, labels = map(list, zip(*chain(*samples)))
            return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), torch.tensor(vs), None, None, None, None, None

    def __str__(self):
        return "\n\t".join([
            f"UnifiedDataLoader mode: {self.mode}",
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
            f"expand_factor: {self.expand_factor}",
            f"cache_refresh_time: {self.cache_refresh_time}",
            f"normalize_embed: {self.normalize_embed}",
        ])


class TaxoExpanDataLoader(DataLoader):
    def __init__(self, mode, data_path, sampling_mode=1, batch_size=10, negative_size=20, max_pos_size=100,
                 expand_factor=50, shuffle=True, num_workers=8, cache_refresh_time=64, normalize_embed=False,
                 test_topk=-1):
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.shuffle = shuffle
        self.cache_refresh_time = cache_refresh_time
        self.normalize_embed = normalize_embed

        raw_graph_dataset = MAGDataset(name="", path=data_path, raw=False)
        msk_graph_dataset = ExpanDataset(raw_graph_dataset, sampling_mode=sampling_mode,
                                             negative_size=negative_size,
                                             max_pos_size=max_pos_size, expand_factor=expand_factor,
                                             cache_refresh_time=cache_refresh_time, normalize_embed=normalize_embed,
                                             test_topk=test_topk)
        self.dataset = msk_graph_dataset
        self.num_workers = num_workers
        super(TaxoExpanDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                collate_fn=self.collate_fn, num_workers=self.num_workers,
                                                pin_memory=True)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader

    def collate_fn(self, samples):
        us, graphs_u, paths_u, lens, queries, labels = map(list, zip(*chain(*samples)))
        if 'g' not in self.mode:
            graphs_u = None
        if 'r' in self.mode:
            lens = torch.tensor(lens)
            max_u = lens.max(dim=0)[0]
            paths_u = [p + [self.dataset.pseudo_leaf_node] * (max_u - len(p)) for p in paths_u]
            paths_u = torch.tensor(paths_u)
        else:
            lens = None
            paths_u = None
        return torch.tensor(queries), torch.tensor(labels), torch.tensor(us), graphs_u, paths_u, lens

    def __str__(self):
        return "\n\t".join([
            f"TaxoExpanDataLoader mode: {self.mode}",
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
            f"expand_factor: {self.expand_factor}",
            f"cache_refresh_time: {self.cache_refresh_time}",
            f"normalize_embed: {self.normalize_embed}",
        ])