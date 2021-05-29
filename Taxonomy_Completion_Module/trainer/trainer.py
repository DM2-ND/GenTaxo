import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
import dgl
from tqdm import tqdm
import time, copy, random
import itertools
import json
import more_itertools as mit
from functools import partial
from collections import defaultdict
from model.model import *
from model.loss import *
from data_loader.data_loaders import *


MAX_CANDIDATE_NUM=100000


def rearrange(energy_scores, candidate_position_idx, true_position_idx):
    tmp = np.array([[x==y for x in candidate_position_idx] for y in true_position_idx]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.test_batch_size = config['trainer']['test_batch_size']
        self.is_infonce_training = config['loss'].startswith("info_nce")
        self.is_focal_loss = config['loss'].startswith("FocalLoss")
        self.data_loader = data_loader
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_mode = self.config['lr_scheduler']['args']['mode']  # "min" or "max"
        self.log_step = len(data_loader) // 5
        self.pre_metric = pre_metric
        self.writer.add_text('Text', 'Model Architecture: {}'.format(self.config['arch']), 0)
        self.writer.add_text('Text', 'Training Data Loader: {}'.format(self.config['train_data_loader']), 0)
        self.writer.add_text('Text', 'Loss Function: {}'.format(self.config['loss']), 0)
        self.writer.add_text('Text', 'Optimizer: {}'.format(self.config['optimizer']), 0)

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        all_ranks = self.pre_metric(output, target)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(all_ranks)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics


class TrainerS(Trainer):
    """
    Trainer class, for one-to-one matching methods on taxonomy completion task

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerS, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.mode = mode

        dataset = self.data_loader.dataset
        self.candidate_positions = data_loader.dataset.all_edges
        if len(self.candidate_positions) > MAX_CANDIDATE_NUM:
            valid_pos = set(itertools.chain.from_iterable(dataset.valid_node2pos.values()))
            valid_neg = list(set(self.candidate_positions).difference(valid_pos))
            valid_sample_size = max(MAX_CANDIDATE_NUM - len(valid_pos), 0)
            self.valid_candidate_positions = random.sample(valid_neg, valid_sample_size) + list(valid_pos)
        else:
            self.valid_candidate_positions = self.candidate_positions
        self.valid_candidate_positions = self.candidate_positions

        if 'g' in mode:
            self.all_nodes = sorted(list(dataset.core_subgraph.nodes))
            self.edge2subgraph = {e: dataset._get_subgraph_and_node_pair(-1, e[0], e[1]) for e in tqdm(self.candidate_positions, desc='collecting nodegraph')}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, v, bgu, bgv, bpu, bpv, lens = batch

            self.optimizer.zero_grad()
            prediction = self.model(nf, u, v, bgu, bgv, bpu, bpv, lens)
            label = label[:, 0].to(self.device)
            if self.is_infonce_training:
                n_batches = label.sum().detach()
                prediction = prediction.reshape(n_batches, -1)
                target = torch.zeros(n_batches, dtype=torch.long).to(self.device)
                loss = self.loss(prediction, target)
            else:
                loss = self.loss(prediction, label)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader)-1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item()))

        log = { 'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                vocab = dataset.test_node_list
                node2pos = dataset.test_node2pos
                candidate_positions = self.candidate_positions
                self.logger.info(f'number of candidate positions: {len(candidate_positions)}')
            else:
                vocab = dataset.valid_node_list
                node2pos = dataset.valid_node2pos
                candidate_positions = self.valid_candidate_positions
            batched_model = [] # save the CPU graph representation
            batched_positions = []
            for edges in tqdm(mit.sliced(candidate_positions, batch_size), desc="Generating graph encoding ..."):
                edges = list(edges)
                us, vs, bgu, bgv, bpu, bpv, lens = None, None, None, None, None, None, None
                if 'r' in self.mode:
                    us, vs = zip(*edges)
                    us = torch.tensor(us)
                    vs = torch.tensor(vs)
                if 'g' in self.mode:
                    bgs = [self.edge2subgraph[e] for e in edges]
                    bgu, bgv = zip(*bgs)
                if 'p' in self.mode:
                    bpu, bpv, lens = dataset._get_batch_edge_node_path(edges)
                    bpu = bpu
                    bpv = bpv
                    lens = lens

                ur, vr = self.model.forward_encoders(us, vs, bgu, bgv, bpu, bpv, lens)
                batched_model.append((ur.detach().cpu(), vr.detach().cpu()))
                batched_positions.append(len(edges))

            # start per query prediction
            all_ranks = []
            for i, query in tqdm(enumerate(vocab), desc='testing'):
                batched_energy_scores = []
                nf = node_features[query, :].to(self.device)
                for (ur, vr), n_position in zip(batched_model, batched_positions):
                    expanded_nf = nf.expand(n_position, -1)
                    ur = ur.to(self.device)
                    vr = vr.to(self.device)
                    energy_scores = model.match(ur, vr, expanded_nf)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_positions, node2pos[query])
                all_ranks.extend(self.pre_metric(batched_energy_scores, labels))
            total_metrics = [metric(all_ranks) for metric in self.metrics]
        return total_metrics


class TrainerT(TrainerS):
    """
    Trainer class, for TMN on taxonomy completion task

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerT, self).__init__(mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.l1 = config['trainer']['l1']
        self.l2 = config['trainer']['l2']
        self.l3 = config['trainer']['l3']

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, v, bgu, bgv, bpu, bpv, lens = batch

            self.optimizer.zero_grad()
            scores, scores_p, scores_c, scores_e = self.model(nf, u, v, bgu, bgv, bpu, bpv, lens)
            label = label.to(self.device)
            loss_p = self.loss(scores_p, label[:, 1])
            loss_c = self.loss(scores_c, label[:, 2])
            loss_e = self.loss(scores_e, label[:, 0])
            loss = self.loss(scores, label[:, 0]) + self.l1 * loss_p + self.l2 * loss_c + self.l3 * loss_e
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} ELoss: {:.6f} PLoss: {:.6f} CLoss: {:.6f}'
                        .format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                                100.0 * batch_idx / len(self.data_loader),
                                loss.item(), loss_e.item(), loss_p.item(), loss_c.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log


class TrainerTExpan(Trainer):
    """
    Trainer class, for TMN on taxonomy expansion task
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerTExpan, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.mode = mode

        self.l1 = config['trainer']['l1']
        self.l2 = config['trainer']['l2']
        self.l3 = config['trainer']['l3']

        dataset = self.data_loader.dataset
        self.pseudo_leaf = dataset.pseudo_leaf_node
        self.candidate_positions = list(set([(p, c) for (p, c) in data_loader.dataset.all_edges if c == self.pseudo_leaf]))
        self.valid_node2pos = {node:set([(p, c) for (p, c) in pos_l if c == self.pseudo_leaf]) for node, pos_l in dataset.valid_node2pos.items()}
        self.test_node2pos = {node:set([(p, c) for (p, c) in pos_l if c == self.pseudo_leaf]) for node, pos_l in dataset.test_node2pos.items()}
        self.valid_vocab = [node for node, pos in self.valid_node2pos.items() if len(pos)]
        self.test_vocab = [node for node, pos in self.test_node2pos.items() if len(pos)]

        if 'g' in mode:
            self.all_nodes = sorted(list(dataset.core_subgraph.nodes))
            self.edge2subgraph = {e: dataset._get_subgraph_and_node_pair(-1, e[0], e[1]) for e in tqdm(self.candidate_positions, desc='collecting nodegraph')}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, v, bgu, bgv, bpu, bpv, lens = batch

            self.optimizer.zero_grad()
            scores, scores_p, scores_c, scores_e = self.model(nf, u, v, bgu, bgv, bpu, bpv, lens)
            label = label.to(self.device)
            loss_p = self.loss(scores_p, label[:, 1])
            loss_c = self.loss(scores_c, label[:, 2])
            loss_e = self.loss(scores_e, label[:, 0])
            loss = self.loss(scores, label[:, 0]) + self.l1*loss_p + self.l2*loss_c + self.l3*loss_e
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} ELoss: {:.6f} PLoss: {:.6f} CLoss: {:.6f}'
                        .format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                                100.0 * batch_idx / len(self.data_loader),
                                loss.item(), loss_e.item(), loss_p.item(), loss_c.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                vocab = self.test_vocab
                node2pos = self.test_node2pos
            else:
                vocab = self.valid_vocab
                node2pos = self.valid_node2pos
            candidate_positions = self.candidate_positions
            batched_model = [] # save the CPU graph representation
            batched_positions = []
            for edges in tqdm(mit.sliced(candidate_positions, batch_size), desc="Generating graph encoding ..."):
                edges = list(edges)
                us, vs, bgu, bgv, bpu, bpv, lens = None, None, None, None, None, None, None
                if 'r' in self.mode:
                    us, vs = zip(*edges)
                    us = torch.tensor(us)
                    vs = torch.tensor(vs)
                if 'g' in self.mode:
                    bgs = [self.edge2subgraph[e] for e in edges]
                    bgu, bgv = zip(*bgs)
                if 'p' in self.mode:
                    bpu, bpv, lens = dataset._get_batch_edge_node_path(edges)
                    bpu = bpu
                    bpv = bpv
                    lens = lens

                ur, vr = self.model.forward_encoders(us, vs, bgu, bgv, bpu, bpv, lens)
                batched_model.append((ur.detach().cpu(), vr.detach().cpu()))
                batched_positions.append(len(edges))

            # start per query prediction
            all_ranks = []
            for i, query in tqdm(enumerate(vocab), desc='testing'):
                batched_energy_scores = []
                nf = node_features[query, :].to(self.device)
                for (ur, vr), n_position in zip(batched_model, batched_positions):
                    expanded_nf = nf.expand(n_position, -1)
                    ur = ur.to(self.device)
                    vr = vr.to(self.device)
                    energy_scores = model.match(ur, vr, expanded_nf)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_positions, node2pos[query])
                all_ranks.extend(self.pre_metric(batched_energy_scores, labels))
            total_metrics = [metric(all_ranks) for metric in self.metrics]

        return total_metrics


class TrainerExpan(Trainer):
    """
    Trainer class, for one-to-one matching methods on taxonomy expansion task

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, mode, model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler=None):
        super(TrainerExpan, self).__init__(model, loss, metrics, pre_metric, optimizer, config, data_loader, lr_scheduler)
        self.mode = mode

        dataset = self.data_loader.dataset
        self.candidate_positions = dataset.all_nodes
        self.valid_node2pos = dataset.valid_node2pos
        self.test_node2pos = dataset.test_node2pos
        self.valid_vocab = dataset.valid_node_list
        self.test_vocab = dataset.test_node_list

        if 'g' in mode:
            self.all_nodes = sorted(list(dataset.core_subgraph.nodes))
            self.node2subgraph = {node: dataset._get_subgraph_and_node_pair(-1, node) for node in tqdm(self.all_nodes, desc='collecting nodegraph')}

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.data_loader):
            nf, label, u, graphs, paths, lens = batch

            self.optimizer.zero_grad()
            scores = self.model(nf, u, graphs, paths, lens)
            label = label.to(self.device)
            if self.is_infonce_training:
                n_batches = label.sum().detach()
                prediction = scores.reshape(n_batches, -1)
                target = torch.zeros(n_batches, dtype=torch.long).to(self.device)
                loss = self.loss(prediction, target)
            else:
                loss = self.loss(scores, label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_step == 0 or batch_idx == len(self.data_loader) - 1:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'
                        .format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples,
                                100.0 * batch_idx / len(self.data_loader),
                                loss.item()))

        log = {'loss': total_loss / len(self.data_loader)}

        ## Validation stage
        if self.do_validation:
            val_log = {'val_metrics': self._test('validation')}
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.lr_scheduler_mode == "min":
                    self.lr_scheduler.step(log['val_metrics'][0])
                else:
                    self.lr_scheduler.step(log['val_metrics'][-1])
            else:
                self.lr_scheduler.step()

        return log

    def _test(self, mode, gpu=True):
        assert mode in ['test', 'validation']
        torch.cuda.empty_cache()
        model = self.model if gpu else self.model.cpu()

        batch_size = self.test_batch_size

        model.eval()
        with torch.no_grad():
            dataset = self.data_loader.dataset
            node_features = dataset.node_features
            if mode == 'test':
                # vocab = self.test_vocab
                # node2pos = self.test_node2pos
                node2pos = dataset.test_node2parent
                vocab = list(node2pos.keys())
            else:
                vocab = self.valid_vocab
                node2pos = self.valid_node2pos
            candidate_positions = self.candidate_positions
            batched_model = [] # save the CPU graph representation
            batched_positions = []
            for us_l in tqdm(mit.sliced(candidate_positions, batch_size), desc="Generating graph encoding ..."):

                bgu, bpu, lens = None, None, None
                if 'r' in self.mode:
                    us = torch.tensor(us_l)
                if 'g' in self.mode:
                    bgu = [self.node2subgraph[e] for e in us_l]
                if 'p' in self.mode:
                    bpu, lens = dataset._get_batch_edge_node_path(us_l)
                    bpu = bpu
                    lens = lens
                ur = self.model.forward_encoders(us, bgu, bpu, lens)
                batched_model.append(ur.detach().cpu())
                batched_positions.append(len(us))

            # start per query prediction
            all_ranks = []
            for i, query in tqdm(enumerate(vocab), desc='testing'):
                batched_energy_scores = []
                nf = node_features[query, :].to(self.device)
                for ur, n_position in zip(batched_model, batched_positions):
                    expanded_nf = nf.expand(n_position, -1)
                    ur = ur.to(self.device)
                    energy_scores = model.match(ur, expanded_nf)
                    batched_energy_scores.append(energy_scores)
                batched_energy_scores = torch.cat(batched_energy_scores)
                batched_energy_scores, labels = rearrange(batched_energy_scores, candidate_positions, node2pos[query])
                all_ranks.extend(self.pre_metric(batched_energy_scores, labels))
            total_metrics = [metric(all_ranks) for metric in self.metrics]

        return total_metrics
