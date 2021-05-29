""" Model inference on completely new taxons
"""
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from gensim.models import KeyedVectors
import numpy as np
import more_itertools as mit
from pathlib import Path


def prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def main(config, args_outer):
    # Load new taxons and normalize embeddings if needed
    vocab = []
    nf = []
    with open(args_outer.taxon, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                segs = line.split("\t")
                vocab.append("_".join(segs[0].split(" ")))
                nf.append([float(ele) for ele in segs[1].split(" ")])
            
    nf = np.array(nf)
    if config['train_data_loader']['args']['normalize_embed']:
        row_sums = nf.sum(axis=1)
        nf = nf / row_sums[:, np.newaxis]
    kv = KeyedVectors(vector_size=nf.shape[1])
    kv.add(vocab, nf)

    # Load trained model and existing taxonomy
    mode = config['mode']
    logger = config.get_logger('test')
    torch.multiprocessing.set_sharing_strategy('file_system')
    test_data_loader = module_data.UnifiedDataLoader(
        mode=mode,
        data_path=config['data_path'],
        sampling_mode=0,
        batch_size=1, 
        expand_factor=config['train_data_loader']['args']['expand_factor'],
        shuffle=True, 
        num_workers=8,
        cache_refresh_time=config['train_data_loader']['args']['cache_refresh_time'],
        normalize_embed=config['train_data_loader']['args']['normalize_embed'],
        # test_topk=args_outer.topk,
        test=1
    )
    logger.info(test_data_loader)
    test_dataset = test_data_loader.dataset
    indice2word = test_dataset.vocab

    # build model architecture
    model = config.initialize('arch', module_arch, mode)
    node_features = test_dataset.node_features
    vocab_size, embed_dim = node_features.size()
    model.set_embedding(vocab_size=vocab_size, embed_dim=embed_dim)
    logger.info(model)

    # load saved model
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for inference
    device, device_ids = prepare_device(config['n_gpu'], logger)
    model = model.to(device)
    model.set_device(device)
    model.eval()

    """Start inference"""
    candidate_positions = test_dataset.all_edges
    if 'g' in mode:
        edge2subgraph = {e: test_dataset._get_subgraph_and_node_pair(-1, e[0], e[1]) for e in tqdm(candidate_positions, desc='collecting nodegraph')}

    batched_model = []  # save the CPU graph representation
    batched_positions = []
    for edges in tqdm(mit.sliced(candidate_positions, args_outer.batch_size), desc="Generating graph encoding ..."):
        edges = list(edges)
        us, vs, bgu, bgv, bpu, bpv, lens = None, None, None, None, None, None, None
        if 'r' in mode:
            us, vs = zip(*edges)
            us = torch.tensor(us)
            vs = torch.tensor(vs)
        if 'g' in mode:
            bgs = [edge2subgraph[e] for e in edges]
            bgu, bgv = zip(*bgs)
        if 'p' in mode:
            bpu, bpv, lens = test_dataset._get_batch_edge_node_path(edges)
            bpu = bpu
            bpv = bpv
            lens = lens

        ur, vr = model.forward_encoders(us, vs, bgu, bgv, bpu, bpv, lens)
        batched_model.append((ur.detach().cpu(), vr.detach().cpu()))
        batched_positions.append(len(edges))

    # start per query prediction
    save_path = Path(args_outer.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), open(args_outer.save, "w") as fout:
        fout.write(f"Query\tPredicted positions\n")
        for i, query in tqdm(enumerate(vocab)):
            batched_energy_scores = []
            nf = torch.tensor(kv[str(query)], dtype=torch.float32).to(device)
            for (ur, vr), n_position in zip(batched_model, batched_positions):
                expanded_nf = nf.expand(n_position, -1)
                ur = ur.to(device)
                vr = vr.to(device)
                energy_scores = model.match(ur, vr, expanded_nf)
                batched_energy_scores.append(energy_scores)
            batched_energy_scores = torch.cat(batched_energy_scores)
            predicted_scores = batched_energy_scores.cpu().squeeze_().tolist()
            if config['loss'].startswith("info_nce") or config['loss'].startswith("bce_loss"):  # select top-5 predicted parents
                predict_candidate_positions = [candidate_positions[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:-x[1])[:args_outer.topm]]
            else:
                predict_candidate_positions = [candidate_positions[ele[0]] for ele in sorted(enumerate(predicted_scores), key=lambda x:x[1])[:args_outer.topm]]
            predict_parents = "\t".join([f'({indice2word[u]}, {indice2word[v]})' for (u, v) in predict_candidate_positions])
            fout.write(f"{query}\t{predict_parents}\n")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing structure expansion model with case study logging')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest model checkpoint (default: None)')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-t', '--taxon', default=None, type=str, help='path to new taxon list  (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    # args.add_argument('-k', '--topk', default=-1, type=int, help='topk retrieved instances for testing, -1 means no retrieval stage (default: -1)')
    args.add_argument('-m', '--topm', default=10, type=int, help='save topm ranked positions (default: 10)')
    args.add_argument('-b', '--batch_size', default=-1, type=int, help='batch size, -1 for small dataset (default: -1), 20000 for larger MAG-Full data')
    args.add_argument('-s', '--save', default="./output/prediction_results.tsv", type=str, help='save file for prediction results (default: ./output/prediction_results.tsv)')
    args_outer = args.parse_args()
    config = ConfigParser(args)
    main(config, args_outer)
