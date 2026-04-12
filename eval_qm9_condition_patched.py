import os
import argparse
import pickle
import yaml
import torch
from easydict import EasyDict
from statistics import mean
from os.path import join

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.chem import *
from eval import retrieve_qm9_smiles, analyze_stability_for_molecules, PropOptEvaluator, diversity, retrieve_geom_smiles

from rdkit.Chem.rdmolfiles import MolToPDBFile
from rdkit.Chem.AllChem import EmbedMolecule

from configs.datasets_config import get_dataset_info
# from qm9.models import DistributionNodes
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction.main_qm9_prop import get_model as get_cls_model
from qm9.utils import prepare_context, compute_mean_mad
from utils.sample import *
from utils.reconstruct import *
from utils.datasets import QM93D

from collections import Counter
from torch_geometric.loader import DataLoader

import torch

class DistributionNodes:
    def __init__(self, histogram):
        """
        histogram: dict or list
        ä¾å¦ dataset_info['n_nodes']
        å¦ææ¯ dictï¼key æ¯åå­æ°ï¼value æ¯é¢æ°
        """
        if isinstance(histogram, dict):
            items = sorted(histogram.items(), key=lambda x: int(x[0]))
            self.n_nodes = torch.tensor([int(k) for k, _ in items], dtype=torch.long)
            probs = torch.tensor([float(v) for _, v in items], dtype=torch.float)
        else:
            self.n_nodes = torch.arange(len(histogram), dtype=torch.long)
            probs = torch.tensor(histogram, dtype=torch.float)

        probs = probs / probs.sum()
        self.prob = probs

    def sample(self, n_samples):
        idx = torch.multinomial(self.prob, n_samples, replacement=True)
        return self.n_nodes[idx]


def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = get_cls_model(args_classifier)
    classifier_state_dict = torch.load(
        join(dir_path, 'best_checkpoint.npy'),
        map_location=torch.device(device)
    )
    classifier.load_state_dict(classifier_state_dict)
    classifier = classifier.to(device)
    classifier.eval()
    return classifier


def padding(data_list, max_nodes):
    padding_list = []
    for data in data_list:
        if data.size(0) < max_nodes:
            padding_data = torch.cat(
                [data, torch.zeros(max_nodes - data.size(0), data.size(1), device=data.device)],
                dim=0
            )
        else:
            padding_data = data
        padding_list.append(padding_data)
    return torch.stack(padding_list)


def sample(args, device, model, dataset_info, prop_dist, nodesxsample, context):
    max_n_nodes = dataset_info['max_n_nodes']
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    datas = []
    num_atom = len(dataset_info['atom_decoder']) + 1

    for n_particles in nodesxsample:
        atom_type = torch.randn(n_particles, num_atom, device=device)
        pos = torch.randn(n_particles, 3, device=device)
        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        rows = torch.LongTensor(rows).unsqueeze(0)
        cols = torch.LongTensor(cols).unsqueeze(0)
        adj = torch.cat([rows, cols], dim=0)
        data = Data(x=atom_type, edge_index=adj, pos=pos)
        datas.append(data)

    batch = Batch.from_data_list(datas).to(device)
    context = context.index_select(0, batch.batch)

    pos_gen, pos_gen_traj, atom_type_list, atom_traj = model.langevin_dynamics_sample(
        atom_type=batch.x,
        pos_init=batch.pos,
        bond_index=batch.edge_index,
        bond_type=None,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
        context=context,
        extend_order=False,
        n_steps=args.n_steps,
        step_lr=1e-6,
        w_global_pos=args.w_global_pos,
        w_global_node=args.w_global_node,
        w_local_pos=args.w_local_pos,
        w_local_node=args.w_local_node,
        global_start_sigma=args.global_start_sigma,
        clip=args.clip,
        clip_local=None,
        sampling_type=args.sampling_type,
        eta=args.eta,
    )

    pos_list = unbatch(pos_gen, batch.batch)
    atom_list = unbatch(F.one_hot(torch.argmax(atom_type_list[:, :-1], dim=1), 5), batch.batch)

    pos_list_pad = padding(pos_list, max_n_nodes)
    atom_list_pad = padding(atom_list, max_n_nodes)
    return pos_list_pad, atom_list_pad


class DiffusionDataloader:
    def __init__(self, args, model, nodes_dist, prop_dist, device, dataset_info,
                 unkown_labels=False, batch_size=1, iterations=200):
        self.args = args
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = dataset_info
        self.i = 0

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)

        pos, atom_type = sample(
            self.args,
            self.device,
            self.model,
            self.dataset_info,
            self.prop_dist,
            nodesxsample=nodesxsample,
            context=context
        )

        bs, n_nodes, _ = pos.size()
        node_mask = torch.zeros(bs, n_nodes, device=self.device)
        for i in range(bs):
            node_mask[i, :nodesxsample[i]] = 1

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=self.device).unsqueeze(0)
        edge_mask = edge_mask * diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        context = context.squeeze(1)

        prop_key = self.prop_dist.properties[0]
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        else:
            context = (
                context * self.prop_dist.normalizer[prop_key]['mad']
                + self.prop_dist.normalizer[prop_key]['mean']
            )

        data = {
            'positions': pos.detach(),
            'one_hot': atom_type.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            prop_key: context.detach()
        }
        return data

    def __next__(self):
        if self.i < self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9', help='qm9, geom')
    parser.add_argument('--ckpt', type=str, required=True, help='path for loading the checkpoint')
    parser.add_argument('--classifiers_path', type=str, required=True, help='directory containing args.pickle and best_checkpoint.npy')
    parser.add_argument('--save_traj', action='store_true', default=False)
    parser.add_argument('--debug_break', type=eval, default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=3.0)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--global_start_sigma', type=float, default=0.5)
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'))
    parser.add_argument('--w_global_pos', type=float, default=1.0)
    parser.add_argument('--w_local_pos', type=float, default=1.5)
    parser.add_argument('--w_global_node', type=float, default=0.4)
    parser.add_argument('--w_local_node', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--sampling_type', type=str, default='generalized')
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--use_ckpt_property_norms', action='store_true', default=True,
                        help='Prefer property_norms stored in checkpoint when available.')
    parser.add_argument('--force_recompute_property_norms', action='store_true', default=False,
                        help='Ignore checkpoint property_norms and recompute from dataset.')
    parser.add_argument('--use_qm9_second_half', action='store_true', default=None,
                        help='When recomputing property_norms, use QM9 second-half split.')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    args.dataset = 'qm9' if 'qm9' in args.ckpt.lower() else 'geom'
    config = ckpt['config']
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    if args.n_steps is None:
        args.n_steps = ckpt['config'].model.num_diffusion_timesteps

    if 'context' in ckpt and ckpt['context'] is not None:
        args.context = ckpt['context']
    else:
        args.context = ckpt['config'].model.context
        if isinstance(args.context, str):
            args.context = [args.context]

    if args.context is None or len(args.context) == 0:
        raise ValueError('No condition context found in checkpoint.')

    output_dir = get_new_log_dir(log_dir, args.sampling_type + '_test', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    logger.info('Building model...')
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    logger.info(f'dataset: {args.dataset}')
    logger.info(f'sample method: {args.sampling_type}')
    logger.info(f'w_global_pos: {args.w_global_pos:.2f}')
    logger.info(f'w_global_node: {args.w_global_node:.2f}')
    logger.info(f'w_local_pos: {args.w_local_pos:.2f}')
    logger.info(f'w_local_node: {args.w_local_node:.2f}')
    logger.info(f'context: {args.context}')

    dataset_info = get_dataset_info(args.dataset, False)
    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)

    transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])

    # Prefer the exact normalization statistics used during training.
    use_ckpt_norms = (
        args.use_ckpt_property_norms
        and (not args.force_recompute_property_norms)
        and ('property_norms' in ckpt)
        and (ckpt['property_norms'] is not None)
    )

    if use_ckpt_norms:
        property_norms = ckpt['property_norms']
        # Keep property normalizers on CPU. DistributionProperty/sample.py
        # normalizes sampled scalar properties before moving the batch to the
        # target device, so CUDA tensors here would cause a CPU/CUDA mismatch.
        for _prop, stats in property_norms.items():
            for _k, _v in list(stats.items()):
                if torch.is_tensor(_v):
                    stats[_k] = _v.detach().cpu()
        logger.info('Using property_norms loaded from checkpoint (moved to CPU for sampling compatibility).')
    else:
        logger.info('Checkpoint property_norms unavailable or disabled; recomputing from dataset.')
        train_set_for_norm = QM93D('train', condition=None, pre_transform=transforms)

        use_second_half = args.use_qm9_second_half
        if use_second_half is None:
            # Follow the training convention when possible.
            data_context = ckpt.get('data_context', None)
            use_second_half = (data_context is not None)

        if use_second_half:
            logger.info('Recomputing property_norms on QM9 second-half split.')
            split_idx = train_set_for_norm.get_half_split(train_set_for_norm.data.y.size(0), 'qm9_second_half')
            train_set_for_norm = train_set_for_norm[split_idx]
        else:
            logger.info('Recomputing property_norms on full QM9 training split.')

        property_norms = compute_mean_mad(train_set_for_norm, args.context, args.dataset)

    mean = property_norms[args.context[0]]['mean']
    mad = property_norms[args.context[0]]['mad']

    # Build the property distribution for conditional sampling.
    train_set_for_propdist = QM93D('train', condition=None, pre_transform=transforms)
    if ckpt.get('data_context', None) is not None:
        logger.info('Using QM9 second-half split for property distribution sampling (matched to conditional training).')
        split_idx = train_set_for_propdist.get_half_split(train_set_for_propdist.data.y.size(0), 'qm9_second_half')
        train_set_for_propdist = train_set_for_propdist[split_idx]
    else:
        logger.info('Using full QM9 training split for property distribution sampling.')

    dataloader_train = DataLoader(train_set_for_propdist, 256, shuffle=False)
    prop_dist = DistributionProperty(dataloader_train, args.context)
    prop_dist.set_normalizer(property_norms)

    diffusion_dataloader = DiffusionDataloader(
        args,
        model,
        nodes_dist,
        prop_dist,
        args.device,
        dataset_info,
        batch_size=args.batch_size,
        iterations=args.iterations
    )

    classifier = get_classifier(args.classifiers_path, args.device)

    print("MDM: We evaluate the classifier on our generated samples")
    loss = test(
        classifier,
        0,
        diffusion_dataloader,
        mean,
        mad,
        args.context[0],
        args.device,
        1,
        args.debug_break
    )
    print(f"Loss classifier on Generated samples: {loss:.4f}")