import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from dgllife.utils import EarlyStopping
from dataset import MolDataSet, collate
from utils import set_random_seed, evaluate
from model import MVP
import config
import warnings

warnings.filterwarnings("ignore")

import math


class CustomOneCycleLR:

    def __init__(self, optimizer, max_lr, steps_per_epoch, epochs, pct_start=0.3,
                 anneal_strategy='cos', div_factor=25., final_div_factor=1e4,
                 cycle_momentum=True, base_momentum=0.85, max_momentum=0.95):

        self.optimizer = optimizer
        self.max_lrs = [max_lr] * len(optimizer.param_groups) if isinstance(max_lr, (float, int)) else max_lr
        self.total_steps = steps_per_epoch * epochs


        self.step_size_up = int(self.total_steps * pct_start)
        self.step_size_down = self.total_steps - self.step_size_up


        self.initial_lrs = [max_lr / div_factor for max_lr in self.max_lrs]
        self.final_lrs = [max_lr / final_div_factor for max_lr in self.max_lrs]


        self.anneal_strategy = anneal_strategy


        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults and 'betas' not in optimizer.defaults:
                raise ValueError('optimizer 必须支持动量设置')


            self.use_beta1 = 'betas' in optimizer.defaults
            self.base_momentums = [base_momentum] * len(optimizer.param_groups) if isinstance(base_momentum, (
                float, int)) else base_momentum
            self.max_momentums = [max_momentum] * len(optimizer.param_groups) if isinstance(max_momentum, (
                float, int)) else max_momentum


            if self.use_beta1:
                self.momentums = [group['betas'][0] for group in optimizer.param_groups]
            else:
                self.momentums = [group['momentum'] for group in optimizer.param_groups]


        self.current_step = 0
        self.lrs = []


        self._set_initial_values()

    def _set_initial_values(self):
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.initial_lrs[i]

            if self.cycle_momentum:
                if self.use_beta1:

                    beta1, beta2 = group['betas']
                    group['betas'] = (self.max_momentums[i], beta2)
                else:

                    group['momentum'] = self.max_momentums[i]

    def _annealing_cos(self, start, end, pct):

        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):

        return end + (start - end) * (1 - pct)

    def get_lr(self):
        lrs = []


        if self.current_step <= self.step_size_up:

            for i, (initial_lr, max_lr) in enumerate(zip(self.initial_lrs, self.max_lrs)):
                computed_lr = initial_lr + (max_lr - initial_lr) * (self.current_step / self.step_size_up)
                lrs.append(computed_lr)
        else:

            down_step = self.current_step - self.step_size_up
            down_pct = down_step / self.step_size_down

            for i, (max_lr, final_lr) in enumerate(zip(self.max_lrs, self.final_lrs)):
                if self.anneal_strategy == 'cos':
                    computed_lr = self._annealing_cos(max_lr, final_lr, down_pct)
                else:
                    computed_lr = self._annealing_linear(max_lr, final_lr, down_pct)
                lrs.append(computed_lr)

        return lrs

    def get_momentum(self):

        momentums = []


        if self.current_step <= self.step_size_up:

            up_pct = self.current_step / self.step_size_up
            for max_momentum, base_momentum in zip(self.max_momentums, self.base_momentums):
                momentum = max_momentum - (max_momentum - base_momentum) * up_pct
                momentums.append(momentum)
        else:

            down_step = self.current_step - self.step_size_up
            down_pct = down_step / self.step_size_down

            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                if self.anneal_strategy == 'cos':
                    momentum = self._annealing_cos(base_momentum, max_momentum, down_pct)
                else:
                    momentum = self._annealing_linear(base_momentum, max_momentum, down_pct)
                momentums.append(momentum)

        return momentums

    def step(self):


        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


        if self.cycle_momentum:
            momentums = self.get_momentum()
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                if self.use_beta1:
                    param_group['betas'] = (momentum, param_group['betas'][1])
                else:
                    param_group['momentum'] = momentum


        self.lrs.append(lrs[0])
        self.current_step += 1

    def state_dict(self):

        return {
            'current_step': self.current_step,
            'initial_lrs': self.initial_lrs,
            'max_lrs': self.max_lrs,
            'final_lrs': self.final_lrs,
            'step_size_up': self.step_size_up,
            'step_size_down': self.step_size_down,
            'anneal_strategy': self.anneal_strategy,
            'cycle_momentum': self.cycle_momentum,
            'base_momentums': self.base_momentums if self.cycle_momentum else None,
            'max_momentums': self.max_momentums if self.cycle_momentum else None,
            'use_beta1': self.use_beta1 if self.cycle_momentum else None,
        }

    def load_state_dict(self, state_dict):

        self.current_step = state_dict['current_step']
        self.initial_lrs = state_dict['initial_lrs']
        self.max_lrs = state_dict['max_lrs']
        self.final_lrs = state_dict['final_lrs']
        self.step_size_up = state_dict['step_size_up']
        self.step_size_down = state_dict['step_size_down']
        self.anneal_strategy = state_dict['anneal_strategy']
        self.cycle_momentum = state_dict['cycle_momentum']

        if self.cycle_momentum:
            self.base_momentums = state_dict['base_momentums']
            self.max_momentums = state_dict['max_momentums']
            self.use_beta1 = state_dict['use_beta1']

def train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper):
    for epoch in range(args.epoch):
        model.train()
        one_batch_bar = tqdm(train_loader, ncols=100)
        one_batch_bar.set_description(f'[iter:{args.iter},epoch:{epoch + 1}/{args.epoch}]')
        cur_lr = optimizer.param_groups[0]["lr"]
        for i, batch in enumerate(one_batch_bar):
            batch_smiles, batch_graph, fps_t, physchem_t, protein_data, ctf_t, labels = batch
            labels = labels.to(args.device)
            batch_graph = batch_graph.to(args.device)
            batch_smiles["smiles"] = batch_smiles["smiles"].to(args.device)
            atom_feats = batch_graph.ndata['h'].to(args.device)
            fps_t = fps_t.to(args.device)
            physchem_t = physchem_t.to(args.device)
            ctf_t = ctf_t.to(args.device)
            protein_data = {
                'protein_cnn': protein_data['protein_cnn'].to(args.device),
                'protein_seq': protein_data['protein_seq'].to(args.device),
                'protein_seq_len': protein_data['protein_seq_len']
            }
            pred = model(batch_smiles, batch_graph, atom_feats, fps_t, physchem_t, protein_data, ctf_t)
            acc, precision, recall, f1score, auroc, auprc, acc_weight = evaluate(labels, pred)
            loss = loss_func(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            one_batch_bar.set_postfix(dict(
                loss=f'{loss.item():.5f}',
                acc=f'{acc * 100:.2f}%',
                auroc=f'{auroc:.4f}',
                auprc=f'{auprc:.4f}'))
        scheduler.step()
        model.eval()
        res = []
        with torch.no_grad():
            for batch in val_loader:
                batch_smiles, batch_graph, fps_t, physchem_t, protein_data, ctf_t, labels = batch
                labels = labels.to(args.device)
                batch_graph = batch_graph.to(args.device)
                batch_smiles["smiles"] = batch_smiles["smiles"].to(args.device)
                atom_feats = batch_graph.ndata['h'].to(args.device)
                fps_t = fps_t.to(args.device)
                physchem_t = physchem_t.to(args.device)
                ctf_t = ctf_t.to(args.device)
                protein_data = {
                    'protein_cnn': protein_data['protein_cnn'].to(args.device),
                    'protein_seq': protein_data['protein_seq'].to(args.device),
                    'protein_seq_len': protein_data['protein_seq_len']
                }
                pred = model(batch_smiles, batch_graph, atom_feats, fps_t, physchem_t, protein_data, ctf_t)
                acc, precision, recall, f1score, auroc, auprc, _ = evaluate(labels, pred)
                res.append([acc, precision, recall, f1score,auroc, auprc])
        val_results = pd.DataFrame(res, columns=['acc', 'precision', 'recall', 'f1_score','auroc', 'auprc'])
        r = val_results.mean()
        print(
            f"epoch:{epoch}---validation---acc:{r['acc']}---precision:{r['precision']}---recall:{r['recall']}-"
            f"--f1_score:{r['f1_score']}----auroc:{r['auroc']:.4f}---auprc:{r['auprc']:.4f}---lr:{cur_lr}")
        early_stop = stopper.step(r['acc'], model)
        if early_stop:
            break
def main(args):
    data_path = './data/dataset.csv'
    dataset = MolDataSet(data_path)

    data_index = []
    file_name = "index.txt"
    with open('./data' + "/" + file_name, "r") as f:
        for line in f.readlines():
            line = eval(line)
            data_index.append(line)

    train_dataset = Subset(dataset, data_index[0])
    validate_dataset = Subset(dataset, data_index[1])
    test_dataset = Subset(dataset, data_index[2])
    n_feats = dataset.node_featurizer.feat_size('h')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate)
    val_loader = DataLoader(validate_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)
    mean_results = []
    for iteration in range(args.iterations):
        args.iter = iteration

        model = MVP(num_classes=args.class_num, in_feats=n_feats, hidden_feats=args.hidden_feats,
                    rnn_embed_dim=args.rnn_embed_dim, blstm_dim=args.rnn_hidden_dim, blstm_layers=args.rnn_layers,
                    fp_2_dim=args.fp_dim, physchem_dim=args.physchem_dim, dropout=args.p, num_heads=args.head,
                    protein_cnn_dim=args.protein_cnn_dim,protein_rnn_dim=args.protein_rnn_dim,protein_vocab_size=args.protein_vocab_size,
                    ctf_dim=args.ctf_dim,
                    device=args.device).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        stopper = EarlyStopping(mode='higher', filename=f'{args.output}/net_{iteration}.pkl', patience=50)
        loss_func = torch.nn.BCEWithLogitsLoss()
        train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper)
        stopper.load_checkpoint(model)
        model.eval()
        res = []

        with torch.no_grad():
            for batch in test_loader:
                batch_smiles, batch_graph, fps_t, physchem_t, protein_data, ctf_t, labels = batch
                labels = labels.to(args.device)
                batch_graph = batch_graph.to(args.device)
                batch_smiles["smiles"] = batch_smiles["smiles"].to(args.device)
                atom_feats = batch_graph.ndata['h'].to(args.device)
                fps_t = fps_t.to(args.device)
                physchem_t = physchem_t.to(args.device)
                ctf_t = ctf_t.to(args.device)

                protein_data = {
                    'protein_cnn': protein_data['protein_cnn'].to(args.device),
                    'protein_seq': protein_data['protein_seq'].to(args.device),
                    'protein_seq_len': protein_data['protein_seq_len']
                }
                pred = model(batch_smiles, batch_graph, atom_feats, fps_t, physchem_t, protein_data,ctf_t)
                acc, precision, recall, f1score,auroc, auprc,  _ = evaluate(labels, pred)
                res.append([acc, precision, recall, f1score,auroc, auprc])

        test_results = pd.DataFrame(res, columns=['acc', 'precision', 'recall', 'f1_score', 'auroc', 'auprc'])
        r = test_results.mean()
        print(f"test_---acc:{r['acc']}---precision:{r['precision']}---recall:{r['recall']}---f1_score:{r['f1_score']}---auroc:{r['auroc']:.4f}---auprc:{r['auprc']:.4f}")
        mean_results.append([r['acc'], r['precision'], r['recall'], r['f1_score'], r['auroc'], r['auprc']])
        test_mean_results = pd.DataFrame(mean_results, columns=['acc', 'precision', 'recall', 'f1_score', 'auroc', 'auprc'])
        r = test_mean_results.mean()
        print(
            f"mean_test_---acc:{r['acc']}---precision:{r['precision']}---recall:{r['recall']}---f1_score:{r['f1_score']}---auroc:{r['auroc']:.4f}---auprc:{r['auprc']:.4f}")
        test_mean_results.to_csv(f'{args.output}/10_test_results.csv', index=False)


if __name__ == '__main__':
    args = config.parse()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    main(args)

