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
import copy

warnings.filterwarnings("ignore")


def ablation_train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper):
    for epoch in range(args.epoch):
        model.train()
        one_batch_bar = tqdm(train_loader, ncols=100)
        one_batch_bar.set_description(f'[epoch:{epoch + 1}/{args.epoch}]')
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
                res.append([acc, precision, recall, f1score, auroc, auprc])

        val_results = pd.DataFrame(res, columns=['acc', 'precision', 'recall', 'f1_score', 'auroc', 'auprc'])
        r = val_results.mean()
        print(
            f"epoch:{epoch}---validation---acc:{r['acc']:.4f}---precision:{r['precision']:.4f}---recall:{r['recall']:.4f}---"
            f"f1_score:{r['f1_score']:.4f}---auroc:{r['auroc']:.4f}---auprc:{r['auprc']:.4f}---lr:{cur_lr:.6f}")

        early_stop = stopper.step(r['auroc'], model)
        if early_stop:
            break


def run_ablation_experiment(args, ablation_type, train_loader, val_loader, test_loader, n_feats):

    print(f"\n{'=' * 60}")
    print(f"Ablation exp begin: {ablation_type}")
    print(f"{'=' * 60}")

    ablation_args = copy.deepcopy(args)
    if ablation_type == "CrossAIa":
        ablation_args.use_smiles = True
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = False
        ablation_args.use_physchem = False
        ablation_args.use_protein = False
        ablation_args.use_ctf = False
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIa"
    elif ablation_type == "CrossAIb":
        ablation_args.use_smiles = True
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = False
        ablation_args.use_physchem = True
        ablation_args.use_protein = True
        ablation_args.use_ctf = False
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIb"
    elif ablation_type == "CrossAIc":
        ablation_args.use_smiles = True
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = True
        ablation_args.use_physchem = False
        ablation_args.use_protein = True
        ablation_args.use_ctf = False
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIc"
    elif ablation_type == "CrossAId":
        ablation_args.use_smiles = False
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = True
        ablation_args.use_physchem = True
        ablation_args.use_protein = True
        ablation_args.use_ctf = False
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAId"
    elif ablation_type == "CrossAIe":
        ablation_args.use_smiles = True
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = False
        ablation_args.use_physchem = True
        ablation_args.use_protein = True
        ablation_args.use_ctf = True
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIe"
    elif ablation_type == "CrossAIf":
        ablation_args.use_smiles = True
        ablation_args.use_graph = False
        ablation_args.use_fingerprint = True
        ablation_args.use_physchem = True
        ablation_args.use_protein = True
        ablation_args.use_ctf = False
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIf"
    elif ablation_type == "CrossAIg":
        ablation_args.use_smiles = True
        ablation_args.use_graph = False
        ablation_args.use_fingerprint = True
        ablation_args.use_physchem = False
        ablation_args.use_protein = True
        ablation_args.use_ctf = True
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIg"
    elif ablation_type == "CrossAIh":
        ablation_args.use_smiles = True
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = True
        ablation_args.use_physchem = False
        ablation_args.use_protein = False
        ablation_args.use_ctf = True
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIh"
    elif ablation_type == "CrossAIi":
        ablation_args.use_smiles = True
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = True
        ablation_args.use_physchem = True
        ablation_args.use_protein = True
        ablation_args.use_ctf = True
        ablation_args.use_attention_fusion = False
        exp_name = "CrossAIi"
    elif ablation_type == "CrossAIj":
        ablation_args.use_smiles = True
        ablation_args.use_graph = True
        ablation_args.use_fingerprint = True
        ablation_args.use_physchem = True
        ablation_args.use_protein = True
        ablation_args.use_ctf = True
        ablation_args.use_attention_fusion = True
        exp_name = "CrossAIj"
    else:
        raise ValueError(f"unknown ablation type: {ablation_type}")

    original_output = ablation_args.output
    ablation_args.output = os.path.join(original_output, f"ablation_{exp_name}")
    if not os.path.isdir(ablation_args.output):
        os.makedirs(ablation_args.output)

    mean_results = []

    for iteration in range(ablation_args.iterations):
        print(f"\n--- iteration {iteration + 1}/{ablation_args.iterations} ---")

        model = MVP(
            num_classes=ablation_args.class_num,
            in_feats=n_feats,
            hidden_feats=ablation_args.hidden_feats,
            rnn_embed_dim=ablation_args.rnn_embed_dim,
            blstm_dim=ablation_args.rnn_hidden_dim,
            blstm_layers=ablation_args.rnn_layers,
            fp_2_dim=ablation_args.fp_dim,
            physchem_dim=ablation_args.physchem_dim,
            dropout=ablation_args.p,
            num_heads=ablation_args.head,
            protein_cnn_dim=ablation_args.protein_cnn_dim,
            protein_rnn_dim=ablation_args.protein_rnn_dim,
            protein_vocab_size=ablation_args.protein_vocab_size,
            ctf_dim=ablation_args.ctf_dim,
            device=ablation_args.device,
            use_smiles=ablation_args.use_smiles,
            use_graph=ablation_args.use_graph,
            use_fingerprint=ablation_args.use_fingerprint,
            use_physchem=ablation_args.use_physchem,
            use_protein=ablation_args.use_protein,
            use_ctf=ablation_args.use_ctf,
            use_attention_fusion=ablation_args.use_attention_fusion
        ).to(ablation_args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=ablation_args.lr, weight_decay=ablation_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        stopper = EarlyStopping(mode='higher', filename=f'{ablation_args.output}/net_{iteration}.pkl', patience=15)
        loss_func = torch.nn.BCEWithLogitsLoss()

        ablation_train(ablation_args, train_loader, val_loader, model, loss_func, optimizer, scheduler, stopper)

        stopper.load_checkpoint(model)
        model.eval()
        res = []
        prediction_details = []

        with torch.no_grad():
            for batch in test_loader:
                batch_smiles, batch_graph, fps_t, physchem_t, protein_data, ctf_t, labels = batch
                labels = labels.to(ablation_args.device)
                batch_graph = batch_graph.to(ablation_args.device)
                batch_smiles["smiles"] = batch_smiles["smiles"].to(ablation_args.device)
                atom_feats = batch_graph.ndata['h'].to(ablation_args.device)
                fps_t = fps_t.to(ablation_args.device)
                physchem_t = physchem_t.to(ablation_args.device)
                ctf_t = ctf_t.to(ablation_args.device)
                protein_data = {
                    'protein_cnn': protein_data['protein_cnn'].to(ablation_args.device),
                    'protein_seq': protein_data['protein_seq'].to(ablation_args.device),
                    'protein_seq_len': protein_data['protein_seq_len']
                }

                pred = model(batch_smiles, batch_graph, atom_feats, fps_t, physchem_t, protein_data, ctf_t)
                acc, precision, recall, f1score, auroc, auprc, _ = evaluate(labels, pred)
                res.append([acc, precision, recall, f1score, auroc, auprc])

                pred_prob = torch.sigmoid(pred).cpu().numpy().flatten()
                pred_label = (pred_prob > 0.5).astype(int)
                true_label = labels.cpu().numpy().flatten()

                for i in range(len(true_label)):
                    prediction_details.append({
                        'true_label': int(true_label[i]),
                        'predicted_label': int(pred_label[i]),
                        'prediction_probability': float(pred_prob[i])
                    })

        test_results = pd.DataFrame(res, columns=['acc', 'precision', 'recall', 'f1_score', 'auroc', 'auprc'])
        r = test_results.mean()
        print(f"test result---acc:{r['acc']:.4f}---precision:{r['precision']:.4f}---recall:{r['recall']:.4f}---"
              f"f1_score:{r['f1_score']:.4f}---auroc:{r['auroc']:.4f}---auprc:{r['auprc']:.4f}")

        mean_results.append([r['acc'], r['precision'], r['recall'], r['f1_score'], r['auroc'], r['auprc']])

        prediction_details_df = pd.DataFrame(prediction_details)
        prediction_details_file = f'{ablation_args.output}/prediction_details_iter_{iteration}.csv'
        prediction_details_df.to_csv(prediction_details_file, index=False)
        print(f"Predict details are stored in: {prediction_details_file}")

    test_mean_results = pd.DataFrame(mean_results, columns=['acc', 'precision', 'recall', 'f1_score', 'auroc', 'auprc'])
    final_mean = test_mean_results.mean()
    final_std = test_mean_results.std()

    print(f"\n{'=' * 60}")
    print(f"exp {ablation_type} average results finally:")
    print(f"acc: {final_mean['acc']:.4f} ± {final_std['acc']:.4f}")
    print(f"pr: {final_mean['precision']:.4f} ± {final_std['precision']:.4f}")
    print(f"recall: {final_mean['recall']:.4f} ± {final_std['recall']:.4f}")
    print(f"F1: {final_mean['f1_score']:.4f} ± {final_std['f1_score']:.4f}")
    print(f"AUROC: {final_mean['auroc']:.4f} ± {final_std['auroc']:.4f}")
    print(f"AUPR: {final_mean['auprc']:.4f} ± {final_std['auprc']:.4f}")
    print(f"{'=' * 60}")

    test_mean_results.to_csv(f'{ablation_args.output}/ablation_results.csv', index=False)

    return {
        'experiment': ablation_type,
        'acc_mean': final_mean['acc'],
        'acc_std': final_std['acc'],
        'precision_mean': final_mean['precision'],
        'precision_std': final_std['precision'],
        'recall_mean': final_mean['recall'],
        'recall_std': final_std['recall'],
        'f1_mean': final_mean['f1_score'],
        'f1_std': final_std['f1_score'],
        'auroc_mean': final_mean['auroc'],
        'auroc_std': final_std['auroc'],
        'auprc_mean': final_mean['auprc'],
        'auprc_std': final_std['auprc']
    }


def main_ablation():
    args = config.parse()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(validate_dataset, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate)

    ablation_types = [
        "CrossAIa",
        "CrossAIb",
        "CrossAIc",
        "CrossAId",
        "CrossAIe",
        "CrossAif",
        "CrossAIg",
        "CrossAIh",
        "CrossAIi",
        "CrossAIj",
    ]

    all_results = []

    for ablation_type in ablation_types:
        result = run_ablation_experiment(args, ablation_type, train_loader, val_loader, test_loader, n_feats)
        all_results.append(result)

    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(args.output, 'ablation_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print(f"\n{'=' * 80}")
    print("All exp are done!")
    print(f"The results are stored in: {summary_file}")
    print(f"{'=' * 80}")

    print("\nablation results:")
    print(summary_df.round(4))

    return summary_df


if __name__ == '__main__':
    main_ablation()