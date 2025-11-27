from math import sqrt
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import NNConv, Set2Set
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import tokens_struct
from torch_geometric.nn import GraphNorm



class MVP(nn.Module):
    def __init__(self, num_classes=1, in_feats=64, hidden_feats=None, num_step_set2set=6,
                 num_layer_set2set=3, rnn_embed_dim=64, blstm_dim=128, blstm_layers=2,
                 fp_2_dim=128, physchem_dim=128, num_heads=4,
                 protein_cnn_dim=343, protein_rnn_dim=128, protein_vocab_size=21,
                 ctf_dim=343,
                 dropout=0.2, device='cpu',
                 use_smiles=True, use_graph=True, use_fingerprint=True,
                 use_physchem=True, use_protein=True, use_ctf=True,
                 use_attention_fusion=True):
        super(MVP, self).__init__()
        self.device = device
        self.dropout = dropout
        self.use_smiles = use_smiles
        self.use_graph = use_graph
        self.use_fingerprint = use_fingerprint
        self.use_physchem = use_physchem
        self.use_protein = use_protein
        self.use_ctf = use_ctf
        self.use_attention_fusion = use_attention_fusion

        self.vocab = tokens_struct()
        if hidden_feats is None:
            hidden_feats = [64, 64]
        self.final_hidden_feats = hidden_feats[-1]
        self.norm_layer_module = nn.LayerNorm(self.final_hidden_feats).to(device)

        if self.use_ctf:
            self.ctf_module = CTFModule(input_dim=ctf_dim, output_dim=self.final_hidden_feats, dropout=dropout)

        if self.use_graph:
            self.gnn = GNNModule(in_feats, hidden_feats, dropout, num_step_set2set, num_layer_set2set)

        if self.use_smiles:
            self.rnn = RNNModule(self.vocab, rnn_embed_dim, blstm_dim, blstm_layers, self.final_hidden_feats, dropout,
                                 bidirectional=True, device=device)

        if self.use_fingerprint:
            self.fp_mlp = FPNModule(fp_2_dim, self.final_hidden_feats)

        if self.use_physchem:
            self.physchem_mlp = PhysChemModule(in_dim=18, hidden_dim=physchem_dim, out_dim=self.final_hidden_feats,
                                               dropout=dropout)

        if self.use_protein:
            self.protein_cnn = ProteinCNN(
                protein_Oridim=20,
                feature_size=200,
                out_features=protein_cnn_dim,
                max_seq_len=3000,
                kernels=[4],
                dropout_rate=dropout
            )

            self.protein_rnn = ProteinRNNModule(
                vocab_size=protein_vocab_size,
                embed_dim=rnn_embed_dim,
                blstm_dim=blstm_dim,
                num_layers=blstm_layers,
                out_dim=protein_rnn_dim,
                dropout=dropout,
                bidirectional=True,
                device=device
            )

            self.protein_fusion = nn.Sequential(
                nn.Linear(protein_cnn_dim + protein_rnn_dim, self.final_hidden_feats),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.final_hidden_feats)
            )

        self.num_views = sum([use_smiles, use_graph, use_fingerprint, use_physchem, use_protein, use_ctf])

        if self.use_attention_fusion:
            self.conv = nn.Sequential(
                nn.Conv2d(num_heads, num_heads, kernel_size=2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            dim_k = self.final_hidden_feats * num_heads
            dim_v = self.final_hidden_feats * num_heads
            dim_in = self.final_hidden_feats
            assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
            self.dim_in = dim_in
            self.dim_k = dim_k
            self.dim_v = dim_v
            self.num_heads = num_heads
            self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
            self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
            self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
            self._norm_fact = 1 / sqrt(dim_k // num_heads)

            if self.use_smiles and self.use_graph:
                self.cross_attention_smiles_to_graph = CrossAttention(
                    query_dim=self.final_hidden_feats,
                    key_dim=self.final_hidden_feats,
                    value_dim=self.final_hidden_feats,
                    num_heads=num_heads,
                    dropout=dropout
                ).to(device)

            if self.use_graph and self.use_fingerprint:
                self.cross_attention_graph_to_fp = CrossAttention(
                    query_dim=self.final_hidden_feats,
                    key_dim=self.final_hidden_feats,
                    value_dim=self.final_hidden_feats,
                    num_heads=num_heads,
                    dropout=dropout
                ).to(device)

            if self.use_fingerprint and self.use_physchem:
                self.cross_attention_fp_to_physchem = CrossAttention(
                    query_dim=self.final_hidden_feats,
                    key_dim=self.final_hidden_feats,
                    value_dim=self.final_hidden_feats,
                    num_heads=num_heads,
                    dropout=dropout
                ).to(device)

            if self.use_physchem and self.use_protein:
                self.cross_attention_physchem_to_protein = CrossAttention(
                    query_dim=self.final_hidden_feats,
                    key_dim=self.final_hidden_feats,
                    value_dim=self.final_hidden_feats,
                    num_heads=num_heads,
                    dropout=dropout
                ).to(device)

            if self.use_protein and self.use_ctf:
                self.cross_attention_protein_to_ctf = CrossAttention(
                    query_dim=self.final_hidden_feats,
                    key_dim=self.final_hidden_feats,
                    value_dim=self.final_hidden_feats,
                    num_heads=num_heads,
                    dropout=dropout
                ).to(device)

            conv_output_height = self.num_views - 2 + 1 if self.num_views >= 3 else 1
            conv_output_width = self.final_hidden_feats - 2 + 1 if self.final_hidden_feats >= 3 else 1
            mlp_input_dim = num_heads * conv_output_height * conv_output_width
        else:
            mlp_input_dim = self.num_views * self.final_hidden_feats

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(1024, num_classes)
        ).to(device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, smiles, graphs, atom_feats, fp_t, physchem_t, protein_data, ctf_t):
        batch_size = smiles['smiles'].size(0)
        view_features = []

        if self.use_smiles:
            smiles_x = self.rnn(smiles)
            smiles_x = self.norm_layer_module(smiles_x).view(batch_size, 1, -1)
            view_features.append(('smiles', smiles_x))

        if self.use_graph:
            graph_x = self.gnn(graphs, atom_feats)
            graph_x = self.norm_layer_module(graph_x).view(batch_size, 1, -1)
            view_features.append(('graph', graph_x))

        if self.use_fingerprint:
            fp_x = self.fp_mlp(fp_t)
            fp_x = self.norm_layer_module(fp_x).view(batch_size, 1, -1)
            view_features.append(('fp', fp_x))

        if self.use_physchem:
            physchem_x = self.physchem_mlp(physchem_t)
            physchem_x = self.norm_layer_module(physchem_x).view(batch_size, 1, -1)
            view_features.append(('physchem', physchem_x))

        if self.use_protein:
            protein_cnn_x = self.protein_cnn(protein_data['protein_cnn'])
            protein_rnn_x = self.protein_rnn({
                'protein_seq': protein_data['protein_seq'],
                'protein_seq_len': protein_data['protein_seq_len']
            })
            protein_x = self.protein_fusion(torch.cat([protein_cnn_x, protein_rnn_x], dim=1))
            protein_x = protein_x.view(batch_size, 1, -1)
            view_features.append(('protein', protein_x))

        if self.use_ctf:
            ctf_x = self.ctf_module(ctf_t)
            ctf_x = self.norm_layer_module(ctf_x).view(batch_size, 1, -1)
            view_features.append(('ctf', ctf_x))

        if len(view_features) == 0:
            return torch.zeros(batch_size, device=self.device)

        if self.use_attention_fusion and len(view_features) >= 1:
            enhanced_features = self._apply_attention_fusion(view_features, batch_size)
            in_tensor = torch.cat(enhanced_features, dim=1)
            batch, n, dim_in = in_tensor.shape
            assert dim_in == self.dim_in
            nh = self.num_heads
            dk = self.dim_k // nh
            dv = self.dim_v // nh

            q = self.linear_q(in_tensor).reshape(batch, n, nh, dk).transpose(1, 2)
            k = self.linear_k(in_tensor).reshape(batch, n, nh, dk).transpose(1, 2)
            v = self.linear_v(in_tensor).reshape(batch, n, nh, dv).transpose(1, 2)

            dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
            dist = torch.softmax(dist, dim=-1)
            att = torch.matmul(dist, v)

            out = self.conv(att)
            out = out.view(batch_size, -1)
        else:
            features = [feature for _, feature in view_features]
            in_tensor = torch.cat(features, dim=1)
            out = in_tensor.view(batch_size, -1)

        out = self.mlp(out)
        return out.squeeze(dim=-1)

    def _apply_attention_fusion(self, view_features, batch_size):
        enhanced_features = []
        feature_dict = dict(view_features)

        views_order = ['smiles', 'graph', 'fp', 'physchem', 'protein', 'ctf']
        available_views = [v for v in views_order if v in feature_dict]

        for i, view_name in enumerate(available_views):
            current_feature = feature_dict[view_name]

            if i < len(available_views) - 1:
                next_view_name = available_views[i + 1]
                next_feature = feature_dict[next_view_name]

                cross_attention_name = f'cross_attention_{view_name}_to_{next_view_name}'
                if hasattr(self, cross_attention_name):
                    cross_attention_layer = getattr(self, cross_attention_name)
                    enhanced_feature = cross_attention_layer(
                        query=current_feature,
                        key=next_feature,
                        value=next_feature
                    )
                    enhanced_features.append(enhanced_feature)
                else:
                    enhanced_features.append(current_feature)
            else:
                enhanced_features.append(current_feature)

        return enhanced_features

    def predict(self, smiles, graphs, atom_feats, fp_t, physchem_t, protein_data, ctf_t):
        return self.sigmoid(self.forward(smiles, graphs, atom_feats, fp_t, physchem_t, protein_data, ctf_t))


class SelfAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim]))

    def forward(self, x):
        bsz, seq_len, dim = x.shape

        q = self.query(x)  # [batch_size, seq_len, input_dim]
        k = self.key(x)  # [batch_size, seq_len, input_dim]
        v = self.value(x)  # [batch_size, seq_len, input_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale.to(x.device)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, v)
        return output + x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        assert value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        self.head_dim = query_dim // num_heads

        self.w_q = nn.Linear(query_dim, query_dim, bias=False)
        self.w_k = nn.Linear(key_dim, query_dim, bias=False)
        self.w_v = nn.Linear(value_dim, query_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        # 输出层
        self.fc_out = nn.Linear(query_dim, query_dim)
        self.layer_norm = nn.LayerNorm(query_dim)

    def forward(self, query, key, value):

        batch_size = query.shape[0]

        Q = self.w_q(query)  # [batch_size, seq_len_q, query_dim]
        K = self.w_k(key)  # [batch_size, seq_len_k, query_dim]
        V = self.w_v(value)  # [batch_size, seq_len_v, query_dim]

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        weighted = torch.matmul(attention_weights, V)
        weighted = weighted.transpose(1, 2).contiguous().view(
            batch_size, -1, self.query_dim
        )

        output = self.fc_out(weighted)
        output = self.layer_norm(output + query)

        return output


class GNNModule(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, dropout=0.2, num_step_set2set=6,
                 num_layer_set2set=3):
        super(GNNModule, self).__init__()
        if hidden_feats is None:
            hidden_feats = [64, 64]

        self.conv_layers = nn.ModuleList()
        prev_dim = in_feats

        for i, hidden_dim in enumerate(hidden_feats):
            edge_network = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, prev_dim * hidden_dim)
            )

            conv_layer = NNConv(
                in_feats=prev_dim,
                out_feats=hidden_dim,
                edge_func=edge_network,
                aggregator_type='mean'
            )
            self.conv_layers.append(conv_layer)
            prev_dim = hidden_dim

        self.readout = Set2Set(input_dim=hidden_feats[-1],
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.norm = GraphNorm(hidden_feats[-1] * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_feats[-1] * 2, hidden_feats[-1] * 2),
            nn.BatchNorm1d(hidden_feats[-1] * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_feats[-1] * 2, hidden_feats[-1]),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.graph_norm = nn.LayerNorm(hidden_feats[-1])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, graphs, atom_feats):
        node_x = atom_feats


        if 'edge_feat' not in graphs.edata:
            graphs.edata['edge_feat'] = torch.ones(graphs.number_of_edges(), 1).to(node_x.device)


        for conv in self.conv_layers:
            node_x = conv(graphs, node_x, graphs.edata['edge_feat'])
            node_x = self.activation(node_x)
            node_x = self.dropout(node_x)


        batch_size = graphs.batch_size
        if graphs.number_of_nodes() > batch_size:
            node_x = self.graph_norm(node_x)

        graph_x = self.readout(graphs, node_x)
        out = self.norm(graph_x)
        out = self.fc(out)
        return out


class RNNModule(nn.Module):
    def __init__(self, vocab, embed_dim, blstm_dim, num_layers, out_dim=2, dropout=0.2, bidirectional=True,
                 device='cpu'):
        super(RNNModule, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.blstm_dim = blstm_dim
        self.hidden_size = blstm_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.device = device
        self.num_dir = 1
        if self.bidirectional:
            self.num_dir += 1
        self.embeddings = nn.Embedding(vocab.tokens_length, self.embed_dim, padding_idx=vocab.pad)
        self.rnn = nn.LSTM(self.embed_dim, self.blstm_dim, num_layers=self.num_layers,
                           bidirectional=self.bidirectional, dropout=dropout,
                           batch_first=True)
        self.drop = nn.Dropout(p=dropout)
        if self.bidirectional:
            self.norm_layer = nn.LayerNorm(2 * self.blstm_dim).to(device)
            self.fc = nn.Sequential(nn.Linear(2 * self.blstm_dim, self.out_dim), nn.ReLU(),
                                    nn.Dropout(p=dropout))
        else:
            self.norm_layer = nn.LayerNorm(2 * self.blstm_dim).to(device)
            self.fc = nn.Sequential(nn.Linear(self.blstm_dim, self.out_dim), nn.ReLU(),
                                    nn.Dropout(p=dropout))

    def forward(self, batch):
        smiles = batch["smiles"]
        seq_lens = batch['seq_len']
        x = self.embeddings(smiles.long())
        packed_input = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        packed_output, states = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), np.array(seq_lens) - 1, :self.blstm_dim]
        out_reverse = output[:, 0, self.blstm_dim:]
        text_fea = torch.cat((out_forward, out_reverse), 1)
        out = self.fc(text_fea)
        return out


class FPNModule(nn.Module):
    def __init__(self, fp_2_dim, out_feats, dropout=0.2):
        super(FPNModule, self).__init__()
        self.fp_2_dim = fp_2_dim
        self.dropout_fpn = dropout
        self.out_feats = out_feats
        self.fp_dim = 2513
        self.fc = nn.Sequential(nn.Linear(self.fp_dim, self.fp_2_dim), nn.ReLU(),
                                nn.Dropout(p=self.dropout_fpn), nn.Linear(self.fp_2_dim, self.out_feats))


    def forward(self, smiles):
        return self.fc(smiles)

class PhysChemModule(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=128, out_dim=384, dropout=0.2):
        super(PhysChemModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.has_residual = (in_dim == out_dim)
        if not self.has_residual:
            self.residual_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        orig_x = x
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            orig_x = orig_x.unsqueeze(0)

        out = self.fc(x)

        if self.has_residual:
            out = out + orig_x
        elif hasattr(self, 'residual_proj'):
            out = out + self.residual_proj(orig_x)

        return out

class ProteinCNN(nn.Module):

    def __init__(self, protein_Oridim=20, feature_size=200, out_features=343,
                 max_seq_len=3000, kernels=[4], dropout_rate=0.1):
        super(ProteinCNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.protein_Oridim = protein_Oridim
        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.kernels = kernels
        self.out_features = out_features

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=self.protein_Oridim,
                          out_channels=self.feature_size,
                          kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_seq_len - ks + 1)
            )
            for ks in self.kernels
        ])
        self.fc = nn.Linear(in_features=self.feature_size * len(self.kernels),
                            out_features=self.out_features)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        embedding_x = x.permute(0, 2, 1)  # [batch_size, protein_Oridim, seq_len]

        out = [conv(embedding_x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [batch_size, feature_size*len(kernels), 1]
        out = out.view(-1, out.size(1))  # [batch_size, feature_size*len(kernels)]
        out = self.dropout(out)
        out = self.fc(out)  # [batch_size, out_features]
        return out


class ProteinRNNModule(nn.Module):

    def __init__(self, vocab_size, embed_dim=64, blstm_dim=128, num_layers=2,
                 out_dim=128, dropout=0.2, bidirectional=True, device='cpu'):
        super(ProteinRNNModule, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.blstm_dim = blstm_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.device = device
        self.num_dir = 2 if bidirectional else 1

        self.embeddings = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(self.embed_dim, self.blstm_dim, num_layers=self.num_layers,
                           bidirectional=self.bidirectional, dropout=dropout,
                           batch_first=True)
        self.drop = nn.Dropout(p=dropout)

        rnn_output_dim = self.blstm_dim * self.num_dir
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.norm_layer = nn.LayerNorm(self.out_dim).to(device)

    def forward(self, batch):
        protein_seq = batch["protein_seq"]
        seq_lens = batch['protein_seq_len']

        x = self.embeddings(protein_seq.long())
        packed_input = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        packed_output, states = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            out_forward = output[range(len(output)), np.array(seq_lens) - 1, :self.blstm_dim]
            out_reverse = output[:, 0, self.blstm_dim:]
            text_fea = torch.cat((out_forward, out_reverse), 1)
        else:
            text_fea = output[range(len(output)), np.array(seq_lens) - 1, :]

        out = self.fc(text_fea)
        out = self.norm_layer(out)
        return out

class CTFModule(nn.Module):

    def __init__(self, input_dim=343, hidden_dim=256, output_dim=128, dropout=0.2):
        super(CTFModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)

