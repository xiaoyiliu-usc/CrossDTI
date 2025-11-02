import pandas as pd
import torch
import dgl
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski

from utils import tokens_struct
from pubchemfp import GetPubChemFPs


class MolDataSet(Dataset):
    def __init__(self, data_path, labels_split=',', max_seq_len=3000, protein_vocab_size=21, ctf_dim=343):
        self.data = pd.read_csv(data_path)
        self.smiles = self.data['SMILES'].to_list()
        self.sequences = self.data['Sequence'].to_list()
        self.labels = self.data['label'].astype(int).to_list()
        self.labels_split = labels_split
        self.max_seq_len = max_seq_len
        self.protein_vocab_size = protein_vocab_size
        self.ctf_dim = ctf_dim
        self.token = tokens_struct()
        self.node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

    def encode_sequence(self, sequence):
        amino_acid_dict = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
            'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20
        }

        seq_indices = [amino_acid_dict.get(aa, 0) for aa in sequence.upper() if aa in amino_acid_dict]
        seq_len = len(seq_indices)

        if len(seq_indices) > self.max_seq_len:
            seq_indices = seq_indices[:self.max_seq_len]
            seq_len = self.max_seq_len
        else:
            seq_indices = seq_indices + [0] * (self.max_seq_len - len(seq_indices))


        one_hot_seq = np.zeros((self.max_seq_len, 20))
        for i, idx in enumerate(seq_indices):
            if idx > 0:
                one_hot_seq[i, idx - 1] = 1

        return {
            'protein_cnn': one_hot_seq,
            'protein_seq': seq_indices,
            'protein_seq_len': seq_len
        }

    def calculate_ctf(self, sequence):

        aa_class_dict = {
            'A': 1, 'G': 1, 'V': 1,
            'I': 2, 'L': 2, 'F': 2, 'P': 2,
            'Y': 3, 'M': 3, 'T': 3, 'S': 3,
            'H': 4, 'N': 4, 'Q': 4, 'W': 4,
            'R': 5, 'K': 5,
            'D': 6, 'E': 6,
            'C': 7
        }


        ctf_vector = np.zeros(self.ctf_dim)


        classified_seq = [aa_class_dict.get(aa, 0) for aa in sequence.upper() if aa in aa_class_dict]


        for i in range(len(classified_seq) - 2):

            triad = (classified_seq[i], classified_seq[i + 1], classified_seq[i + 2])

            index = (triad[0] - 1) * 49 + (triad[1] - 1) * 7 + (triad[2] - 1)
            if 0 <= index < self.ctf_dim:
                ctf_vector[index] += 1

        if len(classified_seq) > 2:
            ctf_vector = ctf_vector / (len(classified_seq) - 2)

        return ctf_vector


    def __getitem__(self, index):
        smiles = self.smiles[index]
        sequence = self.sequences[index]
        labels = self.labels[index]
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_bigraph(mol, add_self_loop=True, node_featurizer=self.node_featurizer,
                               edge_featurizer=self.edge_featurizer)

        fp = []
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
        fp_pubcfp = GetPubChemFPs(mol)
        fp_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp.extend(fp_pubcfp)
        fp.extend(fp_ecfp2)

        physchem = calculate_physchem_features(mol)

        protein_data = self.encode_sequence(sequence)

        ctf = self.calculate_ctf(sequence)

        return self.token.encode(smiles), graph, fp, physchem, protein_data, ctf, torch.tensor(labels, dtype=torch.float32)

def collate(sample):
    encoded_smiles, graphs, fps, physchem, protein_data_list, ctf_list, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)
    seq_len = [len(i) for i in encoded_smiles]
    padded_smiles_batch = pad_sequence([torch.tensor(i) for i in encoded_smiles], batch_first=True)
    fps_t = torch.FloatTensor(fps)
    physchem_t = torch.FloatTensor(physchem)

    ctf_t = torch.FloatTensor(ctf_list)

    protein_cnn_batch = torch.FloatTensor([data['protein_cnn'] for data in protein_data_list])
    protein_seq_batch = torch.LongTensor([data['protein_seq'] for data in protein_data_list])
    protein_seq_len = [data['protein_seq_len'] for data in protein_data_list]

    protein_data = {
        'protein_cnn': protein_cnn_batch,
        'protein_seq': protein_seq_batch,
        'protein_seq_len': protein_seq_len
    }
    return {'smiles': padded_smiles_batch, 'seq_len': seq_len}, batched_graph, fps_t, physchem_t, protein_data, ctf_t, labels


def calculate_physchem_features(mol):
    features = []

    features.append(Descriptors.MolWt(mol))
    features.append(Crippen.MolLogP(mol))
    features.append(Lipinski.NumHDonors(mol))
    features.append(Lipinski.NumHAcceptors(mol))
    features.append(Descriptors.NumRotatableBonds(mol))
    features.append(Lipinski.NumAromaticRings(mol))
    features.append(Descriptors.TPSA(mol))
    features.append(mol.GetNumHeavyAtoms())
    features.append(mol.GetNumAtoms())
    features.append(Descriptors.RingCount(mol))


    features.append(Descriptors.FractionCSP3(mol))
    features.append(Descriptors.NumAliphaticRings(mol))
    features.append(Descriptors.NumSaturatedRings(mol))
    features.append(Descriptors.NumHeteroatoms(mol))
    features.append(Descriptors.NumAromaticHeterocycles(mol))
    features.append(Descriptors.NumSaturatedHeterocycles(mol))
    features.append(Descriptors.NumAliphaticHeterocycles(mol))
    features.append(Descriptors.LabuteASA(mol))

    return features

