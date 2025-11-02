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
        """编码蛋白质序列为one-hot格式"""
        # 氨基酸字典（20种标准氨基酸）
        amino_acid_dict = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
            'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20
        }

        # 将序列转换为索引（0用于padding）
        seq_indices = [amino_acid_dict.get(aa, 0) for aa in sequence.upper() if aa in amino_acid_dict]
        seq_len = len(seq_indices)

        # 截断或填充序列到固定长度
        if len(seq_indices) > self.max_seq_len:
            seq_indices = seq_indices[:self.max_seq_len]
            seq_len = self.max_seq_len
        else:
            seq_indices = seq_indices + [0] * (self.max_seq_len - len(seq_indices))

        # 转换为one-hot编码
        one_hot_seq = np.zeros((self.max_seq_len, 20))
        for i, idx in enumerate(seq_indices):
            if idx > 0:  # 有效氨基酸索引（1-20）
                one_hot_seq[i, idx - 1] = 1

        # 返回字典格式，与collate函数期望的一致
        return {
            'protein_cnn': one_hot_seq,  # 用于CNN的one-hot编码
            'protein_seq': seq_indices,  # 用于序列模型的索引
            'protein_seq_len': seq_len  # 实际序列长度
        }

    def calculate_ctf(self, sequence):
        # 氨基酸分类字典：将20种氨基酸分为7类
        # 分类依据：物理化学性质相似的氨基酸归为一类
        aa_class_dict = {
            'A': 1, 'G': 1, 'V': 1,  # 小疏水氨基酸
            'I': 2, 'L': 2, 'F': 2, 'P': 2,  # 疏水氨基酸
            'Y': 3, 'M': 3, 'T': 3, 'S': 3,  # 极性氨基酸
            'H': 4, 'N': 4, 'Q': 4, 'W': 4,  # 芳香族/碱性氨基酸
            'R': 5, 'K': 5,  # 碱性氨基酸
            'D': 6, 'E': 6,  # 酸性氨基酸
            'C': 7  # 半胱氨酸（特殊）
        }

        # 初始化CTF特征向量（7^3=343维）
        ctf_vector = np.zeros(self.ctf_dim)

        # 将序列中的每个氨基酸映射到其类别
        classified_seq = [aa_class_dict.get(aa, 0) for aa in sequence.upper() if aa in aa_class_dict]

        # 计算三联体频率
        for i in range(len(classified_seq) - 2):
            # 获取连续三个氨基酸的类别
            triad = (classified_seq[i], classified_seq[i + 1], classified_seq[i + 2])
            # 计算在三联体空间中的索引（7进制）
            index = (triad[0] - 1) * 49 + (triad[1] - 1) * 7 + (triad[2] - 1)
            if 0 <= index < self.ctf_dim:
                ctf_vector[index] += 1

        # 归一化处理
        if len(classified_seq) > 2:
            ctf_vector = ctf_vector / (len(classified_seq) - 2)

        return ctf_vector


    def __getitem__(self, index):
        """
        获取指定索引的数据项。
        index: 索引值，用于从数据集中获取对应的分子数据
        """
        smiles = self.smiles[index]  # 获取第 index 个分子的 SMILES 字符串
        sequence = self.sequences[index]
        labels = self.labels[index]  # 获取第 index 个分子的标签
        mol = Chem.MolFromSmiles(smiles)  # 使用 RDKit 将 SMILES 字符串转换为分子对象
        graph = mol_to_bigraph(mol, add_self_loop=True, node_featurizer=self.node_featurizer,
                               edge_featurizer=self.edge_featurizer)

        # 计算多个分子指纹并将它们存入一个列表
        fp = []
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  # 计算 MACCS 指纹，167 维
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # 计算 ErG 指纹，441 维
        fp_pubcfp = GetPubChemFPs(mol)  # 计算 PubChem 指纹，881 维
        fp_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # 计算 ECFP2 指纹，1024 维
        # 将所有的指纹拼接到一起
        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp.extend(fp_pubcfp)
        fp.extend(fp_ecfp2)

        # 添加物理化学特征计算
        physchem = calculate_physchem_features(mol)

        protein_data = self.encode_sequence(sequence)

        # 计算CTF特征
        ctf = self.calculate_ctf(sequence)

        # 返回处理后的数据：SMILES 编码，图结构，指纹，物理化学特征和标签
        return self.token.encode(smiles), graph, fp, physchem, protein_data, ctf, torch.tensor(labels, dtype=torch.float32)

def collate(sample):
    """
    collate_fn 函数，用于处理批量数据
    sample: 输入的样本数据
    """
    # 将样本拆分成编码的 SMILES、图、指纹、物理化学特征和标签
    encoded_smiles, graphs, fps, physchem, protein_data_list, ctf_list, labels = map(list, zip(*sample))
    # 使用 DGL 将多个图结构合并成一个大图
    batched_graph = dgl.batch(graphs)
    # 合并标签并转换为浮点数类型
    labels = torch.stack(labels)
    # 获取每个 SMILES 序列的长度
    seq_len = [len(i) for i in encoded_smiles]
    # 对 SMILES 序列进行填充，使得每个批次中的 SMILES 长度相同
    padded_smiles_batch = pad_sequence([torch.tensor(i) for i in encoded_smiles], batch_first=True)
    # 将指纹转换为浮点型张量
    fps_t = torch.FloatTensor(fps)
    # 将物理化学特征转换为浮点型张量
    physchem_t = torch.FloatTensor(physchem)

    ctf_t = torch.FloatTensor(ctf_list)

    # 处理蛋白质序列数据
    protein_cnn_batch = torch.FloatTensor([data['protein_cnn'] for data in protein_data_list])
    protein_seq_batch = torch.LongTensor([data['protein_seq'] for data in protein_data_list])  # 已经是填充后的
    protein_seq_len = [data['protein_seq_len'] for data in protein_data_list]

    protein_data = {
        'protein_cnn': protein_cnn_batch,
        'protein_seq': protein_seq_batch,
        'protein_seq_len': protein_seq_len
    }
    # 返回一个包含 SMILES、图、指纹、物理化学特征和标签的字典
    return {'smiles': padded_smiles_batch, 'seq_len': seq_len}, batched_graph, fps_t, physchem_t, protein_data, ctf_t, labels


def calculate_physchem_features(mol):
    """计算分子的物理化学特性"""
    features = []
    # 保留原有特征
    features.append(Descriptors.MolWt(mol))  # 分子量
    features.append(Crippen.MolLogP(mol))  # LogP
    features.append(Lipinski.NumHDonors(mol))  # 氢键供体数量
    features.append(Lipinski.NumHAcceptors(mol))  # 氢键受体数量
    features.append(Descriptors.NumRotatableBonds(mol))  # 可旋转键数量
    features.append(Lipinski.NumAromaticRings(mol))  # 芳香环数量
    features.append(Descriptors.TPSA(mol))  # 拓扑极性表面积
    features.append(mol.GetNumHeavyAtoms())  # 重原子数量
    features.append(mol.GetNumAtoms())  # 原子数量
    features.append(Descriptors.RingCount(mol))  # 环数量

    # 添加新特征
    features.append(Descriptors.FractionCSP3(mol))  # sp3碳的比例
    features.append(Descriptors.NumAliphaticRings(mol))  # 脂肪环数量
    features.append(Descriptors.NumSaturatedRings(mol))  # 饱和环数量
    features.append(Descriptors.NumHeteroatoms(mol))  # 杂原子数量
    features.append(Descriptors.NumAromaticHeterocycles(mol))  # 芳香杂环数量
    features.append(Descriptors.NumSaturatedHeterocycles(mol))  # 饱和杂环数量
    features.append(Descriptors.NumAliphaticHeterocycles(mol))  # 脂肪杂环数量
    features.append(Descriptors.LabuteASA(mol))  # 拉布特表面积

    return features

