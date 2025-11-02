import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="./output/", help="the path of output model")
    parser.add_argument("-c", "--class_num", type=int, default=1, help="the dimension of output")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="the number of running model")
    parser.add_argument("-e", "--epoch", type=int, default=60, help="the max number of epoch")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")

    parser.add_argument('--hidden_feats', type=list, default=[192, 384], help="the size of node representations after the i-th GAT layer")
    parser.add_argument('--rnn_embed_dim', type=int, default=128, help="the embedding size of each SMILES token")
    parser.add_argument('--rnn_hidden_dim', type=int, default=384, help="the number of features in the RNN hidden state")
    parser.add_argument('--rnn_layers', type=int, default=2, help="the number of rnn layers")
    parser.add_argument('--fp_dim', type=int, default=256, help="the hidden size of fingerprints module")
    #parser.add_argument('--fp_radius' , type=int, default=2, help="the radius of fingerprints module")
    #新增
    parser.add_argument('--physchem_dim', type=int, default=256, help="the hidden size of physicochemical features module")
    parser.add_argument('--head', type=int, default=16, help="the head size of attention")#24
    parser.add_argument('--p', type=float, default=0.4, help="dropout probability")#实际的dropout率为0.6
    parser.add_argument('--lr', type=float, default=1e-4,  help='learning rate')#8e-4
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=float, default=128, help='batch size')
    # 添加新的参数
    parser.add_argument('--graph_norm', type=bool, default=True, help="是否使用图正则化")
    parser.add_argument('--residual', type=bool, default=True, help="是否使用残差连接")
    parser.add_argument('--physchem_layers', type=int, default=3, help="物理化学特征网络的层数")
    parser.add_argument('--batch_norm', type=bool, default=True, help="是否使用批量归一化")
    parser.add_argument('--protein_cnn_dim', type=int, default=343,
                        help="蛋白质CNN特征的输出维度")
    parser.add_argument('--protein_rnn_dim', type=int, default=128,
                        help="蛋白质RNN特征的输出维度")
    parser.add_argument('--protein_vocab_size', type=int, default=21,
                        help="蛋白质序列词汇表大小（20种氨基酸+padding）")
    parser.add_argument('--max_seq_len', type=int, default=3000,
                        help="蛋白质序列的最大长度")
    parser.add_argument('--protein_kernels', type=list, default=[3],
                        help="蛋白质CNN的卷积核大小列表")
    parser.add_argument('--ctf_dim', type=int, default=343, help='CTF特征维度')
    # 消融实验
    parser.add_argument('--use_smiles', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="是否使用SMILES序列视图")
    parser.add_argument('--use_graph', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="是否使用分子图视图")
    parser.add_argument('--use_fingerprint', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="是否使用分子指纹视图")
    parser.add_argument('--use_physchem', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="是否使用物理化学特征视图")
    parser.add_argument('--use_protein', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="是否使用蛋白质序列视图")
    parser.add_argument('--use_attention_fusion', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="是否使用注意力融合机制，False则使用简单连接")

    return parser.parse_args()
