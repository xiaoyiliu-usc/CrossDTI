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
    parser.add_argument('--physchem_dim', type=int, default=256, help="the hidden size of physicochemical features module")
    parser.add_argument('--head', type=int, default=16, help="the head size of attention")
    parser.add_argument('--p', type=float, default=0.4, help="dropout probability")
    parser.add_argument('--lr', type=float, default=1e-4,  help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=float, default=128, help='batch size')
    parser.add_argument('--graph_norm', type=bool, default=True)
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--physchem_layers', type=int, default=3)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--protein_cnn_dim', type=int, default=343)
    parser.add_argument('--protein_rnn_dim', type=int, default=128)
    parser.add_argument('--protein_vocab_size', type=int, default=21)
    parser.add_argument('--max_seq_len', type=int, default=3000)
    parser.add_argument('--protein_kernels', type=list, default=[3])
    parser.add_argument('--ctf_dim', type=int, default=343)
    # ablation
    parser.add_argument('--use_smiles', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use SMILES features or not')
    parser.add_argument('--use_graph', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use graph features or not')
    parser.add_argument('--use_fingerprint', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use fingerprints or not')
    parser.add_argument('--use_physchem', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use physicochemical features or not')
    parser.add_argument('--use_protein', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use protein features or not')
    parser.add_argument('--use_attention_fusion', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use attention fusion or not')
    parser.add_argument('--use_ctf',default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use ctf or not')
    return parser.parse_args()
