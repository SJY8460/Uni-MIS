import argparse
import torch

import logging

import os

parser = argparse.ArgumentParser(
    description='Joint Multiple Intent Detection and Slot Filling')

# Hyper-param
parser.add_argument('--bert_lr','-br', type=float, default=1e-5)
parser.add_argument('--slot_weight', '-sw', type=float, default=1)
parser.add_argument('--intent_weight', '-iw', type=float, default=1)
parser.add_argument('--window_type', '-wt', type=str, default='tf',help='cnn or transformer')
parser.add_argument('--window_size', '-ws', type=int, default=3)
parser.add_argument('--use_hard_vote', '-hv', action='store_true', default=False)
parser.add_argument('--ablation', '-ab', action='store_true', default=False)
parser.add_argument('--task_type', '-tt', type=str, default='multi')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument("--random_state",
                    '-rs',
                    help='random seed',
                    type=int,
                    default=999)
parser.add_argument('--gpu',
                    '-g',
                    action='store_true',
                    help='use gpu',
                    required=False,
                    default=False)
# File Path
parser.add_argument('--data_dir',
                    '-dd',
                    help='dataset file path',
                    type=str,
                    default='./data/MixATIS_clean')
parser.add_argument('--save_dir', '-sd', type=str, default='./save/MixATIS_clean')
parser.add_argument('--message', type=str)
# Trainning
parser.add_argument('--num_epoch', '-ne', type=int, default=300)
parser.add_argument('--patient', type=int, default=30) 
parser.add_argument('--batch_size', '-bs', type=int, default=32)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.0001)
parser.add_argument('--load_model_dir',
                    '-lmd',
                    help='the path of load model',
                    type=str,
                    required=False)
parser.add_argument('--optimizer',
                    '-op',
                    type=str,
                    default='Adam',
                    required=False)
# Model configuration
parser.add_argument('--tf_layer_num', '-nl', type=int, default=4)
parser.add_argument('--drop_out', '-do', type=float, default=0.4)
parser.add_argument('--embedding_dim', '-ed', type=int, default=128)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--decoder', type=str, default='lstm')
parser.add_argument('--decoder_hidden_dim', '-dhd', type=int, default=128)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=128)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=256)
parser.add_argument('--embed_model_name', '-emn', type=str, required=False)

parser.add_argument('--label_embedding_name', '-len', type=str, default=None)
parser.add_argument('--decoder_gat_hidden_dim', '-dghd', type=int, default=128)
parser.add_argument('--n_head', '-nh', type=int, default=4)
parser.add_argument('--slot_graph_window', '-sgw', type=int, default=2)

parser.add_argument('--embed_type',
                    '-et',
                    type=str,
                    default='None',
                    help='w2v|bert|wpb|elmo|roberta|roberta-large')
parser.add_argument('--bert_path',
                    '-bt',
                    type=str,
                    default='bert-base-uncased',
                    )

parser.add_argument('--freeze_embed',
                    '-fe',
                    action='store_true',
                    help='freeze embedding',
                    required=False,
                    default=False)
parser.add_argument('--use_char',
                    '-uc',
                    action='store_true',
                    help='use char',
                    required=False,
                    default=False)
parser.add_argument('--use_label_char',
                    '-ulc',
                    action='store_true',
                    help='use label char',
                    required=False,
                    default=False)
parser.add_argument('--use_label_enhance',
                    '-ule',
                    action='store_true',
                    help='use enhance',
                    required=False,
                    default=False)

parser.add_argument('--use_bio',
                    '-ubio',
                    action='store_true',
                    help='use bio',
                    required=False,
                    default=False)

parser.add_argument('--use_co',
                    '-uco',
                    action='store_true',
                    help='use co sf',
                    required=False,
                    default=False)
parser.add_argument('--use_gold',
                    '-ugold',
                    action='store_true',
                    help='use gold',
                    required=False,
                    default=False)
parser.add_argument('--use_adaptor',
                    '-uada',
                    action='store_true',
                    help='use adaptor',
                    required=False,
                    default=False)
parser.add_argument('--intent_guide',
                    '-ig',
                    action='store_true',
                    help='use intent guide',
                    required=False,
                    default=False)



args = parser.parse_args()
args.gpu = args.gpu and torch.cuda.is_available()


def init_logger(save_dir):
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    os.makedirs(save_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# logger.info(str(vars(args)))

import numpy as np
import random

# Fix the random seed of package random.
random.seed(args.random_state)
np.random.seed(args.random_state)

# Fix the random seed of Pytorch when using GPU.
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fix the random seed of Pytorch when using CPU.
torch.manual_seed(args.random_state)
torch.random.manual_seed(args.random_state)