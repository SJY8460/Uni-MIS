# import fitlog
import random
import argparse
import os
import json
import numpy as np
import torch
from utils.dataset import DataManager
from models.main_model import MainModel
from utils.wandb_process import Processor
from utils.config import init_logger
import logging
import time
import wandb

import argparse
import torch

import logging

import datetime




parser = argparse.ArgumentParser(
    description='Joint Multiple Intent Detection and Slot Filling')

# Hyper-param
# parser.add_argument('--bert_lr','-br', type=float, default=1e-5)
parser.add_argument('--slot_weight', '-sw', type=float, default=1)
# parser.add_argument('--intent_weight', '-iw', type=float, default=1)
parser.add_argument('--window_type', '-wt', type=str, default='tf',help='cnn or transformer')
parser.add_argument('--window_size', '-ws', type=int, default=3)
parser.add_argument('--use_hard_vote', '-hv', default=True)
parser.add_argument('--ablation', '-ab', action='store_true', default=False)
parser.add_argument('--task_type', '-tt', type=str, default='multi')
parser.add_argument('--mode', type=str, default='train')
# parser.add_argument("--random_state",
#                     '-rs',
#                     help='random seed',
#                     type=int,
#                     default=999)
parser.add_argument('--gpu',
                    '-g',
                    action='store_true',
                    help='use gpu',
                    default=True)
# File Path
# parser.add_argument('--data_dir',
#                     '-dd',
#                     help='dataset file path',
#                     type=str,
#                     default='./data/MixATIS_clean')
# parser.add_argument('--save_dir', '-sd', type=str, default='./save/MixATIS_clean')
parser.add_argument('--message', type=str)
# Trainning
parser.add_argument('--num_epoch', '-ne', type=int, default=200)
parser.add_argument('--patient', type=int, default=30) 
# parser.add_argument('--batch_size', '-bs', type=int, default=32)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
# parser.add_argument("--learning_rate", '-lr', type=float, default=0.0001)
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
# parser.add_argument('--decoder', type=str, default='lstm')
parser.add_argument('--decoder_hidden_dim', '-dhd', type=int, default=128)
# parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=128)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)
# parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=256)
parser.add_argument('--embed_model_name', '-emn', type=str, required=False)

parser.add_argument('--label_embedding_name', '-len', type=str, default=None)
# parser.add_argument('--decoder_gat_hidden_dim', '-dghd', type=int, default=128)
parser.add_argument('--n_head', '-nh', type=int, default=4)
parser.add_argument('--slot_graph_window', '-sgw', type=int, default=2)

# parser.add_argument('--embed_type',
#                     '-et',
#                     type=str,
#                     default='None',
#                     help='w2v|bert|wpb|elmo|roberta|roberta-large')
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


sweep_config = {
     'name': 'Uni-MIS',
    "method": "grid",
     'metric': {
      'name': 'best sem acc',
      'goal': 'maximize'   
    },
    "parameters": {
        "intent_embedding_dim": {"values": [256,384]},
        "slot_decoder_hidden_dim": {"values": [384,512]},
        "decoder_gat_hidden_dim": {"values": [128,256]},
        "save_dir": {"values": ["./save/MixSNIPS_clean"]},
        "data_dir": {"values": ["./data/MixSNIPS_clean"]},
        "batch_size": {"values": [32,16]},
        "random_state": {"values": [120,3407,114514,42,996]},
        "decoder": {"values": ["agif"]},
        "embed_type": {"values": ["roberta"]},
        "bert_lr": {"values": [1e-5,5e-5,2e-5]},
        "learning_rate": {"values": [1e-4,2e-4,5e-4,5e-5]},
        "intent_weight": {"values": [0.1,0.2,0.4,0.8,1.0]},
       
    }
}


def train(config = None):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    import os
    wandb.init(config = None, name = nowtime,group='snips')
    config = wandb.config
    config.update(args.__dict__)


    if config.mode == 'train':
        # fitlog.commit(__file__)
        # fitlog.set_log_dir('logs/')  # set the logging directory
        # fitlog.create_log_folder()
        # fitlog.add_hyper(config)
        # fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters
        # config.save_dir = 'logs/' + fitlog.get_log_folder()
        init_logger(config.save_dir)
        logger = logging.getLogger()
        # if not os.path.exists(config.save_dir):
        #     os.system("mkdir -p " + config.save_dir)

        # log_path = os.path.join(config.save_dir, "param.json")
        # with open(log_path, "w", encoding="utf8") as fw:
        #     fw.write(json.dumps(config.__dict__, indent=True))

        dataset = DataManager(config, [('word', False), ('intent', True),
                                     ('slot', True),('token_intent',True)])

        dataset.quick_build()
        dataset.show_summary()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MainModel(
            dataset,
            config
            )
        
        logger.info(model)
        processor = Processor(config, dataset, model)
        processor.train()

    if config.mode == 'test':
        import os
        file_list = sorted(os.listdir('./logs/'))
        for idx, file_name in enumerate(file_list):
            print("[{}] {}".format(idx, file_name))
        print("请输入需要测试的模型")
        idx = int(input())
        config.save_dir = os.path.join('./logs/', file_list[idx]) + '/'
        dataset = DataManager(config, [('word', False), ('intent', True),
                                     ('slot', True),('token_intent',True)])

        dataset.quick_build()
        dataset.show_summary()
        model = MainModel(
            dataset,
            config
            )
        processor = Processor(config, dataset, model)
        output = processor.prediction(
            config, config.save_dir)
    

        # if config.mode == 'train':
        #     fitlog.finish()  # finish the logging
    wandb.finish()

def sweep():
    sweep_id = wandb.sweep(sweep_config, project="Uni-Joint")
    wandb.agent(sweep_id, function=train)

if __name__ == "__main__":
        wandb.login()
        sweep()