import fitlog
import random
import argparse
import os
from utils.config import args
import json
import numpy as np
import torch
from utils.dataset import DataManager
from models.main_model import MainModel
from utils.process import Processor
from utils.config import init_logger
import logging
import time

if __name__ == '__main__':
    if args.mode == 'train':
        # fitlog.commit(__file__)
        fitlog.set_log_dir('logs/')  # set the logging directory
        fitlog.create_log_folder()
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters
        args.save_dir = 'logs/' + fitlog.get_log_folder()
        init_logger(args.save_dir)
        logger = logging.getLogger()
        if not os.path.exists(args.save_dir):
            os.system("mkdir -p " + args.save_dir)

        log_path = os.path.join(args.save_dir, "param.json")
        with open(log_path, "w", encoding="utf8") as fw:
            fw.write(json.dumps(args.__dict__, indent=True))

        dataset = DataManager(args, [('word', False), ('intent', True),
                                     ('slot', True),('token_intent',True)])

        dataset.quick_build()
        dataset.show_summary()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MainModel(
            dataset,
            args
            )
        
        logger.info(model)
        processor = Processor(args, dataset, model)
        processor.train()

    if args.mode == 'test':
        import os
        file_list = sorted(os.listdir('./logs/'))
        for idx, file_name in enumerate(file_list):
            print("[{}] {}".format(idx, file_name))
        print("请输入需要测试的模型")
        idx = int(input())
        args.save_dir = os.path.join('./logs/', file_list[idx]) + '/'
        dataset = DataManager(args, [('word', False), ('intent', True),
                                     ('slot', True),('token_intent',True)])

        dataset.quick_build()
        dataset.show_summary()
        model = MainModel(
            dataset,
            args
            )
        processor = Processor(args, dataset, model)
        output = processor.prediction(
            args, args.save_dir)
    

if args.mode == 'train':
    fitlog.finish()  # finish the logging
