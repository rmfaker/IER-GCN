import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of EV-GCN')
        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')

        parser.add_argument('--hgc', type=int, default=32, help='hidden units of gconv layer')
        parser.add_argument('--lg', type=int, default=3, help='number of gconv layers')
        parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
        parser.add_argument('--num_iter', default=300, type=int, help='number of epochs for training')
        parser.add_argument('--edropout', type=float, default=0.3, help='edge dropout rate')
        parser.add_argument('--dropout', default=0.2, type=float, help='ratio of dropout')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--ckpt_path', type=str, default='./save_models/ev_gcn', help='checkpoint path to save trained models')

        parser.add_argument('--mode', type=str, default='kfold', choices=['kfold', 'loso'])
        parser.add_argument('--n_folds', type=int, default=10)

        # (so train_eval_evgcn.py can access these)
        parser.add_argument('--abide_root', type=str, default='./data/ABIDE_pcp')
        parser.add_argument('--pipeline', type=str, default='cpac')
        parser.add_argument('--filt', type=str, default='filt_noglobal')
        parser.add_argument('--derivative', type=str, default='rois_cc200')

        
        # DIR flags (default off)
        parser.add_argument('--dir', type=str, default='off', choices=['off','full'],
                            help='enable full DIR training')
        
        parser.add_argument('--dir_topk_ratio', type=float, default=0.6,
                            help='fraction of edges kept as C (Top-k)')
        parser.add_argument('--dir_alpha', type=float, default=1.0,
                            help='scale for residual logits')
        parser.add_argument('--dir_tau', type=float, default=1.0,
                            help='temperature on PAE logit')
        parser.add_argument('--dir_tau_gumbel', type=float, default=1.0,
                            help='gumbel temperature for soft top-k')
        parser.add_argument('--dir_lambda_var', type=float, default=0.1,
                            help='weight for variance term')
        parser.add_argument('--dir_J', type=int, default=6,
                            help='number of interventions per step')
        parser.add_argument('--dir_bank', type=int, default=32,
                            help='memory bank capacity for S')
        
        parser.add_argument('--dir_no_residual', action='store_true',
                            help='If set, disable residual ψ: posterior π uses prior only.')

        parser.add_argument('--mdd_root', type=str, default='./data/MDD_wwh_667')

        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(" Using GPU in torch")

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(123)
        #self.logging_init()
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


