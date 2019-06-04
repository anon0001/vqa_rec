import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import utils

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import utils
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--reconstruction', type=bool, default=False)
    parser.add_argument('--npy', type=bool, default=False)
    parser.add_argument('--layer', type=int, default=4, choices=[4,5], help='res4f,res5c')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--gamma_r', type=float, default=0.5)
    parser.add_argument('--gamma_a', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--variant', type=str, default='1', choices=['0','1'])
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--use_residual', type=bool, default=False)
    parser.add_argument('--use_feat_loss', type=int, default=0)
    parser.add_argument('--dropout_hid', type=float, default=0.0)
    parser.add_argument('--dropout_unet', type=float, default=0.0)
    parser.add_argument('--use_one_cycle', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write("Output dir is %s"%args.output)

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')




    # eval_dset = VQAFeatureDataset('test2015', dictionary, dataroot=args.data_root,
    eval_dset = VQAFeatureDataset('val', dictionary, dataroot=args.data_root,
                                  size=args.size,
                                  npy=args.npy,
                                  layer=args.layer,
                                  finetune=args.finetune)
    if not args.eval:
        train_dset = VQAFeatureDataset('train', dictionary, dataroot=args.data_root,
                                       size=args.size,
                                       npy=args.npy,
                                       layer=args.layer,
                                       finetune=args.finetune)



    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid,
                                             args.reconstruction,
                                             layer=args.layer,
                                             size=args.size,
                                             variant=args.variant,
                                             finetune=args.finetune,
                                             use_residual=args.use_residual,
                                             use_feat_loss=args.use_feat_loss,
                                             dropout_hid=args.dropout_hid,
                                             dropout_unet=args.dropout_unet,
                                             logger=logger).cuda()

    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = (model).cuda()
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False)

    if args.eval:
        evaluate(model, eval_loader, eval_dset, args.gamma_r, args.use_feat_loss, args.output, args.ckpt)

    else:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True)
        train(model, train_loader, eval_loader, args.epochs, args.output,
              args.reconstruction,
              args.lr,
              args.gamma_r,
              args.layer,
              args.size,
              args.early_stop,
              args.finetune,
              args.use_feat_loss,
              args.dropout_hid,
              args.dropout_unet,
              args.use_one_cycle,
              args.ckpt,
              logger)


