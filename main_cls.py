#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from genericpath import exists
import os
from typing import Final
import cv2
import sys
from matplotlib.pyplot import xcorr
from numpy.random import f, sample, shuffle

from torch.utils.data import dataset
from config import parser

if len(sys.argv) > 1:
    # use shell args
    args = parser.parse_args()
    print('Use shell args.')
else:
    # Debug
    args_list = [
        '--dataset',
        'SAMM',
        '--print-freq',
        '1',
        '--snap',
        'debug',
        '--data_option',
        'wt_diff',
        '--gpus',
        '0',
        '--batch_size',
        '2',
        '--input_size',
        '128',
        '--length',
        '64',
        '-L',
        '12',
        '--workers',
        '0',
    ]
    args = parser.parse_args(args_list)
# os setting
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
if args.gpus is not None:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import re
import logging
import time
import torch
import os.path as osp
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from datetime import datetime
from tqdm import tqdm
from pprint import pformat
from timm.utils import setup_default_logging, NativeScaler, reduce_tensor, distribute_bn
from timm.data.distributed_sampler import OrderedDistributedSampler
from contextlib import suppress

from model.network import Two_Stream_RNN_Cls, load_pretrained_model
from dataset.me_dataset import SAMMDataset, CASME_2Dataset
import utils
import trainer_cls as trainer

# torch.multiprocessing.set_start_method('spawn')
torch.backends.cudnn.benchmark = True

# check resume
RESUME = osp.exists(args.resume)

# check finetune
if len(args.finetune_list) > 0:
    assert RESUME
    FINETUNE = True
else:
    FINETUNE = False

_logger = logging.getLogger('train')
# resume
if RESUME:
    setattr(args, 'save_root', 'results/{}'.format(osp.basename(args.resume)))
else:
    snapshot_name = '_'.join(
        [args.snap, datetime.now().strftime("%Y%m%d-%H%M%S")])
    if len(args.store_name) == 0:
        args.store_name = snapshot_name
    setattr(args, 'save_root', 'results/{}'.format(args.store_name))
# make dirs
if args.local_rank == 0:
    utils.check_rootfolders(args)
else:
    time.sleep(1)
# setup logging
setup_default_logging(
    log_path=os.path.join(args.save_root, args.root_log, 'run.log'))
_logger.info("save experiment to :{}".format(args.save_root))
# save args
if args.local_rank == 0:
    args_string = pformat(args.__dict__)
    _logger.info(args_string)

    # reset random
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# if distributed
if args.distributed and 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
args.device = 'cuda'
args.world_size = 1
args.rank = 0  # global rank
if args.distributed:
    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    _logger.info(
        'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
        % (args.rank, args.world_size))
# else:
#     _logger.info('Training with a single process on 1 GPUs.')
assert args.rank >= 0
utils.synchronize()

# loss_fn
criterion = utils.Focal_Loss(alpha=args.focal_alpha)

# leave one subject out cross validation
img_dirs = utils.get_img_dirs(args.dataset)
img_dirs_dict = utils.leave_one_out(
    img_dirs, args.dataset)  # key -> [train_set, val_set]

# finetuen and resume
if RESUME:
    total_MNA = np.load(osp.join(args.resume, args.root_output,
                                 'cross_validation_MNA_dict.npy'),
                        allow_pickle=True).item()
    match_regions_record_all = np.load(osp.join(
        args.resume, args.root_output, 'match_regions_record_all.npy'),
                                       allow_pickle=True).item()
    if not FINETUNE:
        keys1 = list(total_MNA.keys())
        # keys2 = list(match_regions_record_all.keys())
        rm_key = keys1[-1]  # after python 3.6, order is guaranteed
        if args.delete_last:
            # delete the last subject results
            total_MNA, match_regions_record_all = utils.delete_records(
                total_MNA, match_regions_record_all, rm_key)
            if args.local_rank == 0:
                _logger.info('resume from subject {} (include)'.format(rm_key))
        elif args.local_rank == 0:
            _logger.info('resume from subject {} (not include)'.format(rm_key))
    else:
        if args.local_rank == 0:
            _logger.info('finetune subjects: [{}]'.format(','.join(
                args.finetune_list)))
else:
    total_MNA = {}  # store all cross-validation results
    match_regions_record_all = {}
utils.synchronize()

for vi, (val_id, [train_dirs, val_dirs]) in enumerate(img_dirs_dict.items()):
    # leave {val_id} out...

    # FINETUNE has higher priority than RESUME
    if FINETUNE and (val_id not in args.finetune_list):
        continue  # skip subjects that do not need finetune
    if RESUME and (not FINETUNE) and (val_id in total_MNA):
        continue  # skip from resume

    if val_id in args.finetune_list:
        # delete records
        total_MNA, match_regions_record_all = utils.delete_records(
            total_MNA, match_regions_record_all, val_id)

    if args.data_option == 'diff':
        inchannel = args.L
    elif args.data_option == 'wt_diff':
        inchannel = 4 * args.L
    elif args.data_option == 'wt_dr':
        inchannel = (
            args.L + 1 - 11 +
            1) * 2 * 4  # gauss kernel size = 11, *2 = dr1,dr2, *4 = 4 bands

    # amp
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info(
                'Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # model
    model = Two_Stream_RNN_Cls(mlp_hidden_units=args.hidden_units,
                               inchannel=inchannel,
                               outchannel=2)
    # load pretrained
    if osp.exists(args.load_pretrained):
        model = load_pretrained_model(model, args.load_pretrained,
                                      args.load_bn)
        if args.local_rank == 0:
            _logger.info('Load pretrained model from {}[load_bn: {}]'.format(
                args.load_pretrained, args.load_bn))
    # pytorch_total_params = sum(p.numel() for p in model.parameters()
    #                            if p.requires_grad)
    # print("Total Params: {}".format(pytorch_total_params))
    model = model.cuda()

    # setup synchronized BatchNorm for distributed training
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # if args.local_rank == 0:
        #     _logger.info(
        #         'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
        #         'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.'
        #     )

    # optimizer
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # setup distributed training
    if args.distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        find_unused_parameters=True)
    else:
        model = DataParallel(model).cuda()

    # dataset
    Dataset = SAMMDataset if args.dataset == 'SAMM' else CASME_2Dataset

    def create_dataset():
        train_dataset = Dataset(
            mode='train',
            img_dirs=train_dirs,
            seq_len=args.length,
            step=args.step,
            # step=1000,  # !!
            time_len=args.L,
            input_size=args.input_size,
            data_aug=args.data_aug,
            data_option=args.data_option)
        val_dataset = Dataset(
            mode='test',
            img_dirs=val_dirs,
            seq_len=args.length,
            step=args.length,  # assert no overlap
            # step=1000,  # !!
            time_len=args.L,
            input_size=args.input_size,
            data_aug=False,
            data_option=args.data_option)
        return train_dataset, val_dataset

    train_dataset, val_dataset = create_dataset()
    if args.distributed:
        val_sampler = OrderedDistributedSampler(val_dataset)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        val_sampler = None
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=train_sampler is None,
                                               sampler=train_sampler,
                                               batch_size=args.batch_size,
                                               drop_last=False,
                                               num_workers=args.workers,
                                               pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             num_workers=0,
                                             pin_memory=False,
                                             drop_last=False)

    if args.local_rank == 0:
        _logger.info('<' * 10 + ' {} '.format(val_id) + '<' * 10)
    best_f_score = -1000.0
    best_loss = 1000.0
    val_accum_epochs = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        utils.adjust_learning_rate(optimizer, epoch, args.lr,
                                   args.weight_decay, args.lr_steps,
                                   args.lr_decay_factor)
        trainer.train(train_loader, model, criterion, optimizer, epoch,
                      _logger, args, amp_autocast, loss_scaler)
        utils.synchronize()

        # bn syn
        if args.distributed:
            if args.local_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size,
                          True)  # true for reduce, false for broadcast

        # logging
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss_val, pred_and_gt = trainer.validate(val_loader, model,
                                                     criterion, _logger, args,
                                                     amp_autocast)

            # distributed synchronize
            pred_and_gt = utils.synchronize_pred_and_gt(
                pred_and_gt, epoch, args)

            # eval
            if args.local_rank == 0:
                precision, recall, f_score, MNA, match_regions_record = utils.evaluate_bi_labels(
                    pred_and_gt, val_id, epoch, args)
            else:
                f_score = -10.0
                MNA = (0, 0, 0)
            # precision, recall, f_score, MNA, match_regions_record = utils.evaluate_bi_labels(
            #     pred_and_gt, val_id, epoch, args)
            utils.synchronize()

            # synchronize
            f_score = utils.synchronize_f_score(f_score, args)
            _logger.info('f_score of processor {}: {:.4f}'.format(
                args.local_rank, f_score))
            MNA = utils.synchronize_list(MNA, args)
            _logger.info('MNA of processor {}: {}'.format(
                args.local_rank, MNA))

            is_equal_score = f_score == best_f_score
            is_best_loss = loss_val < best_loss
            best_loss = min(loss_val, best_loss)
            is_best_score = f_score > best_f_score
            best_f_score = max(best_f_score, f_score)

            # save checkpoint
            if args.local_rank == 0:
                _logger.info(
                    'Test[{}]: loss_val: {:.4f} (best: {:.4f}), f-score: {:.4f} (best: {:.4f})'
                    .format(epoch, loss_val, best_loss, f_score, best_f_score))
                utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                    },
                    is_best_score,
                    args.save_root,
                    args.root_model,
                    filename=val_id)
            utils.synchronize()

            if is_best_score or (is_equal_score and
                                 MNA[1] < total_MNA.get(val_id, [0, 0, 0])[1]):
                val_accum_epochs = 0
                total_MNA.update(
                    {val_id:
                     MNA})  # processor 0 need this record for branch selection
                if args.local_rank == 0:
                    match_regions_record_all.update(
                        match_regions_record
                    )  # only processor 0 need this record
                    out_dir = osp.join(args.save_root, args.root_output,
                                       val_id)
                    os.makedirs(out_dir, exist_ok=True)
                    np.save(osp.join(out_dir, 'match_regions_record_best.npy'),
                            match_regions_record)
                    # all
                    np.save(
                        osp.join(args.save_root, args.root_output,
                                 'cross_validation_MNA_dict.npy'), total_MNA)
                    np.save(
                        osp.join(args.save_root, args.root_output,
                                 'match_regions_record_all.npy'),
                        match_regions_record_all)
                    precision, recall, f_score = utils.calculate_metric_from_dict_MNA(
                        total_MNA)
                    _logger.info(
                        'Test[all] Avg f-score now: {:.4f}'.format(f_score))
                utils.synchronize()
            else:
                val_accum_epochs += 1

            if val_accum_epochs >= args.early_stop:
                _logger.info(
                    "validation ccc did not improve over {} epochs, stop processor {}"
                    .format(args.early_stop, args.local_rank))
                break
    if args.local_rank == 0:
        precision_all, recall_all, f_score_all = utils.calculate_metric_from_dict_MNA(
            total_MNA)
        _logger.critical(
            '[{}][{}]/[{}] f_score: {:.4f}, precision_all: {:.4f}, recall_all: {:.4f}, f_score_all: {:.4f}'
            .format(val_id, vi + 1, len(img_dirs_dict), best_f_score,
                    precision_all, recall_all, f_score_all))

# store results
if args.local_rank == 0:
    np.save(
        osp.join(args.save_root, args.root_output,
                 'cross_validation_MNA_dict.npy'), total_MNA)
    np.save(
        osp.join(args.save_root, args.root_output,
                 'match_regions_record_all.npy'), match_regions_record_all)
    _logger.info('ALL DONE')
exit()
