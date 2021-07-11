import time
from matplotlib.pyplot import winter
import torch
import torch.nn.functional as F
import numpy as np

import utils
import dataset.utils as dataset_utils
import dataset.params as DATASET_PARAMS


def train(dataloader, model, criterion, optimizer, epoch, logger, args,
          amp_autocast, loss_scaler):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    model.train()
    for i, data_batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        temp_data, img_features, annos, labels, _ = data_batch
        batch_size = temp_data.shape[0]

        # # TODO: skip all zero samples
        # if (labels == 0).all() and np.random.rand() <= 0.7:
        #     end = time.time()
        #     # print('skip all zeros batch...')
        #     continue
        # keep_ids = []
        # for bi in range(batch_size):
        #     if not ((labels[bi] == 0).all() and np.random.rand() <= 0.5):
        #         keep_ids.append(bi)
        # # print('skip {} samples...'.format(batch_size - len(keep_ids)))
        # batch_size = len(keep_ids)  # m batch_size
        # if batch_size == 0:
        #     end = time.time()
        #     # print('skip all zeros batch...')
        #     continue
        # keep_ids = np.asarray(keep_ids)
        # temp_data = temp_data[keep_ids]
        # img_features = img_features[keep_ids]
        # annos = annos[keep_ids]
        # labels = labels[keep_ids]

        # label preprocess
        labels[labels > 0] = 1  # 1, 2 -> 1

        temp_data = temp_data.cuda()
        img_features = img_features.cuda()
        # annos = annos.cuda()
        labels = labels.cuda()

        with amp_autocast():
            out = model(temp_data, img_features)
            # flat labels
            out = out.reshape(batch_size * args.length, -1)
            labels = labels.reshape(-1)
            loss = criterion(out, labels)

        # backward + step
        optimizer.zero_grad()
        if loss_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            loss_scaler(loss, optimizer)

        # distirbuted reduce
        utils.reduce_loss(loss, args)
        losses.update(loss.item(), temp_data.size(0))
        batch_time.update(time.time() - end)

        if args.local_rank == 0 and (i % args.print_freq == 0
                                        or i == len(dataloader) - 1):
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                            epoch,
                            i + 1,
                            len(dataloader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            lr=optimizer.param_groups[-1]['lr']))
            logger.info(output)
        torch.cuda.synchronize()
        end = time.time()


def validate(dataloader, model, criterion, logger, args, amp_autocast):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.eval()
    end = time.time()
    # outs = []
    # annos = []
    # labels = []
    # pred_anno_dict = {}  # imgs_dir -> anno values
    # pred_label_dict = {}  # imgs_dir -> labels
    # anno_dict = {}
    # label_dict = {}
    pred_and_gt = {}  # img_p -> [pred, target]

    for i, data_batch in enumerate(dataloader):
        temp_data, img_features, annos, labels, seq_info = data_batch

        # label preprocess
        labels[labels > 0] = 1  # 1, 2 -> 1

        batch_size = labels.shape[0]
        temp_data = temp_data.cuda()
        img_features = img_features.cuda()
        # annos = annos.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            with amp_autocast():
                out = model(temp_data, img_features)
                loss = criterion(out.reshape(batch_size * args.length, -1),
                                labels.reshape(-1)).float()

        if not torch.isnan(loss).any():
            # distirbuted reduce
            utils.reduce_loss(loss, args)

            losses.update(loss.item(), temp_data.size(0))
            batch_time.update(time.time() - end)
            if args.local_rank == 0 and (i % args.print_freq == 0
                                         or i == len(dataloader) - 1):
                output = ('Val: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                              i + 1,
                              len(dataloader),
                              batch_time=batch_time,
                              loss=losses))
                logger.info(output)
        torch.cuda.synchronize()

        # record
        img_dirs, fronts, tails = seq_info
        for batch_idx in range(batch_size):
            img_dir = img_dirs[batch_idx]
            front = fronts[batch_idx].item()
            tail = tails[batch_idx].item()
            # assert batch_size == 1, 'batch size should be 1'

            img_dir_ps = dataset_utils.scan_jpg_from_img_dir(img_dir)
            # if not img_dir in pred_label_dict:
            # pred_anno_dict[img_dir] = np.zeros(len(img_dir_ps))
            # pred_label_dict[img_dir] = np.zeros(len(img_dir_ps))
            # anno_dict = [img_dir] = np.zeros(len(img_dir_ps))
            # label_dict = [img_dir] = np.zeros(len(img_dir_ps))

            pred_label = torch.argmax(out[batch_idx], dim=-1).reshape(-1)
            label = labels[batch_idx].reshape(-1)

            for j in range(front, tail):
                img_p = img_dir_ps[j]
                pred_and_gt[img_p] = [
                    pred_label[j - front].item(), label[j - front].item()
                ]

            # pred_anno_dict[img_dir][front:tail] += pred_annos
            # assert (pred_label_dict[img_dir][front:tail] == 0
            #         ).all(), 'should be no overlap'
            # pred_label_dict[img_dir][front:tail] += pred_labels
            # anno_dict[img_dir][front:tail] += annos
            # label_dict[img_dir][front:tail] += labels
        end = time.time()

    return losses.avg, pred_and_gt
