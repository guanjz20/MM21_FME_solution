import os
import sys
import cv2
from timm.utils import reduce_tensor
import torch
import shutil
import numpy as np
import os.path as osp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.modules import loss
from datetime import datetime

import paths
import dataset.utils as dataset_utils

sys.setrecursionlimit(10000)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Focal_Loss(torch.nn.Module):
    def __init__(self, alpha=[], gamma=2, num_class=2, epsilon=1e-7):
        super().__init__()
        if alpha == []:
            self.alpha = torch.ones(num_class)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, pred, target):
        assert len(pred.shape) == 2, 'pred shape should be N, num_class'
        assert len(target.shape) == 1, 'target shape should be N'
        pred = torch.softmax(pred, dim=-1)
        target_pred = -F.nll_loss(pred, target, reduction='none')
        loss = -torch.log(target_pred + self.epsilon)
        class_alpha = torch.tensor([self.alpha[c.item()] for c in target],
                                   dtype=torch.float32).to(loss.device)
        weights = ((1 - target_pred)**self.gamma) * class_alpha
        loss = (weights * loss).mean()
        return loss


class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = Focal_Loss(num_class=3)

    def forward(self, out, anno_y, label_y):
        anno_x = out[..., 0]
        label_x = out[..., 1:]

        if len(anno_x.shape) == 2:
            anno_x = anno_x.reshape(-1)
            anno_y = anno_y.reshape(-1)
        # loss_ccc = -ccc(anno_x, anno_y)[0]
        # loss_mse = F.mse_loss(anno_x, anno_y)
        loss_l1 = F.l1_loss(anno_x, anno_y)

        # logits = F.log_softmax(label_x, dim=-1)
        # loss_ce = F.nll_loss(logits, label_y)
        if len(label_x.shape) == 3:
            label_x = label_x.reshape(-1, label_x.shape[-1])
            label_y = label_y.reshape(-1)

        # loss_ce = F.cross_entropy(label_x, label_y, reduce='mean')
        # loss_focal = self.focal_loss(label_x, label_y)

        # loss = loss_ccc + loss_ce
        # loss = loss_mse + loss_ce
        # loss = loss_mse + loss_focal
        # loss = loss_mse * 100
        # loss = loss_l1 * 100 + loss_focal
        loss = loss_l1 * 1000
        return loss


def ccc(y_pred, y_true, epsilon=1e-7):
    assert len(y_pred.shape) == 1
    true_mean = y_true.mean()
    pred_mean = y_pred.mean()
    v_true = y_true - true_mean
    v_pred = y_pred - pred_mean

    rho = (v_pred * v_true).sum() / (torch.sqrt(
        (v_pred**2).sum()) * torch.sqrt((v_true**2).sum()) + epsilon)
    std_predictions = torch.std(y_pred)
    std_gt = torch.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
        (std_predictions**2 + std_gt**2 +
         (pred_mean - true_mean)**2) + epsilon)
    return ccc, rho


def img_dirs_filter(img_dirs, dataset):
    '''
    some clips are not labeled...
    '''
    _img_dirs = []
    if dataset == 'SAMM':
        anno_dict = np.load(osp.join(paths.SAMM_LABEL_DIR, 'anno_dict.npy'),
                            allow_pickle=True).item()
    elif dataset == 'CASME_2':
        anno_dict = np.load(osp.join(paths.CASME_2_LABEL_DIR, 'anno_dict.npy'),
                            allow_pickle=True).item()
    else:
        raise NotImplementedError
    for img_dir in img_dirs:
        if img_dir in anno_dict:
            _img_dirs.append(img_dir)
        else:
            print('clip: {} is not labeled or labeled incorrectly.'.format(
                img_dir))
    return _img_dirs


def get_img_dirs(dataset):
    if dataset == 'SAMM':
        img_dirs = [
            osp.join(paths.SAMM_VIDEO_DIR, name)
            for name in os.listdir(paths.SAMM_VIDEO_DIR)
        ]
    elif dataset == 'CASME_2':
        _img_dirs = [[
            osp.join(paths.CASME_2_VIDEO_DIR, name1, name2)
            for name2 in os.listdir(osp.join(paths.CASME_2_VIDEO_DIR, name1))
        ] for name1 in os.listdir(paths.CASME_2_VIDEO_DIR)]
        img_dirs = []
        for dirs in _img_dirs:
            img_dirs.extend(dirs)
    else:
        raise NotImplementedError
    img_dirs = img_dirs_filter(img_dirs, dataset)
    return img_dirs


def leave_one_out(img_dirs, dataset):
    img_dirs_dict = {}
    img_dirs = sorted(img_dirs)
    if dataset == 'SAMM':
        keys = []
        for img_dir in img_dirs:
            keys.append(osp.basename(img_dir).split('_')[0])  # 006, 007...
        keys = sorted(list(set(keys)))
        for key in keys:
            train_set = []
            val_set = []
            for img_dir in img_dirs:
                if key in img_dir:
                    val_set.append(img_dir)
                else:
                    train_set.append(img_dir)
            img_dirs_dict[key] = [train_set, val_set]
    elif dataset == 'CASME_2':
        keys = []
        for img_dir in img_dirs:
            keys.append(img_dir.split('/')[-2])  # s15, s16...
        keys = sorted(list(set(keys)))
        for key in keys:
            train_set = []
            val_set = []
            for img_dir in img_dirs:
                if img_dir.split('/')[-2] == key:
                    val_set.append(img_dir)
                else:
                    train_set.append(img_dir)
            img_dirs_dict[key] = [train_set, val_set]
    else:
        raise NotImplementedError
    return img_dirs_dict


def adjust_learning_rate(optimizer, epoch, lr_strat, wd, lr_steps, factor=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every N epochs"""
    decay = factor**(sum(epoch >= np.asarray(lr_steps)))
    lr = lr_strat * decay
    decay = wd
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = decay


def save_checkpoint(state, is_best, save_root, root_model, filename='val'):
    torch.save(
        state,
        '%s/%s/%s_checkpoint.pth.tar' % (save_root, root_model, filename))
    if is_best:
        shutil.copyfile(
            '%s/%s/%s_checkpoint.pth.tar' % (save_root, root_model, filename),
            '%s/%s/%s_best_loss.pth.tar' % (save_root, root_model, filename))
        # print("checkpoint saved to",
        #       '%s/%s/%s_best_loss.pth.tar' % (save_root, root_model, filename))


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model, args.root_output, args.root_runs
    ]
    folders_util = [
        "%s/" % (args.save_root) + folder for folder in folders_util
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)


def evaluate(pred_anno_dict,
             pred_label_dict,
             dataset,
             threshold=0.9,
             val_id='all',
             epoch=-1,
             args=None):
    if dataset == 'SAMM':
        pred_gt = np.load(osp.join(paths.SAMM_ROOT, 'pred_gt.npy'),
                          allow_pickle=True).item()
        anno_dict = np.load(osp.join(paths.SAMM_ROOT, 'anno_dict.npy'),
                            allow_pickle=True).item()
        fps = 200
    elif dataset == 'CASME_2':
        pred_gt = np.load(osp.join(paths.CASME_2_LABEL_DIR, 'pred_gt.npy'),
                          allow_pickle=True).item()
        anno_dict = np.load(osp.join(paths.CASME_2_LABEL_DIR, 'anno_dict.npy'),
                            allow_pickle=True).item()
        fps = 30
    else:
        raise NotImplementedError

    result_dict = {}
    for img_dir, pred_annos in pred_anno_dict.items():
        pred_labels = pred_label_dict[img_dir]
        gt_list = pred_gt[img_dir]
        pred_list = []

        # scan all possible peak point
        for peak_idx in range(0, len(pred_annos), fps):
            is_peak = True
            front = peak_idx
            tail = peak_idx
            # label_sum = pred_labels[peak_idx]
            cumsum = pred_annos[peak_idx]
            while is_peak and cumsum < threshold and (
                    front > 0 or tail < len(pred_annos) - 1):
                if front - 1 >= 0:
                    front -= 1
                    cumsum += pred_annos[front]
                    # label_sum += pred_labels[front]
                if tail + 1 < len(pred_annos):
                    tail += 1
                    cumsum += pred_annos[tail]
                    # label_sum += pred_labels[tail]
                is_peak = pred_annos[peak_idx] >= pred_annos[
                    front] and pred_annos[peak_idx] >= pred_annos[tail]
            if is_peak and cumsum >= threshold:
                # TODO: label func
                pred_list.append([front, tail, -1])

        M = len(gt_list)
        N = len(pred_list)
        A = 0
        for [onset, offset, label_gt] in gt_list:
            for [
                    front, tail, _
            ] in pred_list:  # TODO: if one pred could match more than one gt?
                if front < onset:
                    b1 = [front, tail]
                    b2 = [onset, offset]
                else:
                    b2 = [front, tail]
                    b1 = [onset, offset]

                # 1
                if b1[1] >= b2[0] and b2[1] >= b1[1]:
                    overlap = b1[1] - b2[0] + 1
                    union = b2[1] - b1[0] + 1
                elif b1[1] >= b2[1]:
                    overlap = b2[1] - b2[0] + 1
                    union = b1[1] - b1[0] + 1
                else:
                    # no overlap
                    overlap = 0
                    union = 1
                if overlap / union >= 0.5:
                    A += 1
                    break
        result_dict[img_dir] = [M, N, A]

    ret_info = []
    M = 0
    N = 0
    A = 0
    for key, (m, n, a) in result_dict.items():
        # p = a / n
        # r = a / m
        # f = 2 * r * p / (p + r)
        # ret_info.append('[{}] P: {.4f}, R: {:.4f}, F1: {:.4f}'.format(
        #     key, p, r, f))
        M += m
        N += n
        A += a

    if M == 0 or N == 0 or A == 0:
        precision = -1.0
        recall = -1.0
        f_score = -1.0
    else:
        precision = A / N
        recall = A / M
        f_score = 2 * recall * precision / (recall + precision)
    ret_info.append('[over all] P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(
        precision, recall, f_score))

    # save fig
    column = 3
    fig = plt.figure(figsize=(10,
                              ((len(pred_anno_dict) - 1) // column + 1) * 2))
    for i, (img_dir, pred_annos) in enumerate(pred_anno_dict.items()):
        fig.add_subplot((len(pred_anno_dict) - 1) // column + 1, column, i + 1)
        plt.plot(pred_annos, 'b-', alpha=0.5)
        plt.plot(anno_dict[img_dir], 'r-', alpha=0.5)
    fig.tight_layout()
    plt.savefig(
        osp.join(args.save_root, args.root_output,
                 '{}_anno_{}.pdf'.format(val_id, epoch)))
    plt.close('all')

    return ret_info, f_score, (M, N, A)


def evaluate_bi_labels(pred_and_gt, val_id, epoch, args):
    keys = sorted(list(pred_and_gt.keys()))
    imgs_dirs = sorted(list(set([osp.dirname(img_p) for img_p in keys])))
    result_dict = {}
    for imgs_dir in imgs_dirs:
        result_dict[imgs_dir] = []
        img_ps = dataset_utils.scan_jpg_from_img_dir(imgs_dir)
        for img_p in img_ps:
            result_dict[imgs_dir].append(pred_and_gt.get(
                img_p, [0, 0]))  # [pred, target]
        result_dict[imgs_dir] = np.asarray(result_dict[imgs_dir])

    precision, recall, f_score, MNA, result_dict, match_regions_record = evaluate_pred_and_gt(
        result_dict, args)

    # visulization
    if args.local_rank == 0:
        column = 3
        fig = plt.figure(figsize=(10,
                                  ((len(imgs_dirs) - 1) // column + 1) * 2))
        for i, imgs_dir in enumerate(imgs_dirs):
            fig.add_subplot((len(imgs_dirs) - 1) // column + 1, column, i + 1)
            data = result_dict[imgs_dir]
            pred = data[:, 0]
            target = data[:, 1]
            plt.plot(pred, 'b-', alpha=0.5)
            plt.plot(target, 'r-', alpha=0.5)  # gt
            plt.title(osp.basename(imgs_dir))
        fig.tight_layout()
        out_dir = osp.join(args.save_root, args.root_output, val_id)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(osp.join(out_dir, 'bi_label_{}.pdf'.format(epoch)))
        plt.close('all')

    return precision, recall, f_score, MNA, match_regions_record


def extend_front(front, pred, patience):
    assert pred[front] > 0
    d = patience
    while d > 0:
        if front + d < len(pred) and pred[front + d] > 0:
            return extend_front(front + d, pred, patience)
        d -= 1
    return front


def evaluate_pred_and_gt(result_dict, args):
    if args.dataset == 'SAMM':
        # patience = 25
        pred_gt = np.load(osp.join(paths.SAMM_ROOT, 'pred_gt.npy'),
                          allow_pickle=True).item()
    elif args.dataset == 'CASME_2':
        pred_gt = np.load(osp.join(paths.CASME_2_LABEL_DIR, 'pred_gt.npy'),
                          allow_pickle=True).item()
        # patience = 10
    else:
        raise NotImplementedError

    M = 0
    N = 0
    A = 0
    match_regions_record = {}
    for imgs_dir, data in result_dict.items():
        pred = data[:, 0]
        target = data[:, 1]

        found_regions = []
        match_regions = [
        ]  # gt_onset, gt_offset, pred_onset, pred_offset, TP/FP
        front = 0
        while front < len(pred):
            tail = front
            if pred[front] > 0:
                tail = extend_front(front, pred, args.patience)
                if front < tail:  # find one region
                    found_regions.append([front, tail])
            front = tail + args.patience

        # modify result_dict
        pred = np.zeros_like(pred)
        for front, tail in found_regions:
            pred[front:tail] = 1
        data[:, 0] = pred
        result_dict[imgs_dir] = data

        # eval precision, recall, f_score
        gt_list = pred_gt[imgs_dir]
        m = len(gt_list)
        n = len(found_regions)
        a = 0
        # TODO: determine whether one predicted region is macro or micro-expression
        gt_regions_mark = np.zeros(m)
        found_regions_mark = np.zeros(n)
        for mg, [onset, offset, label_gt] in enumerate(gt_list):
            # label_gt: 1->macro, 2->micro
            for mf, [front, tail] in enumerate(
                    found_regions
            ):  # TODO: if one found region can match more than one gt region
                if front < onset:
                    b1 = [front, tail]
                    b2 = [onset, offset]
                else:
                    b1 = [onset, offset]
                    b2 = [front, tail]

                # 1
                if b1[1] >= b2[0] and b2[1] >= b1[1]:
                    overlap = b1[1] - b2[0] + 1
                    union = b2[1] - b1[0] + 1
                elif b1[1] >= b2[1]:
                    overlap = b2[1] - b2[0] + 1
                    union = b1[1] - b1[0] + 1
                else:  # no overlap
                    overlap = 0
                    union = 1
                if overlap / union >= 0.5:
                    a += 1
                    found_regions_mark[mf] = 1
                    gt_regions_mark[mg] = 1
                    match_regions.append([onset, offset, front, tail, 'TP'])
                    break
        for mg in range(m):
            if gt_regions_mark[mg] == 0:
                onset, offset, _ = gt_list[mg]
                match_regions.append([onset, offset, '-', '-', 'FN'])
        for mf in range(n):
            if found_regions_mark[mf] == 0:
                front, tail = found_regions[mf]
                match_regions.append(['-', '-', front, tail, 'FP'])
        match_regions_record[imgs_dir] = match_regions
        M += m
        N += n
        A += a
        # NOTE: if one found region can match more than one gt region, TP+FP may be greater than n

    # result of the participant
    if A == 0 or N == 0:
        precision = -1.0
        recall = -1.0
        f_score = -1.0
    else:
        precision = A / N
        recall = A / M
        f_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f_score, (M, N,
                                        A), result_dict, match_regions_record


def calculate_metric_from_dict_MNA(MNA_all):
    M = 0
    N = 0
    A = 0
    for k, mna in MNA_all.items():
        m, n, a = mna
        M += m
        N += n
        A += a
    try:
        precision = A / N
        recall = A / M
        f_score = 2 * precision * recall / (precision + recall)
    except:
        precision = -1.0
        recall = -1.0
        f_score = -1.0
    return precision, recall, f_score


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def reduce_loss(loss, args):
    if args.distributed:
        loss = reduce_tensor(loss.data, float(args.world_size))
    return loss


def synchronize_pred_and_gt(pred_and_gt, epoch, args, remove=True):
    if args.distributed:
        out_dir = osp.join(args.save_root, args.root_runs,
                           'temp_{}'.format(epoch))
        if args.local_rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        synchronize()  # make dir done
        np.save(
            osp.join(out_dir,
                     'temp_pred_and_gt_{}.npy'.format(args.local_rank)),
            pred_and_gt)
        synchronize()  # save done
        if args.local_rank == 0:
            pred_and_gt = {}
            for name in os.listdir(out_dir):
                data = np.load(osp.join(out_dir, name),
                               allow_pickle=True).item()
                pred_and_gt.update(data)
            np.save(osp.join(out_dir, 'temp_pred_and_gt_merge.npy'),
                    pred_and_gt)
            synchronize()  # merge done
        else:
            synchronize()  # start read
            pred_and_gt = np.load(osp.join(out_dir,
                                           'temp_pred_and_gt_merge.npy'),
                                  allow_pickle=True).item()
        synchronize()  # read done
        if remove and args.local_rank == 0:
            shutil.rmtree(out_dir)
    return pred_and_gt


def synchronize_f_score(f_score, args):
    assert isinstance(f_score, float)
    if args.distributed:
        f_score = torch.tensor(f_score).cuda()
        assert f_score.dtype == torch.float32
        synchronize()  # wait tensor allocation
        dist.broadcast(f_score, src=0)
        f_score = f_score.item()
    return f_score


def synchronize_list(list_obj, args):
    assert isinstance(list_obj, (list, tuple))
    if args.distributed:
        list_obj = torch.tensor(list_obj, dtype=torch.int32).cuda()
        synchronize()  # wait tensor allocation
        dist.broadcast(list_obj, src=0)
        list_obj = list_obj.cpu().numpy().tolist()
    return list_obj


def delete_records(total_MNA, match_regions_record_all, val_id):
    # keys1 = list(total_MNA.keys())
    keys2 = list(match_regions_record_all.keys())
    rm_key = val_id

    del total_MNA[rm_key]
    for k in keys2:
        if k.split('/')[-2] == rm_key or osp.basename(k).split(
                '_')[0] == rm_key:
            del match_regions_record_all[k]
    return total_MNA, match_regions_record_all