'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import math

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter',
           'accuracy_open', 'ova_loss', 'compute_roc',
           'compute_roc_aupr', 'misc_id_ood', 'ova_ent',
           'test_ood', 'test', 'get_pseudo_ova_mask',
           'get_subset']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_open(pred, target, topk=(1,), num_classes=5):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)

def compute_roc_aupr(unk_all, label_all, num_known):
    '''
    return roc, aupr_out, aupr_in
    '''
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all), \
           average_precision_score(Y_test, unk_all), \
           average_precision_score(1 - Y_test, -1.0 * unk_all)


def misc_id_ood(score_id, score_ood):
    '''
    return roc, aupr_out, aupr_in
    '''
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0]+score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all),\
           average_precision_score(Y_test, id_all), \
           average_precision_score(1 - Y_test, -1.0 * id_all)

def ova_loss(logits_open, label, alpha=0.5):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = (1-alpha) * open_loss_neg + alpha * open_loss
    return Lo*2.0



def get_pseudo_ova_mask(args, logits_open, logits):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    logits = F.softmax(logits, 1)
    tmp_range = torch.arange(0, logits_open.size(0)).long().to(args.device)
    pred_close = logits.data.max(1)[1]
    max_id_score = logits_open[tmp_range, 1, pred_close]
    min_ood_score = logits_open[tmp_range, 0, :].min(1)[0]
    mask_id = max_id_score > args.th_id
    mask_ood = min_ood_score > args.th_ood
    mask = mask_id | mask_ood
    return mask, mask_id, mask_ood


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
    return Le


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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

def get_subset(args, dataset, model):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=150)
    model.eval()

    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, outputs_open = model(inputs)
            mask, mask_id, mask_ood = get_pseudo_ova_mask(args, outputs_open, outputs)
            if batch_idx == 0:
                all_conf = mask
                id_conf = mask_id 
            else:
                all_conf = torch.cat([all_conf, mask], 0)
                id_conf = torch.cat([id_conf, mask_id], 0)
        if not args.no_progress:
            test_loader.close()
    
    id_selected = np.where(id_conf.data.cpu().numpy() != 0)[0]
    all_selected = np.where(all_conf.data.cpu().numpy() != 0)[0]

    model.train()
    return id_selected, all_selected


def test(args, test_loader, model, epoch, val=False, feature=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=150)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, outputs_open = model(inputs)
            if feature:
                feat = model(inputs, feat_only = True)
                if batch_idx == 0:
                    feat_all = feat
                else:
                    feat_all = torch.cat([feat_all, feat], 0)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().to(args.device)
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_score = outputs.max(1)[0]
            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1))#[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, min(5, int(outputs.size(1)))))
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])

            ind_unk = unk_score > 0.5
            pred_close[ind_unk] = int(outputs.size(1))
            acc_all, unk_acc, size_unk = accuracy_open(pred_close,
                                                       targets,
                                                       num_classes=int(outputs.size(1)))
            acc.update(acc_all.item(), inputs.shape[0])
            unk.update(unk_acc, size_unk)

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = targets
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} "
                                            "acc: {acc:.3f}. "
                                            "unk: {unk:.3f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    acc=acc.avg,
                    unk=unk.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    unk_all = unk_all.data.cpu().numpy()
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    if not val:
        roc, aupr_out, aupr_in= compute_roc_aupr(unk_all, label_all,
                          num_known=int(outputs.size(1)))
        roc_soft = compute_roc(-known_all, label_all,
                               num_known=int(outputs.size(1)))
        ind_known = np.where(label_all < int(outputs.size(1)))[0]
        id_score = unk_all[ind_known]
        ind_unknown = np.where(label_all >= int(outputs.size(1)))[0]
        ood_score = unk_all[ind_unknown]
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        logger.info("Overall acc: {:.3f}".format(acc.avg))
        logger.info("Unk acc: {:.3f}".format(unk.avg))
        logger.info("ROC: {:.3f}".format(roc))
        logger.info("ROC Softmax: {:.3f}".format(roc_soft))
        logger.info("AUPR(in): {:.3f}".format(aupr_in))
        logger.info("AUPR(out): {:.3f}".format(aupr_out))
        return losses.avg, top1.avg, acc.avg, \
               unk.avg, roc, roc_soft, id_score, ood_score,\
               aupr_in, aupr_out
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg


def test_ood(args, test_id, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0], ncols=150)
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda(args.device)
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
        if not args.no_progress:
            test_loader.close()

    unk_all = unk_all.data.cpu().numpy()
    roc, aupr_out, aupr_in = misc_id_ood(test_id, unk_all) 

    return roc, aupr_in, aupr_out
