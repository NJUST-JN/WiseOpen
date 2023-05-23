import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, TransformFixMatch_Tiny_Weak, \
    Tiny_mean, Tiny_std
from tqdm import tqdm
from utils import AverageMeter, ova_loss, save_checkpoint,\
    ova_ent, get_subset, test, test_ood
import math


logger = logging.getLogger(__name__)

def cosin_warmup_hyper(afrom, ato, total, epoch):
    if epoch >= 0 and epoch <= total:
        return (afrom - ato)/2.*(math.cos(math.pi * epoch/total)+1)+ato
    elif epoch < 0:
        return afrom
    else:
        return ato


def warmup_alpha(args, epoch):
    epoch = epoch - args.sup_warm
    args.alpha = cosin_warmup_hyper(max(0, args.alpha_from), max(0, args.alpha_to), args.ova_warm, epoch) 
    args.writer.add_scalar('hyper/2.alpha', args.alpha, epoch)
                    
def warmup_id_ood_th(args, epoch):
    epoch = epoch - args.sup_warm
    if epoch >= 0:
        t = epoch / args.uc1
        args.th_id = min(args.S_hat_id**(args.C_id * args.gama_id ** (-t)), args.S_max_id)
        args.th_ood = min(args.S_hat_ood**(args.C_ood * args.gama_ood ** (-t)), args.S_max_ood)
    args.writer.add_scalar('hyper/3.emth_id', args.th_id, epoch)
    args.writer.add_scalar('hyper/4.emth_ood', args.th_ood, epoch)

def warmup_threshold(args, epoch):
    epoch = epoch - args.start_fix
    if epoch >= 0:
        t = epoch / args.uc2
        args.threshold = min(args.threshold_hat**(args.C_threshold * args.gama_threshold ** (-t)), args.max_threshold)
    args.writer.add_scalar('hyper/5.threshold', args.threshold, epoch)

def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler, best_acc, best_acc_val, best_roc ):
    if args.amp:
        from apex import amp

    best_aupr_in_ood_dic = {}
    best_aupr_out_ood_dic = {}
    best_roc_ood_dic = {}
    aupr_in_valid = 0
    aupr_out_valid = 0
    test_accs = []

    logger.info("global best_acc: {:.4f} | best_acc_val: {:.4f} | best_roc: {:.4f}".format(
        best_acc, best_acc_val, best_roc 
    ))

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.3f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)
    default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"

    model.train()

    df_subst = copy.deepcopy(unlabeled_dataset)
    cf_subset = unlabeled_dataset

    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformOpenMatch
    elif args.dataset == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformOpenMatch
    elif 'tiny' in args.dataset:
        mean = Tiny_mean
        std = Tiny_std
        func_trans = TransformFixMatch_Tiny_Weak

    df_subst.transform = func_trans(mean=mean, std=std)
    labeled_dataset = copy.deepcopy(labeled_trainloader.dataset)
    labeled_dataset.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_o = AverageMeter()
        losses_oem = AverageMeter()
        losses_socr = AverageMeter()
        losses_fix = AverageMeter()
        mean_lableled_score = AverageMeter()

        
        cf_trainloader = None
        df_trainloader_all = None

        output_args["epoch"] = epoch       
        warmup_id_ood_th(args, epoch)
        warmup_threshold(args, epoch)
        warmup_alpha(args, epoch)
        id_selected, all_selected = get_subset(args, unlabeled_dataset, ema_model.ema)
        
        df_subst.init_index()
        cf_subset.set_index(id_selected)
        df_subst.set_index(all_selected)

        
        if epoch >= args.start_fix and len(cf_subset) >= args.batch_size * args.mu * args.least_set:
            cf_trainloader = DataLoader(cf_subset,
                                        sampler = train_sampler(cf_subset),
                                        batch_size = args.batch_size * args.mu,
                                        num_workers = args.num_workers,
                                        drop_last = True
                                        )
            cf_iter = iter(cf_trainloader)
        if epoch >= args.sup_warm and len(df_subst) >= args.batch_size * args.mu * args.least_set:
            df_trainloader_all = DataLoader(df_subst,
                                            sampler=train_sampler(df_subst),
                                            batch_size=args.batch_size * args.mu,
                                            num_workers=args.num_workers,
                                            drop_last=True)
            df_iter = iter(df_trainloader_all)
        
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0], ncols=150)
        model.train()
        end = time.time()

        for batch_idx in range(args.eval_step):

            try:
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.__next__()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.__next__()
            if cf_trainloader is not None:
                try:
                    (inputs_cf_w, inputs_cf_s, _), _ = cf_iter.__next__()
                except:
                    if args.world_size > 1:
                        unlabeled_epoch += 1
                        cf_trainloader.sampler.set_epoch(unlabeled_epoch)
                    cf_iter = iter(cf_trainloader)
                    (inputs_cf_w, inputs_cf_s, _), _ = cf_iter.__next__()
            if df_trainloader_all is not None:
                try:
                    (inputs_df_w, inputs_df_s, _), _ = df_iter.__next__()
                except:
                    df_iter = iter(df_trainloader_all)
                    (inputs_df_w, inputs_df_s, _), _ = df_iter.__next__()
            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]
            inputs = torch.cat([inputs_x, inputs_x_s], 0)
            if df_trainloader_all is not None:
                inputs_all = torch.cat([inputs_df_w, inputs_df_s], 0)
                inputs = torch.cat([inputs, inputs_all], 0)
            inputs = inputs.to(args.device)
            targets_x = targets_x.to(args.device)


            logits, logits_open = model(inputs)

            if df_trainloader_all is not None:
                logits_open_u1, logits_open_u2 = logits_open[2*b_size:].chunk(2)

            l_logits_open = logits_open[:2*b_size]
            l_logits_open = l_logits_open.view(l_logits_open.size(0), 2, -1)
            l_logits_open = F.softmax(l_logits_open, 1)
            tmp_range = torch.arange(0, l_logits_open.size(0)).long().cuda(args.device)
            id_score = l_logits_open[tmp_range, 1, targets_x.repeat(2)]
            mean_score = id_score.mean().item()
            mean_lableled_score.update(mean_score)

            Lx = F.cross_entropy(logits[:2*b_size], targets_x.repeat(2), reduction='mean')
            Lo = ova_loss(logits_open[:2*b_size], targets_x.repeat(2), alpha=args.alpha)

            if df_trainloader_all is not None:
                L_oem1 = ova_ent(logits_open_u1)
                L_oem2 = ova_ent(logits_open_u2)               
                L_oem = L_oem1 / 2. + L_oem2 / 2.
                logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
                logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
                logits_open_u1 = F.softmax(logits_open_u1, 1)
                logits_open_u2 = F.softmax(logits_open_u2, 1)
                L_socr = torch.mean(torch.sum(torch.sum(torch.abs(logits_open_u1 - logits_open_u2)**2, 1), 1))
            else:
                L_oem = torch.zeros(1).to(args.device).mean()
                L_socr = torch.zeros(1).to(args.device).mean()
          

            if cf_trainloader is not None:
                inputs_ws = torch.cat([inputs_cf_w, inputs_cf_s], 0).to(args.device)
                logits, _ = model(inputs_ws)
                logits_u_w, logits_u_s = logits.chunk(2)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                L_fix = (F.cross_entropy(logits_u_s,
                                         targets_u,
                                         reduction='none') * mask).mean()               
            else:
                L_fix = torch.zeros(1).to(args.device).mean()
           
            loss = Lx + Lo + args.lambda_oem * L_oem  \
                   + args.lambda_socr * L_socr + L_fix
            

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(Lo.item())
            losses_oem.update(L_oem.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_oem"] = losses_oem.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]

            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if epoch == args.start_fix - 1:
            args.threshold_hat = max(min(np.exp(-losses_x.avg), args.max_threshold), args.min_threshold)
        if epoch == args.sup_warm - 1:
            args.S_hat_id = max(min(mean_lableled_score.avg, args.S_max_id), args.S_min_id)
            args.S_hat_ood = max(min(mean_lableled_score.avg, args.S_max_ood), args.S_min_ood)

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            val_acc = test(args, val_loader, test_model, epoch, val=True)
            test_loss, test_acc_close, test_overall, \
            test_unk, test_roc, test_roc_softm, test_id, test_ood_score,\
            test_aupr_in, test_aupr_out \
                = test(args, test_loader, test_model, epoch)
            aupr_in_ood_dic = {}
            aupr_out_ood_dic = {}
            roc_ood_dic = {}
            is_best = val_acc > best_acc_val
            if is_best and args.ood_test:
                for ood in ood_loaders.keys():
                    roc_ood, aupr_in_ood, aupr_out_ood = test_ood(args, test_id, ood_loaders[ood], test_model)
                    aupr_in_ood_dic[ood] = aupr_in_ood
                    aupr_out_ood_dic[ood] = aupr_out_ood
                    roc_ood_dic[ood] = roc_ood
                    logger.info("ROC vs {ood}: {roc:.2f}".format(ood=ood, roc=roc_ood*100))
                    logger.info("AUPR(in) vs {ood}: {aupr_in:.2f}".format(ood=ood, aupr_in=aupr_in_ood*100))
                    logger.info("AUPR(out) vs {ood}: {aupr_out:.2f}".format(ood=ood, aupr_out=aupr_out_ood*100))


            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_o', losses_o.avg, epoch)
            args.writer.add_scalar('train/4.train_loss_oem', losses_oem.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_socr', losses_socr.avg, epoch)
            args.writer.add_scalar('train/6.train_loss_fix', losses_fix.avg, epoch)


            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/3.roc', test_roc, epoch)

            args.writer.add_scalar('val/1.acc', val_acc, epoch)

            args.writer.add_scalar('hyper/1.lr', [group["lr"] for group in optimizer.param_groups][0], epoch)
            
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                best_roc = test_roc
                best_acc = test_acc_close
                aupr_in_valid = test_aupr_in
                aupr_out_valid = test_aupr_out
                best_aupr_out_ood_dic = aupr_out_ood_dic
                best_aupr_in_ood_dic = aupr_in_ood_dic
                best_roc_ood_dic = roc_ood_dic
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'roc': test_roc,
                'best_acc': best_acc,
                'best_acc_val':best_acc_val,
                'best_roc':best_roc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'threshold_hat': args.threshold_hat,
                'S_hat_id': args.S_hat_id,
                'S_hat_ood': args.S_hat_ood,
            }, is_best, args.out)
            
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(best_acc))
            logger.info('Valid roc: {:.2f}'.format(best_roc*100))
            logger.info('Valid aupr_in: {:.2f}'.format(aupr_in_valid*100))
            logger.info('Valid aupr_out: {:.2f}'.format(aupr_out_valid*100))
            logger.info('Mean top-1 acc: {:.2f}'.format(
                np.mean(test_accs[-20:])))
            for ood in best_roc_ood_dic.keys():
                logger.info("Valid ROC vs {ood}: {roc:.2f}".format(ood=ood, roc=best_roc_ood_dic[ood]*100))
                logger.info("Valid AUPR(in) vs {ood}: {aupr_in:.2f}".format(ood=ood, aupr_in=best_aupr_in_ood_dic[ood]*100))
                logger.info("Valid AUPR(out) vs {ood}: {aupr_out:.2f}".format(ood=ood, aupr_out=best_aupr_out_ood_dic[ood]*100))
    if args.local_rank in [-1, 0]:
        args.writer.close()
