#!/usr/bin/env python
# import sys
# sys.path.append("/data1/wuyao/code/BFtD-journal/")

import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from xmuda.common.solver.build import build_optimizer, build_optimizer_small_lr, build_scheduler
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.bftd.build_uni import build_model_2d, build_model_3d, build_model_fusion
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate_fusion
from xmuda.models.losses import entropy_loss
from xmuda.models.fda import FDA_source_to_target
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='xmuda.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

# input train_metric_2d and train_metric_2d (objects of class SegIoU defined in metric.py)
def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    model_2d_ema, _ = build_model_2d(cfg)
    for param in model_2d_ema.parameters():
        param.detach_()
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    logger.info('#Parameters 2D: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    model_3d_ema, _ = build_model_3d(cfg)
    for param in model_3d_ema.parameters():
        param.detach_()
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    logger.info('#Parameters 3D: {:.2e}'.format(num_params))

    # build fusion model
    model_fusion, train_metric_fus = build_model_fusion(cfg)
    model_fusion_ema, _ = build_model_fusion(cfg)
    for param in model_fusion_ema.parameters():
        param.detach_()
    logger.info('Build fusion model:\n{}'.format(str(model_fusion)))
    num_params = sum(param.numel() for param in model_fusion.parameters())
    logger.info('#Parameters Fusion: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    model_fusion = model_fusion.cuda()
    model_2d_ema = model_2d_ema.cuda()
    model_3d_ema = model_3d_ema.cuda()
    model_fusion_ema = model_fusion_ema.cuda()

    # build optimizer
    optimizer_2d = build_optimizer_small_lr(cfg, model_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)
    optimizer_fusion = build_optimizer(cfg, model_fusion)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)
    scheduler_fusion = build_scheduler(cfg, optimizer_fusion)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_fusion = CheckpointerV2(model_fusion,
                                     optimizer=optimizer_fusion,
                                     scheduler=scheduler_fusion,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_fusion',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_fusion = checkpointer_fusion.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None

    # what is metric? metric.segIou
    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d, train_metric_fus])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        model_fusion.train()

        model_2d_ema.train()
        model_3d_ema.train()
        model_fusion_ema.train()

        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        model_fusion.eval()

        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    for iteration in range(start_iteration, max_iteration):
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            # target
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        optimizer_fusion.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Generate data_batch_st
        # ---------------------------------------------------------------------------- #
        img_s = data_batch_src['img']
        img_t = data_batch_trg['img']

        # FDA
        r = random.randint(0, 1)
        if r == 0:  # mixup
            lambda_mix = 0.95
            img_st = img_s * lambda_mix + img_t * (1 - lambda_mix)
        elif r == 1:  # FDA
            img_st = FDA_source_to_target(img_s, img_t)
            img_st = img_st.cuda()

        pc_st = []
        last_number_s = 0
        last_number_t = 0
        for i in range(img_s.shape[0]):
            n_src = data_batch_src['img_indices'][i].shape[0]  # num of points in i-th src
            current_number_s = last_number_s + n_src
            xyz_s = data_batch_src['x'][0][last_number_s:current_number_s, :3]  # coords in i-th src
            x_s = torch.max(xyz_s[:, 0]) - torch.min(xyz_s[:, 0])
            y_s = torch.max(xyz_s[:, 1]) - torch.min(xyz_s[:, 1])
            z_s = torch.max(xyz_s[:, 2]) - torch.min(xyz_s[:, 2])
            density_s = (x_s * y_s * z_s) / n_src

            n_trg = data_batch_trg['img_indices'][i].shape[0]  # num of points in i-th trg
            current_number_t = last_number_t + n_trg
            xyz_t = data_batch_trg['x'][0][last_number_t:current_number_t, :3]  # coords in i-th trg
            x_t = torch.max(xyz_t[:, 0]) - torch.min(xyz_t[:, 0])
            y_t = torch.max(xyz_t[:, 1]) - torch.min(xyz_t[:, 1])
            z_t = torch.max(xyz_t[:, 2]) - torch.min(xyz_t[:, 2])
            density_t = (x_t * y_t * z_t) / n_trg

            alfa = (density_s.item() / density_t.item())**(-(1/3))
            xyz_st = (xyz_s * alfa)
            xyz_st = torch.LongTensor(xyz_st.numpy())

            index_s = data_batch_src['x'][0][last_number_s:current_number_s, 3].view(-1,1)
            pc_st.append(torch.cat([xyz_st, index_s], dim=1))

            last_number_s += n_src
            last_number_t += n_trg

        pc_st = torch.cat(pc_st, 0)
        pc_st = [pc_st, data_batch_src['x'][1]]
        data_batch_st = {
            'x': pc_st,
            'seg_label':  data_batch_src['seg_label'],
            'img': img_st,
            'img_indices': data_batch_src['img_indices']
        }

        # ---------------------------------------------------------------------------- #
        # Inference teacher with data_batch_st
        # ---------------------------------------------------------------------------- #
        feats_2d_st = model_2d_ema(data_batch_st)
        feats_3d_st, preds_3d_st = model_3d_ema(data_batch_st)

        feats_fus_st = model_fusion_ema(feats_2d_st.clone().detach(), feats_3d_st.clone().detach(),
                                        data_batch_st['img_indices'])

        preds_2d_st = model_2d_ema(data_batch_st, feats_2d_st, feats_fus_st)

        probs_2d_st = F.softmax(preds_2d_st['seg_logit'], dim=1)
        probs_3d_st = F.softmax(preds_3d_st['seg_logit'], dim=1)
        avg_probs_st = F.softmax(probs_2d_st + probs_3d_st, dim=1)

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #
        feats_2d_s = model_2d(data_batch_src)
        feats_3d_s, preds_3d_s = model_3d(data_batch_src)

        feats_fus_s = model_fusion(feats_2d_s.clone().detach(), feats_3d_s.clone().detach(),
                                   data_batch_src['img_indices'])

        preds_2d_s = model_2d(data_batch_src, feats_2d_s, feats_fus_s)

        # segmentation loss: cross entropy
        seg_loss_src_2d = F.cross_entropy(preds_2d_s['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src_3d = 0.1 * F.cross_entropy(preds_3d_s['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(seg_loss_src_2d=seg_loss_src_2d, seg_loss_src_3d=seg_loss_src_3d)
        loss_2d_s = seg_loss_src_2d
        loss_3d_s = seg_loss_src_3d

        ps_2d = F.log_softmax(preds_2d_s['seg_logit'], dim=1)
        ps_3d = F.log_softmax(preds_3d_s['seg_logit'], dim=1)

        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            seg_logit_2d = preds_2d_s['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d_s['seg_logit']  # P_2D-Fusion
            seg_logit_3d = preds_3d_s['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d_s['seg_logit']  # P_3D-Fusion

            # fusion-then-distillation
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d_s['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()  # Dkl(P_Fusion||P_2D-Fusion)
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d_s['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()  # Dkl(P_Fusion||P_3D-Fusion)
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d_s += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d_s += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d

        # update metric (e.g. IoU) compute segIoU
        probs_2d_s = F.softmax(preds_2d_s['seg_logit'], dim=1)
        probs_3d_s = F.softmax(preds_3d_s['seg_logit'], dim=1) if model_3d else None
        preds_fus_s = {'seg_logit': F.softmax(probs_2d_s + probs_3d_s, dim=1)}
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d_s, data_batch_src)
            train_metric_3d.update_dict(preds_3d_s, data_batch_src)
            train_metric_fus.update_dict(preds_fus_s, data_batch_src)

        # backward
        loss_s = loss_2d_s + loss_3d_s

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #
        feats_2d_t = model_2d(data_batch_trg)
        feats_3d_t, preds_3d_t = model_3d(data_batch_trg)

        feats_fus_t = model_fusion(feats_2d_t.clone().detach(), feats_3d_t.clone().detach(),
                                   data_batch_trg['img_indices'])

        preds_2d_t = model_2d(data_batch_trg, feats_2d_t, feats_fus_t)

        loss_2d_t = []
        loss_3d_t = []
        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            seg_logit_2d = preds_2d_t['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d_t['seg_logit']
            seg_logit_3d = preds_3d_t['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d_t['seg_logit']

            # fusion-then-distillation
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d_t['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()  # Dkl(P_Fusion||P_2D-Fusion)
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d_t['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()  # Dkl(P_Fusion||P_3D-Fusion)
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d_t.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d_t.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d)
        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            # uni-modal self-training loss with pseudo labels
            pl_loss_trg_2d = F.cross_entropy(preds_2d_t['seg_logit'], data_batch_trg['pseudo_label_2d'])
            pl_loss_trg_3d = F.cross_entropy(preds_3d_t['seg_logit'], data_batch_trg['pseudo_label_3d'])
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d_t.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
            loss_3d_t.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)
        if cfg.TRAIN.XMUDA.lambda_minent > 0:
            # MinEnt
            minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d_t['seg_logit'], dim=1))
            minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d_t['seg_logit'], dim=1))
            train_metric_logger.update(minent_loss_trg_2d=minent_loss_trg_2d,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d_t.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_2d)
            loss_3d_t.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_3d)

        loss_t = sum(loss_2d_t) + sum(loss_3d_t)

        loss_stu = loss_s + loss_t
        st_loss_2d = F.kl_div(ps_2d, probs_2d_st.detach(), reduction='none').sum(1).mean()
        st_loss_3d = F.kl_div(ps_3d, probs_3d_st.detach(), reduction='none').sum(1).mean()
        st_loss_2d_avg = F.kl_div(ps_2d, avg_probs_st.detach(), reduction='none').sum(1).mean()
        st_loss_3d_avg = F.kl_div(ps_3d, avg_probs_st.detach(), reduction='none').sum(1).mean()

        loss_cdkd = 0.1 * (st_loss_2d + st_loss_3d)
        loss_cmkd = cfg.TRAIN.XMUDA.lambda_cmkd * (st_loss_2d_avg + st_loss_3d_avg)
        loss_cons = 0.1 * (loss_cdkd + loss_cmkd)
        train_metric_logger.update(loss_cdkd=0.1 * loss_cdkd, loss_cmkd=0.1 * loss_cmkd)

        loss = loss_stu + loss_cons

        # update student
        loss.backward()
        optimizer_2d.step()
        optimizer_3d.step()
        optimizer_fusion.step()

        # update teacher
        cur_iter = iteration + 1
        update_ema_variables(model_2d, model_2d_ema, 0.999, cur_iter)
        update_ema_variables(model_3d, model_3d_ema, 0.999, cur_iter)
        update_ema_variables(model_fusion, model_fusion_ema, 0.999, cur_iter)

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr_2d: {lr_2d:.2e}',
                        'lr_3d: {lr_3d:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr_2d=optimizer_2d.param_groups[0]['lr'],
                    lr_3d=optimizer_3d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou', 'var')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)
            checkpoint_data_fusion['iteration'] = cur_iter
            checkpoint_data_fusion[best_metric_name] = ''
            checkpointer_fusion.save('model_fusion_{:06d}'.format(cur_iter), **checkpoint_data_fusion)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            # inference with fusion module
            validate_fusion(cfg, model_2d, model_3d, model_fusion, val_dataloader, val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        scheduler_fusion.step()

        end = time.time()

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, output_dir, run_name)


if __name__ == '__main__':
    main()
