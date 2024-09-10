import os
import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from xmuda.data.utils.evaluate import Evaluator
from xmuda.data.utils.visualize import show_raw_image, draw_points_image_labels, draw_bird_eye_view, draw_bird_eye_view_error


def validate_fusion(cfg,
                    model_2d,
                    model_3d,
                    model_fusion,
                    dataloader,
                    val_metric_logger,
                    pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    dual_model = (model_2d is not None) and (model_3d is not None)

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names) if model_2d else None
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if dual_model else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            feats_2d = model_2d(data_batch)
            feats_3d, preds_3d = model_3d(data_batch) if model_3d else None

            feats_fus = model_fusion(feats_2d.clone().detach(), feats_3d.clone().detach(),
                                     data_batch['img_indices']) if model_3d else None

            preds_2d = model_2d(data_batch, feats_2d, feats_fus) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy() if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            probs_fusion = F.softmax(probs_2d + probs_3d, dim=1)
            pred_label_voxel_ensemble = probs_fusion.argmax(1).cpu().numpy() if dual_model else None  # fusion output

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if dual_model else None

                # evaluate
                if model_2d:
                    evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                if dual_model:
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    if model_2d:
                        assert np.all(pred_label_2d >= 0)
                    if model_3d:
                        assert np.all(pred_label_3d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx] if model_2d else None
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy() if model_2d else None,
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8)  if model_2d else None,
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label']) if model_2d else None
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            if seg_loss_2d is not None:
                val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        eval_list = []
        if evaluator_2d is not None:
            val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
            eval_list.append(('2D', evaluator_2d))
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
            eval_list.append(('3D', evaluator_3d))
        if dual_model:
            eval_list.append(('2D+3D', evaluator_ensemble))
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy: {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))


# Special for SSDA in sKITTI, because 11-21 sequences in unlabeled target do not have GT
def validate_fusion_gen_pselab(cfg,
                    model_2d,
                    model_3d,
                    model_fusion,
                    dataloader,
                    val_metric_logger,
                    pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    dual_model = (model_2d is not None) and (model_3d is not None)

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names) if model_2d else None
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if dual_model else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                # data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            feats_2d = model_2d(data_batch)
            feats_3d, preds_3d = model_3d(data_batch) if model_3d else None

            feats_fus = model_fusion(feats_2d.clone().detach(), feats_3d.clone().detach(),
                                     data_batch['img_indices']) if model_3d else None

            preds_2d = model_2d(data_batch, feats_2d, feats_fus) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy() if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            probs_fusion = F.softmax(probs_2d + probs_3d, dim=1)
            pred_label_voxel_ensemble = probs_fusion.argmax(1).cpu().numpy() if dual_model else None  # fusion output

            # get original point cloud from before voxelization
            # seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(points_idx)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                # curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if dual_model else None

                # evaluate
                # if model_2d:
                #     evaluator_2d.update(pred_label_2d, curr_seg_label)
                # if model_3d:
                #     evaluator_3d.update(pred_label_3d, curr_seg_label)
                # if dual_model:
                #     evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    if model_2d:
                        assert np.all(pred_label_2d >= 0)
                    if model_3d:
                        assert np.all(pred_label_3d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx] if model_2d else None
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy() if model_2d else None,
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8)  if model_2d else None,
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            # seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label']) if model_2d else None
            # seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            # if seg_loss_2d is not None:
            #     val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            # if seg_loss_3d is not None:
            #     val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        # eval_list = []
        # if evaluator_2d is not None:
        #     val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
        #     eval_list.append(('2D', evaluator_2d))
        # if evaluator_3d is not None:
        #     val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
        #     eval_list.append(('3D', evaluator_3d))
        # if dual_model:
        #     eval_list.append(('2D+3D', evaluator_ensemble))
        # for modality, evaluator in eval_list:
        #     logger.info('{} overall accuracy: {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
        #     logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
        #     logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))


def viz_fusion(cfg,
               model_2d,
               model_3d,
               model_fusion,
               dataloader,
               val_metric_logger,
               pselab_path=None,
               output_dir=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')
    it = 0
    dual_model = (model_2d is not None) and (model_3d is not None)

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names) if model_2d else None
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if dual_model else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            feats_2d = model_2d(data_batch)
            feats_3d, preds_3d = model_3d(data_batch) if model_3d else None

            feats_fus = model_fusion(feats_2d.clone().detach(), feats_3d.clone().detach(),
                                     data_batch['img_indices']) if model_3d else None

            preds_2d = model_2d(data_batch, feats_2d, feats_fus) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy() if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            probs_fusion = F.softmax(probs_2d + probs_3d, dim=1)
            pred_label_voxel_ensemble = probs_fusion.argmax(1).cpu().numpy() if dual_model else None  # fusion output

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']

            img_indices = data_batch['img_indices']

            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if dual_model else None

                # evaluate
                if model_2d:
                    evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                if dual_model:
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    if model_2d:
                        assert np.all(pred_label_2d >= 0)
                    if model_3d:
                        assert np.all(pred_label_3d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx] if model_2d else None
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy() if model_2d else None,
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8)  if model_2d else None,
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

                # ===========================================================================================

                if it < 500:
                    ### visualization
                    it = it + 1
                    # ipr = output_dir + 'viz/raw/'
                    # igt = output_dir + 'viz/gt/'
                    ipd = output_dir + 'viz/pred/'
                    ipd_2d = output_dir + 'viz/pred_2d/'
                    ipd_3d = output_dir + 'viz/pred_3d/'

                    # if not os.path.exists(ipr):
                    #     os.makedirs(ipr)
                    # if not os.path.exists(igt):
                    #     os.makedirs(igt)
                    if not os.path.exists(ipd):
                        os.makedirs(ipd)
                    if not os.path.exists(ipd_2d):
                        os.makedirs(ipd_2d)
                    if not os.path.exists(ipd_3d):
                        os.makedirs(ipd_3d)
                    # ipr = ipr + str(it)
                    # igt = igt + str(it)
                    ipd = ipd + str(it)
                    ipd_2d = ipd_2d + str(it)
                    ipd_3d = ipd_3d + str(it)

                    color_palette_type = ''
                    if cfg.DATASET_SOURCE.TYPE == 'NuScenesLidarSegSCN':
                        color_palette_type = 'NuScenesLidarSeg'
                        point_size = 3
                    elif cfg.DATASET_SOURCE.TYPE == 'VirtualKITTISCN':
                        color_palette_type = 'VirtualKITTI'
                        point_size = 1
                    elif cfg.DATASET_SOURCE.TYPE == 'A2D2SCN':
                        color_palette_type = 'SemanticKITTI'
                        point_size = 1
                    # show_raw_image((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(), ipr)
                    # draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                    #                          img_indices[batch_ind], curr_seg_label, igt,
                    #                          color_palette_type=color_palette_type, point_size=point_size)
                    draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                                             img_indices[batch_ind], pred_label_ensemble, ipd,
                                             color_palette_type=color_palette_type, point_size=point_size)
                    draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                                             img_indices[batch_ind], pred_label_2d, ipd_2d,
                                             color_palette_type=color_palette_type, point_size=point_size)
                    draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                                             img_indices[batch_ind], pred_label_3d, ipd_3d,
                                             color_palette_type=color_palette_type, point_size=point_size)

                    ### visualization-BEV
                    # it = it + 1
                    # igt = output_dir + 'viz/gt_BEV/'
                    # ipd = output_dir + 'viz/error_BEV/'
                    # if not os.path.exists(igt):
                    #     os.makedirs(igt)
                    # if not os.path.exists(ipd):
                    #     os.makedirs(ipd)
                    # igt = igt + str(it)
                    # ipd = ipd + str(it)
                    #
                    # color_palette_type = ''
                    # if cfg.DATASET_SOURCE.TYPE == 'NuScenesLidarSegSCN':
                    #     color_palette_type = 'NuScenesLidarSeg'
                    #     point_size = 1
                    # elif cfg.DATASET_SOURCE.TYPE == 'VirtualKITTISCN':
                    #     color_palette_type = 'VirtualKITTI'
                    #     point_size = 1
                    # elif cfg.DATASET_SOURCE.TYPE == 'A2D2SCN':
                    #     color_palette_type = 'SemanticKITTI'
                    #     point_size = 1
                    #
                    # coords = data_batch['x'][0][left_idx:right_idx]
                    # draw_bird_eye_view(coords, curr_seg_label, igt, color_palette_type=color_palette_type,
                    #                    point_size=point_size)
                    #
                    # error_result = (pred_label_ensemble == curr_seg_label).astype(int)
                    # draw_bird_eye_view_error(coords, error_result, ipd, point_size=point_size)
                    # draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                    #                          img_indices[batch_ind], curr_seg_label, igt,
                    #                          color_palette_type=color_palette_type, point_size=point_size)
                    # draw_points_image_labels((data_batch['img'][batch_ind].permute(1, 2, 0)).cpu(),
                    #                          img_indices[batch_ind], pred_label_ensemble, ipd,
                    #                          color_palette_type=color_palette_type, point_size=point_size)
                else:
                    continue

                # ===========================================================================================

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label']) if model_2d else None
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            if seg_loss_2d is not None:
                val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        eval_list = []
        if evaluator_2d is not None:
            val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
            eval_list.append(('2D', evaluator_2d))
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
            eval_list.append(('3D', evaluator_3d))
        if dual_model:
            eval_list.append(('2D+3D', evaluator_ensemble))
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy: {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))
