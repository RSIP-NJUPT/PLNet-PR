from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F    
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.models.losses import accuracy
from mmengine.config import ConfigDict
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from .convfc_bbox_head import ConvFCBBoxHead

@MODELS.register_module()
class ContrastShared2FCBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 fc_out_channels: int = 1024,
                 contrast_out_channels: int = 512,
                 loss_contrast=dict(
                                  type='SupConProxyAnchorLoss',
                                  class_num=37,
                                  size_contrast=512,
                                  stage=2,
                                  mrg=0,
                                  alpha=32,
                                  loss_weight=0.5),
                 *args, **kwargs) -> None:
        super().__init__(   
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.contrast_out_channels = contrast_out_channels
        self.encoder = Contrastivebranch(self.shared_out_channels, self.contrast_out_channels)  #self.mlp_head_dim 256 or 128
        self.loss_contrast = MODELS.build(loss_contrast)

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_contrast = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        contrast_feature = self.encoder(x_contrast)
        return cls_score, bbox_pred, contrast_feature
    
    def loss_and_target(self,
                        cls_score: Tensor,
                        bbox_pred: Tensor,
                        contrast_feature: Tensor,
                        rois: Tensor,
                        sampling_results: List[SamplingResult],
                        rcnn_train_cfg: ConfigDict,
                        concat: bool = True,
                        reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the features extracted by the bbox head.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch. Defaults to True.
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """

        cls_reg_targets = self.get_targets(
            sampling_results, rcnn_train_cfg, concat=concat)
        losses = self.loss(
            cls_score,
            bbox_pred,
            contrast_feature,
            rois,
            *cls_reg_targets,
            reduction_override=reduction_override)

        # cls_reg_targets is only for cascade rcnn
        return dict(loss_bbox=losses, bbox_targets=cls_reg_targets)
    
    def loss(self,
                cls_score: Tensor,
                bbox_pred: Tensor,
                contrast_feature: Tensor,
                rois: Tensor,
                labels: Tensor,
                label_weights: Tensor,
                bbox_targets: Tensor,
                bbox_weights: Tensor,
                reduction_override: Optional[str] = None) -> dict:
            """Calculate the loss based on the network predictions and targets.

            Args:
                cls_score (Tensor): Classification prediction
                    results of all class, has shape
                    (batch_size * num_proposals_single_image, num_classes)
                bbox_pred (Tensor): Regression prediction results,
                    has shape
                    (batch_size * num_proposals_single_image, 4), the last
                    dimension 4 represents [tl_x, tl_y, br_x, br_y].
                rois (Tensor): RoIs with the shape
                    (batch_size * num_proposals_single_image, 5) where the first
                    column indicates batch id of each RoI.
                labels (Tensor): Gt_labels for all proposals in a batch, has
                    shape (batch_size * num_proposals_single_image, ).
                label_weights (Tensor): Labels_weights for all proposals in a
                    batch, has shape (batch_size * num_proposals_single_image, ).
                bbox_targets (Tensor): Regression target for all proposals in a
                    batch, has shape (batch_size * num_proposals_single_image, 4),
                    the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                bbox_weights (Tensor): Regression weights for all proposals in a
                    batch, has shape (batch_size * num_proposals_single_image, 4).
                reduction_override (str, optional): The reduction
                    method used to override the original reduction
                    method of the loss. Options are "none",
                    "mean" and "sum". Defaults to None,

            Returns:
                dict: A dictionary of loss.
            """

            losses = dict()

            if cls_score is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if cls_score.numel() > 0:
                    loss_cls_ = self.loss_cls(
                        cls_score,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                    if isinstance(loss_cls_, dict):
                        losses.update(loss_cls_)
                    else:
                        losses['loss_cls'] = loss_cls_
                    if self.custom_activation:
                        acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                        losses.update(acc_)
                    else:
                        losses['acc'] = accuracy(cls_score, labels)
            if bbox_pred is not None:
                bg_class_ind = self.num_classes
                # 0~self.num_classes-1 are FG, self.num_classes is BG
                pos_inds = (labels >= 0) & (labels < bg_class_ind)
                # do not perform bounding box regression for BG anymore.
                if pos_inds.any():
                    if self.reg_decoded_bbox:
                        bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                        bbox_pred = get_box_tensor(bbox_pred)
                    if self.reg_class_agnostic:
                        pos_bbox_pred = bbox_pred.view(
                            bbox_pred.size(0), -1)[pos_inds.type(torch.bool)]
                    else:
                        pos_bbox_pred = bbox_pred.view(
                            bbox_pred.size(0), self.num_classes,
                            -1)[pos_inds.type(torch.bool),
                                labels[pos_inds.type(torch.bool)]]
                    losses['loss_bbox'] = self.loss_bbox(
                        pos_bbox_pred,
                        bbox_targets[pos_inds.type(torch.bool)],
                        bbox_weights[pos_inds.type(torch.bool)],
                        avg_factor=bbox_targets.size(0),
                        reduction_override=reduction_override)
                else:
                    losses['loss_bbox'] = bbox_pred[pos_inds].sum()

            if contrast_feature is not None:
                loss_contrast= self.loss_contrast(contrast_feature, labels)
                losses['loss_contrast'] = loss_contrast

            return losses
    
class Contrastivebranch(nn.Module):
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
    def init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        #nn.init.xavier_uniform_(self.last_fc_weight)
    def forward(self, x):
        feat = self.head(x)
        return feat
