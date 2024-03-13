_base_ = ['./faster-rcnn_r50_fpn_6x_nwpu.py']
# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook', save_best='coco/bbox_mAP'))


# model settings
model = dict(
    type='FasterRCNN_Con',
    neck=dict(
        type='EGPRLFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        size_contrast=512,
        num_outs=5),
    roi_head=dict(
        type='GPRLMHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='ContrastShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            loss_contrast=dict(
                type='SupConProxyAnchorLoss',
                class_num=10,
                size_contrast=512,
                stage=2,
                mrg=0,
                alpha=32,
                loss_weight=0.02)),
        gcn_cfg=dict(
            in_channels=256,
            roi_feat_size=7,
            fc_out_channels=256,
            dropout=0.5,
            n_shift=6,
            init='xavier'),
        )
  )

