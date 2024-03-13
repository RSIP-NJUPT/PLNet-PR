_base_ = ['./faster-rcnn_r50_egrpl_1x_hrrsd.py']


dataset_type = 'CocoDataset'
data_root = '/home/ma_teng_fei/Desktop/data/hrrsd/'
metainfo = {
    'classes':
        ('bridge', 'airplane', 'ground track field', 'vehicle', 'parking lot',
        'T junction', 'baseball diamond', 'tennis court', 'basketball court', 'ship',
          'crossroad', 'harbor', 'storage tank'),
        # palette is a list of color tuples, which is used for visualization.
    'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (250, 0, 0), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175)]
}

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomChoiceResize',scales=[(800,800), (900, 900), (1024,1024)], keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandAugment', aug_space=[dict(type='Rotate', prob=0.5),
    #                              dict(type='RandomFlip', prob=0.5)],
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler', drop_last=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=3,
        dataset=dict(
                    type=dataset_type,
                    data_root=data_root,
                    metainfo=metainfo,
                    ann_file='trainval.json',
                    data_prefix=dict(img='img/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args),
                )
    )
