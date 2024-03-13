# dataset settings
dataset_type = 'CocoDataset'
data_root = '/mnt/e/data/DIOR/'
metainfo = {
    'classes':
        ('golffield', 'Expressway-toll-station', 'vehicle', 'trainstation', 'chimney',
         'storagetank', 'ship', 'harbor', 'airplane', 'groundtrackfield',
         'tenniscourt', 'dam', 'basketballcourt', 'Expressway-Service-area', 'stadium',
         'airport', 'baseballfield', 'bridge', 'windmill', 'overpass'),
        # palette is a list of color tuples, which is used for visualization.
    'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (250, 0, 0), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
             (0, 82, 0), (120, 166, 157)]
}

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    # dict(type='RandomResize',scale=[(800, 800), (1024, 1024)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandAugment', aug_space=[dict(type='Rotate', prob=0.5),
    #                              dict(type='RandomFlip', prob=0.5)],
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
    ]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler', drop_last=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo = metainfo,
        ann_file='trainval.json',
        data_prefix=dict(img='JPEGImages-trainval/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo = metainfo,
        ann_file='second_select_test.json',
        data_prefix=dict(img='JPEGImages-test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'second_select_test.json',
    classwise=True,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
val_evaluator = test_evaluator

