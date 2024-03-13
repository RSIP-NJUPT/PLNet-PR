# dataset settings

dataset_type = 'CocoDataset'
data_root = '/home/ma_teng_fei/Desktop/data/NWPU-10_v2/'
metainfo = {
        'classes': 
        ('airplane', 'storagetank', 'baseball', 'tenniscourt', 'basketball',
        'groundtrackfield', 'bridge', 'harbor', 'ship', 'vehicle'),
        'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),]
        }


backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(500,500), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]



test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(500,500), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_pipeline = test_pipeline

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
        ann_file='train.json',
        data_prefix=dict(img='NWPU-10_v2/JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))


test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo = metainfo,
        ann_file='test.json',
        data_prefix=dict(img='NWPU-10_v2/JPEGImages/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
val_dataloader = test_dataloader

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    classwise=True,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
val_evaluator = test_evaluator

