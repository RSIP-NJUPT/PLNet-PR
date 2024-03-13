_base_ = ['./faster-rcnn_r50_egrpl_6x_nwpu.py']

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
    dict(type='RandomResize',scale=[(400, 400), (600, 600)], keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),

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
            metainfo = metainfo,
            ann_file='train.json',
            data_prefix=dict(img='NWPU-10_v2/JPEGImages/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))
