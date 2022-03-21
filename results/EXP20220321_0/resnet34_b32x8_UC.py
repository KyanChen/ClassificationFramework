prefix = 'G:\ProgramData\Miniconda3\envs\pytorch'
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
num_classes = 21
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth'
        ),
        prefix='backbone'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=21,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'ClsFolderDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True),
    dict(type='Resize', size=(256, 256)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256)),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data_prefix = 'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='ClsFolderDataset',
        data_prefix=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images',
        ann_file=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../train_list.txt',
        classes=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../class_names.txt',
        pipeline=[
            dict(type='LoadImageFromFile', rearrange=True),
            dict(type='Resize', size=(256, 256)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ClsFolderDataset',
        data_prefix=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images',
        ann_file=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../val_list.txt',
        classes=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../class_names.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ClsFolderDataset',
        data_prefix=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images',
        ann_file=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../val_list.txt',
        classes=
        'H:/DataSet\SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../class_names.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.001)
lr_config = dict(
    policy='Poly',
    power=0.9,
    min_lr=1e-07,
    by_epoch=True,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=200)
evaluation = dict(interval=1, metric='accuracy')
load_from = None
resume_from = None
work_dir = 'results/EXP20220321_0'
gpu_ids = [0]
