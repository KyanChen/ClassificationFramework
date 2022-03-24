checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
n_batch = 2
model = dict(
    type='ImageRepresentor',
    img_size=(256, 256),
    backbone=dict(
        type='Siren',
        num_layers=3,
        in_channels=2,
        out_channels=3,
        base_channels=28,
        num_modulation=256,
        bias=True,
        expansions=[1],
        init_cfg=None),
    modulations=dict(type='Modulations', n_dims=256, n_batch=2))
dataset_type = 'RepFolderDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True),
    dict(type='Resize', size=(256, 256)),
    dict(
        type='RandomFlip',
        flip_prob=0.5,
        directions=['horizontal', 'vertical']),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data_prefix = 'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepFolderDataset',
        data_prefix=
        'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images',
        ann_file=
        'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../all_img_list.txt',
        pipeline=[
            dict(type='LoadImageFromFile', rearrange=True),
            dict(type='Resize', size=(256, 256)),
            dict(
                type='RandomFlip',
                flip_prob=0.5,
                directions=['horizontal', 'vertical']),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    val=dict(
        type='RepFolderDataset',
        data_prefix=
        'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images',
        ann_file=
        'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../val_list.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='RepFolderDataset',
        data_prefix=
        'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images',
        ann_file=
        'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images/../val_list.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(
    modulations=dict(type='Adam', lr=0.005, betas=(0.5, 0.999)),
    siren=dict(type='Adam', lr=0.001, betas=(0.5, 0.999)))
lr_config = dict(
    modulations=dict(
        policy='InnerPoly',
        loop_num=10,
        power=0.9,
        min_lr=1e-07,
        by_epoch=True),
    siren=dict(
        policy='OuterPoly',
        power=0.9,
        min_lr=1e-07,
        by_epoch=True,
        warmup='linear',
        warmup_iters=5,
        warmup_ratio=0.1,
        warmup_by_epoch=True))
runner = dict(type='DynamicEpochBasedRunner', max_epochs=200)
evaluation = dict(interval=1, metric='accuracy')
load_from = None
resume_from = None
work_dir = 'results/EXP20220321_0'
gpu_ids = [0]
