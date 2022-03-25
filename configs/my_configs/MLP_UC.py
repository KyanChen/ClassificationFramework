checkpoint_config = dict(interval=100)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [ ('val', 1)]
custom_hooks = [dict(type='VisImg', dir='result')]

n_batch = 1
inner_loop_num = 5000

model = dict(
    type='ImageRepresentor',
    img_size=(256, 256),
    max_sample_per_img=0.5,
    max_inner_iter=inner_loop_num,
    loss=dict(type='MSELoss', loss_weight=1.0),
    backbone=dict(
        type='Siren',
        inner_layers=8,
        in_channels=2,
        out_channels=3,
        base_channels=28,
        num_modulation=256,
        bias=True,
        expansions=[1],
        init_cfg=None
        ),
    modulations=dict(
        type='Modulations',
        n_dims=256,
        n_batch=n_batch
    )
)

# dataset settings
dataset_type = 'RepFolderDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True),
    dict(type='Resize', size=(256, 256)),
    # dict(type='RandomFlip', flip_prob=0.5, directions=['horizontal', 'vertical']),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_prefix = 'data/UC/UCMerced_LandUse/Images'
data_prefix = r'H:\DataSet\SceneCls\UCMerced_LandUse\UCMerced_LandUse\Images'.replace('\\', '/')

data = dict(
    samples_per_gpu=n_batch,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=data_prefix+'/../all_img_list.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=data_prefix+'/../val_list.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=data_prefix+'/../val_list.txt',
        pipeline=test_pipeline))


optimizer_config = dict(type='MyOptimizerHook', grad_clip=None)

# DETR: backbone:1e-5; lr:1e-4; weight_decay:1e-4; betas=(0.9, 0.95)
# Adam L2正则化的最佳学习率是1e-6（最大学习率为1e-3），而0.3是weight decay的最佳值（学习率为3e-3）
# optimizer_inner = dict(type='AdamW', lr=1e-3, weight_decay=1e-3)
# optimizer_outer = dict(type='AdamW', lr=1e-3, weight_decay=1e-3)
optimizer = dict(
    modulations=dict(type='AdamW', lr=1e-2, betas=(0.5, 0.999)),
    siren=dict(type='AdamW', lr=1e-3, betas=(0.5, 0.999))
)

# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    modulations=dict(policy='InnerPoly', inner_loop_num=inner_loop_num, power=0.9, min_lr=1e-5, by_epoch=True
                     ),
    siren=dict(policy='OuterPoly', power=0.9, min_lr=1e-7, by_epoch=True,
               warmup='linear', warmup_iters=5, warmup_ratio=0.1, warmup_by_epoch=True),
    )


runner = dict(type='DynamicEpochBasedRunner', max_epochs=5000)
evaluation = dict(interval=10, metric='accuracy')
load_from = None
resume_from = None