checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

num_classes = 15
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='AlexNet'
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True
    ))

# dataset settings
dataset_type = 'ClsFolderDataset'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True),
    # dict(type='RandomResizedCrop', size=256),
    dict(type='Resize', size=(56, 56)),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True),
    dict(type='Resize', size=(56, 56)),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

ann_data_prefix = 'I:/CodeRep/INRCls/data_list'
data_prefix = 'D:/GID/RGB_15_train'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_data_prefix+'/train_list_56.txt',
        classes=ann_data_prefix+'/class_names.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_data_prefix+'/val_list_56.txt',
        classes=ann_data_prefix+'/class_names.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_data_prefix+'/val_list_112.txt',
        classes=ann_data_prefix+'/class_names.txt',
        pipeline=test_pipeline))


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# DETR: backbone:1e-5; lr:1e-4; weight_decay:1e-4; betas=(0.9, 0.95)
# Adam L2正则化的最佳学习率是1e-6（最大学习率为1e-3），而0.3是weight decay的最佳值（学习率为3e-3）
# optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-3)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# lr_config = dict(
#     policy='Poly', power=0.9, min_lr=1e-7, by_epoch=True,
#     warmup='linear', warmup_iters=5, warmup_ratio=0.1, warmup_by_epoch=True
#     )

# # optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
lr_config = dict(policy='step', step=[45, 90, 120])

runner = dict(type='EpochBasedRunner', max_epochs=150)
evaluation = dict(interval=5, metric='accuracy')
load_from = None
resume_from = None