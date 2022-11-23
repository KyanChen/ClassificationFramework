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

num_classes = 21
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResMLP',
        num_layers=4,
        in_channels=1024,
        out_channels=512,
        base_channels=512,
        bias=True,
        expansions=[1]
        ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True
    ))

# dataset settings
dataset_type = 'ClsVec'
train_pipeline = [
    dict(type='ToTensor', keys=['img', 'gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

ann_data_prefix = 'I:/CodeRep/INRCls/data_list/UC'
data_prefix = 'I:/CodeRep/INRCls/results/EXP20220422_9'
data = dict(
    samples_per_gpu=1024,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_data_prefix+'/train_list.txt',
        classes=ann_data_prefix+'/class_names.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_data_prefix+'/val_list.txt',
        classes=ann_data_prefix+'/class_names.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_prefix,
        ann_file=ann_data_prefix+'/val_list.txt',
        classes=ann_data_prefix+'/class_names.txt',
        pipeline=test_pipeline))


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# DETR: backbone:1e-5; lr:1e-4; weight_decay:1e-4; betas=(0.9, 0.95)
# Adam L2正则化的最佳学习率是1e-6（最大学习率为1e-3），而0.3是weight decay的最佳值（学习率为3e-3）
optimizer = dict(type='AdamW', lr=1e-4)
# optimizer = dict(type='SGD', lr=1e-3)

# lr_config = dict(
#     policy='Poly', power=0.9, min_lr=1e-6, by_epoch=True,
#     warmup='linear', warmup_iters=5, warmup_ratio=0.1, warmup_by_epoch=True
#     )

# # optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(policy='step', step=[50, 100])
lr_config = None

runner = dict(type='EpochBasedRunner', max_epochs=150)
evaluation = dict(interval=5, metric='accuracy', is_save_pred=False)
load_from = None
resume_from = None