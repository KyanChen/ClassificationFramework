checkpoint_config = dict(interval=5)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

img_size = 56
num_classes = 15

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='s',
        img_size=img_size,
        patch_size=2,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=num_classes,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True
    ))

dataset_type = 'ClsFolderDataset'

img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True),
    # dict(type='RandomResizedCrop', size=256),
    # dict(type='Resize', size=(128, 128)),
    # dict(type='CenterCrop', crop_size=64),
    dict(type='Resize', size=(28, 28)),
    # dict(type='RandomFlip', flip_prob=0.5, directions=['horizontal', 'vertical']),
    # dict(type='ColorJitter', brightness=0.2, contrast=0.1, saturation=0.1),

    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', rearrange=True),
    # dict(type='Resize', size=(112, 112)),
    # dict(type='CenterCrop', crop_size=84),
    dict(type='Resize', size=(56, 56)),
    # dict(type='CenterCrop', crop_size=56),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

ann_data_prefix = 'I:/CodeRep/INRCls/data_list/GID/N600'
# ann_data_prefix = 'I:/CodeRep/INRCls/data_list/UC'
data_prefix = 'D:/GID/RGB_15_train'
# data_prefix = 'H:/DataSet/SceneCls/UCMerced_LandUse/UCMerced_LandUse/Images'
# data_prefix = 'I:/CodeRep/INRCls/results/EXP20220407_0'
data = dict(
    samples_per_gpu=32,
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
        # ann_file=ann_data_prefix+'/val_list.txt',
        ann_file=ann_data_prefix+'/val_list_56.txt',
        classes=ann_data_prefix+'/class_names.txt',
        pipeline=test_pipeline))

evaluation = dict(interval=5, metric='accuracy', is_save_pred=False, metric_options={'topk': (1, 5)})

# specific to vit pretrain
paramwise_cfg = dict(custom_keys={
    '.cls_token': dict(decay_mult=0.0),
    '.pos_embed': dict(decay_mult=0.0)
})

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.3,
    paramwise_cfg=paramwise_cfg,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

lr_config = None

runner = dict(type='EpochBasedRunner', max_epochs=150)
