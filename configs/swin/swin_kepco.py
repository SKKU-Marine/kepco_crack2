

norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint= 'data/output/fourth_train/latest.pth'
        )),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=False,
            class_weight=[0.1, 0.9])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=False,
            class_weight=[0.1, 0.9])),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = ['CrackDataset0', 'CrackDataset1']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# define dataset for train
train_tancheon = dict(
        type='CrackDataset0',
        data_root='./data/train/seg/tancheon/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=train_pipeline,
        split=None)

train_aihub_shm = dict(
        type='CrackDataset1',
        data_root='./data/train/seg/aihub_shm/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=train_pipeline,
        split=None)

train_seoul = dict(
        type='CrackDataset0',
        data_root='./data/train/seg/seoul_crack/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=train_pipeline,
        split=None)

train_aihub_welcon = dict(
        type='CrackDataset0',
        data_root='./data/train/seg/aihub_welcon/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=train_pipeline,
        split=None)

train_bridge2 = dict(
        type='CrackDataset0',
        data_root='./data/train/seg/bridge2/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=train_pipeline,
        split=None)

# for validation
val_tancheon = dict(
        type='CrackDataset0',
        data_root='./data/validation/seg/tancheon/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=test_pipeline,
        split=None)

val_aihub_shm = dict(
        type='CrackDataset1',
        data_root='./data/validation/seg/aihub_shm/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=test_pipeline,
        split=None)

val_seoul = dict(
        type='CrackDataset0',
        data_root='./data/validation/seg/seoul_crack/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=test_pipeline,
        split=None)

val_aihub_welcon = dict(
        type='CrackDataset0',
        data_root='./data/validation/seg/aihub_welcon/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=test_pipeline,
        split=None)

val_bridge2 = dict(
        type='CrackDataset0',
        data_root='./data/validation/seg/bridge2/',
        img_dir='image/',
        ann_dir='label/',
        pipeline=test_pipeline,
        split=None)

# Dataset configuration
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train= [train_aihub_shm,train_aihub_welcon,train_seoul,train_tancheon,train_bridge2],
    val=[val_aihub_shm,val_aihub_welcon,val_seoul,val_tancheon,val_bridge2],
    test= val_aihub_shm)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'data/output/fourth_train/latest.pth'
resume_from = None # use this if you want to continue from previous train weight
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(type='OptimizerHook')
lr_config = dict(
    policy = 'poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=0.9,
    min_lr=0.0,
    by_epoch=False
    )
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=1000, type='CheckpointHook')
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True, by_epoch=False)
checkpoint_file = 'data/output/fourth_train/latest.pth'
seed = 0
gpu_ids = range(0, 1)
device = 'cuda'
work_dir = 'data/output/new_train/'
