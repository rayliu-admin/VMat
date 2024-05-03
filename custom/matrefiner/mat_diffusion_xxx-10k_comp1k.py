_base_ = [
    'configs/_base_/datasets/comp1k.py', 'configs/_base_/matting_default_runtime.py'
]

experiment_name = 'mat_diffusion_xxx-10k_comp1k'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='MATREFINER',
    step=6,
    data_preprocessor=dict(
        type='MatdiffPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        proc_trimap='as_is' ,
        num_timesteps=6,
        diffusion_cfg=dict(
            betas=dict(
                type='linear',
                start=0.8,
                stop=0,
                num_timesteps=6),
            diff_iter=False,
        ),
    ),
    backbone=dict(
        type='DenoiseUNet',
        in_channels=4,
        out_channels=1,
        model_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_strides=(16, 32),
        learn_time_embd=True,
        channel_mult = (1, 1, 2, 2, 4, 4),
        dropout=0.0),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.8,
            stop=0,
            num_timesteps=6),
        diff_iter=False),
    loss_alpha=dict(type='L1Loss'),
    # test_cfg=dict(
    #     resize_method='pad',
    #     resize_mode='reflect',
    #     size_divisor=32,
    # )
)

# dataset settings
data_root = r'E:\MAT\DATA\adobe_composition-1k\adobe_composition-1k'
bg_dir = r'E:\MAT\DATA\adobe_composition-1k\adobe_composition-1k\Training_set\bg'

train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', color_type='grayscale'),
    dict(type='LoadImageFromFile', key='merged'),
    # dict(type='RandomLoadResizeBg', bg_dir=bg_dir),
    # dict(
    #     type='CompositeFg',
    #     fg_dirs=[
    #         f'{data_root}/Training_set/Adobe-licensed images/fg',
    #         f'{data_root}/Training_set/Other/fg'
    #     ],
    #     alpha_dirs=[
    #         f'{data_root}/Training_set/Adobe-licensed images/alpha',
    #         f'{data_root}/Training_set/Other/alpha'
    #     ]),
    # dict(
    #     type='RandomAffine',
    #     keys=['alpha', 'fg'],
    #     degrees=30,
    #     scale=(0.8, 1.25),
    #     shear=10,
    #     flip_ratio=0.5),
    # dict(type='GenerateTrimap', kernel_size=(1, 30)),
    # dict(type='Resize', size=256, keep_ratio=True),
    dict(type='Crop', keys=['alpha','merged'],crop_size=(256,256)),
    # dict(type='RandomJitter'),
    # dict(type='MergeFgAndBg'),
    dict(type='LoadCoarseMasks'),
    dict(type='PackInputs',data_keys=['coarse_masks','fine_masks']),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        color_type='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        color_type='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='Crop', keys=['alpha','merged','trimap'],crop_size=(512,512)),
    dict(type='FormatTrimap', to_onehot=True),
    dict(type='LoadCoarseMasks'),
    dict(type='PackInputs',data_keys=['coarse_masks','fine_masks']),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=20,
    dataset=dict(pipeline=train_pipeline,
                 data_root = data_root),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=20,
    dataset=dict(pipeline=test_pipeline,
                     indices=2,data_root=data_root),
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=200_000,
    # val_interval=100,
)

# 关闭验证，有相关继承
val_cfg = None
val_evaluator = None
val_dataloader = None

# val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='AmpOptimWrapper',
    optimizer=dict(type='Adam', lr=4e-4, betas=[0.5, 0.999]))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=2500,
        by_epoch=False,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=200_000,  # TODO, need more check
        eta_min=0,
        begin=0,
        end=200_000,
        by_epoch=False,
    )
]

visualizer = dict(
    img_keys=['pred_alpha', 'coarse_mask', 'gt_merged', 'gt_alpha'])

# custom_hooks = [dict(type='BasicVisualizationHook',on_train=True, interval=100)]

# checkpoint saving
# inheritate from _base_

# runtime settings
# inheritate from _base_