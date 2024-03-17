load_from = "./mmflow/checkpoints/flownet2_8x1_sfine_flyingthings3d_subset_384x768.pth"

FlowNet2css_checkpoint = './mmflow/checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
FlowNet2sd_checkpoint = './mmflow/checkpoints/flownet2sd_8x1_slong_chairssdhom_384x448.pth'
model = dict(
    type='FlowNet2',
    flownetCSS=dict(
        type='FlowNetCSS',
        freeze_net=True,
        flownetC=dict(
            type='FlowNetC',
            encoder=dict(
                type='FlowNetEncoder',
                in_channels=3,
                pyramid_levels=['level1', 'level2', 'level3'],
                out_channels=(64, 128, 256),
                kernel_size=(7, 5, 5),
                strides=(2, 2, 2),
                num_convs=(1, 1, 1),
                dilations=(1, 1, 1),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            corr_level='level3',
            corr_encoder=dict(
                type='CorrEncoder',
                in_channels=473,
                pyramid_levels=['level3', 'level4', 'level5', 'level6'],
                kernel_size=(3, 3, 3, 3),
                num_convs=(1, 2, 2, 2),
                out_channels=(256, 512, 512, 1024),
                redir_in_channels=256,
                redir_channels=32,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                corr_cfg=dict(
                    type='Correlation',
                    kernel_size=1,
                    max_displacement=10,
                    stride=1,
                    padding=0,
                    dilation_patch=2),
                scaled=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            decoder=dict(
                type='FlowNetCDecoder',
                in_channels=dict(
                    level6=1024,
                    level5=1026,
                    level4=770,
                    level3=386,
                    level2=194),
                out_channels=dict(
                    level6=512, level5=256, level4=128, level3=64),
                deconv_bias=True,
                pred_bias=True,
                upsample_bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            train_cfg=dict(),
            test_cfg=dict()),
        flownetS1=dict(
            type='FlowNetS',
            encoder=dict(
                type='FlowNetEncoder',
                in_channels=12,
                pyramid_levels=[
                    'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
                ],
                num_convs=(1, 1, 2, 2, 2, 2),
                out_channels=(64, 128, 256, 512, 512, 1024),
                kernel_size=(7, 5, (5, 3), 3, 3, 3),
                strides=(2, 2, 2, 2, 2, 2),
                dilations=(1, 1, 1, 1, 1, 1),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            decoder=dict(
                type='FlowNetSDecoder',
                in_channels=dict(
                    level6=1024,
                    level5=1026,
                    level4=770,
                    level3=386,
                    level2=194),
                out_channels=dict(
                    level6=512, level5=256, level4=128, level3=64),
                deconv_bias=True,
                pred_bias=True,
                upsample_bias=False,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            train_cfg=dict(),
            test_cfg=dict()),
        flownetS2=dict(
            type='FlowNetS',
            encoder=dict(
                type='FlowNetEncoder',
                in_channels=12,
                pyramid_levels=[
                    'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
                ],
                num_convs=(1, 1, 2, 2, 2, 2),
                out_channels=(64, 128, 256, 512, 512, 1024),
                kernel_size=(7, 5, (5, 3), 3, 3, 3),
                strides=(2, 2, 2, 2, 2, 2),
                dilations=(1, 1, 1, 1, 1, 1),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            decoder=dict(
                type='FlowNetSDecoder',
                in_channels=dict(
                    level6=1024,
                    level5=1026,
                    level4=770,
                    level3=386,
                    level2=194),
                out_channels=dict(
                    level6=512, level5=256, level4=128, level3=64),
                deconv_bias=True,
                pred_bias=True,
                upsample_bias=False,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            train_cfg=dict(),
            test_cfg=dict()),
        link_cfg=dict(scale_factor=4, mode='bilinear'),
        out_level='level2',
        flow_div=20.0,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=FlowNet2css_checkpoint        ),
        train_cfg=dict(),
        test_cfg=dict()),
    flownetSD=dict(
        type='FlowNetS',
        freeze_net=True,
        encoder=dict(
            type='FlowNetSDEncoder',
            in_channels=6,
            plugin_channels=64,
            pyramid_levels=[
                'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
            ],
            num_convs=(2, 2, 2, 2, 2, 2),
            out_channels=((64, 128), 128, 256, 512, 512, 1024),
            kernel_size=3,
            strides=(2, 2, 2, 2, 2, 2),
            dilations=(1, 1, 1, 1, 1, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        decoder=dict(
            type='FlowNetSDecoder',
            in_channels=dict(
                level6=1024, level5=1026, level4=770, level3=386, level2=194),
            out_channels=dict(level6=512, level5=256, level4=128, level3=64),
            inter_channels=dict(level5=512, level4=256, level3=128, level2=64),
            deconv_bias=True,
            pred_bias=True,
            upsample_bias=True,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=FlowNet2sd_checkpoint        ),
        train_cfg=dict(),
        test_cfg=dict()),
    flownet_fusion=dict(
        type='FlowNetS',
        encoder=dict(
            type='FlowNetEncoder',
            in_channels=11,
            pyramid_levels=['level1', 'level2', 'level3'],
            num_convs=(1, 2, 2),
            out_channels=(64, (64, 128), 128),
            kernel_size=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        decoder=dict(
            type='FlowNetSDecoder',
            in_channels=dict(level3=128, level2=162, level1=82),
            out_channels=dict(level3=32, level2=16),
            inter_channels=dict(level2=32, level1=16),
            deconv_bias=True,
            pred_bias=True,
            upsample_bias=True,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            flow_loss=dict(
                type='MultiLevelEPE',
                p=2,
                reduction='sum',
                flow_div=1.0,
                weights=dict(level1=0.005)),
            flow_div=1.0,
            init_cfg=[
                dict(
                    type='Xavier',
                    distribution='uniform',
                    layer=['Conv2d', 'ConvTranspose2d'],
                    bias=0),
                dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
            ])),
    link_cfg=dict(scale_factor=4, mode='nearest'),
    flow_div=20.0,
    out_level='level2',
    train_cfg=dict(),
    test_cfg=dict())

# data
# Training set: Sintel clean/final
# Validation/Training: Sintel clean/final
data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        drop_last=True,
        persistent_workers=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[dict(
        type='Sintel',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_occ=True),
            dict(
                type='ColorJitter',
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5),
            dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
            dict(
                type='Normalize',
                mean=[0., 0., 0.], std=[255., 255., 255.],
                to_rgb=False),
            dict(
                type='GaussianNoise',
                sigma_range=(0, 0.04),
                clamp_range=(0.0, 1.0)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='RandomAffine',
                global_transform=dict(
                    translates=(0.05, 0.05),
                    zoom=(1.0, 1.5),
                    shear=(0.86, 1.16),
                    rotate=(-10., 10.)),
                relative_transform=dict(
                    translates=(0.00375, 0.00375),
                    zoom=(0.985, 1.015),
                    shear=(1.0, 1.0),
                    rotate=(-1.0, 1.0))),
            dict(type='RandomCrop', crop_size = (384, 768)),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                        keys=['imgs', 'flow_gt'],
                        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ])
        ],
        data_root='mmflow/data/Sintel',
        test_mode=False,
        pass_style='clean'),

        dict(
        type='Sintel',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_occ=True),
            dict(
                type='ColorJitter',
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5),
            dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
            dict(
                type='Normalize',
                mean=[0., 0., 0.], std=[255., 255., 255.],
                to_rgb=False),
            dict(
                type='GaussianNoise',
                sigma_range=(0, 0.04),
                clamp_range=(0.0, 1.0)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='RandomAffine',
                global_transform=dict(
                    translates=(0.05, 0.05),
                    zoom=(1.0, 1.5),
                    shear=(0.86, 1.16),
                    rotate=(-10., 10.)),
                relative_transform=dict(
                    translates=(0.00375, 0.00375),
                    zoom=(0.985, 1.015),
                    shear=(1.0, 1.0),
                    rotate=(-1.0, 1.0))),
            dict(type='RandomCrop', crop_size = (384, 768)),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                        keys=['imgs', 'flow_gt'],
                        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ])
        ],
        data_root='mmflow/data/Sintel',
        test_mode=False,
        pass_style='final')],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                data_root='mmflow/data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs', 'flow_gt', 'valid'],
                        meta_keys=[
                            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
                        ])
                ],
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                data_root='mmflow/data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                data_root='mmflow/data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                data_root='mmflow/data/Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape'
                        ])
                ],
                test_mode=True,
                pass_style='final'),
                ],
        separate_eval=True))


# Schedule, settings from sfine flyingthings3d
# optimizer
optimizer = dict(
    type='Adam', lr=0.00001, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[200000, 300000, 400000, 500000])
runner = dict(type='IterBasedRunner', max_iters=600000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')

# Runtime
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
