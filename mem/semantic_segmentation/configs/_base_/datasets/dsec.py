
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True) # this casts to torch.float32...
resize_base_H, resize_base_W = 440, 640 # resize image, so that the 224² sees most of the scene
ratio_range_min, ratio_range_max = 1.0, 1.01
crop_size = (440, 640)

numclassesstr = ["", "19"]
numclassesstr = numclassesstr[0]

# see https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/transforms.py for details
train_pipeline = [
    dict(type='LoadNpy'), # (H, W, C) tensor.
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(resize_base_W, resize_base_H), ratio_range=(ratio_range_min, ratio_range_max), keep_ratio=False), # (W, H)
    dict(type='ToFloat32TorchEvs'), # (C, H, W)
    dict(type='RemoveHotPixelsEvs'), # in (0.0, 255.0)
    dict(type='NormalizeEvs'),
    dict(type='ToUnit8Evs'), 
    dict(type='EventRandAugmentEvs', num_ops=2, magnitude=10, no_geometric_trafos=True), # expects (C, H, W)
    dict(type='ToFloat32NumpyEvs'), # back to (H, W, C)
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [ 
    dict(type='LoadNpy'),  # (H, W, C) tensor
    dict(
        type='MyMultiScaleFlipAug', # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/test_time_aug.py
        img_scale=(resize_base_W, resize_base_H), 
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(resize_base_W, resize_base_H), keep_ratio=False), # (W, H)
            dict(type='ToFloat32TorchEvs'),
            dict(type='RemoveHotPixelsEvs'), # in (0.0, 255.0)
            dict(type='NormalizeEvs'),
            dict(type='ToFloat32NumpyEvs'), # back to (H, W, C)
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),  # need to crop to network input size
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data=dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    shuffle = True,
    train=dict(type='EventDataset',
            data_root = '../../../../dsec/SS_final/',
            img_dir='imgs/train',
            img_suffix='.npy',
            seg_map_suffix='.png',
            ann_dir=f'anns{numclassesstr}/train', 
            pipeline=train_pipeline,
            classes=f'../../../../datasets/dsec/SS_final/labels{numclassesstr}.txt',
            ),
    val=dict(type='EventDataset',
            data_root='../../../../datasets/dsec/SS_final/',
            img_dir='imgs/val',
            img_suffix='.npy',
            seg_map_suffix='.png',
            ann_dir=f'anns{numclassesstr}/val', 
            pipeline=test_pipeline,
            classes=f'../../../../datasets/dsec/SS_final/labels{numclassesstr}.txt',
            ),
    test=dict(type='EventDataset',
            data_root='../../../../datasets/dsec/SS_final/',
            img_dir='imgs/val',
            img_suffix='.npy',
            seg_map_suffix='.png',
            ann_dir=f'anns{numclassesstr}/val', 
            pipeline=test_pipeline, 
            classes=f'../../../../datasets/dsec/SS_final/labels{numclassesstr}.txt',
            ), 
    )
