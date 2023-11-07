# --------------------------------------------------------
# Masked Event Modelling: Self-Supervised Pretraining for Event Cameras
# 
# Based on the following code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'
# recommand use this config for mem models which are self-supervised pretrained and then intermediate fine-tuned on imagenet
_base_ = [
    '../../_base_/datasets/dsec.py', 
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]
seed = 0
norm_cfg = dict(type='SyncBN', requires_grad=True) # SyncBN  BN3d

crop_size = (512, 512)

drop = 0.1
VIT = "base"
if "base" in VIT:
    embed_dim = 768
    num_heads = 12
elif "tiny" in VIT:
    embed_dim = 192
    num_heads = 3
elif "small" in VIT:
    embed_dim = 384
    num_heads = 6

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='EvBEiT',
        img_size=crop_size[0],
        patch_size=16,
        embed_dim=embed_dim, # 768, 192
        depth=12,
        num_heads=num_heads, # 12, 3
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=0.1,
        drop_path_rate=drop,
        out_indices=[8, 9, 10, 11],
        qkv_bias_timm = False,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[embed_dim, embed_dim, embed_dim, embed_dim], 
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        num_classes=11,
        channels=512,
        dropout_ratio=drop,
        norm_cfg=norm_cfg,
        align_corners=False,
            loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=embed_dim,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=drop,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

load_from = None

optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
# fp16 = True
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
