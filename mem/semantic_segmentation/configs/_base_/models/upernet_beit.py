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
norm_cfg = dict(type='SyncBN', requires_grad=True) # SyncBN  BN3d
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='XCiT',
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
# https://github.com/open-mmlab/mmsegmentation/blob/a82ebad0f6aa3cc88452d7442730d60f27c41153/mmseg/models/losses/cross_entropy_loss.py#L12
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)), 
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
