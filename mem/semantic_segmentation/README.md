# ADE20k Semantic segmentation with BEiT

## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.7.0 mmsegmentation==0.30.0 mmengine==0.7.0 
pip install scipy timm==0.4.12
```

## Fine-tuning

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --options model.pretrained=<CHECKPOINT_PATH>
```

---

## Acknowledgment 

Code based on the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository.
This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository, [XCiT](https://github.com/facebookresearch/xcit) and the [SETR](https://github.com/fudan-zvg/SETR) repository.
