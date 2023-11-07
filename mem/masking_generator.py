# --------------------------------------------------------
# Masked Event Modelling: Self-Supervised Pretraining for Event Cameras
# 
# Based on the following code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/microsoft/unilm/tree/master/beit
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
import random
import math
import numpy as np

class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect  # is alway 1/0.3 = 3
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask



class MaskingGeneratorRandomLocation:
    def __init__(
            self, input_size, num_masking_patches):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        print(f"Masking Ration for RandomLocation-Masker is = {self.num_masking_patches/self.num_patches}")

        assert self.num_masking_patches < self.num_patches


    def __repr__(self):
        repr_str = "Generator(patchesY: %d, patchesX %d, numMaskingPatches: %d" % (self.height, self.width, self.num_masking_patches)
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def __call__(self):
        mask = np.zeros(shape=self.height * self.width, dtype=np.int)
        
        max_idx = self.height * self.width - 1
        all_idxs = np.arange(max_idx) # 0, ..., max_idx - 1
        idxs = random.sample(sorted(all_idxs), self.num_masking_patches) # performs rand sampling without replacement
        
        mask[idxs] = 1
        mask = mask.reshape((self.height, self.width))
    
        return mask
