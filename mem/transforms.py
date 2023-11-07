# --------------------------------------------------------
# Masked Event Modelling: Self-Supervised Pretraining for Event Cameras
# 
# Based on the following code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import warnings
import math
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import copy

class ToNumpy:
    """Convert torch.Tensor (C, H, W) in (0,1) => numpy (C, H, W) in (0, 255)"""

    def __init__(self):
        pass

    def __call__(self, x):
        x = x.numpy() # (C, H, W)
        return x * 255 


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)


_pil_interpolation_to_str = {
    InterpolationMode.NEAREST: 'PIL.Image.NEAREST',
    InterpolationMode.BILINEAR: 'PIL.Image.BILINEAR',
    InterpolationMode.BICUBIC: 'PIL.Image.BICUBIC',
    InterpolationMode.LANCZOS: 'PIL.Image.LANCZOS',
    InterpolationMode.HAMMING: 'PIL.Image.HAMMING',
    InterpolationMode.BOX: 'PIL.Image.BOX',
}


def _pil_interp(method):
    if method == 'bicubic':
        return InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return InterpolationMode.LANCZOS
    elif method == 'hamming':
        return InterpolationMode.HAMMING
    else:
        return InterpolationMode.BILINEAR


_RANDOM_INTERPOLATION = (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)

class CreateTwoPic:
    def __init__(self):
        pass

    def __call__(self, img):
        return img, img

class RandomResizedCropAndInterpolationWithTwoPic:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear', second_interpolation='lanczos'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = _pil_interp(second_interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation), \
                   F.resized_crop(img, i, j, h, w, self.second_size, self.second_interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0}'.format(interpolate_str)
        if self.second_size is not None:
            format_string += ', second_size={0}'.format(self.second_size)
            format_string += ', second_interpolation={0}'.format(_pil_interpolation_to_str[self.second_interpolation])
        format_string += ')'
        return format_string

class FixedResizeTransform:
    """Rotate by one of the given angles."""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return F.resize(x, (int(x.size[1]/self.factor),int(x.size[0]/self.factor)),interpolation=InterpolationMode.BILINEAR)


# x: (C, H, W). C0 = pos, C1 = time, C2 = neg
class LogTransform:
    """Log-transform an event Image, pos and neg channel separately."""

    def __init__(self):
        pass

    def __call__(self, x):
        ones = torch.ones(x[0:1,:,:].shape)
        x[0:1,:,:] = torch.log(x[0:1,:,:]+ones)
        x[2:3,:,:] = torch.log(x[2:3,:,:]+ones)
        return x.float()

class GammaTransform:
    """Gamma-transform an event Image, pos and neg channel separately."""

    def __init__(self, gamma=0.5):
        self.gamma = gamma
        pass

    def __call__(self, x):
        x[0:1,:,:] = x[0:1,:,:]**self.gamma
        x[2:3,:,:] = x[2:3,:,:]**self.gamma
        return x.float()
   
# x: (C, H, W)
class NormalizeEvent:
    """Normalize a event Image.
        x = (3, H, W) -> or (W, H)?
    """

    def __init__(self):
        pass

    def __call__(self, x):
        if x[0::2,:,:].max() != 0:
            factor = 1.0 / x[0::2,:,:].max()
            x[0::2,:,:] = x[0::2,:,:] * factor
        return x.float()

class RemoveTimesurface:
    """Remove the green dimention of a event Image."""

    def __init__(self):
        pass

    def __call__(self, x):
        x[1,:,:]= 0.0
        return x.float()

class RemoveHotPixels:
    """Remove hot pixels."""

    def __init__(self, num_stds=10, num_hot_pixels=None):
        self.num_stds = num_stds
        self.num_hot_pixels = num_hot_pixels

    def detect_hot_pixels(self, hist_yx, num_hot_pixels=None, num_stds=None):
        hist_yx_flatten = hist_yx[0::2,:,:].flatten() # (2*H*W)

        if num_hot_pixels is not None:
            if num_hot_pixels >= hist_yx[0::2,:,:].sum() / 4:
                num_hot_pixels = hist_yx[0::2,:,:].sum() / 4
                print(f"Setting num_hot_pixels to {num_hot_pixels}")
            hot_pixel_inds = torch.atleast_1d(torch.argsort(hist_yx_flatten)[len(hist_yx_flatten)-int(num_hot_pixels):])
        else:
            mean, std = torch.mean(hist_yx[0::2,:,:]), torch.std(hist_yx[0::2,:,:])
            threshold_filter = mean + num_stds * std
            hot_pixel_inds = torch.atleast_1d(torch.squeeze(torch.argwhere(hist_yx_flatten > threshold_filter)))

        return hot_pixel_inds

    def __call__(self, x):
        hot_pixel_inds = self.detect_hot_pixels(x, num_stds=self.num_stds, num_hot_pixels=self.num_hot_pixels)
        hot_pixel_index_2d = np.asarray(np.unravel_index(hot_pixel_inds, x.shape)).T
        x[0::2, hot_pixel_index_2d[:, 1], hot_pixel_index_2d[:, 2]] = 0
        return x

class EventJitter:
    """Rotate by one of the given angles."""

    def __init__(self, factor=0.1, dropout=0.8):
        self.factor = factor
        self.dropout = dropout

    def __call__(self, x):
        with torch.no_grad():
            jitter = torch.clone(x) * self.factor
            jitter = torch.nn.functional.dropout(jitter, p=self.dropout, training=False)
            jitter = jitter * (torch.rand(x.shape) - 0.5)

            return x + jitter

def _apply_op(img: torch.Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img

class ToFloat32:
    """Convert numpy in (0, 255) => torch.Tensor in (0,1)"""

    def __init__(self):
        pass

    def __call__(self, x):
        return x.to(torch.float32)/255

class ToUnit8:
    """Remove the green dimention of a event Image."""

    def __init__(self):
        pass

    def __call__(self, x):
        return (255*x).to(torch.uint8)


class EventRandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
        verbose: bool = False,
        gen: Optional[torch.Generator] = None, 
        small=False
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.verbose = verbose
        self.gen = gen

        if small:
            self.names = ['Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize']
        else:
            self.names = ['Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness', 'Color', \
                'Contrast', 'Sharpness', 'Posterize', 'Solarize', 'AutoContrast', 'Equalize']

        print(f"Created RandAug with {self.names}")

    def set_names(self, x):
        if isinstance(x, list):
            self.names = copy.copy(x)
        elif isinstance(x, str):
            if x == "all":
                self.names = ['Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize', 'Solarize', 'AutoContrast', 'Equalize']
            elif x == "small":
                self.names = ['Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize']
            else:
                raise RuntimeError(f"Not implemented for {x}")
        else:
            raise RuntimeError(f"Not implemented for {x}")
    
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        d = {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        
        return {k: v for k, v in d.items() if k in self.names}

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        assert img.dtype == torch.uint8
        
        fill = self.fill
        channels, height, width = list(img.shape)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,), generator=self.gen).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            
            # Seems stupid but ensures that two instances behave similarly
            randi0 = torch.randint(self.magnitude+1, (1,), generator=self.gen).item()
            randi1 = torch.randint(2, (1,), generator=self.gen)
            
            if magnitudes.ndim > 0:
                magnitude = float(magnitudes[randi0].item())
            else:
                magnitude = 0.0
                
            if signed and randi1:
                magnitude *= -1.0
                
            if self.verbose:
                print(f"EventRandAug: {op_name:20s} {magnitude:7.2f}")
                
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s
