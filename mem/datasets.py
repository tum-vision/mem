# --------------------------------------------------------
# Masked Event Modelling: Self-Supervised Pretraining for Event Cameras
# 
# Based on the following code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'
import os
import random
import numpy as np

from torchvision import datasets, transforms
import mmcv

from transforms import FixedResizeTransform, RandomResizedCropAndInterpolationWithTwoPic, CreateTwoPic, NormalizeEvent, RemoveHotPixels, RemoveTimesurface, LogTransform, GammaTransform
from transforms import EventRandAugment, ToUnit8, ToFloat32
from timm.data import create_transform

from utils import map_pixels
from masking_generator import MaskingGenerator, MaskingGeneratorRandomLocation
from dataset_folder import ImageFolder, npyFolder
from dataset_folder import imgnet_npy_loader, default_loader, caltech_npy_loader, ncars_npy_loader, dsec_npy_loader

class DataAugmentationForPT(object):
    def __init__(self, args, is_train=True):
        if args.data_set == "dsec_semseg":
            transform = build_transform_dsec(is_train, args)
        else:
            transform = build_transformNPY(is_train, args)
        
        
        self.common_transform = transforms.Compose(list(filter(lambda item: item is not None, [
            transform,
            transforms.ColorJitter(args.color_jitter, 0, args.color_jitter),
            CreateTwoPic()
        ])))

        self.patch_transform = transforms.Compose([

        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels, # This effectively does x*0.8 + 0.2
            ])
        elif args.discrete_vae_type == "event":
            self.visual_token_transform = transforms.Compose([
            ])
        else:
            raise NotImplementedError()

        if args.masking == "random":
            self.masked_position_generator = MaskingGeneratorRandomLocation(
                args.window_size, num_masking_patches=args.num_mask_patches
            )           
        elif args.masking == "block":
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        else:
            print(f"Need to chose proper masking scheme. {args.masking} does not exist.")

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), \
            self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForPT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr



class DataAugmentationForPTE2V(object):
    def __init__(self, args):
    
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=224, second_size=224,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0,0,0),
                    std=(1,1,1),
                ),
            ])
        elif args.discrete_vae_type == "event":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            raise NotImplementedError()

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForPT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr




def build_pretraining_dataset(args, is_train=True):
    transform = DataAugmentationForPT(args, is_train) if args.data_set != "IMNET" else DataAugmentationForPTE2V(args)
    print("Data Aug = %s" % str(transform))
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    if not os.path.exists(root):
        root = os.path.join(args.data_path, 'extracted_train' if is_train else 'extracted_val')
        if not os.path.exists(root):
            root = os.path.join(args.data_path, 'train_events' if is_train else 'test_events')
        assert os.path.exists(root)

    if args.data_set == "IMNET":
        ds = datasets.ImageFolder(root, transform=transform)
    elif args.data_set == "npy":
        if "caltech" in args.data_path or "Caltech" in args.data_path:
            loader = caltech_npy_loader
            print(f"using caltech 101 npy loader")
        elif "ncars" in args.data_path or "N-Cars" in args.data_path:
            loader = ncars_npy_loader
            print(f"using ncars npy loader")
        elif "imagenet" in args.data_path and "npy" in args.data_path:
            loader = imgnet_npy_loader
            print(f"using imagenet NPY loader")
        ds = npyFolder(root, transform=transform, loader=loader)
    elif args.data_set == "dsec_semseg":
        assert "dsec" in args.data_path or "DSEC" in args.data_path or "SS_final" in args.data_path
        loader = dsec_npy_loader
        print(f"using DSEC npy loader")
        ds = npyFolder(root, transform=transform, loader=loader, is_valid_file=is_valid_file_dsec)
    return ds

def is_valid_file_dsec(path):
    if ".npy" in path:
        return True
    else:
        return False

def is_valid_file_mvsec(path):
    if "indoor" in path: # or "indoor_flying1" in path or "indoor_flying2" in path or "indoor_flying3" in path:
        fn = int(path.split("/")[-1][:-4])
        if fn > 150:
            return True
    else: 
        return False

class EventPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=np.random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                alpha=np.random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=np.random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, img):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img = np.moveaxis(img, 0, -1) # (C, H, W) => (H, W, C)
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)
        
        return img # (H, W, C)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

def build_transform_dsec(is_train, args):
    t = []    
    
    t += [ SliceRandomMaxEvs(args.slice_max_evs) ]

    H, W = 440, 640
    if is_train:
        DataAugmentations = [
            RandomTimeFlip(),
            Aug_FlipEvsAlongX(H=H, W=W),
            Aug_RandomShiftEvs(H=H, W=W, max_shift=args.max_random_shift_evs)
        ]
        t += DataAugmentations

    t += [ EventArrToImg(H, W) ]
    t += [ transforms.ToTensor() ]

    t += [ transforms.Resize((args.input_H, args.input_W), transforms.InterpolationMode.BILINEAR, antialias=True) ]

    if not args.timesurface:
        t.append(RemoveTimesurface())
    if args.hotpixfilter:
        t.append(RemoveHotPixels(num_stds=args.hotpix_num_stds))
    if args.logtrafo:
        t.append(LogTransform())
    if args.gammatrafo:
        t.append(GammaTransform(args.gamma))
    if args.normalize_events:
        t.append(NormalizeEvent())

    if is_train and args.rand_aug:
        t.append(ToUnit8())
        t.append(EventRandAugment(small=False, magnitude=20))
        t.append(ToFloat32())

    return transforms.Compose(t)

def build_transform_e2v2(is_train, args):
    input_size = (args.input_H, args.input_W)
    t = []
    if args.resize:
        t.append(FixedResizeTransform(2))
        t.append(transforms.RandomCrop(input_size, pad_if_needed=True))
    else:
        t.append(transforms.RandomCrop(input_size, pad_if_needed=True))
    
    if is_train:
        t.append(transforms.RandomHorizontalFlip(p=0.5))

    t.append(transforms.ToTensor())


    return transforms.Compose(t)



def build_transform_e2v(is_train, args):
    resize_im = args.input_size > 32
    mean = (0, 0, 0)
    std = (1, 1, 1)

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        args.crop_pct = None
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)    


def build_dataset(is_train, args):
    if args.data_set == "dsec_semseg":
        transform = build_transform_dsec(is_train, args)
    elif args.data_set == "IMNET":
        transform = build_transform_e2v(is_train, args)
    else:
        transform = build_transformNPY(is_train, args)


    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'IMNET': # splits in train/val
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if not os.path.exists(root):
            root = os.path.join(args.data_path, 'extracted_train' if is_train else 'extracted_val')
            assert os.path.exists(root)
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.data_set == "image_folder": # takes all subfolders
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
    elif args.data_set == "npy":
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if not os.path.exists(root):
            root = os.path.join(args.data_path, 'extracted_train' if is_train else 'extracted_val')
        if not os.path.exists(root):
            root = os.path.join(args.data_path, 'training' if is_train else 'validation')
        if not os.path.exists(root):
            raise ValueError(f"Path {root} does not exist.")

        if "caltech" in args.data_path or "Caltech" in args.data_path:
            loader = caltech_npy_loader
            print(f"using caltech 101 npy loader")
        elif "ncars" in args.data_path or "N-Cars" in args.data_path:
            loader = ncars_npy_loader
            print(f"using ncars npy loader")
        elif "imagenet" in args.data_path:
            loader = imgnet_npy_loader
            print(f"using imagenet NPY loader")
        dataset = npyFolder(root, transform=transform, loader=loader)
    elif args.data_set == "dsec_semseg":
        assert "dsec" in args.data_path or "DSEC" in args.data_path or "SS_final" in args.data_path or "raw_npy" in args.data_path
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        if not os.path.exists(root):
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            assert os.path.exists(root)
        loader = dsec_npy_loader
        print(f"using DSEC npy loader")
        dataset = npyFolder(root, transform=transform, loader=loader, is_valid_file=is_valid_file_dsec)
    else:
        raise NotImplementedError()

    if args.data_set != "mvsec":
        nb_classes = len(dataset.class_to_idx)
    else:
        nb_classes = 0
    
    print("Number of classses = %d" % nb_classes)
    return dataset, nb_classes


class ReshapeScaleXandY():
    def __init__(self, newH=224, newW=224, oldH=480, oldW=640, is_train=False):
        assert newH >= 100 and newW >= 100
        assert newH <= 640 and newW <= 640
        assert oldH >= 100 and oldW >= 100
        assert oldH <= 640 and oldW <= 640

        if is_train:
            oldHWs = [oldH, oldW]
            minsize = np.argmin(oldHWs)

            scale = 256 / oldHWs[minsize]
            self.scale_x = scale
            self.scale_y = scale
        else:
            self.scale_x = (newW/oldW)
            self.scale_y = (newH/oldH)

    def __call__(self, x):
        x[:, 0] *= self.scale_x
        x[:, 1] *= self.scale_y
        return x


class SliceRandomMaxEvs():
    def __init__(self, keep_max_num_evs=30000):
        self.keep_max_N_evs = keep_max_num_evs
        assert keep_max_num_evs >= 5000 and keep_max_num_evs < 200000
        print(f"Slicing max {keep_max_num_evs} num evs.")

    def __call__(self, x):
        if len(x) > self.keep_max_N_evs:
            rand_start = random.choice(range(len(x) - self.keep_max_N_evs + 1)) # (0, ... , L-max_N)
            x = x[rand_start:(rand_start+self.keep_max_N_evs), :]
        return x
    

class Aug_FlipEvsAlongX():
    def __init__(self, H=None, W=None, p=0.5):
        if H is not None:
            assert H >= 100 and H <= 640
        if W is not None:
            assert W >= 100 and W <= 640
        assert p >= 0.0 and p <= 1.0
        self.H = H
        self.W = W
        self.p = p
        
    
    def __call__(self, x): 
        W = self.W
        if self.W is None:
            W = x[:, 0].max().astype(np.int64) + 1

        if np.random.random() < self.p:
            x[:, 0] = W - 1 - x[:, 0]

        return x


class Aug_RandomShiftEvs():
    def __init__(self, H=None, W=None, max_shift=20):
        if H is not None:
            assert H >= 100 and H <= 640
        if W is not None:
            assert W >= 100 and W <= 640
        assert max_shift >= 0 and max_shift <= 200
        self.H = H
        self.W = W
        self.max_shift = max_shift
    
    def __call__(self, x): 
        H, W = self.H, self.W
        if W is None:
            W = x[:, 0].max().astype(np.int64) + 1
        if H is None:
            H = x[:, 1].max().astype(np.int64) + 1

        x_shift, y_shift = np.random.randint(-self.max_shift, self.max_shift + 1, size=(2,))
        x[:, 0] += x_shift
        x[:, 1] += y_shift   

        valid_events = (x[:, 0] >= 0) & (x[:, 0] < W) & (x[:, 1] >= 0) & (x[:, 1] < H)
        x = x[valid_events]
        
        return x


class EventArrToImg():
    def __init__(self, H=None, W=None, timeSurface=False):
        if H is not None:
            assert H >= 100 and H <= 640
        if W is not None:
            assert W >= 100 and W <= 640
        self.H = H
        self.W = W
        self.mean = 0
        self.counter = 0
        self.timeSurface = timeSurface
        if self.timeSurface:
            print(f"Using Time Surface!")

    def __call__(self, x):
        xs, ys, ts, ps = x.T
        xs = xs.astype(np.int)
        ys = ys.astype(np.int)

        H, W = self.H, self.W
        if W is None:
            W = (xs.max()).astype(np.int64) + 1
        if H is None:
            H = (ys.max()).astype(np.int64) + 1

        img_pos = np.zeros((H*W,), dtype=np.uint8)
        img_tss = np.zeros((H*W,), dtype=np.uint8)
        img_neg = np.zeros((H*W,), dtype=np.uint8)

        np.add.at(img_pos, xs[ps == 1] + W * ys[ps == 1], 1)
        np.add.at(img_neg, xs[ps == -1] + W * ys[ps == -1], 1)
        
        # inspired by 
        # https://github.com/uzh-rpg/rpg_ev-transfer/blob/8aec426f8548a84262da46c07d4d5d475224edd8/datasets/data_util.py
        # similar to https://github.com/82magnolia/n_imagenet/blob/a0b1f7338e427688b89495a90cf6f00f0422a5c1/real_cnn_model/data/imagenet.py#L282
        if self.timeSurface:
            idxs = (xs + ys * W)
            ts_norm = (ts-ts.min())
            img_tss[idxs] = (ts_norm / ts_norm.max() * 255) 
            hist = np.stack([img_pos, img_tss, img_neg]).reshape((3, H, W)).transpose(1, 2, 0) # (H, W, 3)
        else:
            hist = np.stack([img_pos, img_tss, img_neg]).reshape((3, H, W)).transpose(1, 2, 0) # (H, W, 3)

        return hist # (H, W, 3)        


class RandomTimeFlip():
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x): 
        # (N_evs, 4)
        if np.random.random() < self.p:
            x = np.flip(x, axis=0)
            x[:, 2] = x[0, 2] - x[:, 2]
            x[:, 3] = - x[:, 3]  # Inversion in time means inversion in polarity
        
        return x

def build_transformNPY(is_train, args):
    t = []

    H, W = None, None # variable H, W
    if "imagenet" in args.data_path:
        H, W = args.input_H, args.input_W # fixed H, W
        t += [ ReshapeScaleXandY(newH=H, newW=W, oldH=480, oldW=640, is_train=is_train) ]
        if is_train:
            scale = 256 / 480
            H = int(480 * scale)
            W = int(640 * scale)
    elif "SS_final" in args.data_path or "dsec" in args.data_path or "DSEC" in args.data_path:
        H, W = 440, 640

    t += [ SliceRandomMaxEvs(args.slice_max_evs) ]

    if is_train:
        DataAugmentations = [
            RandomTimeFlip(),
            Aug_FlipEvsAlongX(H=H, W=W),
            Aug_RandomShiftEvs(H=H, W=W, max_shift=args.max_random_shift_evs),
        ]
        t += DataAugmentations

    t += [ EventArrToImg(H, W, args.timesurface) ]

    t += [ transforms.ToTensor() ]
    if "caltech" in args.data_path or "Caltech" in args.data_path or "ncars" in args.data_path or "N-Cars" in args.data_path or "SS_final" in args.data_path or "dsec" in args.data_path or "DSEC" in args.data_path:
        t += [ transforms.Resize((args.input_H, args.input_W), transforms.InterpolationMode.BILINEAR, antialias=True) ]
    
    if is_train:
        t += [ transforms.RandomCrop((args.input_H, args.input_W), pad_if_needed=True) ]

    if not args.timesurface:
        t.append(RemoveTimesurface())
    if args.hotpixfilter:
        t.append(RemoveHotPixels(num_stds=args.hotpix_num_stds))
    if args.logtrafo:
        t.append(LogTransform())
    if args.gammatrafo:
        t.append(GammaTransform(args.gamma))
    if args.normalize_events:
        t.append(NormalizeEvent())

    if is_train and args.rand_aug:
        t.append(ToUnit8())
        t.append(EventRandAugment(small=False, magnitude=20))
        t.append(ToFloat32())

    return transforms.Compose(t)
