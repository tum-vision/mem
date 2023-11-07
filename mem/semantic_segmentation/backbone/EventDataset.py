import numpy as np
from mmseg.datasets import PIPELINES
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose, LoadAnnotations

import os.path as osp
import warnings
from collections import OrderedDict
import random
import torch
import torchvision.transforms.functional as F
import copy
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import matplotlib
import math 
from torchvision.transforms import InterpolationMode
matplotlib.use('Agg')

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger


def render(x, y, pol, H, W):
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img


@DATASETS.register_module()
class EventDataset():
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict, optional): build LoadAnnotations to
            load gt for evaluation, load from disk by default. Default: None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.npy',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend='disk')):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
        ) if gt_seg_map_loader_cfg is None else LoadAnnotations(
            **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                # if "train" in img_dir and int(img[-10:-4]) % 10 == 0:
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results



@PIPELINES.register_module()
class RandomTimeFlip():
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x): 
        if np.random.random() < self.p:
            x = np.flip(x, axis=0)
            x[:, 2] = x[0, 2] - x[:, 2]
            x[:, 3] = - x[:, 3]  # Inversion in time means inversion in polarity
        
        return x


@PIPELINES.register_module()
class RemoveHotPixels:
    """Remove hot pixels."""

    def __init__(self, num_stds=10, num_hot_pixels=None):
        self.num_stds = num_stds
        self.num_hot_pixels = num_hot_pixels

    def detect_hot_pixels(self, hist_yx, num_hot_pixels=None, num_stds=None):
        # hist_yx: (C, H, W)
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

@PIPELINES.register_module()
class RemoveHotPixelsEvs:
    """Remove hot pixels."""

    def __init__(self, num_stds=10, num_hot_pixels=None):
        self.num_stds = num_stds
        self.num_hot_pixels = num_hot_pixels

    def detect_hot_pixels(self, hist_yx, num_hot_pixels=None, num_stds=None):
        # hist_yx: (C, H, W)
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

    def __call__(self, results):
        x = results['img']
        hot_pixel_inds = self.detect_hot_pixels(x, num_stds=self.num_stds, num_hot_pixels=self.num_hot_pixels)
        hot_pixel_index_2d = np.asarray(np.unravel_index(hot_pixel_inds, x.shape)).T
        x[0::2, hot_pixel_index_2d[:, 1], hot_pixel_index_2d[:, 2]] = 0

        results['img'] = x
        return results


@PIPELINES.register_module()
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
        return x


@PIPELINES.register_module()
class SliceRandomMaxEvs():
    def __init__(self, keep_max_num_evs=30000):
        self.keep_max_N_evs = keep_max_num_evs
        assert keep_max_num_evs >= 5000 and keep_max_num_evs < 2000000

    def __call__(self, x):
        if len(x) > self.keep_max_N_evs:
            rand_start = random.choice(range(len(x) - self.keep_max_N_evs + 1)) # (0, ... , L-max_N)
            x = x[rand_start:(rand_start+self.keep_max_N_evs), :]
        return x
    
@PIPELINES.register_module()
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
    

@PIPELINES.register_module()
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
        # x = np.array (N_evs, 4). float64
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

import os

@PIPELINES.register_module()
class SaveToDisk():
    def __init__(self, val=True, dir="/usr/wiss/klenk/Documents/default_vizDataSS_TrainPipelinePhoto/"):

            self.i = 0
            self.dir = dir
            self.val = val
            os.makedirs(os.path.join(self.dir, "evs"), exist_ok=True)
            if not self.val:
                os.makedirs(os.path.join(self.dir, "ans"), exist_ok=True)

    def __call__(self, results):
        img = results["img"]
            
        plt.figure()    
        plt.imshow(img.astype(np.uint8))
        plt.savefig(f"{os.path.join(self.dir, 'evs', str(self.i)+'.png')}")
        plt.close()   
    
        if not self.val:
            ann = results["gt_semantic_seg"]
            plt.figure()    
            plt.imshow(ann.astype(np.uint8))
            plt.savefig(f"{os.path.join(self.dir, 'ans', str(self.i)+'.png')}")
            plt.close()
        self.i += 1
        
        return results


@PIPELINES.register_module()
class LoadNpy():
    def __init__(self,
                    to_float32=True,
                    color_type='color',
                    file_client_args=dict(backend='disk'),
                    imdecode_backend='cv2'):
            self.to_float32 = to_float32
            self.color_type = color_type
            self.file_client_args = file_client_args.copy()
            self.file_client = None
            self.imdecode_backend = imdecode_backend
            self.i = 0

            H, W = 440, 640
            self.SliceRandomMaxEvs = SliceRandomMaxEvs(180000)
            self.EventArrToImg = EventArrToImg(H, W)


    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = np.load(filename).astype(np.float32)
        img[:, 3] = 2*img[:, 3]-1  # ps in (-1, 1)
        # cut the event data 
        keep = img[:, 1] < 440 # due to front of car
        img = img[keep]
        trafos = [self.SliceRandomMaxEvs]
        trafos += [self.EventArrToImg]

        for t in trafos:
            img = t(img)

        img = img.astype(np.uint8)  # should be (440, 640, 3) here!

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results


@PIPELINES.register_module()
class EventArrToImg():
    def __init__(self, H=None, W=None):
        if H is not None:
            assert H >= 100 and H <= 640
        if W is not None:
            assert W >= 100 and W <= 640
        self.H = H
        self.W = W

    def __call__(self, x): 
        # x = torch (N_evs, 4). float64
        xs, ys, _, ps = x.T
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
        hist = np.stack([img_pos, img_tss, img_neg]).reshape((3, H, W)).transpose(1, 2, 0) # (H, W, 3)

        return hist # (H, W, 3)  img values in (0,...,max_event_count)



@PIPELINES.register_module()
class LoadAnnFloat(object):
    """Load annotations for semantic segmentation.
    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.float32)  # (440, 640)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str




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


class ToNumpy:
    """Convert torch.Tensor (H, W, C) in (0,1) to numpy (C, H, W) in (0, 255)"""

    def __init__(self):
        pass

    def __call__(self, x):
        x = x.numpy()
        x = x.transpose((1, 2, 0)) # (C, H, W)
        return x * 255 

class ToFloat32:
    """Remove the green dimention of a event Image."""

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





@PIPELINES.register_module()
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

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class MyMultiScaleFlipAug:
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=[(1333, 400), (1333, 800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    """

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        assert (img_scale is None) ^ (scale_factor is None), (
            'Must have but only one variable can be set')
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            self.img_scale = scale_factor if isinstance(
                scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """
        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list

        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str


@PIPELINES.register_module()
class ResizeImgs(object):
    """Resize images 
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 min_size=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.min_size = min_size

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            if self.min_size is not None:
                if min(results['scale']) < self.min_size:
                    new_short = self.min_size
                else:
                    new_short = min(results['scale'])

                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio


    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str



@PIPELINES.register_module()
class NormalizeEvs(object):
    """Normalizes event images & leaves segs.
    This transform Normalizes the input image to max_value == 1.0
    """

    def __init__(self):
        pass

    def __call__(self, results):
        max = 1.0
        if results['img'].max() != 0:
            max = results['img'].max()

        results['img'] = results['img'] / max * 255. 

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'normalizing (resized) event histogram')
        return repr_str

@PIPELINES.register_module()
class EventRandAugmentEvs(torch.nn.Module):
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
        small=False, 
        no_geometric_trafos=False
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.verbose = verbose
        self.gen = gen

        self.no_geometric_trafos = no_geometric_trafos

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
        if self.no_geometric_trafos:
            d = {
                "Identity": (torch.tensor(0.0), False),
                "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
                "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
                "AutoContrast": (torch.tensor(0.0), False),
                "Equalize": (torch.tensor(0.0), False),
            }
        else:
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

    def forward(self, results):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        img = results['img']
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

        results['img'] = img
        return results


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


@PIPELINES.register_module()
class ToFloat32TorchEvs():
    def __init__(self):
        pass
    
    def __call__(self, x):
        x['img'] = torch.from_numpy(x['img']).to(torch.float32).permute((2, 0, 1))
        return x

@PIPELINES.register_module()
class ToUnit8Evs():
    def __init__(self):
        pass
    
    def __call__(self, x):
        # x is (H, W, C) => (H, W, C) out for EventRandAugmentEvs()
        # x is in (0, 255) => output as well, which we want for EventRandAugmentEvs()
        x['img'] = x['img'].to(torch.uint8)

        return x
    
@PIPELINES.register_module()
class ToFloat32NumpyEvs():
    def __init__(self):
        pass
    
    def __call__(self, x): 
        # input x is (C, H, W). output x is (H, W, C)
        # x is in (0.0, 255.0)
        x['img'] = np.asarray(x['img']).astype(np.float32).transpose((1, 2, 0))
        return x
