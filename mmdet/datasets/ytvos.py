import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
import random


@DATASETS.register_module()
class YTVOSDataset(CustomDataset):
    CLASSES=('person','giant_panda','lizard','parrot','skateboard','sedan',
        'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
        'train','horse','turtle','bear','motorbike','giraffe','leopard',
        'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
        'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
        'tennis_racket')
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

         # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(self.ann_file)

        img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                img_ids.append((idx, frame_id))
        self.img_ids = img_ids
        # filter data infos if classes are customized
        if self.custom_classes:
            self.data_infos = self.get_subset_by_classes()

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)


    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(self.img_ids[idx])
        while True:
            data = self.prepare_train_img(self.img_ids[idx])
            if data is None:
                idx = self._rand_another(self.img_ids[idx])
                continue
            
            return data

    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i+1 # TODO:
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    #TODO:
    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]
        
        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:,:2]+ bbox[:,2:])/2.
        sizes = bbox[:,2:] - bbox[:,:2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes/2.
        new_x2y2 = new_centers + new_sizes/2.
        c_min = [0,0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1,new_x2y2)).astype(np.float32)
        return bbox

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in sample_range:
          # check if the frame id is valid
          ref_idx = (vid, i)
          if i != frame_id and ref_idx in self.img_ids:
              valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def prepare_train_img(self, idx):
        # prepare a pair of image in a sequence
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        img_info = vid_info
        _, ref_frame_id = self.sample_ref(idx)
        
        #TODO:
        # frame_id = 2
        # ref_frame_id = 3
        
        ann = self.get_ann_info(vid, frame_id)
        ref_ann = self.get_ann_info(vid, ref_frame_id)
        # print(f"annotation {ann}")
        # print(f"ref annotation {ref_ann}")
        results = dict(
            img_info=img_info,
            vid_info=vid_info, 
            ann_info=ann,
            ref_ann_info=ref_ann,
            frame_id=frame_id,
            ref_frame_id=ref_frame_id)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        img_info = vid_info
        results = dict(
            img_info=img_info,
            vid_info=vid_info,
            frame_id=frame_id,
            ref_frame_id=None)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                mask_polys = [
                    p for p in segm if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, obj_ids=gt_ids, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens


        return ann
